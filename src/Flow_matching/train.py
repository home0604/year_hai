# -*- coding: utf-8 -*-
"""
Flow Matching DiT training script for 10x10 patch-antenna inverse design.

Supports:
  --arch: smalldit (AR-comparable) / stable3dit (SD3-style)
  --flow_type: cfm (continuous) / dfm (discrete)

Experiment layout:
  experiments/YYYYMMDD/exp_name/
    args.json, run.sh, train.log, best_model.pth, last_model.pth
"""
import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import wandb
import random

from model import build_model
from flow_matching import (
    bits01_to_pm1, cfm_loss, sample_fm_cfg,
    cfm_xpred_ste_loss, sample_fm_xpred_cfg,
    dfm_loss, sample_dfm_cfg, tokens_to_bits,
    dfm_mask_loss, sample_dfm_mask_cfg,
    make_spatial_patch_indices,
)


class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1)  # line-buffered

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_exp_dir(args):
    """Create experiments/YYYYMMDD/exp_name/ and save args + run command."""
    date_str = datetime.now().strftime("%Y%m%d")
    exp_dir = os.path.join(args.exp_root, date_str, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    cmd = " ".join(sys.argv)
    with open(os.path.join(exp_dir, "run.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# {datetime.now().isoformat()}\n")
        f.write(f"cd {os.getcwd()}\n")
        cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_dev:
            f.write(f"export CUDA_VISIBLE_DEVICES={cuda_dev}\n")
        f.write(f"python -u {cmd}\n")

    sys.stdout = Tee(os.path.join(exp_dir, "train.log"))

    return exp_dir


# ---------------------------
# EMA (Exponential Moving Average)
# ---------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach()
                       for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        """Swap model params with EMA params. Call again to restore."""
        for name, p in model.named_parameters():
            p.data, self.shadow[name] = self.shadow[name], p.data

    def state_dict(self):
        return self.shadow


# ---------------------------
# Seed
# ---------------------------
def set_seed(seed: int):
    print(f"Setting global seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def restore_pixel_order(X_flat, height, width):
    N, num_bits = X_flat.shape
    assert num_bits == height * width
    return X_flat.reshape(N, 1, height, width).astype(np.float32)


# ---------------------------
# Canonicalization (y-axis symmetry)
# ---------------------------
def horizontal_flip_y_axis(X_flat, height, width):
    N, num_bits = X_flat.shape
    assert num_bits == height * width
    X_reshaped = X_flat.reshape(N, height, width)
    X_flipped = X_reshaped[:, :, ::-1]
    return X_flipped.reshape(N, num_bits)


def canonicalize_under_yflip(X_flat, height, width):
    X_flat = X_flat.copy()
    X_flip = horizontal_flip_y_axis(X_flat, height, width)
    N = X_flat.shape[0]
    X_can = np.empty_like(X_flat)
    for i in range(N):
        a, b = X_flat[i], X_flip[i]
        X_can[i] = a if list(a) <= list(b) else b
    return X_can


# ---------------------------
# Train / Eval (flow_type-aware)
# ---------------------------
def _cfm_loss_dispatch(model, yb, x1, sigma_min, group_size,
                       cfm_param, lambda_ste, ste_t_weight):
    if cfm_param == "xpred_ste":
        return cfm_xpred_ste_loss(model, yb, x1, lambda_ste=lambda_ste,
                                  ste_t_weight=ste_t_weight, group_size=group_size)
    return cfm_loss(model, yb, x1, sigma_min=sigma_min, group_size=group_size)


def train_one_epoch(model, loader, optimizer, device, flow_type, sigma_min,
                    ot_coupling=False, group_size=1, flip_pred=False,
                    cfm_param="velocity", lambda_ste=0.0, ste_t_weight=True):
    model.train()
    total_loss, total = 0.0, 0
    for yb, xb in loader:
        yb = yb.to(device)
        xb = xb.to(device).float()
        optimizer.zero_grad()
        if flow_type == "cfm":
            x1 = bits01_to_pm1(xb)
            loss = _cfm_loss_dispatch(model, yb, x1, sigma_min, group_size,
                                      cfm_param, lambda_ste, ste_t_weight)
        elif flow_type == "dfm_mask":
            loss = dfm_mask_loss(model, yb, xb, group_size=group_size)
        else:  # dfm
            loss = dfm_loss(model, yb, xb, ot=ot_coupling, group_size=group_size,
                            flip_pred=flip_pred)
        loss.backward()
        optimizer.step()
        B = xb.size(0)
        total_loss += loss.item() * B
        total += B
    return total_loss / total


@torch.no_grad()
def eval_loss(model, loader, device, flow_type, sigma_min,
              ot_coupling=False, group_size=1, flip_pred=False,
              cfm_param="velocity", lambda_ste=0.0, ste_t_weight=True):
    model.eval()
    total_loss, total = 0.0, 0
    for yb, xb in loader:
        yb = yb.to(device)
        xb = xb.to(device).float()
        if flow_type == "cfm":
            x1 = bits01_to_pm1(xb)
            loss = _cfm_loss_dispatch(model, yb, x1, sigma_min, group_size,
                                      cfm_param, lambda_ste, ste_t_weight)
        elif flow_type == "dfm_mask":
            loss = dfm_mask_loss(model, yb, xb, group_size=group_size)
        else:  # dfm
            loss = dfm_loss(model, yb, xb, ot=ot_coupling, group_size=group_size,
                            flip_pred=flip_pred)
        B = xb.size(0)
        total_loss += loss.item() * B
        total += B
    return total_loss / total


def _sample_fn(model, y, num_bits, steps, guidance_scale, flow_type,
               group_size=1, flip_pred=False, cfm_param="velocity"):
    """Dispatch to cfm or dfm sampling."""
    if flow_type == "cfm":
        if cfm_param == "xpred_ste":
            return sample_fm_xpred_cfg(model, y, num_bits, steps, guidance_scale,
                                       group_size=group_size)
        return sample_fm_cfg(model, y, num_bits, steps, guidance_scale,
                             group_size=group_size)
    elif flow_type == "dfm_mask":
        return sample_dfm_mask_cfg(model, y, num_bits, steps, guidance_scale,
                                   group_size=group_size)
    else:
        return sample_dfm_cfg(model, y, num_bits, steps, guidance_scale,
                              group_size=group_size, flip_pred=flip_pred)


@torch.no_grad()
def eval_reconstruction(model, loader, device, num_bits, steps, guidance_scale,
                        flow_type, group_size=1, flip_pred=False, cfm_param="velocity"):
    """1-shot sampling bit/pattern accuracy."""
    model.eval()
    total_bit, total_pat, total = 0.0, 0.0, 0
    for yb, xb in loader:
        yb = yb.to(device)
        gt = xb.to(device).long()
        bits = _sample_fn(model, yb, num_bits, steps, guidance_scale, flow_type,
                          group_size, flip_pred, cfm_param)
        bit_acc = (bits == gt).float().mean().item()
        pat_acc = (bits == gt).all(dim=1).float().mean().item()
        B = xb.size(0)
        total_bit += bit_acc * B
        total_pat += pat_acc * B
        total += B
    return total_bit / total, total_pat / total


@torch.no_grad()
def eval_topk_pattern_accuracy(model, Y_test_t, X_test_flat, k, device,
                                num_bits, steps, guidance_scale, flow_type,
                                group_size=1, flip_pred=False, cfm_param="velocity"):
    """각 test y에 대해 k번 샘플링, 하나라도 GT와 완전 일치하면 correct."""
    model.eval()
    N = Y_test_t.size(0)
    X_test_t = torch.tensor(X_test_flat, device=device).long()
    correct = 0
    for i in range(N):
        y_rep = Y_test_t[i:i+1].to(device).repeat(k, 1)
        samples = _sample_fn(model, y_rep, num_bits, steps, guidance_scale, flow_type,
                              group_size, flip_pred, cfm_param)
        gt = X_test_t[i].unsqueeze(0).repeat(k, 1)
        if (samples == gt).all(dim=1).any():
            correct += 1
    return correct / N


# ---------------------------
# Main
# ---------------------------
def main(args):
    exp_dir = setup_exp_dir(args)
    print(f"Experiment directory: {exp_dir}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    group_size = args.group_size
    vocab_size = 2 ** group_size if group_size > 1 else 2
    flip_pred = args.flip_pred
    cfm_param = args.cfm_param
    ste_t_weight = not args.ste_no_t_weight
    ste_warmup_ep = int(args.ste_warmup_frac * args.epochs)
    print(f"Architecture: {args.arch}, Flow type: {args.flow_type}, "
          f"OT coupling: {args.ot_coupling}, group_size: {group_size} (V={vocab_size}), "
          f"spatial: {args.spatial}, flip_pred: {flip_pred}")
    if args.flow_type == "cfm":
        print(f"CFM parameterization: {cfm_param}"
              + (f" | lambda_ste={args.lambda_ste} (warmup {ste_warmup_ep} ep), "
                 f"ste_t_weight={ste_t_weight}" if cfm_param == "xpred_ste" else ""))

    # --- Data ---
    train_npz = np.load(os.path.join(args.data_root, "dataset_train.npz"))
    valid_npz = np.load(os.path.join(args.data_root, "dataset_valid.npz"))
    test_npz  = np.load(os.path.join(args.data_root, "dataset_test.npz"))

    X_train_2d = train_npz["X"].astype(np.float32)
    Y_train    = train_npz["Y"].astype(np.float32)
    X_val_2d   = valid_npz["X"].astype(np.float32)
    Y_val      = valid_npz["Y"].astype(np.float32)
    X_test_2d  = test_npz["X"].astype(np.float32)
    Y_test     = test_npz["Y"].astype(np.float32)
    freq = train_npz["freq"].astype(np.float32)

    N_train, H, W = X_train_2d.shape
    num_points = Y_train.shape[1]
    num_bits = H * W
    print(f"Dataset: train={len(X_train_2d)}, val={len(X_val_2d)}, test={len(X_test_2d)}, "
          f"H={H}, W={W}, num_bits={num_bits}, num_points={num_points}")

    # flatten raster
    X_train_flat = X_train_2d.reshape(N_train, num_bits)
    X_val_flat   = X_val_2d.reshape(len(X_val_2d), num_bits)
    X_test_flat  = X_test_2d.reshape(len(X_test_2d), num_bits)

    if args.canonical_target:
        print("Applying y-flip canonicalization to targets.")
        X_train_flat = canonicalize_under_yflip(X_train_flat, H, W)
        X_val_flat   = canonicalize_under_yflip(X_val_flat, H, W)
        X_test_flat  = canonicalize_under_yflip(X_test_flat, H, W)

    # Spatial patch reordering (raster → patch 순서)
    inv_reorder_idx = None
    if args.spatial and group_size > 1:
        ph = pw = int(group_size ** 0.5)
        assert ph * pw == group_size, f"spatial requires square patches: group_size={group_size} is not a perfect square"
        fwd_idx, inv_reorder_idx = make_spatial_patch_indices(H, W, ph, pw)
        fwd_np = fwd_idx.numpy()
        X_train_flat = X_train_flat[:, fwd_np]
        X_val_flat   = X_val_flat[:, fwd_np]
        X_test_flat  = X_test_flat[:, fwd_np]
        print(f"Applied spatial {ph}x{pw} patch reordering ({num_bits} bits → {num_bits//group_size} patches)")

    Y_train_t, X_train_t = torch.tensor(Y_train), torch.tensor(X_train_flat)
    Y_val_t,   X_val_t   = torch.tensor(Y_val),   torch.tensor(X_val_flat)
    Y_test_t,  X_test_t  = torch.tensor(Y_test),  torch.tensor(X_test_flat)

    train_loader = DataLoader(TensorDataset(Y_train_t, X_train_t),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(TensorDataset(Y_val_t, X_val_t),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader  = DataLoader(TensorDataset(Y_test_t, X_test_t),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # --- W&B ---
    wandb.init(project=args.project, name=args.exp_name)
    wandb.config.update(vars(args))

    # --- Model / Optim ---
    num_tokens = num_bits // group_size if group_size > 1 else num_bits
    model = build_model(
        args.arch,
        num_points=num_points,
        num_bits=num_tokens,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        seq_len_y=args.seq_len_y,
        drop_prob=args.drop_prob,
        vocab_size=vocab_size if args.flow_type in ("dfm", "dfm_mask") else 2,
        group_size=group_size if args.flow_type == "cfm" else 1,
        mask_token=(args.flow_type == "dfm_mask"),
        cond_mode=args.cond_mode,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"#params = {n_params/1e6:.2f}M")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup + cosine decay (train_ordering_no_sample_images.py와 동일)
    scheduler = None
    if args.lr_scheduler == "cosine":
        warmup_epochs = 5
        lr_warmup = 1e-4
        lr_max = args.lr

        def lr_lambda(epoch):
            ep = epoch + 1
            if ep <= warmup_epochs:
                return lr_warmup / lr_max
            T = max(args.epochs - warmup_epochs, 1)
            t = float(ep - warmup_epochs - 1) / float(T - 1) if T > 1 else 0.0
            cos_factor = 0.5 * (1.0 + np.cos(np.pi * t))
            return cos_factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(f"Using warm-up + cosine LR schedule: warmup_epochs={warmup_epochs}, "
              f"lr_warmup={lr_warmup}, lr_max={lr_max}")

    save_path = os.path.join(exp_dir, "best_model.pth")     # 주 best = min val_loss (기존 규약)
    bitacc_path = os.path.join(exp_dir, "best_bitacc.pth")  # 대안 best = max val_bit_acc (참고용)
    last_path = os.path.join(exp_dir, "last_model.pth")

    use_ema = args.ema_decay > 0
    ema = ModelEMA(model, decay=args.ema_decay) if use_ema else None
    print(f"EMA: {'enabled' if use_ema else 'disabled'} (decay={args.ema_decay})")

    best_val = float("inf")
    best_bit = -1.0          # best_model 선택 기준: val_bit_acc 최대 (val_loss 는 sample 품질과 어긋나 미사용)
    for epoch in range(1, args.epochs + 1):
        # STE 가중치 워밍업: 초반은 순수 x-pred 회귀(=유효한 CFM)로 흐름부터 학습
        if cfm_param == "xpred_ste" and ste_warmup_ep > 0:
            lambda_ste_ep = args.lambda_ste * min(1.0, epoch / ste_warmup_ep)
        else:
            lambda_ste_ep = args.lambda_ste
        tr_loss = train_one_epoch(model, train_loader, optimizer, device,
                                  args.flow_type, args.sigma_min, args.ot_coupling,
                                  group_size, flip_pred,
                                  cfm_param, lambda_ste_ep, ste_t_weight)
        if ema:
            ema.update(model)

        if ema:
            ema.apply(model)
        # val 은 best-model 선택 기준이라 최종 lambda_ste 로 고정(에폭 간 스케일 안정)
        va_loss = eval_loss(model, val_loader, device, args.flow_type, args.sigma_min,
                            args.ot_coupling, group_size, flip_pred,
                            cfm_param, args.lambda_ste, ste_t_weight)
        if ema:
            ema.apply(model)

        cur_lr = optimizer.param_groups[0]["lr"]
        log = {"epoch": epoch, "lr": cur_lr, "train_loss": tr_loss, "val_loss": va_loss}

        msg = f"[Epoch {epoch:03d}/{args.epochs}] Train {tr_loss:.6f} | Val {va_loss:.6f} | lr {cur_lr:.2e}"

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if ema:
                ema.apply(model)
            bit_acc, pat_acc = eval_reconstruction(
                model, val_loader, device, num_bits, args.sample_steps,
                args.guidance_scale, args.flow_type, group_size, flip_pred, cfm_param,
            )
            if ema:
                ema.apply(model)
            log.update({"val_bit_acc": bit_acc, "val_pattern_acc": pat_acc})
            msg += f" | BitAcc {bit_acc:.4f} PatAcc {pat_acc:.4f}"

            # 대안 best (참고용): max val_bit_acc. min-val_loss 가 sample 품질과 어긋나는 경우
            # (dfm_g4: ep250 PatAcc0.08 vs ep500 0.47) 대비용으로 별도 저장.
            if bit_acc > best_bit:
                best_bit = bit_acc
                if ema:
                    ema.apply(model)
                torch.save(model.state_dict(), bitacc_path)
                if ema:
                    ema.apply(model)
                wandb.run.summary["best_val_bit_acc"] = best_bit
                wandb.run.summary["best_val_pattern_acc"] = pat_acc

        # 주 best_model = min val_loss (기존 규약; 다운스트림 eval 이 로드하는 파일)
        if va_loss < best_val:
            best_val = va_loss
            if ema:
                ema.apply(model)
            torch.save(model.state_dict(), save_path)
            if ema:
                ema.apply(model)
            wandb.run.summary["best_val_loss"] = best_val
            print(f"  Best saved (val_loss {va_loss:.6f}) -> {save_path}")

        print(msg)
        wandb.log(log)

        if scheduler is not None:
            scheduler.step()

        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val,
        }
        if ema:
            last_ckpt["ema_state_dict"] = ema.state_dict()
        torch.save(last_ckpt, last_path)

    # --- Test (best EMA model) ---
    print("Evaluating best EMA model on test set...")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_loss = eval_loss(model, test_loader, device, args.flow_type, args.sigma_min,
                          args.ot_coupling, group_size, flip_pred,
                          cfm_param, args.lambda_ste, ste_t_weight)
    test_bit, test_pat = eval_reconstruction(
        model, test_loader, device, num_bits, args.sample_steps,
        args.guidance_scale, args.flow_type, group_size, flip_pred, cfm_param,
    )
    print(f"Test Loss {test_loss:.6f} | BitAcc {test_bit:.4f} | PatAcc {test_pat:.4f}")
    wandb.log({"test_loss": test_loss, "test_bit_acc": test_bit, "test_pattern_acc": test_pat})

    # --- Top-k pattern accuracy ---
    print(f"Top-{args.topk} pattern accuracy via sampling...")
    n_eval = min(args.topk_eval_n, len(X_test_flat))
    topk_acc = eval_topk_pattern_accuracy(
        model, Y_test_t[:n_eval], X_test_flat[:n_eval], k=args.topk,
        device=device, num_bits=num_bits, steps=args.sample_steps,
        guidance_scale=args.guidance_scale, flow_type=args.flow_type,
        group_size=group_size, flip_pred=flip_pred, cfm_param=cfm_param,
    )
    print(f"Top-{args.topk} Pattern Accuracy (on {n_eval}): {topk_acc:.4f}")
    wandb.log({f"top{args.topk}_pattern_acc": topk_acc})
    wandb.run.summary[f"top{args.topk}_pattern_acc"] = topk_acc

    # --- Forward surrogate evaluation (best-of-k) ---
    try:
        from eval import evaluate_inverse_model, load_forward_surrogate
        print("Forward-surrogate best-of-k evaluation...")
        fwd = load_forward_surrogate(num_points=num_points, device=device)

        def inverse_fn(y_batch):
            bits = _sample_fn(model, y_batch, num_bits, args.sample_steps,
                              args.guidance_scale, args.flow_type, group_size,
                              flip_pred, cfm_param)
            if inv_reorder_idx is not None:
                bits = bits[:, inv_reorder_idx.to(bits.device)]
            return bits

        fwd_metrics = evaluate_inverse_model(
            inverse_fn, Y_test_t[:n_eval], H, W, freq,
            k=args.topk, n_eval=n_eval, fwd=fwd, device=device,
        )
        print(f"Forward surrogate metrics: {fwd_metrics}")
        wandb.log({f"fwd_{k}": v for k, v in fwd_metrics.items()})
        for k, v in fwd_metrics.items():
            wandb.run.summary[f"fwd_{k}"] = v
    except Exception as e:
        print(f"Forward surrogate eval skipped: {e}")

    # --- GT vs Pred images ---
    model.eval()
    with torch.no_grad():
        num_samples = min(32, len(X_test_flat))
        idx = torch.randperm(len(X_test_flat))[:num_samples]
        Y_sample = Y_test_t[idx].to(device)
        X_true = X_test_flat[idx.numpy()]
        X_gen = _sample_fn(model, Y_sample, num_bits, args.sample_steps,
                           args.guidance_scale, args.flow_type, group_size,
                           flip_pred, cfm_param).cpu()
        if inv_reorder_idx is not None:
            X_gen = X_gen[:, inv_reorder_idx]
        X_gen = X_gen.numpy()

    # X_true is in patch order if spatial; reorder back for images
    if inv_reorder_idx is not None:
        X_true = X_true[:, inv_reorder_idx.numpy()]
    X_true_img = restore_pixel_order(X_true, H, W)
    X_gen_img = restore_pixel_order(X_gen, H, W)
    images = []
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        axes[0].imshow(X_true_img[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("GT"); axes[0].axis("off")
        axes[1].imshow(X_gen_img[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Pred"); axes[1].axis("off")
        plt.tight_layout()
        images.append(wandb.Image(fig, caption=f"Sample #{int(idx[i])}"))
        plt.close(fig)
    wandb.log({"GT_vs_Pred_Layouts": images})
    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, required=True,
                   help="Experiment name (e.g. dfm_spg4_nocfg)")
    p.add_argument("--exp_root", type=str, default="./experiments",
                   help="Root directory for experiments")
    p.add_argument("--data_root", type=str,
                   default="/hai/home/lsh/antenna/year_hai/data/datasets")

    # Architecture
    p.add_argument("--arch", type=str, default="stable3dit",
                   choices=["stable3dit", "smalldit"],
                   help="stable3dit: SD3-style (adaLN+cross-attn), smalldit: AR-comparable (additive)")

    # Flow type
    p.add_argument("--flow_type", type=str, default="dfm",
                   choices=["cfm", "dfm", "dfm_mask"],
                   help="cfm: continuous flow matching, dfm: discrete flow matching")

    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=15)
    p.add_argument("--dim_ff", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seq_len_y", type=int, default=50,
                   help="spectral condition sequence length (stable3dit only)")

    # flow matching / CFG
    p.add_argument("--sigma_min", type=float, default=0.0)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--drop_prob", type=float, default=0.1,
                   help="probability of dropping condition for CFG training")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG guidance scale (1.0 = no guidance)")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay rate for model weights (0 to disable)")
    p.add_argument("--ot_coupling", action="store_true", default=False,
                   help="Use mini-batch OT coupling for DFM (Hamming distance matching)")
    p.add_argument("--group_size", type=int, default=1,
                   choices=[1, 2, 4, 5, 10],
                   help="Bit grouping for DFM: 1=binary(V=2), 2=V=4, 4=V=16, 5=V=32, 10=V=1024")
    p.add_argument("--spatial", action="store_true", default=False,
                   help="Use spatial 2D patch grouping instead of consecutive bits")
    p.add_argument("--flip_pred", action="store_true", default=False,
                   help="Residual logit parameterization: model predicts correction from x_t")
    p.add_argument("--cond_mode", type=str, default="crossattn",
                   choices=["crossattn", "adaln"],
                   help="crossattn: t→adaLN + y_seq→cross-attn (기존) / "
                        "adaln: (t+pooled_y)→adaLN, cross-attn 제거 (정통 DiT식)")

    # CFM parameterization (velocity vs x-prediction + STE)
    p.add_argument("--cfm_param", type=str, default="velocity",
                   choices=["velocity", "xpred_ste"],
                   help="velocity: 기존 속도 회귀 / xpred_ste: 출력=x̂₁ + STE 이산화 손실")
    p.add_argument("--lambda_ste", type=float, default=0.5,
                   help="STE 이산화 손실 가중치 (cfm_param=xpred_ste)")
    p.add_argument("--ste_warmup_frac", type=float, default=0.2,
                   help="lambda_ste 를 0→목표값으로 램프하는 에폭 비율")
    p.add_argument("--ste_no_t_weight", action="store_true", default=False,
                   help="STE 항의 t-가중(작은 t 다운웨이트) 비활성화")

    # data options
    p.add_argument("--canonical_target", action="store_true", default=True)

    # train
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=5)

    # eval
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--topk_eval_n", type=int, default=2000)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", type=str, default="rogers_inverse_dit")

    args = p.parse_args()
    main(args)
