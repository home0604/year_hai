# -*- coding: utf-8 -*-
"""
Forward-surrogate 기반 inverse 모델 평가.

아이디어:
  inverse 모델이 목표 스펙트럼 Y*로부터 패턴 X̂ 를 생성하면,
  freeze 된 forward surrogate f(·) 로 X̂ 의 스펙트럼 Ŝ = f(X̂) 를 예측하고
  Ŝ 가 목표 Y* 와 얼마나 가까운지(re-simulation error)로 품질을 잰다.

  - 비트 완전 일치(top-k exact match)보다 안테나 관점에서 의미 있는 지표.
  - FM / AR 어느 inverse 모델이든 "생성한 bits + 목표 Y" 만 주면 동일하게 평가 가능.

forward surrogate (train.py의 DeepHybridCNN_Res, model_1218.pth):
  - 구조: stage_blocks=(3,4,4,5), stage_channels=(128,256,512,512), strides=(1,1,1,1)
  - 입력: (B,1,H,W), 픽셀 {0,1} raster(row-major) 그대로 (정규화 없음)
  - 출력: (B, num_points) S11 (dB)
  - 검증된 입력 규약: raster {0,1}  (test MSE≈0.82 dB² == 모델 고유 오차/ noise floor)
"""
import os
import numpy as np
import torch
import torch.nn as nn


DEFAULT_FWD_CKPT = "/hai/home/lsh/antenna/year_hai/data/year/models/pa10x10/fwd/model_1218.pth"


# ---------------------------
# Forward surrogate 정의 (train.py 재현)
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, negative_slope=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        idt = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            idt = self.proj(idt)
        return self.act(out + idt)


def _make_stage(in_ch, out_ch, num_blocks, stride=1):
    blocks = [ResidualBlock(in_ch, out_ch, stride)]
    blocks += [ResidualBlock(out_ch, out_ch, 1) for _ in range(num_blocks - 1)]
    return nn.Sequential(*blocks)


class DeepHybridCNN_Res(nn.Module):
    def __init__(self, num_points, stage_blocks=(3, 4, 4, 5),
                 stage_channels=(128, 256, 512, 512), strides=(1, 1, 1, 1)):
        super().__init__()
        in_ch = 1
        feats = []
        for nb, oc, st in zip(stage_blocks, stage_channels, strides):
            feats.append(_make_stage(in_ch, oc, nb, st))
            in_ch = oc
        feats.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*feats)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(stage_channels[-1], num_points))

    def forward(self, x):
        return self.fc(self.features(x))


# ---------------------------
# Load (frozen)
# ---------------------------
def load_forward_surrogate(ckpt_path=DEFAULT_FWD_CKPT, num_points=201, device="cuda"):
    """forward surrogate 를 로드하고 완전히 freeze (eval + requires_grad=False)."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Forward surrogate checkpoint not found: {ckpt_path}")
    model = DeepHybridCNN_Res(num_points=num_points)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"🧊 Forward surrogate loaded (frozen) from {ckpt_path} "
          f"[missing={len(missing)}, unexpected={len(unexpected)}]")
    return model


@torch.no_grad()
def predict_spectrum(fwd, bits, height, width, batch_size=4096):
    """
    bits: (M, num_bits) in {0,1} (long/float) raster(row-major)
    return: (M, num_points) 예측 S11
    """
    device = next(fwd.parameters()).device
    if not torch.is_tensor(bits):
        bits = torch.tensor(bits)
    bits = bits.float()
    outs = []
    for i in range(0, bits.size(0), batch_size):
        xb = bits[i:i + batch_size].to(device).view(-1, 1, height, width)
        outs.append(fwd(xb))
    return torch.cat(outs, dim=0)


# ---------------------------
# Antenna-relevant 지표
# ---------------------------
def spectrum_metrics(pred_S, target_Y, freq=None):
    """
    pred_S, target_Y: (..., num_points)  (같은 shape)
    return dict (per-sample 텐서):
      - mse, mae           : 스펙트럼 오차
      - res_freq_err       : 공진 주파수(=S11 최저점) 오차 (freq 단위; freq 없으면 index 단위)
      - res_depth_err      : 공진 깊이(최저 S11 값) 오차
    """
    mse = ((pred_S - target_Y) ** 2).mean(dim=-1)
    mae = (pred_S - target_Y).abs().mean(dim=-1)

    pred_argmin = pred_S.argmin(dim=-1)        # 공진 위치(인덱스)
    tgt_argmin = target_Y.argmin(dim=-1)
    if freq is not None:
        f = freq.to(pred_S.device) if torch.is_tensor(freq) else torch.tensor(freq, device=pred_S.device)
        res_freq_err = (f[pred_argmin] - f[tgt_argmin]).abs()
    else:
        res_freq_err = (pred_argmin - tgt_argmin).abs().float()

    res_depth_err = (pred_S.min(dim=-1).values - target_Y.min(dim=-1).values).abs()
    return {"mse": mse, "mae": mae, "res_freq_err": res_freq_err, "res_depth_err": res_depth_err}


# ---------------------------
# best-of-k 평가 (one-to-many inverse에 적합)
# ---------------------------
@torch.no_grad()
def evaluate_best_of_k(fwd, gen_bits, target_Y, height, width, freq=None,
                       res_freq_tol=0.05):
    """
    gen_bits : (N, k, num_bits)  각 목표마다 k개 생성
    target_Y : (N, num_points)   목표 스펙트럼
    선택 기준: forward surrogate 스펙트럼 MSE 가 가장 낮은 후보(best-of-k)를 채택.

    return: dict (스칼라 평균 지표)
      - mse / mae / res_freq_err / res_depth_err : best-of-k 후보 기준 평균
      - res_match_rate : 공진 주파수 오차 <= res_freq_tol 비율 (matched)
    """
    device = next(fwd.parameters()).device
    N, k, nb = gen_bits.shape
    target_Y = target_Y.to(device) if torch.is_tensor(target_Y) else torch.tensor(target_Y, device=device)
    if freq is not None and not torch.is_tensor(freq):
        freq = torch.tensor(freq, device=device)

    flat = gen_bits.reshape(N * k, nb)
    pred = predict_spectrum(fwd, flat, height, width).reshape(N, k, -1)  # (N,k,P)
    tgt = target_Y.unsqueeze(1)                                          # (N,1,P)

    mse_nk = ((pred - tgt) ** 2).mean(dim=-1)        # (N,k)
    best = mse_nk.argmin(dim=1)                       # (N,) surrogate 기준 best 후보
    idx = best.view(N, 1, 1).expand(N, 1, pred.size(-1))
    pred_best = pred.gather(1, idx).squeeze(1)        # (N,P)

    m = spectrum_metrics(pred_best, target_Y, freq=freq)
    res_match = (m["res_freq_err"] <= res_freq_tol).float().mean().item()
    return {
        "fwd_mse": m["mse"].mean().item(),
        "fwd_mae": m["mae"].mean().item(),
        "res_freq_err": m["res_freq_err"].mean().item(),
        "res_depth_err": m["res_depth_err"].mean().item(),
        "res_match_rate": res_match,
        "k": k,
    }


@torch.no_grad()
def evaluate_inverse_model(inverse_sample_fn, Y_test_t, height, width, freq,
                           k=10, n_eval=2000, fwd=None, fwd_ckpt=DEFAULT_FWD_CKPT,
                           res_freq_tol=0.05, device="cuda", sample_batch=256):
    """
    inverse_sample_fn(y_batch) -> bits (B, num_bits) in {0,1}  를 받아
    forward-surrogate 기반 best-of-k 지표를 계산.

    inverse_sample_fn 은 FM/AR 어느 쪽이든 "조건 y배치 -> 생성 bits" 만 맞추면 됨.
    """
    if fwd is None:
        fwd = load_forward_surrogate(fwd_ckpt, num_points=Y_test_t.size(1), device=device)
    n = min(n_eval, Y_test_t.size(0))
    num_bits = height * width

    gen = torch.empty(n, k, num_bits, dtype=torch.long)
    for i in range(n):
        y_rep = Y_test_t[i:i + 1].to(device).repeat(k, 1)
        bits = inverse_sample_fn(y_rep)                 # (k, num_bits)
        gen[i] = bits.cpu().long()

    return evaluate_best_of_k(fwd, gen, Y_test_t[:n], height, width,
                              freq=freq, res_freq_tol=res_freq_tol)


# ===========================================================================
# Standalone deterministic evaluation
# ===========================================================================
if __name__ == "__main__":
    import argparse
    import random
    from model import build_model
    from flow_matching import (
        bits01_to_pm1, sample_fm_cfg, sample_dfm_cfg, sample_dfm_mask_cfg,
        make_spatial_patch_indices, tokens_to_bits,
    )

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def horizontal_flip_y_axis(X_flat, height, width):
        N, num_bits = X_flat.shape
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

    def sample_fn(model, y, num_bits, steps, guidance_scale, flow_type,
                  group_size=1, flip_pred=False):
        if flow_type == "cfm":
            return sample_fm_cfg(model, y, num_bits, steps, guidance_scale,
                                 group_size=group_size)
        elif flow_type == "dfm_mask":
            return sample_dfm_mask_cfg(model, y, num_bits, steps, guidance_scale,
                                       group_size=group_size)
        else:
            return sample_dfm_cfg(model, y, num_bits, steps, guidance_scale,
                                  group_size=group_size, flip_pred=flip_pred)

    p = argparse.ArgumentParser(description="Deterministic evaluation of FM checkpoints")
    p.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pth")
    p.add_argument("--data_root", type=str,
                   default="/hai/home/lsh/antenna/year_hai/data/datasets")
    p.add_argument("--arch", type=str, default="stable3dit",
                   choices=["stable3dit", "smalldit"])
    p.add_argument("--flow_type", type=str, default="dfm", choices=["cfm", "dfm", "dfm_mask"])
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=15)
    p.add_argument("--dim_ff", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seq_len_y", type=int, default=50)
    p.add_argument("--drop_prob", type=float, default=0.1)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--spatial", action="store_true", default=False)
    p.add_argument("--flip_pred", action="store_true", default=False)
    p.add_argument("--canonical_target", action="store_true", default=True)
    p.add_argument("--sample_steps", type=int, nargs="+", default=[50, 25, 10],
                   help="Steps to sweep (multiple values)")
    p.add_argument("--guidance_scale", type=float, nargs="+", default=[1.0],
                   help="Guidance scales to sweep (multiple values)")
    p.add_argument("--quick", action="store_true", default=False,
                   help="Only compute bit/pat accuracy (skip top-k and forward surrogate)")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--topk_eval_n", type=int, default=2000)
    p.add_argument("--acc_eval_n", type=int, default=4096,
                   help="Targets used for BitAcc/PatAcc (0 = full test set)")
    p.add_argument("--sample_batch", type=int, default=512,
                   help="Sampling batch size (k-replicas are batched together)")
    p.add_argument("--csv", type=str, default=None,
                   help="Append results as CSV rows to this file")
    p.add_argument("--tag", type=str, default="", help="Run name for the CSV rows")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_size = args.group_size
    vocab_size = 2 ** group_size if group_size > 1 else 2

    # --- Data ---
    test_npz = np.load(os.path.join(args.data_root, "dataset_test.npz"))
    X_test_2d = test_npz["X"].astype(np.float32)
    Y_test = test_npz["Y"].astype(np.float32)
    freq = test_npz["freq"].astype(np.float32)
    _, H, W = X_test_2d.shape
    num_bits = H * W
    num_points = Y_test.shape[1]
    X_test_flat = X_test_2d.reshape(len(X_test_2d), num_bits)

    if args.canonical_target:
        X_test_flat = canonicalize_under_yflip(X_test_flat, H, W)

    inv_reorder_idx = None
    if args.spatial and group_size > 1:
        ph = pw = int(group_size ** 0.5)
        fwd_idx, inv_reorder_idx = make_spatial_patch_indices(H, W, ph, pw)
        X_test_flat = X_test_flat[:, fwd_idx.numpy()]

    Y_test_t = torch.tensor(Y_test)
    X_test_t = torch.tensor(X_test_flat)

    # --- Model ---
    num_tokens = num_bits // group_size if group_size > 1 else num_bits
    model = build_model(
        args.arch, num_points=num_points, num_bits=num_tokens,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_ff, dropout=args.dropout,
        seq_len_y=args.seq_len_y, drop_prob=args.drop_prob,
        vocab_size=vocab_size,
        group_size=group_size if args.flow_type == "cfm" else 1,
        mask_token=(args.flow_type == "dfm_mask"),
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch}, {n_params/1e6:.2f}M params")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Config: flow={args.flow_type}, gs={group_size}, V={vocab_size}, "
          f"spatial={args.spatial}, flip_pred={args.flip_pred}")

    # --- Forward surrogate ---
    fwd = load_forward_surrogate(num_points=num_points, device=device)
    n_eval = min(args.topk_eval_n, len(X_test_flat))

    # --- Sweep ---
    import time

    if args.quick:
        print(f"\n{'steps':>5} {'cfg':>5} | {'BitAcc':>7} {'PatAcc':>7} | {'Time':>7}")
        print("-" * 45)
    else:
        print(f"\n{'steps':>5} {'cfg':>5} | {'BitAcc':>7} {'PatAcc':>7} "
              f"{'Top{0}Pat':>8} | {'FwdMSE':>8} {'ResFreqErr':>10} {'ResMatch':>9} | {'Time':>7}".format(args.topk))
        print("-" * 90)

    for steps in args.sample_steps:
        for gs in args.guidance_scale:
            t_start = time.time()
            set_seed(args.seed)

            # Bit / Pattern accuracy
            total_bit, total_pat, total = 0.0, 0.0, 0
            bs = args.sample_batch
            n_acc = (len(X_test_flat) if args.acc_eval_n <= 0
                     else min(args.acc_eval_n, len(X_test_flat)))
            for i in range(0, n_acc, bs):
                yb = Y_test_t[i:i+bs].to(device)
                gt = X_test_t[i:i+bs].to(device).long()
                bits = sample_fn(model, yb, num_bits, steps, gs,
                                 args.flow_type, group_size, args.flip_pred)
                total_bit += (bits == gt).float().mean().item() * len(yb)
                total_pat += (bits == gt).all(dim=1).float().mean().item() * len(yb)
                total += len(yb)
            bit_acc = total_bit / total
            pat_acc = total_pat / total

            if args.quick:
                elapsed = time.time() - t_start
                print(f"{steps:>5} {gs:>5.1f} | {bit_acc:>7.4f} {pat_acc:>7.4f} | {elapsed:>6.1f}s")
                continue

            # k candidates per target — sampled ONCE, reused for both
            # top-k pattern accuracy and forward-surrogate best-of-k.
            set_seed(args.seed)
            k = args.topk
            chunk = max(1, args.sample_batch // k)          # targets per batch
            gen = torch.empty(n_eval, k, num_bits, dtype=torch.long)
            for i in range(0, n_eval, chunk):
                j = min(i + chunk, n_eval)
                y_rep = Y_test_t[i:j].to(device).repeat_interleave(k, dim=0)
                bits = sample_fn(model, y_rep, num_bits, steps, gs,
                                 args.flow_type, group_size, args.flip_pred)
                gen[i:j] = bits.view(j - i, k, num_bits).cpu().long()

            # Top-k pattern accuracy (model's bit ordering)
            gt_k = X_test_t[:n_eval].long().unsqueeze(1)            # (n,1,D)
            topk_acc = (gen == gt_k).all(dim=2).any(dim=1).float().mean().item()

            # Forward surrogate best-of-k (needs raster ordering)
            gen_raster = gen
            if inv_reorder_idx is not None:
                gen_raster = gen[:, :, inv_reorder_idx]
            fwd_metrics = evaluate_best_of_k(
                fwd, gen_raster, Y_test_t[:n_eval], H, W, freq=freq)

            elapsed = time.time() - t_start
            print(f"{steps:>5} {gs:>5.1f} | {bit_acc:>7.4f} {pat_acc:>7.4f} "
                  f"{topk_acc:>8.4f} | {fwd_metrics['fwd_mse']:>8.4f} "
                  f"{fwd_metrics['res_freq_err']:>10.6f} "
                  f"{fwd_metrics['res_match_rate']:>9.4f} | {elapsed:>6.1f}s")

            if args.csv:
                new = not os.path.exists(args.csv)
                with open(args.csv, "a") as f:
                    if new:
                        f.write("tag,arch,flow,group_size,spatial,steps,cfg,bit_acc,"
                                "pat_acc,topk_pat_acc,fwd_mse,fwd_mae,res_freq_err,"
                                "res_depth_err,res_match_rate,k,sec\n")
                    f.write(f"{args.tag},{args.arch},{args.flow_type},{group_size},"
                            f"{int(args.spatial)},{steps},{gs},{bit_acc:.6f},"
                            f"{pat_acc:.6f},{topk_acc:.6f},{fwd_metrics['fwd_mse']:.6f},"
                            f"{fwd_metrics['fwd_mae']:.6f},{fwd_metrics['res_freq_err']:.6f},"
                            f"{fwd_metrics['res_depth_err']:.6f},"
                            f"{fwd_metrics['res_match_rate']:.6f},{k},{elapsed:.1f}\n")
