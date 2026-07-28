# -*- coding: utf-8 -*-
"""
AR baseline 평가 — Flow Matching 과 동일한 프로토콜(canonicalize → best-of-k → surrogate).

학습 스크립트(train_ordering_no_sample_images.py)에는 teacher-forcing 만 있고
autoregressive decode 루프가 없어서 여기서 구현한다.

NFE = num_bits = 100 (순차). FM 의 steps 와 직접 비교되는 축이다.
주의: KV cache 를 쓰지 않으므로 wall-clock 은 AR 에 불리하다.
      NFE(=순차 깊이 100)는 캐시 유무와 무관한 구조적 수치다.
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_ordering_no_sample_images import (          # noqa: E402
    SmallTransformerAR, get_order_indices, canonicalize_under_yflip, BOS_IDX,
)
from eval import load_forward_surrogate, evaluate_best_of_k   # noqa: E402


@torch.no_grad()
def sample_ar(model, y, num_bits, temperature=1.0, greedy=False):
    """Ancestral decoding. returns (B, num_bits) long, in the model's (snake) order."""
    B = y.size(0)
    device = y.device
    tokens = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
    for _ in range(num_bits):                       # NFE = num_bits, 순차
        logits = model(y, tokens)                   # (B, L)
        p = torch.sigmoid(logits[:, -1] / temperature)
        nxt = (p > 0.5).long() if greedy else (torch.rand_like(p) < p).long()
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
    return tokens[:, 1:]                            # drop BOS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AR baseline eval (FM 과 동일 프로토콜)")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str,
                   default="/hai/home/lsh/antenna/year_hai/data/datasets")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=15)
    p.add_argument("--dim_ff", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ordering", type=str, default="snake")
    p.add_argument("--spectral_cond", type=str, default="resnet1d")
    p.add_argument("--temperature", type=float, nargs="+", default=[1.0],
                   help="Ancestral sampling temperature(s) to sweep")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--topk_eval_n", type=int, default=2000)
    p.add_argument("--acc_eval_n", type=int, default=4096)
    p.add_argument("--sample_batch", type=int, default=512)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--tag", type=str, default="ar_snake")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data (FM eval 과 동일한 전처리: canonicalize → snake reorder) ---
    te = np.load(os.path.join(args.data_root, "dataset_test.npz"))
    X2d = te["X"].astype(np.float32)
    Y = te["Y"].astype(np.float32)
    freq = te["freq"].astype(np.float32)
    _, H, W = X2d.shape
    num_bits, num_points = H * W, Y.shape[1]

    Xf = canonicalize_under_yflip(X2d.reshape(len(X2d), num_bits), H, W)
    order_idx = get_order_indices(args.ordering, num_bits, H, W)   # raster -> snake
    X_ord = Xf[:, order_idx]

    Y_t = torch.tensor(Y)
    X_t = torch.tensor(X_ord)

    # --- Model ---
    chain2spatial = torch.from_numpy(order_idx).long()
    model = SmallTransformerAR(
        num_points=num_points, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        max_len=num_bits, vocab_size=3, dropout=args.dropout,
        spectral_cond=args.spectral_cond, use_2d_pos=False,
        chain2spatial=chain2spatial, height=H, width=W,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Model: SmallTransformerAR, {sum(q.numel() for q in model.parameters())/1e6:.2f}M params")
    print(f"Checkpoint: {args.ckpt}")
    print(f"NFE = {num_bits} (sequential); ordering = {args.ordering}")

    fwd = load_forward_surrogate(num_points=num_points, device=device)
    n_eval = min(args.topk_eval_n, len(X_ord))
    n_acc = min(args.acc_eval_n, len(X_ord))

    print(f"\n{'T':>5} {'mode':>7} | {'BitAcc':>7} {'PatAcc':>7} {'Top10Pat':>8} | "
          f"{'FwdMSE':>8} {'ResFreqErr':>10} {'ResMatch':>9} | {'ones':>12} | {'Time':>7}")
    print("-" * 104)

    for temp in args.temperature:
        for greedy in ([True, False] if temp == args.temperature[0] else [False]):
            t0 = time.time()
            set_seed(args.seed)

            # --- BitAcc / PatAcc (1 candidate per target) ---
            tb = tp = 0.0
            for i in range(0, n_acc, args.sample_batch):
                j = min(i + args.sample_batch, n_acc)
                bits = sample_ar(model, Y_t[i:j].to(device), num_bits,
                                 temperature=temp, greedy=greedy)
                gt = X_t[i:j].to(device).long()
                tb += (bits == gt).float().mean().item() * (j - i)
                tp += (bits == gt).all(dim=1).float().mean().item() * (j - i)
            bit_acc, pat_acc = tb / n_acc, tp / n_acc

            # --- k candidates per target (best-of-k) ---
            # greedy 는 k 개가 모두 동일하므로 best-of-k 가 무의미 → 건너뛴다.
            if greedy:
                print(f"{temp:>5.2f} {'greedy':>7} | {bit_acc:>7.4f} {pat_acc:>7.4f} "
                      f"{'--':>8} | {'--':>8} {'--':>10} {'--':>9} | {'--':>12} | "
                      f"{time.time()-t0:>6.1f}s")
                continue

            set_seed(args.seed)
            k = args.topk
            chunk = max(1, args.sample_batch // k)
            gen = torch.empty(n_eval, k, num_bits, dtype=torch.long)
            for i in range(0, n_eval, chunk):
                j = min(i + chunk, n_eval)
                y_rep = Y_t[i:j].to(device).repeat_interleave(k, dim=0)
                bits = sample_ar(model, y_rep, num_bits, temperature=temp)
                gen[i:j] = bits.view(j - i, k, num_bits).cpu()

            topk_acc = (gen == X_t[:n_eval].long().unsqueeze(1)).all(dim=2)\
                       .any(dim=1).float().mean().item()

            # snake -> raster 로 되돌려 surrogate 에 넣는다
            gen_raster = torch.zeros_like(gen)
            gen_raster[:, :, torch.from_numpy(order_idx)] = gen
            fm = evaluate_best_of_k(fwd, gen_raster, Y_t[:n_eval], H, W, freq=freq)

            n1 = gen.float().sum(-1)
            ones = f"{n1.mean():.1f}±{n1.std():.2f}"
            p60 = (n1 == 60).float().mean().item()
            elapsed = time.time() - t0

            print(f"{temp:>5.2f} {'sample':>7} | {bit_acc:>7.4f} {pat_acc:>7.4f} "
                  f"{topk_acc:>8.4f} | {fm['fwd_mse']:>8.4f} {fm['res_freq_err']:>10.6f} "
                  f"{fm['res_match_rate']:>9.4f} | {ones:>12} | {elapsed:>6.1f}s")
            print(f"      {'':>7} | P(ones=60) = {p60:.3f}")

            if args.csv:
                new = not os.path.exists(args.csv)
                with open(args.csv, "a") as f:
                    if new:
                        f.write("tag,arch,flow,group_size,spatial,steps,cfg,bit_acc,"
                                "pat_acc,topk_pat_acc,fwd_mse,fwd_mae,res_freq_err,"
                                "res_depth_err,res_match_rate,k,sec\n")
                    f.write(f"{args.tag},ar,ar,1,0,{num_bits},{temp},{bit_acc:.6f},"
                            f"{pat_acc:.6f},{topk_acc:.6f},{fm['fwd_mse']:.6f},"
                            f"{fm['fwd_mae']:.6f},{fm['res_freq_err']:.6f},"
                            f"{fm['res_depth_err']:.6f},{fm['res_match_rate']:.6f},"
                            f"{k},{elapsed:.1f}\n")
