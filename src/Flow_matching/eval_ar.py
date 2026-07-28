# -*- coding: utf-8 -*-
"""
AR transformer inverse 모델을 forward-surrogate 기반으로 평가.

대상 체크포인트(둘 다 L15-d512-resnet1d-no2dpos, vocab3, max_len100):
  - trained   : .../inverse/transformer_ar/ar10x10-L15-...-ordhilbert-...-canon1-...pth
  - fine_tuned: .../inverse/fine_tuned/best_model.pth   (ordering 미표기 → 자동 검증)

지표:
  - forward-surrogate best-of-k : fwd_mse / fwd_mae / res_freq_err / res_depth_err / res_match_rate
  - (참고) exact-match top-k pattern accuracy (canonical 기준)

ordering 자동 검증: hilbert/snake/raster 로 소수 샘플 생성 후 fwd_mse 최저를 채택.
(잘못된 ordering 이면 패턴이 뒤섞여 S11 오차가 크게 나오므로 구분 가능)
"""
import os
import sys
import argparse
import numpy as np
import torch

# ../ (src) 에서 AR 정의/샘플러 import
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from train_inverse_10x10 import (  # noqa: E402
    SmallTransformerAR, sample_transformer_ar,
    canonicalize_under_yflip,
)
import eval as E  # noqa: E402


CKPTS = {
    # Jun 25 최종 코드(train_ordering_no_sample_images.py)로 60_density_500000 에 학습 (in-distribution)
    "local_snake": "/hai/home/lsh/antenna/year_hai/src/inverse_models_changing_ordering/"
                   "ar10x10-L15-d512-h4-ff768-dr0.1-ordsnake-specresnet1d-2dpos0-canon0-lr0.0001-bs128-seed42/"
                   "best_model.pth",
    # Feb 24 예전 체크포인트 (분포 불일치 의심 — 참고용)
    "trained": "/hai/home/lsh/antenna/year_hai/data/year/models/pa10x10/inverse/transformer_ar/"
               "ar10x10-L15-d512-h4-ff768-dr0.1-ordhilbert-specresnet1d-2dpos0-canon1-lr0.0001-bs128-seed42.pth",
    "fine_tuned": "/hai/home/lsh/antenna/year_hai/data/year/models/pa10x10/inverse/fine_tuned/best_model.pth",
}


def build_ar_model(num_points, device):
    model = SmallTransformerAR(
        num_points=num_points, d_model=512, nhead=4, num_layers=15,
        dim_feedforward=768, max_len=100, vocab_size=3, dropout=0.1,
        spectral_cond="resnet1d", use_2d_pos=False, height=10, width=10,
    ).to(device)
    return model


@torch.no_grad()
def generate_topk(model, Y, k, ordering, H, W, num_bits, device, batch_samples=64):
    """
    Y: (N, num_points) -> gen: (N, k, num_bits) in {0,1} (original raster ordering)
    test 샘플을 batch_samples 개씩 묶고 각 k배 복제하여 배치 샘플링(속도).
    """
    N = Y.size(0)
    gen = torch.empty(N, k, num_bits, dtype=torch.long)
    for s in range(0, N, batch_samples):
        yb = Y[s:s + batch_samples]
        b = yb.size(0)
        y_rep = yb.repeat_interleave(k, dim=0).to(device)        # (b*k, P): [y0]*k,[y1]*k,...
        bits = sample_transformer_ar(model, y_rep, num_bits=num_bits, greedy=False,
                                     ordering=ordering, height=H, width=W)  # (b*k, num_bits)
        gen[s:s + b] = bits.view(b, k, num_bits).cpu()
    return gen


def detect_ordering(model, fwd, Y_probe, H, W, num_bits, freq, device,
                    candidates=("hilbert", "snake", "raster"), n=200):
    """소수 샘플 k=1 생성 후 fwd_mse 최저 ordering 채택."""
    print(f"🔎 Detecting ordering on {n} samples (k=1)...")
    best, best_mse = None, float("inf")
    Yp = Y_probe[:n]
    for ordr in candidates:
        gen = generate_topk(model, Yp, k=1, ordering=ordr, H=H, W=W,
                            num_bits=num_bits, device=device)
        r = E.evaluate_best_of_k(fwd, gen, Yp, H, W, freq=freq)
        print(f"   ordering={ordr:8s} -> fwd_mse={r['fwd_mse']:.4f} res_match={r['res_match_rate']:.3f}")
        if r["fwd_mse"] < best_mse:
            best, best_mse = ordr, r["fwd_mse"]
    print(f"   ✅ selected ordering = {best}")
    return best


def exact_match_topk(gen, X_test_flat, H, W, k):
    """canonical 기준 exact-match top-k pattern accuracy (참고용)."""
    N = gen.shape[0]
    gt_can = canonicalize_under_yflip(X_test_flat[:N].astype(np.int64), H, W)  # (N, nb)
    gt = torch.tensor(gt_can)
    correct = 0
    for i in range(N):
        cand = gen[i].numpy().astype(np.int64)                 # (k, nb)
        cand_can = canonicalize_under_yflip(cand, H, W)
        if (torch.tensor(cand_can) == gt[i].unsqueeze(0)).all(dim=1).any():
            correct += 1
    return correct / N


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 device={device}")

    d = np.load(os.path.join(args.data_root, "dataset_test.npz"))
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    freq = d["freq"].astype(np.float32)
    N_all, H, W = X.shape
    num_bits, num_points = H * W, Y.shape[1]
    n_eval = min(args.n_eval, N_all)
    X_flat = X.reshape(N_all, num_bits)
    Y_t = torch.tensor(Y)
    print(f"✅ test={N_all}, eval on {n_eval}, k={args.k}, H={H},W={W}, P={num_points}")

    fwd = E.load_forward_surrogate(args.fwd_ckpt, num_points=num_points, device=device)

    # GT-as-candidate oracle floor (참조 상한)
    gt_bits = torch.tensor(X_flat[:n_eval].astype(np.int64)).unsqueeze(1)
    oracle = E.evaluate_best_of_k(fwd, gt_bits, Y_t[:n_eval], H, W, freq=freq,
                                  res_freq_tol=args.res_freq_tol)
    print(f"\n🧭 ORACLE (GT pattern→fwd): fwd_mse={oracle['fwd_mse']:.4f} "
          f"res_freq_err={oracle['res_freq_err']:.4f} res_match={oracle['res_match_rate']:.3f}\n")

    results = {"__oracle__": oracle}
    for name, path in CKPTS.items():
        if args.only and name != args.only:
            continue
        print("=" * 70)
        print(f"▶ {name}: {path}")
        model = build_ar_model(num_points, device)
        sd = torch.load(path, map_location=device, weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        model.eval()

        ordering = args.ordering
        if ordering == "auto":
            ordering = detect_ordering(model, fwd, Y_t, H, W, num_bits, freq, device)

        print(f"⏳ generating {n_eval}×{args.k} samples (ordering={ordering})...")
        gen = generate_topk(model, Y_t[:n_eval], k=args.k, ordering=ordering,
                            H=H, W=W, num_bits=num_bits, device=device)

        fwd_metrics = E.evaluate_best_of_k(fwd, gen, Y_t[:n_eval], H, W, freq=freq,
                                           res_freq_tol=args.res_freq_tol)
        em = exact_match_topk(gen, X_flat, H, W, args.k)
        fwd_metrics["exact_match_topk"] = em
        fwd_metrics["ordering"] = ordering
        results[name] = fwd_metrics
        print(f"  fwd_mse={fwd_metrics['fwd_mse']:.4f}  fwd_mae={fwd_metrics['fwd_mae']:.4f}  "
              f"res_freq_err={fwd_metrics['res_freq_err']:.4f}  res_match={fwd_metrics['res_match_rate']:.3f}  "
              f"exact_top{args.k}={em:.4f}")

    # 요약표
    print("\n" + "=" * 70)
    print(f"📊 SUMMARY (n_eval={n_eval}, k={args.k}, best-of-k by surrogate MSE)")
    print(f"{'model':<12}{'ord':<9}{'fwd_mse':>9}{'fwd_mae':>9}{'res_ferr':>9}{'res_match':>10}{'exactTopK':>10}")
    o = results["__oracle__"]
    print(f"{'ORACLE':<12}{'-':<9}{o['fwd_mse']:>9.4f}{o['fwd_mae']:>9.4f}{o['res_freq_err']:>9.4f}{o['res_match_rate']:>10.3f}{'-':>10}")
    for name in CKPTS:
        if name in results:
            r = results[name]
            print(f"{name:<12}{r['ordering']:<9}{r['fwd_mse']:>9.4f}{r['fwd_mae']:>9.4f}"
                  f"{r['res_freq_err']:>9.4f}{r['res_match_rate']:>10.3f}{r['exact_match_topk']:>10.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str,
                   default="/hai/year/datasets/pa10x10/60_density_500000_dataset/")
    p.add_argument("--fwd_ckpt", type=str, default=E.DEFAULT_FWD_CKPT)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--ordering", type=str, default="auto",
                   choices=["auto", "hilbert", "snake", "raster"])
    p.add_argument("--res_freq_tol", type=float, default=0.05, help="공진 주파수 매칭 허용오차 (GHz)")
    p.add_argument("--only", type=str, default=None, choices=[None, "local_snake", "trained", "fine_tuned"])
    args = p.parse_args()
    main(args)
