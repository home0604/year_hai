# -*- coding: utf-8 -*-
"""dfm_g4(uniform) vs dfm_mask_g4(absorbing) head-to-head — dual-10 (판별 regime).
   candidate-parity best-of-N, T∈{1,2}. notch-MSE(-10) best + valid rate(-12)."""
import os, sys, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from table1_search import FM_CFG, build_fm, make_fm_sampler, valid_mask_vec, notch_mse_vec, H, W, SCRATCH
from eval import load_forward_surrogate, predict_spectrum
from inverse_from_csv_10x10 import _find_true_segments

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
N, STEPS = 4096, 10
MODELS = ["dfm_g4", "dfm_mask_g4"]
TEMPS = [1.0, 2.0]

fwd = load_forward_surrogate(num_points=201, device=device)
te = np.load("/hai/home/lsh/antenna/year_hai/data/datasets/dataset_test.npz")
Y = te["Y"].astype(np.float32)

for kind, n in [("dual", 10), ("single", 10)]:
    idx = np.load(os.path.join(SCRATCH, f"table1_{kind}_idx.npy"))[:n]
    print(f"\n===== {kind.upper()} ({n} targets), best-of-{N}, steps={STEPS} =====")
    print(f"{'model':<14}{'T':>4}{'mean bestMSE':>14}{'median':>9}{'valid rate':>12}")
    print("-" * 53)
    for mn in MODELS:
        cfg = FM_CFG[mn]; m = build_fm(cfg, device)
        for T in TEMPS:
            sampler = make_fm_sampler(m, cfg, STEPS, 1.0, temperature=T)
            best_mses, valids = [], []
            for gi in idx:
                gi = int(gi)
                y = torch.as_tensor(Y[gi], device=device).view(1, -1).repeat(N, 1)
                target_t = torch.as_tensor(Y[gi], device=device)
                segs = _find_true_segments(Y[gi] <= -10.0)
                torch.manual_seed(gi)
                bits = sampler(y)
                pred = predict_spectrum(fwd, bits, H, W)
                mse = notch_mse_vec(pred, target_t, -10.0)
                best_mses.append(float(mse.min()))
                valids.append(int(valid_mask_vec(pred, segs, -12.0).any()))
            print(f"{mn:<14}{T:>4.0f}{np.mean(best_mses):>14.2f}"
                  f"{np.median(best_mses):>9.2f}{np.mean(valids):>12.2f}")
        del m; torch.cuda.empty_cache()
