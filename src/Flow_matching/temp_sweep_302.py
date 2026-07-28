# -*- coding: utf-8 -*-
"""hard target 302 에서 temperature 를 올리면 best notch-MSE 가 내려가나?
   (탐색 반경↑ 이 좋은 공진 마스크를 건지는지) — dfm_g4, dfm_mask_g4, N=4096, steps=10."""
import os, sys, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from table1_search import FM_CFG, build_fm, make_fm_sampler, valid_mask_vec, notch_mse_vec, H, W
from eval import load_forward_surrogate, predict_spectrum
from inverse_from_csv_10x10 import _find_true_segments

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
N, STEPS, TGT = 4096, 10, 302
TEMPS = [1.0, 1.5, 2.0, 3.0, 4.0]
MODELS = ["dfm_g4", "dfm_mask_g4"]

fwd = load_forward_surrogate(num_points=201, device=device)
te = np.load("/hai/home/lsh/antenna/year_hai/data/datasets/dataset_test.npz")
Y = te["Y"].astype(np.float32)
target_t = torch.as_tensor(Y[TGT], device=device)
segs = _find_true_segments(Y[TGT] <= -10.0)

print(f"target {TGT} (hard, T=1 에선 best≈145~219, valid 0), N={N}, steps={STEPS}")
print(f"GT notch segs(-10): {segs}\n")
print(f"{'model':<13}{'T':>5}{'bestMSE↓':>10}{'medianMSE':>11}{'valid#(-12)':>12}{'주notch최저깊이':>16}")
print("-" * 68)
for mn in MODELS:
    cfg = FM_CFG[mn]
    m = build_fm(cfg, device)
    y = torch.as_tensor(Y[TGT], device=device).view(1, -1).repeat(N, 1)
    for T in TEMPS:
        sampler = make_fm_sampler(m, cfg, STEPS, 1.0, temperature=T)
        torch.manual_seed(TGT)
        bits = sampler(y)
        pred = predict_spectrum(fwd, bits, H, W)                    # (N,P)
        mse = notch_mse_vec(pred, target_t, -10.0)                  # (N,)
        nvalid = int(valid_mask_vec(pred, segs, -12.0).sum().item())
        best = int(mse.argmin().item())
        # best 마스크가 주 notch(첫 segment) 를 얼마나 깊게 내나
        s, e = segs[0]
        deep = float(pred[best, s:e + 1].min().item())
        print(f"{mn:<13}{T:>5.1f}{float(mse[best]):>10.1f}{float(mse.median()):>11.1f}"
              f"{nvalid:>12}{deep:>14.1f}dB")
    del m; torch.cuda.empty_cache()
    print()
