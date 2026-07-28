# -*- coding: utf-8 -*-
"""sampling temperature 를 올리면 다양성이 회복되나? (완료 모델 dfm_g4, dfm_mask_g4)
  target 1666(풀림)에서 T∈{1,1.5,2,3} × N=4096 → valid%, ALL/VALID 다양성, GT거리.
  가설: T↑ → ALL 다양성↑ 하지만 valid%↓, distinct-VALID 는 안 늘어남(학습 안 된 mode 못 만듦)."""
import os, sys, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from table1_search import FM_CFG, build_fm, make_fm_sampler, valid_mask_vec, H, W
from eval import load_forward_surrogate, predict_spectrum
from inverse_from_csv_10x10 import _find_true_segments, canonicalize_under_yflip

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
N, STEPS, TGT = 4096, 10, 1666
TEMPS = [1.0, 1.5, 2.0, 3.0]
MODELS = ["dfm_g4", "dfm_mask_g4"]

fwd = load_forward_surrogate(num_points=201, device=device)
te = np.load("/hai/home/lsh/antenna/year_hai/data/datasets/dataset_test.npz")
Y = te["Y"].astype(np.float32); X = te["X"].astype(np.int32)
gt = canonicalize_under_yflip(X[TGT].reshape(1, -1).astype(np.int64), H, W)[0]
segs = _find_true_segments(Y[TGT] <= -10.0)


def pham(M):
    if len(M) < 2: return 0.0
    if len(M) > 500: M = M[np.random.default_rng(0).choice(len(M), 500, replace=False)]
    Hd = (M[:, None, :] != M[None, :, :]).sum(-1)
    return float(Hd[np.triu_indices(len(M), 1)].mean())


def vendi(M):
    n = len(M)
    if n < 2: return float(n)
    if n > 500: M = M[np.random.default_rng(0).choice(n, 500, replace=False)]; n = 500
    K = (1.0 - (M[:, None, :] != M[None, :, :]).sum(-1) / 100.0) / n
    w = np.linalg.eigvalsh(K); w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


print(f"target {TGT}, N={N}, steps={STEPS}\n")
print(f"{'model':<13}{'T':>5}{'valid%':>8} | {'ALL pHam/Vendi':>16} | {'VALID pHam/Vendi/#dist':>24} | {'GTham a/min':>12}")
print("-" * 88)
for mn in MODELS:
    cfg = FM_CFG[mn]
    m = build_fm(cfg, device)
    y = torch.as_tensor(Y[TGT], device=device).view(1, -1).repeat(N, 1)
    for T in TEMPS:
        sampler = make_fm_sampler(m, cfg, STEPS, 1.0, temperature=T)
        torch.manual_seed(TGT)
        bits = sampler(y)
        pred = predict_spectrum(fwd, bits, H, W)
        ok = valid_mask_vec(pred, segs, -12.0).cpu().numpy()
        B = canonicalize_under_yflip(bits.cpu().numpy().astype(np.int64), H, W)
        aH, aV = pham(B), vendi(B)
        V = B[ok]
        if len(V) == 0:
            print(f"{mn:<13}{T:>5.1f}{ok.mean()*100:>7.1f}% | {aH:>7.1f}/{aV:<7.1f} | {'(0 valid)':>24} | {'—':>12}")
            continue
        Vu = np.unique(V, axis=0); gth = (V != gt).sum(1)
        print(f"{mn:<13}{T:>5.1f}{ok.mean()*100:>7.1f}% | {aH:>7.1f}/{aV:<7.1f} | "
              f"{pham(V):>7.1f}/{vendi(V):<6.1f}/{len(Vu):>4} | {gth.mean():>7.1f}/{gth.min():<4}")
    del m; torch.cuda.empty_cache()
    print()
