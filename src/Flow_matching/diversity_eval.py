# -*- coding: utf-8 -*-
"""1666(풀림)·302(실패) target 에서 FM 모델들의 생성 마스크 **다양성** 평가.
  · N개 생성 → canonicalize(y-flip 대칭 제거) → valid(-12dB) 추림.
  · 지표: valid rate, #distinct valid, 평균 pairwise Hamming(valid), Vendi score,
          GT까지 Hamming(평균/최소).  → one-to-many(다양한 valid) vs GT-재현 판정.
"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from table1_search import FM_CFG, build_fm, make_fm_sampler, valid_mask_vec, notch_mse_vec, H, W, NB
from eval import load_forward_surrogate, predict_spectrum
from inverse_from_csv_10x10 import _find_true_segments, canonicalize_under_yflip

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
N = 4096; STEPS = 10
fwd = load_forward_surrogate(num_points=201, device=device)
te = np.load("/hai/home/lsh/antenna/year_hai/data/datasets/dataset_test.npz")
Y = te["Y"].astype(np.float32); X = te["X"].astype(np.int32)

MODELS = ["dfm_g4", "dfm_g1", "cfm_g1_xpred", "dfm_mask_g4", "dfm_mask_g1"]
TARGETS = [1666, 302]


def pairwise_hamming(M):                      # M (n,100) → 평균 쌍 bit차이
    if len(M) < 2: return 0.0
    Hd = (M[:, None, :] != M[None, :, :]).sum(-1)
    iu = np.triu_indices(len(M), 1)
    return float(Hd[iu].mean())


def vendi(M):                                 # 유효 설계 실효 개수
    n = len(M)
    if n < 2: return float(n)
    Hf = (M[:, None, :] != M[None, :, :]).sum(-1) / 100.0
    K = (1.0 - Hf) / n
    w = np.linalg.eigvalsh(K); w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


@torch.no_grad()
def gen(model_name, tgt):
    cfg = FM_CFG[model_name]
    m = build_fm(cfg, device)
    sampler = make_fm_sampler(m, cfg, STEPS, 1.0)
    y = torch.as_tensor(Y[tgt], device=device).view(1, -1).repeat(N, 1)
    torch.manual_seed(tgt)
    bits = sampler(y)                                    # (N,100) raster
    pred = predict_spectrum(fwd, bits, H, W)
    segs = _find_true_segments(Y[tgt] <= -10.0)
    ok = valid_mask_vec(pred, segs, -12.0).cpu().numpy()
    B = canonicalize_under_yflip(bits.cpu().numpy().astype(np.int64), H, W)  # 대칭 제거
    del m; torch.cuda.empty_cache()
    return B, ok


gt_all = {t: canonicalize_under_yflip(X[t].reshape(1, -1).astype(np.int64), H, W)[0] for t in TARGETS}
print(f"N={N}, steps={STEPS}\n")
hist = {}
def sub(M, k=500):
    if len(M) <= k: return M
    return M[np.random.default_rng(0).choice(len(M), k, replace=False)]

for tgt in TARGETS:
    print(f"===== target {tgt} ({'풀림' if tgt==1666 else '실패'}) =====")
    print(f"{'model':<14}{'valid%':>7} | {'ALL: pairHam / Vendi':>22} | "
          f"{'VALID: pairHam / Vendi / #distinct':>34} | {'GTham a/min':>12}")
    for mn in MODELS:
        B, ok = gen(mn, tgt)                             # B: 전체 4096 (canonical), ok: valid mask
        vr = ok.mean()
        aH = pairwise_hamming(sub(B)); aV = vendi(sub(B))       # 전체-샘플 다양성
        V = B[ok]
        if len(V) == 0:
            print(f"{mn:<14}{vr*100:>6.1f}% | {aH:>9.1f} / {aV:>8.1f} | "
                  f"{'(0 valid)':>34} | {'—':>12}")
            continue
        Vu = np.unique(V, axis=0)
        vH = pairwise_hamming(sub(V)); vV = vendi(sub(V))       # valid-샘플 다양성
        gth = (V != gt_all[tgt]).sum(1)
        print(f"{mn:<14}{vr*100:>6.1f}% | {aH:>9.1f} / {aV:>8.1f} | "
              f"{vH:>9.1f} / {vV:>8.1f} / {len(Vu):>6} | {gth.mean():>7.1f}/{gth.min():<4}")
        hist[(tgt, mn)] = gth

# 1666 valid 후보의 GT까지 Hamming 분포 → 재현 vs 다양성 시각화
plt.figure(figsize=(7, 4))
for mn in MODELS:
    if (1666, mn) in hist:
        plt.hist(hist[(1666, mn)], bins=range(0, 60, 2), alpha=0.5, label=mn)
plt.xlabel("Hamming distance to GT mask (valid candidates, target 1666)")
plt.ylabel("count"); plt.legend(); plt.title("valid 후보가 GT 근처로 뭉치나 vs 퍼지나 (1666)")
plt.tight_layout(); plt.savefig("table1_out/diversity_1666.png", dpi=130)
print("\nsaved -> table1_out/diversity_1666.png")
