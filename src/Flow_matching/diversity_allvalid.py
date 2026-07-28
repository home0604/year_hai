# -*- coding: utf-8 -*-
"""valid 여부와 무관하게 ALL / valid / invalid 부분집합별로 pairHam, GT-Ham 측정.
   → notch 안 생긴(invalid) 케이스도 mask 패턴이 뭉치는지(collapse) vs 퍼지는지 확인."""
import sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from diversity_analysis import Runner, pairwise_hamming

run = Runner(gpu=0, n=4096, steps=10)
TARGETS = {887:"single",1372:"single",1662:"single",952:"dual",1403:"dual",1666:"dual"}

def gtham(M, gt):
    return float((M != gt).sum(1).mean()) if len(M) else float("nan")

print(f"dfm_g4, N={run.N}, steps={run.STEPS}, T=1  — 부분집합별 pairHam / GT-Ham\n")
print(f"{'idx':>6}{'band':>8}{'valid%':>8} | {'ALL pH/GT':>14} | {'VALID pH/GT':>14} | {'INVALID pH/GT':>16}")
print("-"*74)
for tgt, band in TARGETS.items():
    B, ok, mse, pred = run.generate("dfm_g4", tgt)
    gt = run.gt(tgt)
    a = (pairwise_hamming(B), gtham(B, gt))
    v = (pairwise_hamming(B[ok]), gtham(B[ok], gt)) if ok.sum() > 1 else (float("nan"), gtham(B[ok], gt))
    iv = (pairwise_hamming(B[~ok]), gtham(B[~ok], gt)) if (~ok).sum() > 1 else (float("nan"), float("nan"))
    print(f"{tgt:>6}{band:>8}{ok.mean()*100:>7.1f}% | {a[0]:>6.1f}/{a[1]:<6.1f} | "
          f"{v[0]:>6.1f}/{v[1]:<6.1f} | {iv[0]:>7.1f}/{iv[1]:<7.1f}")
