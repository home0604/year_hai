# -*- coding: utf-8 -*-
"""
FM(DFM/CFM) 생성 마스크 **다양성 / collapse** 분석.

배경:
  · 하나의 target 스펙트럼 y 에 대해 물리적으로는 여러 마스크가 같은 응답을 냄(one-to-many).
  · 하지만 학습 데이터는 y 당 마스크 1개(one-to-one supervision) → 모델이 배운 p(x|y) 가
    GT 마스크에 near-델타로 집중돼 있는지(=재현/collapse), 아니면 다양한 valid 설계를
    내는지(=진짜 one-to-many)를 정량화한다.

지표:
  · pairwise Hamming : 마스크 쌍 평균 bit 차이 (/100). 0 = 동일, 큼 = 다양.
  · Vendi score      : 유사도 커널(1-Hamming/100) 고유값 엔트로피의 exp
                       = "실효 distinct 샘플 개수". 1 이면 사실상 1가지.
  · #distinct valid  : canonicalize(y-flip 대칭 제거) 후 unique valid 개수.
  · GT Hamming       : valid 마스크가 GT 에서 몇 bit 떨어졌나 (재현 여부).
  · valid            : target 의 모든 notch 구간(-10dB 정의)에서 예측 S11 <= valid_db(-12).

통제:
  · canonicalize_under_yflip 로 y-flip 대칭(=같은 스펙트럼) 제거 후 다양성 측정(가짜 다양성 방지).
  · 다양성은 반드시 VALID 부분집합에서도 따로 본다(diverse-but-invalid 는 one-to-many 아님).

모드:
  diversity : 모델 × target 별 ALL vs VALID 다양성 (temperature=1)
  temp      : target 하나에서 temperature sweep — valid% + ALL/VALID 다양성
  temp_mse  : hard target 에서 temperature sweep — best notch-MSE (탐색이 fat-tail 줄이나)

예:
  python diversity_analysis.py --mode diversity --targets 1666 302
  python diversity_analysis.py --mode temp      --target 1666 --models dfm_g4 dfm_mask_g4 --temps 1 1.5 2 3
  python diversity_analysis.py --mode temp_mse  --target 302  --models dfm_g4 dfm_mask_g4 --temps 1 2 3 4

주의: dfm_mask_g1(20260726) 은 학습 미완(ep355)이라 수치가 과대평가될 수 있음.
"""
import os
import sys
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "MAB_code"))

from table1_search import (FM_CFG, build_fm, make_fm_sampler,               # noqa: E402
                           valid_mask_vec, notch_mse_vec, H, W)
from eval import load_forward_surrogate, predict_spectrum                    # noqa: E402
from inverse_from_csv_10x10 import _find_true_segments, canonicalize_under_yflip  # noqa: E402

DATA = "/hai/home/lsh/antenna/year_hai/data/datasets"
NOTCH_DB, VALID_DB = -10.0, -12.0
_RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------
def _cap(M, k=500):
    return M if len(M) <= k else M[_RNG.choice(len(M), k, replace=False)]


def pairwise_hamming(M):
    """마스크(n,100) 쌍 평균 bit 차이."""
    M = _cap(M)
    if len(M) < 2:
        return 0.0
    Hd = (M[:, None, :] != M[None, :, :]).sum(-1)
    return float(Hd[np.triu_indices(len(M), 1)].mean())


def vendi(M):
    """Vendi score = exp(엔트로피(유사도커널 고유값)) = 실효 distinct 개수."""
    M = _cap(M)
    n = len(M)
    if n < 2:
        return float(n)
    K = (1.0 - (M[:, None, :] != M[None, :, :]).sum(-1) / 100.0) / n
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


# --------------------------------------------------------------------------
# 생성 + 평가
# --------------------------------------------------------------------------
class Runner:
    def __init__(self, gpu=0, n=4096, steps=10):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
        self.device = torch.device("cuda")
        self.N, self.STEPS = n, steps
        self.fwd = load_forward_surrogate(num_points=201, device=self.device)
        te = np.load(os.path.join(DATA, "dataset_test.npz"))
        self.Y = te["Y"].astype(np.float32)
        self.X = te["X"].astype(np.int32)
        self.freq = te["freq"]
        self._cache = {}

    def _model(self, name):
        if name not in self._cache:
            self._cache[name] = build_fm(FM_CFG[name], self.device)
        return self._cache[name]

    def gt(self, tgt):
        return canonicalize_under_yflip(self.X[tgt].reshape(1, -1).astype(np.int64), H, W)[0]

    def segs(self, tgt):
        return _find_true_segments(self.Y[tgt] <= NOTCH_DB)

    @torch.no_grad()
    def generate(self, name, tgt, temperature=1.0, cfg_scale=1.0):
        """N개 생성 → (canonical 마스크 B(N,100), valid bool, notch-MSE(N,), pred(N,201))."""
        cfg = FM_CFG[name]
        m = self._model(name)
        sampler = make_fm_sampler(m, cfg, self.STEPS, cfg_scale, temperature=temperature)
        y = torch.as_tensor(self.Y[tgt], device=self.device).view(1, -1).repeat(self.N, 1)
        torch.manual_seed(tgt)
        bits = sampler(y)                                     # (N,100) raster
        pred = predict_spectrum(self.fwd, bits, H, W)         # (N,201)
        mse = notch_mse_vec(pred, torch.as_tensor(self.Y[tgt], device=self.device), NOTCH_DB)
        ok = valid_mask_vec(pred, self.segs(tgt), VALID_DB).cpu().numpy()
        B = canonicalize_under_yflip(bits.cpu().numpy().astype(np.int64), H, W)
        return B, ok, mse.cpu().numpy(), pred.cpu().numpy()


# --------------------------------------------------------------------------
# 모드
# --------------------------------------------------------------------------
def mode_diversity(run, models, targets):
    print(f"[diversity] N={run.N}, steps={run.STEPS}, T=1  (notch{NOTCH_DB}/valid{VALID_DB})\n")
    for tgt in targets:
        print(f"===== target {tgt} =====")
        print(f"{'model':<14}{'valid%':>7} | {'ALL pHam/Vendi':>16} | "
              f"{'VALID pHam/Vendi/#dist':>24} | {'GTham a/min':>12}")
        for mn in models:
            B, ok, _, _ = run.generate(mn, tgt)
            aH, aV = pairwise_hamming(B), vendi(B)
            V = B[ok]
            if len(V) == 0:
                print(f"{mn:<14}{ok.mean()*100:>6.1f}% | {aH:>7.1f}/{aV:<7.1f} | "
                      f"{'(0 valid)':>24} | {'-':>12}")
                continue
            gth = (V != run.gt(tgt)).sum(1)
            print(f"{mn:<14}{ok.mean()*100:>6.1f}% | {aH:>7.1f}/{aV:<7.1f} | "
                  f"{pairwise_hamming(V):>7.1f}/{vendi(V):<6.1f}/{len(np.unique(V,axis=0)):>4} | "
                  f"{gth.mean():>7.1f}/{gth.min():<4}")
        print()


def mode_temp(run, models, tgt, temps):
    print(f"[temp] target {tgt}, N={run.N}, steps={run.STEPS}\n")
    print(f"{'model':<13}{'T':>5}{'valid%':>8} | {'ALL pHam/Vendi':>16} | "
          f"{'VALID pHam/Vendi/#dist':>24} | {'GTham a/min':>12}")
    print("-" * 88)
    for mn in models:
        for T in temps:
            B, ok, _, _ = run.generate(mn, tgt, temperature=T)
            aH, aV = pairwise_hamming(B), vendi(B)
            V = B[ok]
            if len(V) == 0:
                print(f"{mn:<13}{T:>5.1f}{ok.mean()*100:>7.1f}% | {aH:>7.1f}/{aV:<7.1f} | "
                      f"{'(0 valid)':>24} | {'-':>12}")
                continue
            gth = (V != run.gt(tgt)).sum(1)
            print(f"{mn:<13}{T:>5.1f}{ok.mean()*100:>7.1f}% | {aH:>7.1f}/{aV:<7.1f} | "
                  f"{pairwise_hamming(V):>7.1f}/{vendi(V):<6.1f}/{len(np.unique(V,axis=0)):>4} | "
                  f"{gth.mean():>7.1f}/{gth.min():<4}")
        print()


def mode_temp_mse(run, models, tgt, temps):
    print(f"[temp_mse] target {tgt}, N={run.N}, steps={run.STEPS}  (best-of-N notch-MSE)\n")
    segs = run.segs(tgt)
    print(f"GT notch segs(-10): {segs}")
    print(f"{'model':<13}{'T':>5}{'bestMSE':>10}{'medMSE':>9}{'valid#':>8}{'주notch깊이':>13}")
    print("-" * 60)
    for mn in models:
        for T in temps:
            B, ok, mse, pred = run.generate(mn, tgt, temperature=T)
            b = int(mse.argmin())
            s, e = segs[0]
            deep = float(pred[b, s:e + 1].min())
            print(f"{mn:<13}{T:>5.1f}{mse[b]:>10.1f}{np.median(mse):>9.1f}"
                  f"{int(ok.sum()):>8}{deep:>11.1f}dB")
        print()


def mode_cfg(run, models, targets, cfgs):
    """guidance scale sweep — 조건 부합(best-MSE, valid%) vs 다양성(Vendi) 트레이드오프."""
    print(f"[cfg] N={run.N}, steps={run.STEPS}, T=1\n")
    for tgt in targets:
        print(f"===== target {tgt} =====")
        print(f"{'model':<13}{'cfg':>5}{'bestMSE':>9}{'valid%':>8} | "
              f"{'ALL Vendi':>10}{'VALID Vendi/#dist':>18}")
        for mn in models:
            for w in cfgs:
                B, ok, mse, _ = run.generate(mn, tgt, temperature=1.0, cfg_scale=w)
                aV = vendi(B)
                V = B[ok]
                vv = f"{vendi(V):.1f}/{len(np.unique(V,axis=0))}" if len(V) else "-"
                print(f"{mn:<13}{w:>5.1f}{mse.min():>9.1f}{ok.mean()*100:>7.1f}% | "
                      f"{aV:>10.1f}{vv:>18}")
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["diversity", "temp", "temp_mse", "cfg"], default="diversity")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--models", nargs="+",
                    default=["dfm_g4", "dfm_g1", "cfm_g1_xpred", "dfm_mask_g4"])
    ap.add_argument("--targets", nargs="+", type=int, default=[1666, 302])  # diversity 모드
    ap.add_argument("--target", type=int, default=1666)                     # temp/temp_mse 모드
    ap.add_argument("--temps", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--cfgs", nargs="+", type=float, default=[1.0, 2.0, 4.0])  # cfg 모드
    args = ap.parse_args()

    run = Runner(gpu=args.gpu, n=args.n, steps=args.steps)
    if args.mode == "diversity":
        mode_diversity(run, args.models, args.targets)
    elif args.mode == "temp":
        mode_temp(run, args.models, args.target, args.temps)
    elif args.mode == "temp_mse":
        mode_temp_mse(run, args.models, args.target, args.temps)
    else:
        mode_cfg(run, args.models, args.targets, args.cfgs)
