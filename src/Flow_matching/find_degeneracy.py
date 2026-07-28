# -*- coding: utf-8 -*-
"""데이터에 many-to-one(물리적 축퇴)이 실재하는가?

학습셋에서 || f(x) - f(x') || 작은데 Hamming(x,x') 큰 쌍을 전수 탐색.
  · f = 실제 EM 스펙트럼(dataset Y). (surrogate 근사 아님 — 진짜 물리, 이미 계산돼 있음)
  · y-flip 대칭(=동일 스펙트럼)은 canonicalize 로 제거 → 비자명 축퇴만.
  · 효율: hamming/specMSE 둘 다 matmul 로 (b,N) 한 번에.

판정:
  far-Hamming(>H_THR) 쌍이 near-duplicate 수준의 작은 specMSE 를 달성하면 → 축퇴 실재.
  그런 쌍이 거의 없으면 → 이 해상도에서 스펙이 설계를 거의 결정.
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from inverse_from_csv_10x10 import canonicalize_under_yflip

_ap = argparse.ArgumentParser()
_ap.add_argument("--resonant", action="store_true",
                 help="resonant(min S11 < notch_db) 마스크끼리만 — non-resonant 평평-평평 아티팩트 제거")
_ap.add_argument("--single", action="store_true", help="single(정확히 1 notch)만")
_ap.add_argument("--dual", action="store_true", help="dual(>=2 notch)만")
_ap.add_argument("--notch_db", type=float, default=-10.0)
_args = _ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device = torch.device("cuda")
H = W = 10
H_THR = 20          # 이보다 많은 픽셀이 다르면 "구조적으로 다른 설계"
CHUNK = 512

tr = np.load("/hai/home/lsh/antenna/year_hai/data/datasets/dataset_train.npz")
_Yall = tr["Y"].astype(np.float32)
_Xall = tr["X"].reshape(len(tr["X"]), -1).astype(np.int64)
if _args.resonant or _args.dual or _args.single:
    if _args.dual or _args.single:
        from inverse_from_csv_10x10 import _find_true_segments
        nseg = np.array([len(_find_true_segments(_Yall[i] <= _args.notch_db))
                         for i in range(len(_Yall))])
        keep = (nseg >= 2) if _args.dual else (nseg == 1)
        print(f"[{'dual' if _args.dual else 'single'} 필터] {keep.sum()} / {len(_Yall)}")
    else:
        keep = _Yall.min(1) < _args.notch_db
        print(f"[resonant 필터] min S11 < {_args.notch_db}dB : {keep.sum()} / {len(_Yall)}")
    _Xall, _Yall = _Xall[keep], _Yall[keep]
Xc = canonicalize_under_yflip(_Xall, H, W)
# canonical 중복 제거(완전 동일 마스크는 하나만 — y-flip/중복 표본 제거)
Xc, uniq = np.unique(Xc, axis=0, return_index=True)
Y = _Yall[uniq]
N = len(Xc)
print(f"canonical unique 마스크: {N} (필터후 {len(_Xall)})\n")

Xg = torch.as_tensor(Xc, dtype=torch.float32, device=device)      # (N,100)
Yg = torch.as_tensor(Y, device=device)                            # (N,201)
ones = Xg.sum(1)                                                  # (N,)
ynorm = (Yg * Yg).sum(1)                                          # (N,)

nn_spec = torch.empty(N, device=device)          # 전체 최근접 specMSE (자기 제외)
far_spec = torch.full((N,), float("inf"), device=device)  # far-Hamming 중 최소 specMSE
far_ham = torch.zeros(N, device=device)          # 그 때의 Hamming
far_j = torch.full((N,), -1, dtype=torch.long, device=device)

for i0 in range(0, N, CHUNK):
    i1 = min(i0 + CHUNK, N)
    Xq, Yq = Xg[i0:i1], Yg[i0:i1]
    dot = Xq @ Xg.T                                              # (b,N)
    ham = ones[i0:i1, None] + ones[None, :] - 2 * dot            # (b,N)
    spec = (ynorm[i0:i1, None] + ynorm[None, :] - 2 * (Yq @ Yg.T)) / 201.0
    spec = spec.clamp_min(0.0)
    idx = torch.arange(i0, i1, device=device)
    spec[torch.arange(i1 - i0), idx] = float("inf")              # 자기 제외
    nn_spec[i0:i1] = spec.min(1).values
    spec_far = torch.where(ham > H_THR, spec, torch.full_like(spec, float("inf")))
    mv, mj = spec_far.min(1)
    far_spec[i0:i1] = mv
    far_ham[i0:i1] = ham[torch.arange(i1 - i0), mj]
    far_j[i0:i1] = mj

nn = nn_spec.cpu().numpy()
fs = far_spec.cpu().numpy()
fh = far_ham.cpu().numpy()
valid = np.isfinite(fs)

print("=== specMSE(dB^2) 분포 ===")
print(f"전체 최근접(any Hamming)  : median {np.median(nn):.3f}, p10 {np.percentile(nn,10):.3f}, min {nn.min():.4f}")
print(f"far-Hamming(>{H_THR}) 최근접 : median {np.median(fs[valid]):.3f}, p10 {np.percentile(fs[valid],10):.3f}, min {fs[valid].min():.4f}")
print(f"  (far 쌍이 존재하는 마스크: {valid.sum()}/{N})\n")

print("=== 축퇴 판정: far-Hamming 쌍이 specMSE 얼마나 작아지나 ===")
for tau in [0.1, 0.5, 1.0, 2.0, 5.0]:
    c = (fs[valid] < tau).sum()
    print(f"  specMSE < {tau:>4} dB^2  &  Hamming>{H_THR} 인 마스크: {c} ({c/N*100:.2f}%)")

# 가장 강한 축퇴 예시(작은 specMSE + 큰 Hamming)
order = np.argsort(fs)
print(f"\n=== 최강 축퇴 예시 (작은 specMSE, Hamming>{H_THR}) ===")
print(f"{'i':>7}{'j':>7}{'Hamming':>9}{'specMSE':>10}")
for i in order[:12]:
    print(f"{i:>7}{far_j[i].item():>7}{int(fh[i]):>9}{fs[i]:>10.4f}")
np.savez(os.path.join(os.path.dirname(__file__), "degeneracy_result.npz"),
         nn_spec=nn, far_spec=fs, far_ham=fh, far_j=far_j.cpu().numpy())
print("\nsaved -> degeneracy_result.npz")
