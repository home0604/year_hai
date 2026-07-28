# -*- coding: utf-8 -*-
"""
step-function 타겟 데이터셋 생성 — 논문(260622_paper.pdf) §II 의 fine-tuning 셋.

변환 규칙 (논문 p.5):
  "each retained S11 response is converted into a simplified step-function target
   spectrum by assigning rectangular notch intervals to the frequency regions
   satisfying S11 < -10 dB. During this conversion, the pixelated layout itself
   is not modified but is simply paired with the newly assigned step-function target."

  → step[s11 <= NOTCH_DB] = DEPTH_DB,  나머지 0 dB.  마스크는 불변.
  → EM 시뮬레이션 불필요. 기존 데이터셋만으로 만든다.

값 규약은 추론용 타겟(specs/desired/step_mask/masks.zip, 153 개)에서 그대로 읽었다:
  201 점 / 5.0–7.0 GHz / S11 ∈ {0, -20} 이진 계단.

논문과의 차이 (의도적):
  논문의 fine-tuning 700 쌍은 **AR 이 생성한 후보**에서 나왔다 → AR 출력 분포에 편향된
  self-distillation 셋이다. 그걸로 DFM 을 fine-tune 하면 DFM 에 불리하다.
  여기서는 **GT 샘플에서 직접** step 타겟을 만든다 → 모델 중립적이고 양도 더 많다.
"""
import os
import argparse
import numpy as np

DATA = "/hai/home/lsh/antenna/year_hai/data/datasets"
NOTCH_DB = -10.0     # 이 값 이하를 notch 로 본다 (저자 코드 기본값)
DEPTH_DB = -20.0     # step 타겟의 notch 깊이 (masks.zip 규약)


def n_segments(below):
    """연속 True 구간 개수."""
    n, prev = 0, False
    for b in below:
        if b and not prev:
            n += 1
        prev = b
    return n


def to_step(Y, notch_db=NOTCH_DB, depth_db=DEPTH_DB):
    """(N,201) 실제 S11 → (N,201) 계단 타겟."""
    step = np.zeros_like(Y)
    step[Y <= notch_db] = depth_db
    return step


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=os.path.join(DATA, "step"))
    p.add_argument("--mode", type=str, default="resonant",
                   choices=["resonant", "dual"],
                   help="resonant = notch 1개 이상 (single+dual) / dual = notch 2개 이상")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"변환 규칙: step[S11 <= {NOTCH_DB} dB] = {DEPTH_DB} dB, 나머지 0 dB")
    print(f"선별 기준: {args.mode}\n")

    for sp in ["train", "valid", "test"]:
        z = np.load(os.path.join(DATA, f"dataset_{sp}.npz"))
        X, Y, freq = z["X"], z["Y"].astype(np.float32), z["freq"]

        below = Y <= NOTCH_DB
        nseg = np.array([n_segments(b) for b in below])
        keep = nseg >= (2 if args.mode == "dual" else 1)

        Xk, Yk = X[keep], Y[keep]
        Sk = to_step(Yk)

        out = os.path.join(args.out, f"step_{args.mode}_{sp}.npz")
        np.savez_compressed(out, X=Xk, Y=Sk, Y_orig=Yk, freq=freq,
                            n_notch=nseg[keep])

        n1 = int((nseg[keep] == 1).sum())
        n2 = int((nseg[keep] >= 2).sum())
        pts = below[keep].sum(1)
        print(f"[{sp:>5}] {keep.sum():>6} / {len(Y):>6} 유지 "
              f"(single {n1}, dual {n2})  "
              f"notch 폭 평균 {pts.mean():.1f} / 201 점  →  {out}")

    # ---- 추론용 153 개 dual-band 타겟도 npz 로 변환해 둔다 ----
    import zipfile
    zp = "/hai/home/lsh/antenna/year_hai/specs/desired/step_mask/masks.zip"
    if os.path.exists(zp):
        z = zipfile.ZipFile(zp)
        names = sorted(n for n in z.namelist() if n.endswith(".csv"))
        specs, tags = [], []
        for n in names:
            rows = z.read(n).decode("utf-8", "replace").strip().splitlines()[1:]
            specs.append([float(r.split(",")[1]) for r in rows])
            tags.append(os.path.basename(n)[:-4])          # e.g. "5.1_5.3"
        S = np.asarray(specs, dtype=np.float32)
        out = os.path.join(args.out, "step_eval_153.npz")
        np.savez_compressed(out, Y=S, tags=np.array(tags), freq=freq)
        print(f"\n[eval ] 153 dual-band 타겟 → {out}")
        print(f"        S11 고유값 {np.unique(S)}  notch 폭 평균 "
              f"{(S < 0).sum(1).mean():.1f} / 201 점")
