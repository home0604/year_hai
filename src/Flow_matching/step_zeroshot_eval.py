# -*- coding: utf-8 -*-
"""
Zero-shot: fine-tuning 없이 step-function 타겟을 그대로 넣으면 얼마나 손해인가?

핵심 ablation — 같은 타겟, 같은 모델, **조건 입력만** 바꾼다:
  (A) 실제 S11 곡선으로 조건       ← 학습 분포 (in-distribution)
  (B) step-function 사양으로 조건  ← 논문의 추론 형식 (out-of-distribution)

평가는 양쪽 모두 **동일하게** 한다 (저자 코드):
  notch MSE  : loss_function_notch_mse(pred, step_target)  — 두 조건 모두 step 타겟 기준
  valid      : check_overlap_ok_from_segments(pred, step 의 notch 구간, -12 dB)
  ResFreqErr : |argmin(pred) − argmin(실제 S11)| — 진짜 공진이 어디 나야 하는가

기준선:
  GT 마스크  : 달성 가능한 하한 (surrogate 오차 + step 타겟의 이상화 오차)
  무작위 60-ones : chance
"""
import os
import sys
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (_HERE, os.path.dirname(_HERE), os.path.join(os.path.dirname(_HERE), "MAB_code")):
    sys.path.insert(0, p_)

from train_ordering_no_sample_images import (               # noqa: E402
    SmallTransformerAR, get_order_indices, canonicalize_under_yflip,
)
from inverse_from_csv_10x10 import (                        # noqa: E402  ← 저자 코드
    loss_function_notch_mse, check_overlap_ok_from_segments, _find_true_segments,
)
from model import build_model                               # noqa: E402
from flow_matching import make_spatial_patch_indices        # noqa: E402
from eval import load_forward_surrogate, predict_spectrum   # noqa: E402
from measurement_modes import free_running_ar, free_running_fm   # noqa: E402

STEP = "/hai/home/lsh/antenna/year_hai/data/datasets/step"
FM_EXP = os.path.join(_HERE, "experiments/20260709")
AR_CKPT = ("/hai/home/lsh/antenna/year_hai/src/inverse_models_changing_ordering/"
           "ar10x10-L15-d512-h4-ff768-dr0.1-ordsnake-specresnet1d-2dpos0-canon1"
           "-lr0.0001-bs128-seed42/best_model.pth")
H = W = 10
D = 100


@torch.no_grad()
def score(fwd, bits, step_np, orig_np, freq, valid_db):
    """bits (N,k,100) raster → 논문식 지표. 타겟은 step, 공진위치는 실제 S11 기준."""
    dev = next(fwd.parameters()).device
    N, k, _ = bits.shape
    pred = predict_spectrum(fwd, bits.reshape(N * k, D), H, W).reshape(N, k, -1)
    f = torch.tensor(freq)
    ms, va, vb, rfe = [], [], [], []
    for i in range(N):
        t = torch.tensor(step_np[i], device=dev)
        mk = loss_function_notch_mse(pred[i], t, notch_threshold_db=-10.0)
        b = int(mk.argmin())
        ms.append(float(mk[b]))
        segs = _find_true_segments(step_np[i] <= -10.0)
        pk = pred[i].cpu().numpy()
        oks = [check_overlap_ok_from_segments(pk[j], segs, threshold_db=valid_db)
               for j in range(k)]
        va.append(any(oks))
        vb.append(oks[b])
        rfe.append(float((f[int(pred[i, b].cpu().argmin())]
                          - f[int(orig_np[i].argmin())]).abs()))
    return dict(notch_mse=float(np.mean(ms)), valid_any=float(np.mean(va)),
                valid_best=float(np.mean(vb)), res_freq_err=float(np.mean(rfe)),
                res_match=float(np.mean(np.array(rfe) <= 0.05)))


def load_fm(tag, flow, gsz, dev):
    V = 2 ** gsz if gsz > 1 else 2
    nt = D // gsz if gsz > 1 else D
    m = build_model("stable3dit", num_points=201, num_bits=nt, d_model=512,
                    nhead=4, num_layers=15, dim_feedforward=768, dropout=0.1,
                    seq_len_y=50, drop_prob=0.1, vocab_size=V,
                    group_size=gsz if flow == "cfm" else 1).to(dev)
    m.load_state_dict(torch.load(f"{FM_EXP}/{tag}/best_model.pth", map_location=dev))
    return m.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--valid_db", type=float, default=-12.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda")
    z = np.load(os.path.join(STEP, "step_resonant_test.npz"))
    X, Ystep, Yorig, freq = z["X"], z["Y"], z["Y_orig"], z["freq"]
    n = min(args.n, len(X))
    X, Ystep, Yorig = X[:n], Ystep[:n], Yorig[:n]
    Xflat = X.reshape(n, D)

    print(f"공진 test 타겟 {n} 개 · best-of-{args.k} · FM steps={args.steps}, cfg=1.0")
    print(f"valid 판정: 모든 notch 구간에서 예측 S11 < {args.valid_db} dB\n")

    fwd = load_forward_surrogate(num_points=201, device=dev)
    oi = get_order_indices("snake", D, H, W)
    rows = []

    def add(model_name, cond_name, bits_raster):
        rows.append((model_name, cond_name,
                     score(fwd, bits_raster, Ystep, Yorig, freq, args.valid_db)))

    # ---------- 기준선 ----------
    gt = torch.tensor(Xflat).long().unsqueeze(1).repeat(1, args.k, 1)
    add("GT 마스크", "—", gt)
    torch.manual_seed(args.seed)
    rnd = torch.zeros(n, args.k, D, dtype=torch.long)
    for i in range(n):
        for j in range(args.k):
            rnd[i, j, torch.randperm(D)[:60]] = 1
    add("무작위 60-ones", "—", rnd)

    # ---------- AR ----------
    ar = SmallTransformerAR(
        num_points=201, d_model=512, nhead=4, num_layers=15, dim_feedforward=768,
        max_len=D, vocab_size=3, dropout=0.1, spectral_cond="resnet1d",
        use_2d_pos=False, chain2spatial=torch.from_numpy(oi).long(),
        height=H, width=W).to(dev).eval()
    ar.load_state_dict(torch.load(AR_CKPT, map_location=dev))
    for cond_name, Ycond in [("실제 S11", Yorig), ("step 사양", Ystep)]:
        torch.manual_seed(args.seed)
        G = []
        Yt = torch.tensor(Ycond)
        for i in range(0, n, args.bs):
            j = min(i + args.bs, n)
            yr = Yt[i:j].to(dev).repeat_interleave(args.k, 0)
            G.append(free_running_ar(ar, yr, D, temperature=1.0)
                     .view(j - i, args.k, D).cpu())
        G = torch.cat(G)
        Gr = torch.zeros_like(G)
        Gr[:, :, torch.from_numpy(oi)] = G                  # snake → raster
        add("AR (snake)", cond_name, Gr)
    del ar
    torch.cuda.empty_cache()

    # ---------- FM ----------
    for tag, name, flow, gsz, spatial in [("dfm_g1", "DFM V=2", "dfm", 1, False),
                                          ("dfm_g4", "DFM V=16", "dfm", 4, True)]:
        m = load_fm(tag, flow, gsz, dev)
        fidx = None
        if spatial:
            ph = pw = int(gsz ** 0.5)
            fidx, _ = make_spatial_patch_indices(H, W, ph, pw)
        for cond_name, Ycond in [("실제 S11", Yorig), ("step 사양", Ystep)]:
            torch.manual_seed(args.seed)
            G = []
            Yt = torch.tensor(Ycond)
            for i in range(0, n, args.bs):
                j = min(i + args.bs, n)
                yr = Yt[i:j].to(dev).repeat_interleave(args.k, 0)
                G.append(free_running_fm(m, yr, D, args.steps, 1.0, flow, gsz)
                         .view(j - i, args.k, D).cpu())
            G = torch.cat(G)
            if fidx is not None:                            # 2×2 patch → raster
                Gr = torch.zeros_like(G)
                Gr[:, :, fidx] = G
                G = Gr
            add(name, cond_name, G)
        del m
        torch.cuda.empty_cache()

    # ---------- 보고 ----------
    print(f"{'모델':<15}{'조건 입력':<12}{'notch MSE↓':>11}{'valid@10↑':>10}"
          f"{'valid(best)↑':>13}{'ResFreqErr↓':>12}{'ResMatch↑':>10}")
    print("-" * 84)
    prev = None
    for name, cond, r in rows:
        if prev and prev != name:
            print("-" * 84)
        prev = name
        print(f"{name:<15}{cond:<12}{r['notch_mse']:>11.2f}{r['valid_any']:>10.3f}"
              f"{r['valid_best']:>13.3f}{r['res_freq_err']:>12.4f}{r['res_match']:>10.3f}")
    print("-" * 84)
    print("notch MSE 는 두 조건 모두 **step 타겟 기준**이라 비교 가능하다.")
    print("ResFreqErr 는 실제 S11 의 공진 위치 기준 (GHz).")
