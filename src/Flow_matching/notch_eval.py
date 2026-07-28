# -*- coding: utf-8 -*-
"""
논문(260622_paper.pdf) 과 같은 방식의 평가 — notch-region MSE + valid-candidate rate.

동기:
  eval.py 의 FwdMSE 는 201 점 **전체** 평균이고, test 타겟의 **74% 는 공진이 아예 없다**
  (논문 §III-A: non-resonant 73.18%; 우리 test set 실측 73.5%).
  평평한 스펙트럼을 평평하게 예측하는 것만으로 점수가 나오므로 주 지표가 희석된다.
  ResFreqErr 도 공진 없는 타겟에서는 argmin 이 노이즈의 최저점이라 의미가 없다.

  논문은 **notch 영역에만** MSE 를 매기고, **valid candidate 여부**로 성능을 센다.

지표 (전부 저자 코드 MAB_code/inverse_from_csv_10x10.py 를 그대로 호출):
  notch MSE   : loss_function_notch_mse  — 타겟이 -10dB 이하인 주파수점에서만 MSE
  valid rate  : check_overlap_ok_from_segments — 타겟의 **모든** notch 구간에서
                예측 S11 이 threshold 아래로 내려가야 유효 (논문은 -12dB 를 씀)
  best-of-k   : notch MSE 최소 후보를 채택 (논문 §III-B 와 동일한 선택 기준)
"""
import os
import sys
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "MAB_code"))

from train_ordering_no_sample_images import (              # noqa: E402
    SmallTransformerAR, get_order_indices, canonicalize_under_yflip,
)
from inverse_from_csv_10x10 import (                       # noqa: E402  ← 저자 코드
    loss_function_notch_mse, check_overlap_ok_from_segments, _find_true_segments,
)
from model import build_model                              # noqa: E402
from flow_matching import make_spatial_patch_indices       # noqa: E402
from eval import load_forward_surrogate, predict_spectrum  # noqa: E402
from measurement_modes import free_running_ar, free_running_fm  # noqa: E402

DATA = "/hai/home/lsh/antenna/year_hai/data/datasets"
FM_EXP = os.path.join(_HERE, "experiments/20260709")
AR_CKPT = ("/hai/home/lsh/antenna/year_hai/src/inverse_models_changing_ordering/"
           "ar10x10-L15-d512-h4-ff768-dr0.1-ordsnake-specresnet1d-2dpos0-canon1"
           "-lr0.0001-bs128-seed42/best_model.pth")


@torch.no_grad()
def evaluate(fwd, gen_bits, Y_np, freq_np, H, W, notch_db, valid_db):
    """gen_bits (N,k,D) long, raster 순서. → 논문식 지표 dict."""
    device = next(fwd.parameters()).device
    N, k, D = gen_bits.shape
    pred = predict_spectrum(fwd, gen_bits.reshape(N * k, D), H, W).reshape(N, k, -1)

    notch_mses, valid_any, valid_best = [], [], []
    rf_err, depth_err = [], []
    f = torch.tensor(freq_np)

    for i in range(N):
        tgt = torch.tensor(Y_np[i], device=device)
        # --- 저자 코드: notch-region MSE (타겟 <= notch_db 인 점들만) ---
        mse_k = loss_function_notch_mse(pred[i], tgt, notch_threshold_db=notch_db)  # (k,)
        b = int(mse_k.argmin())                       # 논문과 동일: notch MSE 최소 후보
        notch_mses.append(float(mse_k[b]))

        # --- 저자 코드: valid candidate (모든 타겟 notch 구간에서 threshold 아래) ---
        segs = _find_true_segments((Y_np[i] <= notch_db))
        pk = pred[i].cpu().numpy()
        oks = [check_overlap_ok_from_segments(pk[j], segs, threshold_db=valid_db)
               for j in range(k)]
        valid_any.append(any(oks))                    # k 개 중 하나라도 유효한가
        valid_best.append(oks[b])                     # best 후보가 유효한가

        bp = pred[i, b].cpu()
        rf_err.append(float((f[int(bp.argmin())] - f[int(Y_np[i].argmin())]).abs()))
        depth_err.append(float(abs(bp.min().item() - Y_np[i].min())))

    return {
        "notch_mse": float(np.mean(notch_mses)),
        "valid_any": float(np.mean(valid_any)),
        "valid_best": float(np.mean(valid_best)),
        "res_freq_err": float(np.mean(rf_err)),
        "res_depth_err": float(np.mean(depth_err)),
    }


def load_fm(tag, flow, gsz, device):
    V = 2 ** gsz if gsz > 1 else 2
    nt = 100 // gsz if gsz > 1 else 100
    m = build_model("stable3dit", num_points=201, num_bits=nt, d_model=512,
                    nhead=4, num_layers=15, dim_feedforward=768, dropout=0.1,
                    seq_len_y=50, drop_prob=0.1, vocab_size=V,
                    group_size=gsz if flow == "cfm" else 1).to(device)
    m.load_state_dict(torch.load(f"{FM_EXP}/{tag}/best_model.pth", map_location=device))
    return m.eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--notch_db", type=float, default=-10.0, help="notch 영역 정의 (저자 기본값)")
    p.add_argument("--valid_db", type=float, default=-12.0, help="valid candidate 판정 (논문 §III-C)")
    p.add_argument("--n", type=int, default=1000, help="공진 타겟 개수")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--bs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda")
    te = np.load(os.path.join(DATA, "dataset_test.npz"))
    X2d = te["X"].astype(np.float32)
    Y = te["Y"].astype(np.float32)
    freq = te["freq"].astype(np.float32)
    _, H, W = X2d.shape
    D = H * W

    res = Y.min(1) < args.notch_db
    idx = np.where(res)[0][:args.n]
    Y_res = Y[idx]
    Yt = torch.tensor(Y_res)
    n = len(idx)
    nseg = [len(_find_true_segments(Y_res[i] <= args.notch_db)) for i in range(n)]
    npts = (Y_res <= args.notch_db).sum(1)

    print(f"공진 타겟 (min S11 < {args.notch_db} dB): {res.sum()} / {len(Y)} "
          f"({res.mean()*100:.1f}%)  → 앞의 {n} 개 사용")
    print(f"타겟당 notch 구간 {np.mean(nseg):.2f} 개 / 주파수점 {npts.mean():.1f} 개 (201 중)")
    print(f"valid 판정: 모든 notch 구간에서 예측 S11 < {args.valid_db} dB\n")

    fwd = load_forward_surrogate(num_points=201, device=device)
    rows = []

    # ---------------- AR ----------------
    oi = get_order_indices("snake", D, H, W)
    ar = SmallTransformerAR(
        num_points=201, d_model=512, nhead=4, num_layers=15, dim_feedforward=768,
        max_len=D, vocab_size=3, dropout=0.1, spectral_cond="resnet1d",
        use_2d_pos=False, chain2spatial=torch.from_numpy(oi).long(),
        height=H, width=W).to(device).eval()
    ar.load_state_dict(torch.load(AR_CKPT, map_location=device))
    torch.manual_seed(args.seed)
    G = []
    for i in range(0, n, args.bs):
        j = min(i + args.bs, n)
        yr = Yt[i:j].to(device).repeat_interleave(args.k, 0)
        G.append(free_running_ar(ar, yr, D, temperature=1.0).view(j - i, args.k, D).cpu())
    G = torch.cat(G)
    Gr = torch.zeros_like(G)
    Gr[:, :, torch.from_numpy(oi)] = G                      # snake → raster
    rows.append(("AR (snake)", "100 순차",
                 evaluate(fwd, Gr, Y_res, freq, H, W, args.notch_db, args.valid_db)))
    del ar
    torch.cuda.empty_cache()

    # ---------------- FM ----------------
    FM = [("cfm_g1", "CFM", "cfm", 1, False, [1.0]),
          ("dfm_g1", "DFM V=2", "dfm", 1, False, [1.0]),
          ("dfm_g4", "DFM V=16", "dfm", 4, True, [1.0, 1.5])]
    for tag, name, flow, gsz, spatial, cfgs in FM:
        m = load_fm(tag, flow, gsz, device)
        fidx = None
        if spatial:
            ph = pw = int(gsz ** 0.5)
            fidx, _ = make_spatial_patch_indices(H, W, ph, pw)
        for cfg in cfgs:
            torch.manual_seed(args.seed)
            G = []
            for i in range(0, n, args.bs):
                j = min(i + args.bs, n)
                yr = Yt[i:j].to(device).repeat_interleave(args.k, 0)
                G.append(free_running_fm(m, yr, D, args.steps, cfg, flow, gsz)
                         .view(j - i, args.k, D).cpu())
            G = torch.cat(G)
            if fidx is not None:                            # 2×2 patch → raster
                Gr = torch.zeros_like(G)
                Gr[:, :, fidx] = G
                G = Gr
            label = f"{args.steps} 병렬" + (f" · cfg {cfg}" if cfg != 1.0 else "")
            rows.append((name, label,
                         evaluate(fwd, G, Y_res, freq, H, W, args.notch_db, args.valid_db)))
        del m
        torch.cuda.empty_cache()

    print(f"공진 타겟 {n} 개 · best-of-{args.k} (선택기준: notch MSE — 논문과 동일)\n")
    print(f"{'모델':<11}{'NFE':<17}{'notch MSE↓':>11}{'valid@10↑':>11}"
          f"{'valid(best)↑':>13}{'ResFreqErr↓':>12}{'ResDepth↓':>10}")
    print('-' * 86)
    for name, nfe, r in rows:
        print(f"{name:<11}{nfe:<17}{r['notch_mse']:>11.2f}{r['valid_any']:>11.3f}"
              f"{r['valid_best']:>13.3f}{r['res_freq_err']:>12.4f}{r['res_depth_err']:>10.3f}")
