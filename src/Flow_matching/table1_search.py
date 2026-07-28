# -*- coding: utf-8 -*-
"""
논문(260622_paper.pdf) Table I / Fig.16 재현용 **시간예산 탐색** 하네스.

프로토콜:
  · 각 target 마다 T초 wall-clock 예산으로 후보를 생성·평가.
  · backend 2종:
      - AR-MAB  : prefix(3bit) UCB bandit + stochastic suffix sampling (논문 §III-B, 저자 코드 재현)
      - FM      : MAB 없이 단순 반복 sampling (dfm / cfm / cfm-xpred)
  · notch 영역 정의 = notch_db(-10dB), valid 판정 = valid_db(-12dB)  ← 논문 정의대로 분리
  · 선택 기준 = notch-region MSE 최소 (best-of-search).

로깅 (사후에 Table I 숫자 + Fig.16 곡선 전부 재현 가능):
  · timeline : (config,set,target_idx, elapsed_sec, cum_cand, best_mse_so_far, cum_valid)  — pull/batch 마다 1행
  · summary  : (config,model,steps,set,target_idx, final_best_mse, valid_any@-12, total_cand, elapsed)
  · best_bits.npy : target별 best 후보 (raster {0,1})

  시간을 candidate 축과 함께 남기므로 T=5분 한 번으로 더 짧은 T 결과도 사후 재현된다.
  (warmup 등 탐색 궤적을 바꾸는 파라미터만 재실행 필요.)

사용:
  python table1_search.py --config dfm_g4_s10 --gpu 0 --T 300 --n_single 20 --n_dual 10 --out table1_out
  python table1_search.py --config ar         --gpu 1 --T 300 --n_single 20 --n_dual 10 --out table1_out
"""
import os
import sys
import csv
import time
import math
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "MAB_code"))

from eval import load_forward_surrogate, predict_spectrum                 # noqa: E402
from model import build_model                                             # noqa: E402
from flow_matching import (                                               # noqa: E402
    sample_fm_cfg, sample_dfm_cfg, sample_fm_xpred_cfg, sample_dfm_mask_cfg,
    make_spatial_patch_indices,
)
from inverse_from_csv_10x10 import (                                      # noqa: E402  ← 저자 코드
    loss_function_notch_mse, _find_true_segments,
    SmallTransformerAR, get_order_indices, canonicalize_under_yflip,
    _arm_to_prefix_bits, ar_sample_with_fixed_prefix_total,
)

DATA = "/hai/home/lsh/antenna/year_hai/data/datasets"
SCRATCH = ("/hai/home/lsh/.claude_temp/claude-1006/-hai-home-lsh-antenna/"
           "9bf495f2-7c3e-4f32-821b-aefb04235628/scratchpad")
H = W = 10
NB = H * W

AR_CKPT = ("/hai/home/lsh/antenna/year_hai/src/inverse_models_changing_ordering/"
           "ar10x10-L15-d512-h4-ff768-dr0.1-ordsnake-specresnet1d-2dpos0-canon1"
           "-lr0.0001-bs128-seed42/best_model.pth")

FM_CFG = {
    "dfm_g4":       dict(ckpt="experiments/20260709/dfm_g4/best_model.pth",       flow="dfm",       gsz=4, spatial=True),
    "dfm_g1":       dict(ckpt="experiments/20260709/dfm_g1/best_model.pth",       flow="dfm",       gsz=1, spatial=False),
    "cfm_g1":       dict(ckpt="experiments/20260709/cfm_g1/best_model.pth",       flow="cfm",       gsz=1, spatial=False),
    "cfm_g1_xpred": dict(ckpt="experiments/20260718/cfm_g1_xpred/best_model.pth", flow="cfm_xpred", gsz=1, spatial=False),
    "dfm_mask_g4":  dict(ckpt="experiments/20260726/dfm_mask_g4/best_model.pth",   flow="dfm_mask",  gsz=4, spatial=True),
    "dfm_mask_g1":  dict(ckpt="experiments/20260726/dfm_mask_g1/best_model.pth",   flow="dfm_mask",  gsz=1, spatial=False),
}

# MAB 하이퍼 (앞선 분석: 5분 예산에서 UCB 정상동작하도록 warmup=1, pull_batch=250)
PREFIX_LEN = 3
WARMUP = 1
PULL_BATCH = 250
MAB_C = 1.0
LOSS_CLIP = 50.0
PENALTY = 20.0


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def valid_mask_vec(pred, segs, thr):
    """check_overlap_ok_from_segments 의 벡터화: 모든 target notch 구간에서 pred_min<=thr.
    pred (B,P) 텐서 → bool (B,)."""
    B = pred.size(0)
    if not segs:
        return torch.ones(B, dtype=torch.bool, device=pred.device)
    ok = torch.ones(B, dtype=torch.bool, device=pred.device)
    for (s, e) in segs:
        ok &= (pred[:, s:e + 1].min(dim=1).values <= thr)
    return ok


def notch_mse_vec(pred, target_t, notch_db):
    """loss_function_notch_mse 를 배치로 (B,)."""
    return loss_function_notch_mse(pred, target_t, notch_threshold_db=notch_db)


# ---------------------------------------------------------------------------
# FM backend
# ---------------------------------------------------------------------------
def build_fm(cfg, device):
    flow, gsz = cfg["flow"], cfg["gsz"]
    build_flow = "cfm" if flow in ("cfm", "cfm_xpred") else "dfm"
    V = 2 ** gsz if gsz > 1 else 2
    nt = NB // gsz if gsz > 1 else NB
    m = build_model("stable3dit", num_points=201, num_bits=nt, d_model=512, nhead=4,
                    num_layers=15, dim_feedforward=768, dropout=0.1, seq_len_y=50,
                    drop_prob=0.1, vocab_size=V,
                    group_size=gsz if build_flow == "cfm" else 1,
                    mask_token=(flow == "dfm_mask")).to(device)
    m.load_state_dict(torch.load(cfg["ckpt"], map_location=device))
    return m.eval()


def make_fm_sampler(m, cfg, steps, cfg_scale=1.0, temperature=1.0):
    flow, gsz, spatial = cfg["flow"], cfg["gsz"], cfg["spatial"]
    fidx = None
    if spatial:
        ph = pw = int(round(gsz ** 0.5))
        fidx = torch.as_tensor(make_spatial_patch_indices(H, W, ph, pw)[0]).long()

    @torch.no_grad()
    def sampler(yb):
        if flow == "dfm":
            G = sample_dfm_cfg(m, yb, NB, steps, cfg_scale, group_size=gsz, temperature=temperature)
        elif flow == "dfm_mask":
            G = sample_dfm_mask_cfg(m, yb, NB, steps, cfg_scale, group_size=gsz, temperature=temperature)
        elif flow == "cfm":
            G = sample_fm_cfg(m, yb, NB, steps, cfg_scale, group_size=gsz)
        else:  # cfm_xpred
            G = sample_fm_xpred_cfg(m, yb, NB, steps, cfg_scale, group_size=gsz)
        G = G.long()
        if fidx is not None:                         # patch-scramble → raster
            Gr = torch.zeros_like(G)
            Gr[:, fidx.to(G.device)] = G
            G = Gr
        return G
    return sampler


@torch.no_grad()
def fm_search(sampler, fwd, y_row, target_np, T, notch_db, valid_db, batch, seed, device):
    y = torch.as_tensor(y_row, dtype=torch.float32, device=device).view(1, -1)
    target_t = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    segs = _find_true_segments(target_np <= notch_db)
    torch.manual_seed(seed)

    best_mse = math.inf
    best_bits = None
    cum = 0
    cum_valid = 0
    timeline = []
    t0 = time.perf_counter()
    deadline = t0 + T
    while time.perf_counter() < deadline:
        yb = y.repeat(batch, 1)
        bits = sampler(yb)                                  # (batch, NB) {0,1}
        pred = predict_spectrum(fwd, bits, H, W)            # (batch, P) on device
        mse = notch_mse_vec(pred, target_t, notch_db)       # (batch,)
        cum_valid += int(valid_mask_vec(pred, segs, valid_db).sum().item())
        jmin = int(mse.argmin().item())
        if float(mse[jmin]) < best_mse:
            best_mse = float(mse[jmin])
            best_bits = bits[jmin].detach().cpu().numpy().astype(np.uint8)
        cum += batch
        timeline.append((time.perf_counter() - t0, cum, best_mse, cum_valid))
    return timeline, best_mse, best_bits, cum, cum_valid, (time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# AR-MAB backend (저자 코드 재현: prefix UCB + stochastic suffix)
# ---------------------------------------------------------------------------
@torch.no_grad()
def armab_search(model, order_idx, fwd, y_row, target_np, T, notch_db, valid_db, seed, device):
    y = torch.as_tensor(y_row, dtype=torch.float32, device=device).view(1, -1)
    target_t = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    segs = _find_true_segments(target_np <= notch_db)
    order_idx_np = np.asarray(order_idx)

    num_arms = 1 << PREFIX_LEN
    N = np.zeros(num_arms, dtype=np.int64)
    S = np.zeros(num_arms, dtype=np.float64)
    mu = np.zeros(num_arms, dtype=np.float64)
    rng = np.random.default_rng(seed)

    def select_arm(total):
        if WARMUP > 0:
            need = np.flatnonzero(N < WARMUP)
            if need.size:
                minN = int(N[need].min())
                return int(rng.choice(need[N[need] == minN]))
        untried = np.flatnonzero(N == 0)
        if untried.size:
            return int(rng.choice(untried))
        ln_t = math.log(float(max(1, total)) + 1.0)
        ucb = mu + MAB_C * np.sqrt(ln_t / N.astype(np.float64))
        m = float(ucb.max())
        return int(rng.choice(np.flatnonzero(np.isclose(ucb, m))))

    best_mse = math.inf
    best_bits = None
    cum = 0
    cum_valid = 0
    timeline = []
    t0 = time.perf_counter()
    deadline = t0 + T
    while time.perf_counter() < deadline:
        arm = select_arm(int(N.sum()) + 1)
        prefix_bits = _arm_to_prefix_bits(arm, PREFIX_LEN)
        chain = ar_sample_with_fixed_prefix_total(
            model=model, y_1xP=y, num_bits=NB, prefix_bits=prefix_bits,
            total_n=PULL_BATCH, sampling="stochastic",
            micro_batch=PULL_BATCH, deadline=deadline)
        if chain.shape[0] == 0:
            break
        bits_flat = np.zeros_like(chain, dtype=np.int64)
        bits_flat[:, order_idx_np] = chain                 # chain → raster
        bits_flat = canonicalize_under_yflip(bits_flat, height=H, width=W)
        bits_t = torch.as_tensor(bits_flat, dtype=torch.long, device=device)
        pred = predict_spectrum(fwd, bits_t, H, W)          # (B,P)
        mse = notch_mse_vec(pred, target_t, notch_db)       # (B,)
        ok10 = valid_mask_vec(pred, segs, notch_db)         # reward 용(-10, 저자와 동일)
        cum_valid += int(valid_mask_vec(pred, segs, valid_db).sum().item())  # 보고용(-12)

        lv = torch.clamp(mse, max=LOSS_CLIP)
        reward = -lv - torch.where(ok10, torch.zeros_like(lv),
                                   torch.full_like(lv, PENALTY))
        r_max = float(reward.max().item())
        N[arm] += 1
        S[arm] += r_max
        mu[arm] = S[arm] / N[arm]

        jmin = int(mse.argmin().item())
        if float(mse[jmin]) < best_mse:
            best_mse = float(mse[jmin])
            best_bits = bits_flat[jmin].astype(np.uint8)
        cum += bits_flat.shape[0]
        timeline.append((time.perf_counter() - t0, cum, best_mse, cum_valid))
    return timeline, best_mse, best_bits, cum, cum_valid, (time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def load_targets(kind, n):
    te = np.load(os.path.join(DATA, "dataset_test.npz"))
    Y = te["Y"].astype(np.float32)
    idx = np.load(os.path.join(SCRATCH, f"table1_{kind}_idx.npy"))[:n]
    return idx, Y


def main():
    global WARMUP, PULL_BATCH, PREFIX_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="ar | <model>_s<steps>  (model: dfm_g4/dfm_g1/cfm_g1/cfm_g1_xpred)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--n_single", type=int, default=20)
    ap.add_argument("--n_dual", type=int, default=10)
    ap.add_argument("--notch_db", type=float, default=-10.0)
    ap.add_argument("--valid_db", type=float, default=-12.0)
    ap.add_argument("--fm_batch", type=int, default=0, help="0 = steps 로 자동")
    ap.add_argument("--seed", type=int, default=0)
    # MAB(AR) 하이퍼 override — 원본 저자 default: warmup=10, pull_batch=500
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--pull_batch", type=int, default=PULL_BATCH)
    ap.add_argument("--prefix_len", type=int, default=PREFIX_LEN)
    ap.add_argument("--out", type=str, default="table1_out")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    WARMUP, PULL_BATCH, PREFIX_LEN = args.warmup, args.pull_batch, args.prefix_len
    if args.config == "ar":
        print(f"[MAB] prefix_len={PREFIX_LEN} arms={1<<PREFIX_LEN} warmup={WARMUP} pull_batch={PULL_BATCH}",
              flush=True)

    is_ar = (args.config == "ar")
    if is_ar:
        model_name, steps = "ar", 0
    else:
        model_name, s = args.config.rsplit("_s", 1)
        steps = int(s)
        assert model_name in FM_CFG, f"unknown model {model_name}"

    fwd = load_forward_surrogate(num_points=201, device=device)

    if is_ar:
        oi = get_order_indices("snake", NB, H, W)
        model = SmallTransformerAR(
            num_points=201, d_model=512, nhead=4, num_layers=15, dim_feedforward=768,
            max_len=NB, vocab_size=3, dropout=0.1, spectral_cond="resnet1d",
            use_2d_pos=False, chain2spatial=torch.from_numpy(oi).long(),
            height=H, width=W).to(device).eval()
        model.load_state_dict(torch.load(AR_CKPT, map_location=device))
    else:
        cfg = FM_CFG[model_name]
        model = build_fm(cfg, device)
        sampler = make_fm_sampler(model, cfg, steps, cfg_scale=1.0)
        fm_batch = args.fm_batch or (2000 if steps <= 4 else 500)

    sum_path = os.path.join(args.out, f"{args.config}.summary.csv")
    tl_path = os.path.join(args.out, f"{args.config}.timeline.csv")
    sf = open(sum_path, "w", newline="")
    tf = open(tl_path, "w", newline="")
    sw = csv.writer(sf); tw = csv.writer(tf)
    sw.writerow(["config", "model", "steps", "set", "target_idx",
                 "final_best_mse", "valid_any", "total_cand", "elapsed_sec"])
    tw.writerow(["config", "set", "target_idx", "elapsed_sec", "cum_cand",
                 "best_mse", "cum_valid"])

    best_bits_all = {}
    for kind, n in (("single", args.n_single), ("dual", args.n_dual)):
        idx, Y = load_targets(kind, n)
        for ti, gi in enumerate(idx):
            gi = int(gi)
            y_row = Y[gi]
            target_np = Y[gi]
            seed = args.seed * 100003 + gi
            t_start = time.perf_counter()
            if is_ar:
                tl, bmse, bbits, cum, cvalid, el = armab_search(
                    model, oi, fwd, y_row, target_np, args.T,
                    args.notch_db, args.valid_db, seed, device)
            else:
                tl, bmse, bbits, cum, cvalid, el = fm_search(
                    sampler, fwd, y_row, target_np, args.T,
                    args.notch_db, args.valid_db, fm_batch, seed, device)
            valid_any = 1 if cvalid > 0 else 0
            sw.writerow([args.config, model_name, steps, kind, gi,
                         f"{bmse:.6f}", valid_any, cum, f"{el:.2f}"])
            for (esec, cc, bm, cv) in tl:
                tw.writerow([args.config, kind, gi, f"{esec:.3f}", cc,
                             f"{bm:.6f}", cv])
            sf.flush(); tf.flush()
            if bbits is not None:
                best_bits_all[f"{kind}_{gi}"] = bbits
            print(f"[{args.config}] {kind} {ti+1}/{n} idx={gi} "
                  f"best_mse={bmse:.3f} valid={valid_any} cand={cum} "
                  f"({el:.0f}s)", flush=True)

    sf.close(); tf.close()
    np.savez(os.path.join(args.out, f"{args.config}.best_bits.npz"), **best_bits_all)
    print(f"✅ done {args.config} → {sum_path}", flush=True)


if __name__ == "__main__":
    main()
