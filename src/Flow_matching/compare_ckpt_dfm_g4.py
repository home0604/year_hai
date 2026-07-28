# -*- coding: utf-8 -*-
"""
dfm_g4 best_model(min val_loss, ep250) vs last_model(ep500) 를 **진짜 목표**
(forward-surrogate notch-MSE best-of-N + valid rate) 로 직접 비교.
  · candidate-parity: 같은 N, 같은 steps → 모델 품질만 격리.
  · target: table1 고정 인덱스 (single-100 + dual-50).
  · 지표 정의는 Table I 과 동일 (notch 영역 -10, valid -12).
"""
import os, sys, argparse, numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "MAB_code"))

from table1_search import (FM_CFG, make_fm_sampler, valid_mask_vec, notch_mse_vec,
                           H, W, NB, DATA, SCRATCH)
from eval import load_forward_surrogate, predict_spectrum
from model import build_model
from inverse_from_csv_10x10 import _find_true_segments


def build_dfm_g4(ckpt_path, device):
    m = build_model("stable3dit", num_points=201, num_bits=NB // 4, d_model=512, nhead=4,
                    num_layers=15, dim_feedforward=768, dropout=0.1, seq_len_y=50,
                    drop_prob=0.1, vocab_size=16, group_size=1).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]     # last_model.pth 포맷
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def eval_ckpt(ckpt_path, N, steps, device):
    m = build_dfm_g4(ckpt_path, device)
    sampler = make_fm_sampler(m, FM_CFG["dfm_g4"], steps, cfg_scale=1.0)
    fwd = eval_ckpt.fwd
    te = np.load(os.path.join(DATA, "dataset_test.npz"))
    Y = te["Y"].astype(np.float32)
    out = {}
    for kind in ("single", "dual"):
        idx = np.load(os.path.join(SCRATCH, f"table1_{kind}_idx.npy"))
        mses, valids = [], []
        for gi in idx:
            gi = int(gi)
            target_np = Y[gi]
            y = torch.as_tensor(target_np, device=device).view(1, -1)
            target_t = torch.as_tensor(target_np, device=device)
            segs = _find_true_segments(target_np <= -10.0)
            torch.manual_seed(gi)
            bits = sampler(y.repeat(N, 1))
            pred = predict_spectrum(fwd, bits, H, W)
            mse = notch_mse_vec(pred, target_t, -10.0)
            mses.append(float(mse.min().item()))
            valids.append(bool(valid_mask_vec(pred, segs, -12.0).any().item()))
        out[kind] = (float(np.mean(mses)), float(np.mean(valids)), len(idx))
    del m; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    eval_ckpt.fwd = load_forward_surrogate(num_points=201, device=device)

    ck = {
        "best_model(ep250,minValLoss)": "experiments/20260709/dfm_g4/best_model.pth",
        "last_model(ep500)":            "experiments/20260709/dfm_g4/last_model.pth",
    }
    print(f"dfm_g4 selector 비교 — best-of-{args.N}, steps={args.steps}, notch-MSE(-10)/valid(-12)\n")
    print(f"{'ckpt':<30}{'set':<8}{'notchMSE↓':>11}{'valid↑':>9}{'n':>5}")
    print("-" * 63)
    for name, path in ck.items():
        res = eval_ckpt(path, args.N, args.steps, device)
        for kind in ("single", "dual"):
            mse, val, n = res[kind]
            print(f"{name:<30}{kind:<8}{mse:>11.2f}{val:>9.3f}{n:>5}")
