# -*- coding: utf-8 -*-
"""
AR / DFM / CFM 을 **같은 측정방식(자유생성)** 으로 나란히 잰다.

wandb 의 AR BitAcc 는 teacher-forced 값이라 FM 의 생성 BitAcc 와 비교할 수 없다.
맨 아래 대조행에 그 값을 같이 찍어 격차를 보여준다.

usage:
  python compare_modes.py --n 2048 --steps 10
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_ordering_no_sample_images import (        # noqa: E402
    SmallTransformerAR, get_order_indices, canonicalize_under_yflip,
)
from model import build_model                         # noqa: E402
from flow_matching import make_spatial_patch_indices  # noqa: E402
from measurement_modes import (                       # noqa: E402
    free_running_ar, free_running_fm, teacher_forced_ar, support_rate,
)

DATA = "/hai/home/lsh/antenna/year_hai/data/datasets"
FM_EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments/20260709")
AR_CKPT = ("/hai/home/lsh/antenna/year_hai/src/inverse_models_changing_ordering/"
           "ar10x10-L15-d512-h4-ff768-dr0.1-ordsnake-specresnet1d-2dpos0-canon1"
           "-lr0.0001-bs128-seed42/best_model.pth")

FM_MODELS = [           # (tag, flow_type, group_size, spatial)
    ("dfm_g1", "dfm", 1, False),
    ("dfm_g4", "dfm", 4, True),
    ("cfm_g1", "cfm", 1, False),
]


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
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 10, 50],
                   help="FM sampling steps (= NFE, 병렬)")
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--ar_temp", type=float, nargs="+", default=[1.0])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    te = np.load(os.path.join(DATA, "dataset_test.npz"))
    X2d = te["X"].astype(np.float32)
    Y = te["Y"].astype(np.float32)
    _, H, W = X2d.shape
    D = H * W
    Xcanon = canonicalize_under_yflip(X2d.reshape(len(X2d), D), H, W)
    Y_t = torch.tensor(Y)[:args.n]
    n = len(Y_t)

    rows = []       # (model, sampler, NFE, bit, pat, support)

    def score(gen, gt):
        return ((gen == gt).float().mean().item(),
                (gen == gt).all(1).float().mean().item())

    # ---------------- AR ----------------
    oi = get_order_indices("snake", D, H, W)
    X_ar = torch.tensor(Xcanon[:, oi])[:n].long()
    ar = SmallTransformerAR(
        num_points=201, d_model=512, nhead=4, num_layers=15, dim_feedforward=768,
        max_len=D, vocab_size=3, dropout=0.1, spectral_cond="resnet1d",
        use_2d_pos=False, chain2spatial=torch.from_numpy(oi).long(),
        height=H, width=W).to(device).eval()
    ar.load_state_dict(torch.load(AR_CKPT, map_location=device))

    def ar_gen(**kw):
        g = []
        for i in range(0, n, args.bs):
            j = min(i + args.bs, n)
            g.append(free_running_ar(ar, Y_t[i:j].to(device), D, **kw).cpu())
        return torch.cat(g)

    gen = ar_gen(greedy=True)
    b, pa = score(gen, X_ar)
    rows.append(("AR (snake)", "greedy", 100, b, pa, support_rate(gen)))
    for T in args.ar_temp:
        gen = ar_gen(temperature=T)
        b, pa = score(gen, X_ar)
        rows.append(("AR (snake)", f"ancestral T={T}", 100, b, pa, support_rate(gen)))

    # 대조행: wandb 가 보는 값 (생성 지표가 아님)
    tb = tp = 0.0
    for i in range(0, n, args.bs):
        j = min(i + args.bs, n)
        b, pa = teacher_forced_ar(ar, Y_t[i:j].to(device), X_ar[i:j].to(device))
        tb += b * (j - i)
        tp += pa * (j - i)
    ar_tf = (tb / n, tp / n)
    del ar
    torch.cuda.empty_cache()

    # ---------------- FM ----------------
    for tag, flow, gsz, spatial in FM_MODELS:
        Xf = Xcanon
        if spatial and gsz > 1:
            ph = pw = int(gsz ** 0.5)
            fwd_idx, _ = make_spatial_patch_indices(H, W, ph, pw)
            Xf = Xcanon[:, fwd_idx.numpy()]
        X_fm = torch.tensor(Xf)[:n].long()
        m = load_fm(tag, flow, gsz, device)

        for steps in args.steps:
            g = []
            for i in range(0, n, args.bs):
                j = min(i + args.bs, n)
                g.append(free_running_fm(m, Y_t[i:j].to(device), D, steps,
                                         args.cfg, flow, gsz).cpu())
            gen = torch.cat(g)
            b, pa = score(gen, X_fm)
            rows.append((tag, f"{steps}-step cfg={args.cfg}", steps, b, pa,
                         support_rate(gen)))
        del m
        torch.cuda.empty_cache()

    # ---------------- report ----------------
    W_ = 92
    print(f"\n생성(free-running) 성능 — n = {n} test targets, seed = {args.seed}\n")
    print(f"{'model':<11} {'sampler':<18} {'NFE':>4} {'par?':>5} "
          f"{'BitAcc':>7} {'PatAcc':>7} {'ones':>12} {'P(=60)':>7}")
    print("-" * W_)
    last = None
    for tag, samp, nfe, b, pa, sup in rows:
        if last and last != tag:
            print("-" * W_)
        last = tag
        par = "순차" if tag.startswith("AR") else "병렬"
        print(f"{tag:<11} {samp:<18} {nfe:>4} {par:>5} {b:>7.4f} {pa:>7.4f} "
              f"{sup['ones_mean']:>6.1f}±{sup['ones_std']:<5.2f} "
              f"{sup['support_rate']:>7.3f}")
    print("=" * W_)
    print(f"{'AR (snake)':<11} {'teacher-forced':<18} {'—':>4} {'—':>5} "
          f"{ar_tf[0]:>7.4f} {ar_tf[1]:>7.4f} {'—':>12} {'—':>7}   ← wandb 가 보는 값")
    print("-" * W_)
    print("teacher-forced 는 정답 prefix 를 넣고 다음 비트를 맞히는 값 — 생성 지표가 아니다.")
    print("모델 간 비교는 위쪽 생성 행으로만 한다. P(=60) 은 데이터 support 준수율.")
