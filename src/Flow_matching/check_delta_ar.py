# -*- coding: utf-8 -*-
"""AR 의 per-token 예측분포 뾰족함 측정 — FM(check_delta) 과 직접 비교.

AR 은 픽셀마다 Bernoulli(p=sigmoid(logit)) 로 순차 생성. 각 픽셀 생성 시점의 p 를 캡처해
confidence=max(p,1-p), entropy 를 집계. loose fit(=diversity) 가설이면 FM(max-prob 0.99,
near-delta)보다 확실히 soft 해야 함.
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from inverse_from_csv_10x10 import SmallTransformerAR, get_order_indices, BOS_IDX
from table1_search import AR_CKPT, H, W, NB

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device = torch.device("cuda")


def load_ar():
    oi = get_order_indices("snake", NB, H, W)
    m = SmallTransformerAR(
        num_points=201, d_model=512, nhead=4, num_layers=15, dim_feedforward=768,
        max_len=NB, vocab_size=3, dropout=0.1, spectral_cond="resnet1d",
        use_2d_pos=False, chain2spatial=torch.from_numpy(oi).long(),
        height=H, width=W).to(device).eval()
    m.load_state_dict(torch.load(AR_CKPT, map_location=device))
    return m


@torch.no_grad()
def ar_token_stats(m, y_row, Y, B=512, temperature=1.0):
    y = torch.as_tensor(y_row, dtype=torch.float32, device=device).view(1, -1).repeat(B, 1)
    tokens = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
    confs, ents = [], []
    for _ in range(NB):
        p = torch.sigmoid(m(y, tokens)[:, -1] / temperature).clamp(1e-6, 1 - 1e-6)  # (B,) fp32 안전
        confs.append(torch.maximum(p, 1 - p))
        ents.append(-(p * p.log() + (1 - p) * (1 - p).log()))
        nxt = (torch.rand_like(p) < p).long()                    # stochastic (생성과 동일)
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
    conf = torch.stack(confs, 1).reshape(-1).cpu().numpy()
    ent = torch.stack(ents, 1).reshape(-1).cpu().numpy()
    return conf, ent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", type=int, default=[1666, 302])
    ap.add_argument("--B", type=int, default=512)
    args = ap.parse_args()
    te = np.load("/hai/home/lsh/antenna/year_hai/data/datasets/dataset_test.npz")
    Y = te["Y"].astype(np.float32)
    m = load_ar()
    hmax = np.log(2)   # binary 균등 엔트로피
    print(f"AR per-token(픽셀) Bernoulli 뾰족함 (B={args.B})")
    print("비교: dfm_g1 = maxProb 0.998, ent/균등 0.010, %>0.9 = 99.4% (near-delta)\n")
    print(f"{'target':>7}{'maxProb 평균':>13}{'entropy 평균':>13}{'ent/균등':>10}{'%conf>0.9':>11}")
    print("-" * 54)
    for tgt in args.targets:
        conf, ent = ar_token_stats(m, Y[tgt], Y, args.B)
        print(f"{tgt:>7}{conf.mean():>13.3f}{ent.mean():>13.3f}"
              f"{ent.mean()/hmax:>10.3f}{(conf>0.9).mean()*100:>10.1f}%")
