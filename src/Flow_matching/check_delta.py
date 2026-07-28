# -*- coding: utf-8 -*-
"""p(x|y) 가 진짜 near-delta 인지 **직접** 확인.

간접(출력 다양성) 이 아니라, DFM 샘플링 마지막 스텝에서 모델이 내놓는
per-token categorical(softmax)/Bernoulli(sigmoid) 분포의 뾰족함을 측정:
  · max-prob : 토큰별 최대확률 (1.0 이면 one-hot = delta)
  · entropy  : 토큰별 엔트로피 (nats). 0 = delta, 균등이면 log V
  · %near-det: max-prob > 0.9 인 토큰 비율
delta 면 max-prob≈1, entropy≈0. 우리 가설: 1666(풀림)에서 near-delta.
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from table1_search import FM_CFG, build_fm, H, W, NB
from flow_matching import _keep_schedule, _cfg_logits
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device = torch.device("cuda")


@torch.no_grad()
def last_step_probs(name, tgt, Y, B=512, steps=10):
    """샘플러를 실제로 돌려 마지막 스텝의 per-token 예측분포를 반환.
       grouped: probs (B,T,V) / binary: p1 (B,100)."""
    cfg = FM_CFG[name]; gsz = cfg["gsz"]; flow = cfg["flow"]
    m = build_fm(cfg, device)
    y = torch.as_tensor(Y[tgt], device=device).view(1, -1).repeat(B, 1)
    alphas, jumps = _keep_schedule(steps)
    torch.manual_seed(tgt)

    if gsz <= 1:                                   # binary DFM (dfm_g1)
        x = torch.randint(0, 2, (B, NB), device=device).float()
        cap = None
        for i in range(steps):
            t = torch.full((B,), alphas[i], device=device)
            logits = _cfg_logits(m, x * 2.0 - 1.0, t, y, 1.0)
            p1 = torch.sigmoid(logits)
            if i == steps - 1:
                cap = p1                            # 마지막 스텝 예측
            resample = ((p1 > 0.5).float() if i == steps - 1
                        else (torch.rand_like(p1) < p1).float())
            do = torch.rand(B, NB, device=device) < jumps[i]
            x = torch.where(do, resample, x)
        conf = torch.maximum(cap, 1 - cap)          # Bernoulli confidence
        ent = -(cap.clamp(1e-9) * cap.clamp(1e-9).log()
                + (1 - cap).clamp(1e-9) * (1 - cap).clamp(1e-9).log())
        return conf.reshape(-1).cpu().numpy(), ent.reshape(-1).cpu().numpy(), np.log(2)
    else:                                           # grouped DFM (dfm_g4, dfm_mask_g4)
        V = 2 ** gsz; T = NB // gsz
        is_mask = (flow == "dfm_mask")
        x = (torch.full((B, T), V, device=device, dtype=torch.long) if is_mask
             else torch.randint(0, V, (B, T), device=device))
        cap = None
        for i in range(steps):
            t = torch.full((B,), alphas[i], device=device)
            logits = _cfg_logits(m, x, t, y, 1.0)
            probs = F.softmax(logits, dim=-1)       # (B,T,V)
            if i == steps - 1:
                cap = probs
                cand = probs.argmax(-1)
            else:
                cand = torch.multinomial(probs.reshape(-1, V), 1).reshape(B, T)
            if is_mask:
                do = (torch.rand(B, T, device=device) < jumps[i]) & (x == V)
            else:
                do = torch.rand(B, T, device=device) < jumps[i]
            x = torch.where(do, cand, x)
        maxp = cap.max(-1).values
        ent = -(cap.clamp(1e-9) * cap.clamp(1e-9).log()).sum(-1)
        return maxp.reshape(-1).cpu().numpy(), ent.reshape(-1).cpu().numpy(), float(np.log(V))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dfm_g4", "dfm_g1", "dfm_mask_g4"])
    ap.add_argument("--targets", nargs="+", type=int, default=[1666, 302])
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    te = np.load("/hai/home/lsh/antenna/year_hai/data/datasets/dataset_test.npz")
    Y = te["Y"].astype(np.float32)
    print(f"마지막 스텝 per-token 예측분포 뾰족함 (B={args.B}, steps={args.steps})")
    print("delta = max-prob≈1, entropy≈0\n")
    print(f"{'model':<14}{'target':>7}{'maxProb 평균':>13}{'entropy 평균':>13}"
          f"{'ent/균등':>10}{'%maxP>0.9':>11}")
    print("-" * 68)
    for mn in args.models:
        for tgt in args.targets:
            conf, ent, hmax = last_step_probs(mn, tgt, Y, args.B, args.steps)
            print(f"{mn:<14}{tgt:>7}{conf.mean():>13.3f}{ent.mean():>13.3f}"
                  f"{ent.mean()/hmax:>10.3f}{(conf>0.9).mean()*100:>10.1f}%")
