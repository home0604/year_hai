# -*- coding: utf-8 -*-
"""
새 in-distribution AR 체크포인트(local_snake, ordering=snake, canon0)의
teacher-forced bit_acc / pattern_acc 를 학습 코드와 동일 파이프라인으로 재현.
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from train_ordering_no_sample_images import (  # noqa: E402
    SmallTransformerAR, get_order_indices, canonicalize_under_yflip, BOS_IDX,
)

DEFAULT_CKPT = ("/hai/home/lsh/antenna/year_hai/src/inverse_models_changing_ordering/"
                "ar10x10-L15-d512-h4-ff768-dr0.1-ordsnake-specresnet1d-2dpos0-canon0-lr0.0001-bs128-seed42/"
                "best_model.pth")


@torch.no_grad()
def eval_bitacc(model, Y_t, X_t, device, bs=256):
    model.eval()
    N = Y_t.size(0)
    tot_bit = tot_pat = 0.0
    for s in range(0, N, bs):
        yb = Y_t[s:s+bs].to(device)
        xb = X_t[s:s+bs].to(device)
        B, _ = xb.shape
        bos = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
        tokens_in = torch.cat([bos, xb[:, :-1].long()], dim=1)
        targets = xb
        logits = model(yb, tokens_in)
        preds = (torch.sigmoid(logits) > 0.5).float()
        tot_bit += (preds == targets).float().mean().item() * B
        tot_pat += (preds == targets).all(dim=1).float().mean().item() * B
    return tot_bit / N, tot_pat / N


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(os.path.join(a.data_root, "dataset_test.npz"))
    X = d["X"].astype(np.float32); Y = d["Y"].astype(np.float32)
    N, H, W = X.shape; nb = H * W; P = Y.shape[1]
    Xf = X.reshape(N, nb)
    if a.canonical:
        Xf = canonicalize_under_yflip(Xf, H, W)
    if a.normalize_input:
        Y = Y - Y.mean(axis=1, keepdims=True)
        print("🔧 Per-sample normalization applied to Y.")
    order_idx = get_order_indices(a.ordering, nb, H, W)
    Xord = Xf[:, order_idx]
    Y_t = torch.tensor(Y); X_t = torch.tensor(Xord)
    print(f"✅ test={N}, H={H},W={W}, P={P}, ordering={a.ordering}, canon={a.canonical}")

    model = SmallTransformerAR(
        num_points=P, d_model=512, nhead=4, num_layers=15, dim_feedforward=768,
        max_len=100, vocab_size=3, dropout=0.1, spectral_cond="resnet1d",
        use_2d_pos=False, height=H, width=W,
    ).to(device)
    print(f"📂 checkpoint: {a.ckpt}")
    sd = torch.load(a.ckpt, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)

    bit, pat = eval_bitacc(model, Y_t, X_t, device)
    print(f"\n📊 teacher-forced  bit_acc={bit:.4f}  pattern_acc={pat:.4f}  (n={N})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="AR checkpoint path")
    p.add_argument("--data_root", type=str,
                   default="/hai/home/lsh/antenna/year_hai/data/year/datasets/pa10x10/60_density_500000_dataset")
    p.add_argument("--ordering", type=str, default="snake")
    p.add_argument("--canonical", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--normalize_input", type=lambda x: x.lower() == "true", default=False)
    main(p.parse_args())
