# -*- coding: utf-8 -*-
"""302 에서 temperature 별 best 후보의 스펙트럼 변화 시각화 (dfm_mask_g4)."""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from table1_search import FM_CFG, build_fm, make_fm_sampler, notch_mse_vec, valid_mask_vec, H, W
from eval import load_forward_surrogate, predict_spectrum
from inverse_from_csv_10x10 import _find_true_segments

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
N, STEPS, TGT = 4096, 10, 302
MODEL = "dfm_mask_g4"
TEMPS = [1.0, 2.0, 4.0]

fwd = load_forward_surrogate(num_points=201, device=device)
te = np.load("/hai/home/lsh/antenna/year_hai/data/datasets/dataset_test.npz")
Y = te["Y"].astype(np.float32); X = te["X"].astype(np.int32); freq = te["freq"]
gt_spec = Y[TGT]; gt_mask = X[TGT].reshape(-1)
target_t = torch.as_tensor(gt_spec, device=device)
segs = _find_true_segments(gt_spec <= -10.0)

cfg = FM_CFG[MODEL]; m = build_fm(cfg, device)
y = torch.as_tensor(gt_spec, device=device).view(1, -1).repeat(N, 1)
masks, specs, labels = [gt_mask], [gt_spec], ["GT"]
print(f"target {TGT}, {MODEL}, N={N}, steps={STEPS}, notch segs {segs}\n")
for T in TEMPS:
    sampler = make_fm_sampler(m, cfg, STEPS, 1.0, temperature=T)
    torch.manual_seed(TGT)
    bits = sampler(y)
    pred = predict_spectrum(fwd, bits, H, W)
    mse = notch_mse_vec(pred, target_t, -10.0)
    b = int(mse.argmin().item())
    sp = pred[b].cpu().numpy(); mk = bits[b].cpu().numpy()
    depths = [f"{freq[s]:.1f}-{freq[e]:.1f}:{sp[s:e+1].min():.1f}dB" for s, e in segs]
    print(f"T={T}: bestMSE={float(mse[b]):.1f} | notch depths {depths}")
    masks.append(mk); specs.append(sp); labels.append(f"T={T} (MSE{float(mse[b]):.0f})")

n = len(masks)
fig = plt.figure(figsize=(3 * n, 6))
for i, (mk, lab) in enumerate(zip(masks, labels)):
    ax = fig.add_subplot(2, n, i + 1)
    ax.imshow(mk.reshape(H, W), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"{lab}\n({int(mk.sum())}/100)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
axb = fig.add_subplot(2, 1, 2)
axb.plot(freq, gt_spec, "k", lw=2.5, label="GT", zorder=5)
for sp, lab in zip(specs[1:], labels[1:]):
    axb.plot(freq, sp, lw=1.4, label=lab, alpha=0.85)
for s, e in segs:
    axb.axvspan(freq[s], freq[e], color="orange", alpha=0.15)
axb.axhline(-10, ls="--", c="gray", lw=0.8); axb.axhline(-12, ls=":", c="red", lw=0.8)
axb.set_xlabel("Freq (GHz)"); axb.set_ylabel("S11 (dB)")
axb.set_title(f"target {TGT}: {MODEL} best-candidate spectrum vs temperature "
              f"(orange=target notch, --=-10 ..=-12)", fontsize=9)
axb.legend(fontsize=8); axb.grid(alpha=0.3)
out = f"table1_out/temp_diag_302.png"
plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nsaved -> {out}")
