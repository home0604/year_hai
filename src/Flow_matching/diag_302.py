# -*- coding: utf-8 -*-
"""target 302(dual, MSE 폭발) 진단: GT 마스크·스펙트럼 vs 각 모델 best 마스크·surrogate 스펙트럼."""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from eval import load_forward_surrogate, predict_spectrum
from inverse_from_csv_10x10 import _find_true_segments

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
TGT = int(sys.argv[1]) if len(sys.argv) > 1 else 302
KIND = sys.argv[2] if len(sys.argv) > 2 else "dual"
H = W = 10
fwd = load_forward_surrogate(num_points=201, device=device)

te = np.load("/hai/home/lsh/antenna/year_hai/data/datasets/dataset_test.npz")
Y = te["Y"].astype(np.float32); X = te["X"].astype(np.float32); freq = te["freq"]
gt_mask = X[TGT]; gt_spec = Y[TGT]
segs = _find_true_segments(gt_spec <= -10.0)

MODELS = [("ar", "AR-MAB"), ("dfm_g4_s10", "DFM_g4")]

def surrogate(mask_flat):
    b = torch.as_tensor(mask_flat, dtype=torch.long, device=device).view(1, -1)
    return predict_spectrum(fwd, b, H, W)[0].cpu().numpy()

masks, specs, labels = [gt_mask.reshape(-1)], [gt_spec], ["GT"]
print(f"=== target {TGT} ({KIND}) — GT notch segs(-10dB): {segs} ===")
print(f"GT: density={int(gt_mask.sum())}/100, res(min)@{freq[gt_spec.argmin()]:.2f}GHz {gt_spec.min():.1f}dB")
for key, lab in MODELS:
    p = f"table1_out/{key}.best_bits.npz"
    if not os.path.exists(p): continue
    z = np.load(p); k = f"{KIND}_{TGT}"
    if k not in z: continue
    m = z[k].astype(np.float32)
    sp = surrogate(m)
    # notch-region MSE
    mask_pts = gt_spec <= -10.0
    nmse = float(((sp[mask_pts] - gt_spec[mask_pts]) ** 2).mean()) if mask_pts.any() else 0
    seg_mins = [f"{freq[s]:.1f}-{freq[e]:.1f}GHz:min{sp[s:e+1].min():.1f}" for s, e in segs]
    print(f"{lab:<16} density={int(m.sum())}/100 notchMSE={nmse:.1f} | 예측 min in target segs: {seg_mins}")
    masks.append(m); specs.append(sp); labels.append(lab)

# ---- plot ----
n = len(masks)
fig, axes = plt.subplots(2, n, figsize=(3 * n, 6),
                         gridspec_kw={"height_ratios": [1, 1.3]})
for i, (m, lab) in enumerate(zip(masks, labels)):
    axes[0, i].imshow(m.reshape(H, W), cmap="gray_r", vmin=0, vmax=1)
    axes[0, i].set_title(f"{lab}\n({int(m.sum())}/100)", fontsize=9)
    axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
ax = axes[1, 0]
for i in range(1, n):
    axes[1, i].axis("off")
gs = axes[1, 0].get_gridspec()
for a in axes[1, :]:
    a.remove()
axbig = fig.add_subplot(gs[1, :])
axbig.plot(freq, gt_spec, "k", lw=2.5, label="GT spectrum", zorder=5)
for sp, lab in zip(specs[1:], labels[1:]):
    axbig.plot(freq, sp, lw=1.3, label=lab, alpha=0.85)
for s, e in segs:
    axbig.axvspan(freq[s], freq[e], color="orange", alpha=0.15)
axbig.axhline(-10, ls="--", c="gray", lw=0.8); axbig.axhline(-12, ls=":", c="red", lw=0.8)
axbig.set_xlabel("Freq (GHz)"); axbig.set_ylabel("S11 (dB)")
axbig.set_title(f"target {TGT}: GT vs generated-mask surrogate spectra "
                f"(orange=target notch band, --=-10dB ..=-12dB)", fontsize=9)
axbig.legend(fontsize=8); axbig.grid(alpha=0.3)
out = f"table1_out/diag_{KIND}_{TGT}.png"
plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nsaved -> {out}")
