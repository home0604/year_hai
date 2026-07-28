# -*- coding: utf-8 -*-
"""single-band one-to-many 시각화: 한 타깃에 대해 dfm_g4 가 생성한
   **서로 다른 여러 valid 마스크**(다양한 레이아웃) + 각 surrogate 스펙트럼이 모두 GT 공진에 일치.
   → "같은 응답, 다른 설계" 를 직접 보임."""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); sys.path.insert(0, "../MAB_code")
from table1_search import FM_CFG, build_fm, make_fm_sampler, valid_mask_vec, notch_mse_vec, H, W
from eval import load_forward_surrogate, predict_spectrum
from inverse_from_csv_10x10 import _find_true_segments

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
TGT = int(sys.argv[1]) if len(sys.argv) > 1 else 887
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5     # 보여줄 서로 다른 설계 수
N, STEPS = 4096, 10

fwd = load_forward_surrogate(num_points=201, device=device)
te = np.load("/hai/home/lsh/antenna/year_hai/data/datasets/dataset_test.npz")
Y = te["Y"].astype(np.float32); X = te["X"].astype(np.int32); freq = te["freq"]
gt_mask, gt_spec = X[TGT].reshape(-1), Y[TGT]
segs = _find_true_segments(gt_spec <= -10.0)

cfg = FM_CFG["dfm_g4"]; m = build_fm(cfg, device)
sampler = make_fm_sampler(m, cfg, STEPS, 1.0)
y = torch.as_tensor(gt_spec, device=device).view(1, -1).repeat(N, 1)
torch.manual_seed(TGT)
bits = sampler(y)
pred = predict_spectrum(fwd, bits, H, W)
mse = notch_mse_vec(pred, torch.as_tensor(gt_spec, device=device), -10.0)
ok = valid_mask_vec(pred, segs, -12.0).cpu().numpy()

Vb = bits[ok].cpu().numpy()          # valid 마스크 (raster)
Vp = pred[ok].cpu().numpy()          # valid 스펙트럼
Vm = mse[ok].cpu().numpy()
print(f"target {TGT}: valid {ok.sum()}/{N}  (res@{freq[gt_spec.argmin()]:.2f}GHz {gt_spec.min():.1f}dB)")

# GT 공진과 tight 하게 일치하는 valid 만 남긴 뒤(스펙트럼 겹침 깔끔) 다양성 선택
keep = Vm <= np.quantile(Vm, 0.3)    # notch-MSE 하위 30% (GT 매칭 좋은 것)
Vb, Vp, Vm = Vb[keep], Vp[keep], Vm[keep]

# farthest-point sampling: 서로 최대한 다른 valid 설계 K개 (best-MSE 부터 시작)
sel = [int(Vm.argmin())]
while len(sel) < min(K, len(Vb)):
    d = ((Vb[:, None, :] != Vb[sel][None, :, :]).sum(-1)).min(1)   # 선택집합까지 최소 Hamming
    d[sel] = -1
    sel.append(int(d.argmax()))
sel = np.array(sel)
# 선택된 것들의 상호 Hamming
ph = [(Vb[sel][a] != Vb[sel][b]).sum() for a in range(len(sel)) for b in range(a + 1, len(sel))]
print(f"선택 {len(sel)}개 valid 설계: 상호 Hamming 평균 {np.mean(ph):.1f}/100, GT거리 {[int((Vb[i]!=gt_mask).sum()) for i in sel]}")

# ---- plot ----
ncol = len(sel) + 1
fig = plt.figure(figsize=(2.4 * ncol, 5.6))
# 위: GT + 생성 마스크들
ax = fig.add_subplot(2, ncol, 1)
ax.imshow(gt_mask.reshape(H, W), cmap="gray_r", vmin=0, vmax=1)
ax.set_title(f"GT\n({int(gt_mask.sum())}/100)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
for c, i in enumerate(sel):
    ax = fig.add_subplot(2, ncol, c + 2)
    ax.imshow(Vb[i].reshape(H, W), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"design #{c+1}\nΔGT={int((Vb[i]!=gt_mask).sum())}bit", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
# 아래: 스펙트럼 (GT + 생성들 — 모두 같은 공진에 겹침)
axb = fig.add_subplot(2, 1, 2)
axb.plot(freq, gt_spec, "k", lw=2.5, label="GT", zorder=5)
for c, i in enumerate(sel):
    axb.plot(freq, Vp[i], lw=1.3, alpha=0.8, label=f"design #{c+1}")
for s, e in segs:
    axb.axvspan(freq[s], freq[e], color="orange", alpha=0.15)
axb.axhline(-10, ls="--", c="gray", lw=0.8); axb.axhline(-12, ls=":", c="red", lw=0.8)
axb.set_xlabel("Freq (GHz)"); axb.set_ylabel("S11 (dB)")
axb.set_title(f"single-band target {TGT}: {len(sel)} distinct DFM_g4 designs "
              f"(mutual Hamming ~{np.mean(ph):.0f}/100) → same resonance, different layouts", fontsize=9)
axb.legend(fontsize=8, ncol=2); axb.grid(alpha=0.3)
out = f"table1_out/single_o2m_{TGT}.png"
plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved -> {out}")
