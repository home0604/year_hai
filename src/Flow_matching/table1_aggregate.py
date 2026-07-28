# -*- coding: utf-8 -*-
"""
table1_search.py 산출 로그 → 논문 Table I 숫자 + Fig.16/convergence 곡선.

  · Average notch-region MSE : target별 final best_mse 평균 (set별)
  · valid rate               : valid_any(-12dB) 평균
  · top-ranked (win) count   : target별 min best_mse 를 낸 config 에 +1  (AR vs FM 맞대결)
  · 곡선(time-resampled, target 평균):
        best_mse   vs elapsed   (convergence)
        cum_valid  vs elapsed   (Fig.16 스타일 — 유효 후보 누적)
    → table1_out/curve_<set>_<metric>.csv 로 저장(플롯은 --plot).

사용:
  python table1_aggregate.py --out table1_out [--plot]
"""
import os
import csv
import glob
import argparse
import numpy as np
from collections import defaultdict


def read_summaries(out):
    rows = []
    for f in glob.glob(os.path.join(out, "*.summary.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r["final_best_mse"] = float(r["final_best_mse"])
                r["valid_any"] = int(r["valid_any"])
                r["total_cand"] = int(r["total_cand"])
                rows.append(r)
    return rows


def read_timelines(out):
    tl = defaultdict(list)   # (config,set,target_idx) -> [(t,cum,mse,valid)]
    for f in glob.glob(os.path.join(out, "*.timeline.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                tl[(r["config"], r["set"], r["target_idx"])].append(
                    (float(r["elapsed_sec"]), int(r["cum_cand"]),
                     float(r["best_mse"]), int(r["cum_valid"])))
    return tl


def table_block(rows, kind):
    sub = [r for r in rows if r["set"] == kind]
    if not sub:
        return
    configs = sorted(set(r["config"] for r in sub))
    # per-target min mse → win count
    per_tgt = defaultdict(dict)   # target_idx -> {config: mse}
    for r in sub:
        per_tgt[r["target_idx"]][r["config"]] = r["final_best_mse"]
    wins = defaultdict(int)
    n_tgt = len(per_tgt)
    for t, d in per_tgt.items():
        win_cfg = min(d, key=d.get)
        wins[win_cfg] += 1

    print(f"\n===== {kind.upper()}  (targets={n_tgt}) =====")
    print(f"{'config':<20}{'avg notchMSE↓':>14}{'valid@-12↑':>12}"
          f"{'win/tgt':>10}{'avg cand':>11}")
    print("-" * 67)
    agg = []
    for c in configs:
        cr = [r for r in sub if r["config"] == c]
        avg_mse = np.mean([r["final_best_mse"] for r in cr])
        vrate = np.mean([r["valid_any"] for r in cr])
        acand = np.mean([r["total_cand"] for r in cr])
        agg.append((c, avg_mse, vrate, wins[c], acand))
    for c, m, v, w, ac in sorted(agg, key=lambda x: x[1]):
        print(f"{c:<20}{m:>14.2f}{v:>12.3f}{w:>7}/{n_tgt:<2}{ac:>11.0f}")


def resample_curves(tl, out, grid_dt=5.0, T=300.0):
    grid = np.arange(0.0, T + 1e-6, grid_dt)
    # group by (config,set)
    groups = defaultdict(list)   # (config,set) -> list of per-target arrays
    for (cfg, kind, tgt), seq in tl.items():
        seq = sorted(seq)
        ts = np.array([s[0] for s in seq])
        mse = np.array([s[2] for s in seq])     # best-so-far (이미 단조 감소)
        val = np.array([s[3] for s in seq])     # cum_valid (단조 증가)
        # step-interpolate onto grid: value at last event with t<=grid point
        g_mse = np.full_like(grid, np.nan)
        g_val = np.full_like(grid, np.nan)
        for gi, gt in enumerate(grid):
            j = np.searchsorted(ts, gt, side="right") - 1
            if j >= 0:
                g_mse[gi] = mse[j]
                g_val[gi] = val[j]
        groups[(cfg, kind)].append((g_mse, g_val))

    for kind in ("single", "dual"):
        cfgs = sorted(set(c for (c, k) in groups if k == kind))
        if not cfgs:
            continue
        for metric, mi in (("bestmse", 0), ("cumvalid", 1)):
            path = os.path.join(out, f"curve_{kind}_{metric}.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["elapsed_sec"] + cfgs)
                mats = {c: np.vstack([g[mi] for g in groups[(c, kind)]]) for c in cfgs}
                for gi, gt in enumerate(grid):
                    row = [f"{gt:.0f}"]
                    for c in cfgs:
                        col = mats[c][:, gi]
                        row.append(f"{np.nanmean(col):.4f}" if np.any(~np.isnan(col)) else "")
                    w.writerow(row)
            print(f"  saved {path}")


def maybe_plot(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped: {e}]")
        return
    for kind in ("single", "dual"):
        for metric, ylab, logy in (("bestmse", "best notch-MSE (avg)", True),
                                   ("cumvalid", "cumulative valid (avg)", False)):
            p = os.path.join(out, f"curve_{kind}_{metric}.csv")
            if not os.path.exists(p):
                continue
            data = np.genfromtxt(p, delimiter=",", names=True)
            names = data.dtype.names
            x = data[names[0]]
            plt.figure(figsize=(8, 5))
            for c in names[1:]:
                plt.plot(x, data[c], label=c, linewidth=1.3)
            if logy:
                plt.yscale("log")
            plt.xlabel("elapsed time (s)")
            plt.ylabel(ylab)
            plt.title(f"{kind} — {ylab}")
            plt.legend(fontsize=6, ncol=2)
            plt.grid(alpha=0.3)
            png = os.path.join(out, f"curve_{kind}_{metric}.png")
            plt.savefig(png, dpi=130, bbox_inches="tight")
            plt.close()
            print(f"  plotted {png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="table1_out")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--T", type=float, default=300.0)
    args = ap.parse_args()

    rows = read_summaries(args.out)
    if not rows:
        print("no summary files yet.")
        raise SystemExit
    for kind in ("single", "dual"):
        table_block(rows, kind)
    print("\n[curves]")
    resample_curves(read_timelines(args.out), args.out, T=args.T)
    if args.plot:
        maybe_plot(args.out)
