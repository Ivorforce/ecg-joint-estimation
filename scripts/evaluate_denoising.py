"""Denoising and morphology stability under NSTDB noise (paper: Noise robustness, Fig. 2).

Denoising: SNR of `noisy - baselines` against the clean record.
Stability: Pearson correlation between the morphology fitted to the noisy record
and the morphology fitted to the clean record (from fit_ludb.py).

    python scripts/evaluate_denoising.py
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
import numpy as np
from scipy.stats import binomtest

from common import pearson, snr_db, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--fits", default="fits/denoising")
ap.add_argument("--clean_fits", default="fits/ludb")
ap.add_argument("--out", default="results")
args = ap.parse_args()

rows = []
for path in sorted(glob.glob(os.path.join(args.fits, "ludb_*.npz"))):
    record, noise, snr = re.match(r"ludb_(\d+)_(\w+)_snr([+-]\d+)", os.path.basename(path)).groups()
    d = np.load(path)
    clean_morph = np.load(os.path.join(args.clean_fits, f"ludb_{record}.npz"))["sig_morph"]
    row = {
        "record": int(record),
        "noise": noise,
        "target_snr_db": int(snr),
        "input_snr_db": snr_db(d["sig_clean"], d["sig"]),
        "debased_snr_db": snr_db(d["sig_clean"], d["sig"] - d["baselines"]),
        "morph_stability_pearson": pearson(clean_morph, d["sig_morph"]),
    }
    row["delta_snr_db"] = row["debased_snr_db"] - row["input_snr_db"]
    rows.append(row)

groups = defaultdict(list)
for r in rows:
    groups[(r["noise"], r["target_snr_db"])].append(r)

summary = []
print(f"{'noise':<6}{'SNR':>5}{'n':>4}{'input':>9}{'debased':>9}{'delta':>9}{'Pearson':>9}{'sign p':>10}")
for (noise, snr), group in sorted(groups.items()):
    stats = {key: {"mean": float(np.mean(v)), "std": float(np.std(v)), "min": float(np.min(v)), "max": float(np.max(v))}
             for key in ("input_snr_db", "debased_snr_db", "delta_snr_db", "morph_stability_pearson")
             for v in [[r[key] for r in group]]}
    # One-tailed sign test: does baseline subtraction raise the SNR?
    n_better = sum(r["delta_snr_db"] > 0 for r in group)
    sign_p = binomtest(n_better, len(group), 0.5, alternative="greater").pvalue
    summary.append({"noise": noise, "target_snr_db": snr, "n": len(group),
                    "n_improved": n_better, "sign_test_p": sign_p, **stats})
    print(f"{noise:<6}{snr:>+5}{len(group):>4}{stats['input_snr_db']['mean']:>9.2f}"
          f"{stats['debased_snr_db']['mean']:>9.2f}{stats['delta_snr_db']['mean']:>+9.2f}"
          f"{stats['morph_stability_pearson']['mean']:>9.4f}{sign_p:>10.2g}")

write_json(os.path.join(args.out, "denoising.json"), {"groups": summary, "per_condition": rows})

# Fig. 2
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType, as IEEE Xplore requires
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "lines.linewidth": 1.0, "grid.alpha": 0.3,
})
colors = {"bw": "tab:blue", "em": "tab:orange", "ma": "tab:green"}
fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.5))
for noise in sorted({r["noise"] for r in rows}):
    group = [g for g in summary if g["noise"] == noise]
    x_in = [g["input_snr_db"]["mean"] for g in group]
    axs[0].errorbar(x_in, [g["debased_snr_db"]["mean"] for g in group],
                    yerr=[g["debased_snr_db"]["std"] for g in group],
                    marker="o", label=noise, color=colors[noise], capsize=3)
    axs[1].errorbar([g["target_snr_db"] for g in group], [g["morph_stability_pearson"]["mean"] for g in group],
                    yerr=[g["morph_stability_pearson"]["std"] for g in group],
                    marker="o", label=noise, color=colors[noise], capsize=3)
all_snrs = [r["input_snr_db"] for r in rows] + [r["debased_snr_db"] for r in rows]
lim = [min(all_snrs) - 1, max(all_snrs) + 1]
axs[0].plot(lim, lim, "k--", alpha=0.4)
axs[0].set_xlabel("input SNR (dB)")
axs[0].set_ylabel("debased SNR (dB)")
axs[1].set_xlabel("input SNR (dB)")
axs[1].set_ylabel("Pearson vs clean")
axs[1].set_ylim(0.9, 1.001)
for ax in axs:
    ax.grid(True, alpha=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(args.out, "fig2_denoising.pdf"), bbox_inches="tight")
print(f"wrote {os.path.join(args.out, 'fig2_denoising.pdf')}")
