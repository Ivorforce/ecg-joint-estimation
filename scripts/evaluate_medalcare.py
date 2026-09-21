"""Fidelity and cross-lead consistency on MedalCare-XL (paper: Fidelity on
MedalCare-XL, Cross-lead consistency check).

Fidelity: Pearson correlation of `sig_morph + baselines` with the simulated signal.
Cross-lead: the simulated leads satisfy Einthoven's and Goldberger's identities
exactly, so any residual in the fitted morphology comes from fitting leads
independently.

    python scripts/evaluate_medalcare.py
"""
import argparse
import glob
import os

import numpy as np

from common import limb_lead_residuals, pearson, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--fits", default="fits/medalcare")
ap.add_argument("--out", default="results")
args = ap.parse_args()

rows = []
for path in sorted(glob.glob(os.path.join(args.fits, "*", "run_*.npz"))):
    d = np.load(path)
    sig, morph = d["sig"], d["sig_morph"]
    reconst = morph + d["baselines"]
    rows.append({
        "pathology": os.path.basename(os.path.dirname(path)),
        "record": os.path.basename(path)[:-4],
        "pearson_reconst": pearson(sig, reconst),
        "pearson_morph": pearson(sig, morph),
        "lead_ii_std": float(np.std(sig[:, 1])),
        **{f"raw_{k}": v for k, v in limb_lead_residuals(sig).items()},
        **{f"morph_{k}": v for k, v in limb_lead_residuals(morph).items()},
    })


def describe(values):
    values = np.asarray(values)
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "median": float(np.median(values)), "p90": float(np.percentile(values, 90))}


fidelity = {p: describe([r["pearson_reconst"] for r in rows if r["pathology"] == p])
            for p in sorted({r["pathology"] for r in rows})}
fidelity["pooled"] = describe([r["pearson_reconst"] for r in rows])
print("Pearson(sig_morph + baselines, simulated signal)")
for p, s in fidelity.items():
    print(f"  {p:<8} n={s['n']:<3} {s['mean']:.3f} ± {s['std']:.3f}")

# Residuals in microvolts; Einthoven also relative to the lead II standard deviation.
uv = {f"{which}_{k}": describe([r[f"{which}_{k}"] * 1e3 for r in rows])
      for which in ("raw", "morph") for k in ("einthoven", "aVR", "aVL", "aVF")}
percent = describe([r["morph_einthoven"] / r["lead_ii_std"] * 100 for r in rows])
print("Cross-lead residual RMS, median (90th percentile), uV")
for k, s in uv.items():
    print(f"  {k:<16} {s['median']:.4g} ({s['p90']:.3g})")
print(f"  morph Einthoven, % of lead II std: median {percent['median']:.2f}%")

write_json(os.path.join(args.out, "medalcare.json"), {
    "fidelity_pearson_reconst": fidelity,
    "residual_rms_uV": uv,
    "morph_einthoven_percent_of_lead_ii_std": percent,
    "per_record": rows,
})
