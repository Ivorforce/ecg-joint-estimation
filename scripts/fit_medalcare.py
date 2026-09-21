"""Fit noise-free MedalCare-XL simulations (paper: 30 sinus, 20 each AV block,
LBBB, RBBB). R peaks from wfdb's XQRS.

    python scripts/fit_medalcare.py --medalcare data/medalcare-xl
"""
import argparse
import os

from common import detect_qrs, load_medalcare, medalcare_path, parse_ids, save_fit, shard
from ecgfit import fit

ap = argparse.ArgumentParser()
ap.add_argument("--medalcare", default="data/medalcare-xl",
                help="MedalCare-XL's WP2_largeDataset_Noise directory")
ap.add_argument("--sets", nargs="+", default=["sinus:1-30", "avblock:1-20", "lbbb:1-20", "rbbb:1-20"],
                help="class:runs")
ap.add_argument("--out", default="fits/medalcare")
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()

jobs = [(pathology, run)
        for spec in args.sets
        for pathology, runs in [spec.split(":")]
        for run in parse_ids(runs)]

for pathology, run in shard(jobs, args.worker, args.n_workers):
    out = os.path.join(args.out, pathology, f"run_{run:06d}.npz")
    if os.path.exists(out):
        continue
    sig = load_medalcare(medalcare_path(args.medalcare, pathology, run))
    result = fit(sig, detect_qrs(sig), fs=500)
    save_fit(out, sig, result)
    print(f"MedalCare {pathology} {run}: {result.elapsed_s:.0f} s", flush=True)
