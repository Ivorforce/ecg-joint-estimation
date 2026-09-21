"""Fit LUDB records (paper: all 200). One file per record in --out.

    python scripts/fit_ludb.py --ludb data/ludb --records 1-200
"""
import argparse
import os

from common import load_ludb, parse_ids, save_fit, shard
from ecgfit import fit

ap = argparse.ArgumentParser()
ap.add_argument("--ludb", default="data/ludb", help="directory with LUDB's 1.hea, 1.dat, 1.ii, ...")
ap.add_argument("--records", default="1-200")
ap.add_argument("--out", default="fits/ludb")
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()

for record in shard(parse_ids(args.records), args.worker, args.n_workers):
    out = os.path.join(args.out, f"ludb_{record:03d}.npz")
    if os.path.exists(out):
        continue
    sig, qrs_peaks = load_ludb(args.ludb, record)
    result = fit(sig, qrs_peaks, fs=500)
    save_fit(out, sig, result)
    print(f"LUDB {record}: {result.elapsed_s:.0f} s, final loss {result.losses[-1]:.4g}", flush=True)
