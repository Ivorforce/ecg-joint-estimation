"""Fit LUDB records with NSTDB noise added at several SNRs (paper: records 1-20,
bw and em, 0/6/12 dB). Uses the paper's noise segments, see nstdb_segments.json.

    python scripts/fit_denoising.py --ludb data/ludb --nstdb data/nstdb
"""
import argparse
import os

from common import (add_noise_at_snr, load_ludb, load_nstdb, noise_segment_start,
                    parse_ids, save_fit, shard)
from ecgfit import fit

ap = argparse.ArgumentParser()
ap.add_argument("--ludb", default="data/ludb")
ap.add_argument("--nstdb", default="data/nstdb", help="directory with NSTDB's bw.hea, em.hea, ...")
ap.add_argument("--records", default="1-20")
ap.add_argument("--noises", default="bw,em")
ap.add_argument("--snrs", default="0,6,12", help="input SNRs in dB")
ap.add_argument("--out", default="fits/denoising")
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()

noises = {kind: load_nstdb(args.nstdb, kind) for kind in args.noises.split(",")}
jobs = [(record, kind, int(snr))
        for record in parse_ids(args.records)
        for kind in noises
        for snr in args.snrs.split(",")]

for record, kind, snr in shard(jobs, args.worker, args.n_workers):
    out = os.path.join(args.out, f"ludb_{record:03d}_{kind}_snr{snr:+03d}.npz")
    if os.path.exists(out):
        continue
    sig, qrs_peaks = load_ludb(args.ludb, record)
    start = noise_segment_start(kind, record)
    noisy = add_noise_at_snr(sig, noises[kind][start:start + len(sig)], float(snr))
    # R peaks stay the expert ones: the experiment tests the decomposition, not beat detection.
    result = fit(noisy, qrs_peaks, fs=500)
    save_fit(out, noisy, result, sig_clean=sig.astype("float32"))
    print(f"LUDB {record} {kind} {snr:+d} dB: {result.elapsed_s:.0f} s", flush=True)
