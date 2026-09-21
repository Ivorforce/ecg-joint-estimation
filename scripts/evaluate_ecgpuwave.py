"""Fiducial localization by ecgpuwave on raw and fitted signals, against the
LUDB lead II expert annotations (paper: Downstream fiducial recovery, Table 1).

Needs the `ecgpuwave` binary from the PhysioNet WFDB software on PATH.

    python scripts/evaluate_ecgpuwave.py --ludb data/ludb
"""
import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np
import wfdb

from common import LEADS, wave_peaks, write_json

TOLERANCE_S = 0.1
FIDUCIALS = ("qrs_peak", "p_peak", "t_peak")

ap = argparse.ArgumentParser()
ap.add_argument("--ludb", default="data/ludb")
ap.add_argument("--fits", default="fits/ludb")
ap.add_argument("--out", default="results")
args = ap.parse_args()


def ecgpuwave(sig, fs, workdir, name):
    """Delineate lead II of a [time, lead] signal with ecgpuwave."""
    wfdb.wrsamp(name, fs=int(fs), units=["mV"] * 12, sig_name=list(LEADS),
                p_signal=sig.astype(np.float64), fmt=["16"] * 12, write_dir=workdir)
    # ecgpuwave only accepts record names relative to the working directory.
    subprocess.check_output(["ecgpuwave", "-r", name, "-a", "puwave", "-s", "1"], cwd=workdir)
    return wfdb.rdann(os.path.join(workdir, name), "puwave")


def match(expert, detected, tolerance):
    """Signed error of the nearest detection within `tolerance` of each expert point."""
    if len(expert) == 0 or len(detected) == 0:
        return []
    errors = []
    for e in expert:
        nearest = detected[np.argmin(np.abs(detected - e))]
        if abs(nearest - e) <= tolerance:
            errors.append(int(nearest - e))
    return errors


results = []
with tempfile.TemporaryDirectory() as workdir:
    for path in sorted(glob.glob(os.path.join(args.fits, "ludb_*.npz"))):
        record = int(os.path.basename(path)[5:8])
        d = np.load(path)
        fs = float(d["fs"])
        expert_ann = wfdb.rdann(os.path.join(args.ludb, str(record)), "ii")
        expert = dict(zip(FIDUCIALS, wave_peaks(expert_ann)))
        variants = {
            "raw": d["sig"],
            "debased": d["sig"] - d["baselines"],
            "reconst": d["sig_morph"] + d["baselines"],
            "morph": d["sig_morph"],
        }
        row = {"record": record}
        for name, sig in variants.items():
            try:
                detected = dict(zip(FIDUCIALS, wave_peaks(ecgpuwave(sig, fs, workdir, f"rec_{record}_{name}"))))
            except Exception as e:  # ecgpuwave fails or writes symbols we cannot read
                row[name] = {"error": str(e)}
                continue
            row[name] = {}
            for fiducial in FIDUCIALS:
                errors = np.array(match(expert[fiducial], detected[fiducial], int(TOLERANCE_S * fs))) * 1000 / fs
                row[name][fiducial] = {"n_expert": len(expert[fiducial]), "n_matched": len(errors),
                                       **({"rms_error_ms": float(np.sqrt(np.mean(errors ** 2)))} if len(errors) else {})}
        results.append(row)
        print(f"LUDB {record}: QRS peak RMS (ms) "
              + ", ".join(f"{v} {row[v].get('qrs_peak', {}).get('rms_error_ms', float('nan')):.2f}" for v in variants),
              flush=True)

# Table 1: mean over records of the per-record RMS error, and total matched beats.
table = {}
print(f"\n{'':<20}" + "".join(f"{v:>10}" for v in variants))
for fiducial in FIDUCIALS:
    table[fiducial] = {}
    for variant in variants:
        cells = [r[variant][fiducial] for r in results if fiducial in r.get(variant, {})]
        cells = [c for c in cells if c["n_matched"] > 0]
        table[fiducial][variant] = {
            "rms_mean_ms": float(np.mean([c["rms_error_ms"] for c in cells])),
            "rms_median_ms": float(np.median([c["rms_error_ms"] for c in cells])),
            "matches": int(sum(c["n_matched"] for c in cells)),
        }
    print(f"{fiducial + ' RMS (ms)':<20}" + "".join(f"{table[fiducial][v]['rms_mean_ms']:>10.2f}" for v in variants))
    print(f"{fiducial + ' matches':<20}" + "".join(f"{table[fiducial][v]['matches']:>10}" for v in variants))

write_json(os.path.join(args.out, "ecgpuwave.json"), {"table": table, "per_record": results})
