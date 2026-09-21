"""Energy left in the residual `sig - sig_morph - baselines`, per frequency band
(paper: Residual spectrum). Beat-centred windows on lead II, Hann-tapered,
power spectra averaged over beats, then over records.

    python scripts/evaluate_residual_spectrum.py
"""
import argparse
import glob
import os

import numpy as np

from common import write_json

WINDOW_S = (-0.3, 0.7)
BANDS_HZ = {"baseline": (0, 0.5), "low": (0.5, 4), "mid": (4, 15), "high": (15, 50), "very_high": (50, 125)}

ap = argparse.ArgumentParser()
ap.add_argument("--fits", default="fits/ludb")
ap.add_argument("--out", default="results")
ap.add_argument("--lead", type=int, default=1, help="lead index, 1 = II")
args = ap.parse_args()


def mean_psd(x, qrs_peaks, fs):
    """Power spectrum averaged over the beat windows that lie inside the record."""
    start, end = int(WINDOW_S[0] * fs), int(WINDOW_S[1] * fs)
    windows = np.stack([x[p + start:p + end] for p in qrs_peaks if p + start >= 0 and p + end <= len(x)])
    taper = np.hanning(windows.shape[1])
    psd = np.abs(np.fft.rfft(windows * taper, axis=1)) ** 2 / (np.sum(taper ** 2) * fs)
    return np.fft.rfftfreq(windows.shape[1], d=1 / fs), psd.mean(axis=0)


raw, residual = [], []
for path in sorted(glob.glob(os.path.join(args.fits, "ludb_*.npz"))):
    d = np.load(path)
    fs = float(d["fs"])
    sig = d["sig"][:, args.lead]
    freqs, psd = mean_psd(sig, d["qrs_peaks"], fs)
    raw.append(psd)
    residual.append(mean_psd(sig - d["sig_morph"][:, args.lead] - d["baselines"][:, args.lead], d["qrs_peaks"], fs)[1])
raw, residual = np.mean(raw, axis=0), np.mean(residual, axis=0)

bands = {}
for name, (lo, hi) in BANDS_HZ.items():
    band = (freqs >= lo) & (freqs < hi)
    bands[name] = {"lo_hz": lo, "hi_hz": hi, "residual_over_raw": float(residual[band].sum() / raw[band].sum())}
    print(f"{name:<10} {lo:>5}-{hi:<4} Hz: residual {bands[name]['residual_over_raw']:6.1%} of raw energy")
write_json(os.path.join(args.out, "residual_spectrum.json"), {"n_records": len(glob.glob(os.path.join(args.fits, "ludb_*.npz"))), "bands": bands})
