"""Data loading, noise and metrics shared by the experiment scripts."""
import contextlib
import io
import json
import os
from fractions import Fraction

import numpy as np
import wfdb
import wfdb.processing
from scipy.signal import resample_poly

LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
FS = 500
HERE = os.path.dirname(os.path.abspath(__file__))

# WFDB beat symbols (N, V, A, ...), in wfdb's annotation label table.
QRS_SYMBOLS = tuple(
    wfdb.io.annotation.ann_label_table["symbol"][np.flatnonzero(wfdb.io.annotation.is_qrs)]
)


def parse_ids(spec):
    """'1-3,7' -> [1, 2, 3, 7]"""
    ids = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        ids.extend(range(int(a), int(b or a) + 1))
    return ids


def shard(jobs, worker, n_workers):
    """Every n_workers-th job, so several processes can share a run."""
    return jobs[worker::n_workers]


def save_fit(path, sig, result, **extra):
    """Save the input signal and the fit, float32 like the paper's fit files."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        sig=sig.astype(np.float32),
        sig_morph=result.sig_morph.astype(np.float32),
        baselines=result.baselines.astype(np.float32),
        templates=result.templates.astype(np.float32),
        template_window_s=result.template_window_s,
        qrs_peaks=result.qrs_peaks,
        clusters=result.clusters,
        losses=result.losses.astype(np.float32),
        fs=result.fs,
        **extra,
    )


# ---------------------------------------------------------------- LUDB

def load_ludb(ludb_dir, record):
    """Signal float[time, lead] in mV and R peaks from the lead II expert annotation."""
    path = os.path.join(ludb_dir, str(record))
    rec = wfdb.rdrecord(path)
    assert int(rec.fs) == FS, f"LUDB {record}: fs {rec.fs}"
    assert tuple(s.lower() for s in rec.sig_name) == LEADS, f"LUDB {record}: leads {rec.sig_name}"
    qrs_peaks, _, _ = wave_peaks(wfdb.rdann(path, "ii"))
    return rec.p_signal, qrs_peaks


def wave_peaks(ann):
    """R, P and T peak samples from a WFDB wave annotation (LUDB or ecgpuwave).

    Beats closer than 0.1 s to a neighbouring beat are ambiguous and dropped.
    Between two kept beats, a P peak belongs to the next beat and a T peak to
    the previous one; a peak is kept only if it is the only one of its kind in
    that interval and the beat it belongs to exists.
    """
    symbol = np.array(ann.symbol)
    sample = np.array(ann.sample)
    unknown = set(symbol) - {"p", "t", "(", ")", *QRS_SYMBOLS}
    if unknown:
        raise ValueError(f"unknown annotation symbols: {unknown}")

    beats = np.flatnonzero(np.isin(symbol, QRS_SYMBOLS))
    groups = np.split(beats, np.flatnonzero(np.diff(sample[beats]) > int(0.1 * ann.fs)) + 1)
    beats = np.array([g[0] for g in groups if len(g) == 1], dtype=int)

    p_peaks, t_peaks = [], []
    for i, interval in enumerate(np.split(np.arange(len(symbol)), beats)):
        def unique(kind):
            hits = sample[interval][symbol[interval] == kind]
            return hits[0] if len(hits) == 1 else None
        p, t = unique("p"), unique("t")
        if p is not None and i < len(beats):
            p_peaks.append(p)
        if t is not None and i > 0:
            t_peaks.append(t)
    return sample[beats], np.array(p_peaks, dtype=int), np.array(t_peaks, dtype=int)


# ---------------------------------------------------------------- MedalCare-XL

def medalcare_path(medalcare_dir, pathology, run):
    """The paper uses the first runs of simulation S62 in each class's test split."""
    return os.path.join(medalcare_dir, pathology, "test", "run_S62", f"run_{run:06d}_raw.csv")


def load_medalcare(path):
    """Noise-free simulated signal, float[time, lead] in mV."""
    sig = np.loadtxt(path, delimiter=",").T
    assert sig.shape == (5000, 12) and np.all(np.isfinite(sig)), f"{path}: {sig.shape}"
    return sig


def detect_qrs(sig):
    """wfdb XQRS on lead II, falling back to I, V2 and III when it finds < 3 beats."""
    for lead in (1, 0, 7, 2):
        with contextlib.redirect_stdout(io.StringIO()):
            xqrs = wfdb.processing.XQRS(sig=sig[:, lead], fs=FS)
            try:
                xqrs.detect()
            except Exception:
                continue
        if len(xqrs.qrs_inds) >= 3:
            return np.asarray(xqrs.qrs_inds, dtype=np.int64)
    return np.array([], dtype=np.int64)


# ---------------------------------------------------------------- NSTDB noise

def load_nstdb(nstdb_dir, kind):
    """Channel 0 of an NSTDB noise record (bw, em, ma), resampled from 360 to 500 Hz."""
    rec = wfdb.rdrecord(os.path.join(nstdb_dir, kind))
    ratio = Fraction(FS / rec.fs).limit_denominator(100)
    noise = resample_poly(rec.p_signal, up=ratio.numerator, down=ratio.denominator, axis=0)
    return noise[:, 0]


def noise_segment_start(kind, record):
    """Where the paper's noise segment for this record starts in `load_nstdb(kind)`.

    The paper drew the segments at random. The draws are recorded here, recovered
    from the paper's fit files, so that re-running reproduces the paper.
    """
    with open(os.path.join(HERE, "nstdb_segments.json")) as f:
        return json.load(f)[kind][str(record)]


def add_noise_at_snr(sig, noise, snr_db):
    """Add the same noise series to every lead, scaled to the target SNR against
    the lead-averaged signal power."""
    scale = np.sqrt(np.mean(np.mean(sig ** 2, axis=0)) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return sig + scale * noise[:, None]


# ---------------------------------------------------------------- metrics

def snr_db(clean, other):
    """SNR of `other` against `clean` per lead, in dB, averaged over leads."""
    signal = np.mean(clean ** 2, axis=0)
    error = np.maximum(np.mean((other - clean) ** 2, axis=0), 1e-20)
    return float(np.mean(10 * np.log10(signal / error)))


def pearson(a, b):
    """Pearson correlation per lead, averaged over leads (constant leads skipped)."""
    corrs = []
    for x, y in zip(a.T, b.T):
        x, y = x - x.mean(), y - y.mean()
        denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
        if denom > 0:
            corrs.append((x * y).sum() / denom)
    return float(np.mean(corrs))


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def limb_lead_residuals(sig):
    """RMS of the Einthoven and Goldberger identities in a [time, lead] signal."""
    i, ii, iii, avr, avl, avf = sig[:, :6].T
    return {
        "einthoven": rms(ii - i - iii),
        "aVR": rms(avr + (i + ii) / 2),
        "aVL": rms(avl - (i - iii) / 2),
        "aVF": rms(avf - (ii + iii) / 2),
    }


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {path}")
