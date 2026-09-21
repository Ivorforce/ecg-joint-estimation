"""Smoke test on a synthetic record: the fit separates beats from a slow drift."""
import numpy as np

from ecgfit import fit


def synthetic_record(fs=500, seconds=10, n_lead=3):
    t = np.arange(seconds * fs) / fs
    qrs_peaks = np.arange(int(0.5 * fs), len(t) - int(0.7 * fs), int(0.9 * fs))
    beat = np.zeros_like(t)
    for p in qrs_peaks:
        dt = t - t[p]
        beat += np.exp(-(dt / 0.012) ** 2) + 0.3 * np.exp(-((dt - 0.3) / 0.05) ** 2) + 0.1 * np.exp(-((dt + 0.16) / 0.03) ** 2)
    drift = 0.4 * np.sin(2 * np.pi * 0.15 * t)
    gains = np.linspace(0.5, 1.5, n_lead)
    return beat[:, None] * gains + drift[:, None], beat[:, None] * gains, drift, qrs_peaks, fs


def test_fit_separates_beats_and_drift():
    sig, beats, drift, qrs_peaks, fs = synthetic_record()
    result = fit(sig, qrs_peaks, fs)

    assert result.sig_morph.shape == sig.shape and result.baselines.shape == sig.shape
    assert result.templates.shape[1:] == (3, 500)
    assert result.losses[-1] < result.losses[0] / 10
    for lead in range(sig.shape[1]):
        assert np.corrcoef(result.sig_morph[:, lead], beats[:, lead])[0, 1] > 0.9
        assert np.corrcoef(result.baselines[:, lead], drift)[0, 1] > 0.99
