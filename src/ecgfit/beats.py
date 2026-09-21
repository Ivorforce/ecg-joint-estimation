"""Beat windows and beat clustering."""
import numpy as np
from scipy.signal.windows import tukey
from sklearn.cluster import dbscan


def windows_at(x, locations, window):
    """Cut windows out of x around each location.

    x: float[..., time]; locations: int[beat]; window: (start, end) in samples
    relative to each location, end exclusive.
    Returns float[..., beat, end - start], NaN where a window leaves the signal.
    """
    idx = np.asarray(locations)[:, None] + np.arange(window[0], window[1])[None, :]
    inside = (idx >= 0) & (idx < x.shape[-1])
    return np.where(inside, x[..., np.clip(idx, 0, x.shape[-1] - 1)], np.nan)


def cluster_beats(sig, qrs_peaks, fs):
    """Assign each beat a cluster id 0..C-1, by DBSCAN on QRS spectra.

    sig: float[time, lead]; qrs_peaks: int[beat].
    Beats are compared by the 4-45 Hz magnitude spectrum of a Tukey-tapered
    -0.15..+0.2 s window around the R peak, per lead. The DBSCAN radius is the
    95th percentile of all per-lead pairwise distances.
    """
    window = (int(-0.15 * fs), int(0.2 * fs))
    # [lead, beat, time]
    windows = np.nan_to_num(windows_at(sig.T, qrs_peaks, window), nan=0.0)
    windows = windows * tukey(windows.shape[-1], alpha=0.2)
    spectra = np.abs(np.fft.rfft(windows, axis=-1))
    freqs = np.fft.rfftfreq(windows.shape[-1], d=1 / fs)
    spectra = spectra[..., (freqs >= 4) & (freqs < 45)]

    # [lead, beat, beat]
    distances = np.linalg.norm(spectra[:, :, None, :] - spectra[:, None, :, :], axis=-1)
    eps = np.percentile(distances, q=95)
    eps = eps if eps > 0 else 1

    # The rows of the lead-averaged distance matrix are clustered as feature
    # vectors (sklearn's default metric), not as precomputed distances.
    _, labels = dbscan(np.mean(distances, axis=0), eps=eps, min_samples=1)
    return labels
