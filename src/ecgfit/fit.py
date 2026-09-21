"""Joint fit of beat morphology and baseline for one ECG record.

The record is modelled as a template per beat cluster and lead, placed at every
R peak, plus a slowly varying baseline per lead. Both are parameterized by
Daubechies-5 wavelet coefficients and fitted by gradient descent on the
reconstruction error plus five regularization terms. Leads are fitted
independently; batching them is a convenience.
"""
import dataclasses
import time

import keras
import numpy as np
import pywt
import tensorflow as tf

from .beats import cluster_beats
from .wavelets import IDWT1D

WAVELET = "db5"
# Beat template window around the R peak, in seconds.
TEMPLATE_WINDOW_S = (-0.3, 0.7)
# Inverse-DWT levels between the trainable coefficients and the signal.
TEMPLATE_LEVELS = 1
BASELINE_LEVELS = 5
LEARNING_RATE = 0.025


@dataclasses.dataclass
class FitResult:
    sig_morph: np.ndarray          # float[time, lead]: templates placed at every R peak
    baselines: np.ndarray          # float[time, lead]
    templates: np.ndarray          # float[cluster, lead, template_time]
    template_window_s: np.ndarray  # float[2]: template start/end relative to the R peak
    qrs_peaks: np.ndarray          # int[beat]
    clusters: np.ndarray           # int[beat]: template index of each beat
    losses: np.ndarray             # float[step]: total loss before each step
    fs: int
    elapsed_s: float


def fit(
    sig,
    qrs_peaks,
    fs,
    n_steps: int = 300,
    baseline_inertia_weight: float = 2.5,
    verbose: bool = False,
) -> FitResult:
    """Fit templates and baselines to one record.

    sig: float[time, lead], unfiltered, in mV; qrs_peaks: int[beat], R peak
    sample indices from any detector; fs: sampling rate, ~500 Hz.
    """
    sig = np.asarray(sig)
    qrs_peaks = np.asarray(qrs_peaks, dtype=np.int64)
    assert sig.ndim == 2 and sig.dtype in (np.float32, np.float64), f"sig must be float[time, lead], got {sig.dtype} {sig.shape}"
    assert np.all(np.isfinite(sig)), "sig contains NaN or inf"
    # Every length in the model is in samples, tuned at 500 Hz.
    assert float(fs).is_integer() and abs(fs - 500) <= 50, f"fs must be an integer near 500 Hz, got {fs}"
    fs = int(fs)
    n_time, n_lead = sig.shape
    assert n_time >= 2 * fs, f"record too short: {n_time} samples"
    assert qrs_peaks.ndim == 1 and len(qrs_peaks) >= 2, f"need at least 2 R peaks, got {qrs_peaks.shape}"
    assert np.all((qrs_peaks >= 0) & (qrs_peaks < n_time)), "R peaks outside the record"
    assert np.all(np.diff(qrs_peaks) >= int(0.2 * fs)), "R peaks must be increasing and at least 200 ms apart"

    start_time = time.time()
    clusters = cluster_beats(sig, qrs_peaks, fs)
    n_clusters = int(clusters.max()) + 1
    window = template_window(fs)
    if verbose:
        print(f"{len(qrs_peaks)} beats in {n_clusters} clusters")

    # Both coefficient sets start at zero. Each covers its signal plus `pad`
    # samples on either side, cropped after the inverse DWT.
    wavelet = pywt.Wavelet(WAVELET)
    pad = wavelet.dec_len // 2 - 1
    n_template_coefs = window[1] - window[0] + 2 * pad
    for _ in range(TEMPLATE_LEVELS):
        n_template_coefs = pywt.dwt_coeff_len(n_template_coefs, wavelet.dec_len, "symmetric")
    n_baseline_coefs = -(-n_time // 2 ** BASELINE_LEVELS) + 2 * pad

    keras.backend.clear_session(free_memory=True)
    # The model has no real input; its output is a function of its weights only.
    dummy = keras.Input(shape=(1,))
    template_coefs = Coefficients((n_clusters * n_lead, n_template_coefs))
    baseline_coefs = Coefficients((n_lead, n_baseline_coefs))

    templates = InverseWavelet(TEMPLATE_LEVELS)(template_coefs(dummy))[:, pad:-pad]
    templates = TemplatePenalties(n_clusters, n_lead, window, fs)(templates)
    beats = PlaceTemplates(qrs_peaks + window[0], clusters, n_time)(
        keras.ops.reshape(templates, (n_clusters, n_lead, -1))[None, :, :, :])
    baselines = InverseWavelet(BASELINE_LEVELS)(baseline_coefs(dummy))[None, :, pad:-pad][..., :n_time]
    baselines = BaselineInertia(fs, baseline_inertia_weight)(baselines)
    model = keras.Model(inputs=dummy, outputs=beats + baselines)

    sig_tf = tf.constant(sig.T[None], dtype=tf.float32)
    one = tf.constant([1], dtype=tf.float32)
    mse = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            loss = mse(sig_tf, model(one)) + tf.reduce_sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    losses = np.array([float(step()) for _ in range(n_steps)])

    fitted_templates = np.asarray(InverseWavelet(TEMPLATE_LEVELS)(template_coefs.coefs))[:, pad:-pad]
    fitted_baselines = np.asarray(InverseWavelet(BASELINE_LEVELS)(baseline_coefs.coefs))[:, pad:-pad][:, :n_time]
    sig_morph = np.asarray(model(one))[0] - fitted_baselines

    elapsed_s = time.time() - start_time
    if verbose:
        print(f"fit done in {elapsed_s:.0f} s, final loss {losses[-1]:.4g}")
    return FitResult(
        sig_morph=sig_morph.T,
        baselines=fitted_baselines.T,
        templates=fitted_templates.reshape(n_clusters, n_lead, -1),
        template_window_s=np.asarray(window, dtype=np.float64) / fs,
        qrs_peaks=qrs_peaks,
        clusters=clusters,
        losses=losses,
        fs=fs,
        elapsed_s=elapsed_s,
    )


def template_window(fs):
    """TEMPLATE_WINDOW_S in samples, widened to a length the inverse DWT can produce."""
    start, end = (np.array(TEMPLATE_WINDOW_S) * fs).astype(int)
    length = -(-(end - start) // 2 ** TEMPLATE_LEVELS) * 2 ** TEMPLATE_LEVELS
    # Spread the extra samples over both sides in proportion to the window.
    extra = length - (end - start)
    start = int(start - extra / (TEMPLATE_WINDOW_S[1] - TEMPLATE_WINDOW_S[0]) * -TEMPLATE_WINDOW_S[0])
    return np.array([start, start + length])


class Coefficients(keras.layers.Layer):
    """Trainable wavelet coefficients, zero at the start. Ignores its input."""

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.coefs = self.add_weight(shape=shape, initializer="zeros")

    def call(self, inputs):
        return tf.convert_to_tensor(self.coefs)


class InverseWavelet(keras.layers.Layer):
    """`levels` inverse DWT steps with zero detail coefficients. [channel, time] -> [channel, time * 2**levels]"""

    def __init__(self, levels, **kwargs):
        super().__init__(**kwargs)
        self.levels = levels
        self.idwt = IDWT1D(WAVELET)

    def call(self, x):
        for _ in range(self.levels):
            x = self.idwt(keras.ops.stack([x, np.zeros(x.shape, dtype=np.float32)], axis=-1))[:, :, 0]
        return x


class PlaceTemplates(keras.layers.Layer):
    """Sum each beat's cluster template into the record. [1, cluster, lead, W] -> [1, lead, time]"""

    def __init__(self, starts, clusters, n_time, **kwargs):
        super().__init__(**kwargs)
        self.starts = np.asarray(starts)
        self.clusters = np.asarray(clusters)
        self.n_time = n_time

    def call(self, templates):
        n = self.n_time
        placed = []
        for c in sorted(set(self.clusters)):
            padded = keras.ops.pad(templates[:, c], pad_width=((0, 0), (0, 0), (n, n)))
            placed.extend(padded[:, :, n - s:2 * n - s] for s in self.starts[self.clusters == c])
        return keras.ops.sum(placed, axis=0)


class TemplatePenalties(keras.layers.Layer):
    """Adds the four template terms as losses. [cluster * lead, W] -> unchanged"""

    def __init__(self, n_clusters, n_lead, window, fs, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters, self.n_lead, self.window, self.fs = n_clusters, n_lead, window, fs

    def call(self, templates):
        fs = self.fs
        before, after = -self.window[0], self.window[1]

        # L1 decay after the T wave (from +0.65 s), cubic ramp.
        n_late = after - int(0.65 * fs)
        ramp = (np.arange(n_late) / n_late) ** 3 * 0.01
        self.add_loss(keras.ops.mean(tf.abs(templates[..., -n_late:] * ramp)))

        # L1 decay before the P wave (until -0.25 s), quadratic ramp, heaviest at the window start.
        n_early = before - int(0.25 * fs)
        ramp = (np.arange(n_early) / n_early) ** 2 * (0.01 * n_early / n_late)
        self.add_loss(keras.ops.mean(tf.abs(templates[..., :n_early] * ramp[::-1])))

        # Group shrinkage outside the QRS (+-0.05 s): L2 across leads, mean over time.
        by_lead = keras.ops.reshape(templates, (self.n_clusters, self.n_lead, -1))

        def l2_across_leads(x):
            return keras.ops.mean(tf.sqrt(tf.maximum(0.0, tf.reduce_sum(x ** 2, axis=1))))

        n_post = after - int(0.05 * fs)
        n_pre = before - int(0.05 * fs)
        self.add_loss(l2_across_leads(by_lead[..., -n_post:]) * 0.008)
        self.add_loss(l2_across_leads(by_lead[..., :n_pre]) * (0.008 * n_pre / n_post))
        return templates


class BaselineInertia(keras.layers.Layer):
    """Adds the baseline term as a loss: the std of the baseline's
    sample-to-sample difference in 1 s windows (10 ms hop). [1, lead, time] -> unchanged"""

    def __init__(self, fs, weight, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.weight = fs, weight

    def call(self, baselines):
        slope = baselines[..., 1:] - baselines[..., :-1]
        frames = tf.signal.frame(slope, frame_length=int(self.fs * 1), frame_step=int(self.fs // 100), axis=2)
        self.add_loss(keras.ops.mean(keras.ops.abs(keras.ops.std(frames, axis=-1)) * self.weight))
        return baselines
