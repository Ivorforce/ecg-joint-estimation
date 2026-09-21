# ecgfit

Joint estimation of beat morphology and baseline for the 12-lead ECG, and the
experiments of the paper

> L. Tenbrink, K. D. Rizas, A. Loewe. *Joint Morphology and Baseline Estimation
> for the 12-Lead ECG via an Interpretable Cost Function.* Computing in
> Cardiology 2026.

The fit models a recording as one beat template per beat cluster and lead,
placed at every R peak, plus a slowly drifting baseline per lead. Both are
estimated together by gradient descent on the reconstruction error and five
regularization terms. The result is the beat morphology and the baseline as two
separate signals.

## Install

```bash
pip install -e .                  # the fitter
pip install -e ".[experiments]"   # plus what the paper's scripts need
```

To reproduce the paper's fits bit for bit, use Python 3.9 and the pinned
versions: `pip install -r requirements.txt && pip install -e . --no-deps`.

## Use

```python
from ecgfit import fit

result = fit(sig, qrs_peaks, fs=500)
# sig:        float[time, lead], unfiltered, in mV (the paper uses 10 s records)
# qrs_peaks:  R peak sample indices, from any detector
# fs:         sampling rate; the model is tuned for 500 Hz

result.sig_morph      # float[time, lead]: the beat templates placed at every R peak
result.baselines      # float[time, lead]: the baseline
result.templates      # float[cluster, lead, time]: one template per beat cluster
sig - result.baselines  # the recording with only the baseline removed
```

Fitting a 10-second 12-lead record takes about 10 s on a laptop CPU.

## Reproduce the paper

Data, into `data/` (or pass other paths to the scripts):

| dataset | source | path |
|---|---|---|
| LUDB | https://physionet.org/content/ludb/ | `data/ludb` (the folder with `1.hea`, `1.ii`, ...) |
| MIT-BIH NSTDB | https://physionet.org/content/nstdb/ | `data/nstdb` (the folder with `bw.hea`, `em.hea`) |
| MedalCare-XL | https://doi.org/10.5281/zenodo.7293655 | `data/medalcare-xl` (the `WP2_largeDataset_Noise` folder) |

The fiducial experiment also needs the `ecgpuwave` binary on `PATH`
(https://physionet.org/content/ecgpuwave/).

```bash
# Fits, one file per record in fits/. Existing files are skipped; run several
# processes with --worker i --n_workers k to split the work.
python scripts/fit_ludb.py         # 200 LUDB records
python scripts/fit_denoising.py    # LUDB 1-20 with NSTDB bw and em at 0, 6 and 12 dB
python scripts/fit_medalcare.py    # 30 sinus, 20 each AV block, LBBB, RBBB

# Numbers and figures, into results/
python scripts/evaluate_denoising.py          # Noise robustness, Fig. 2
python scripts/evaluate_medalcare.py          # Fidelity and cross-lead consistency on MedalCare-XL
python scripts/evaluate_ecgpuwave.py          # Downstream fiducial recovery, Table 1
python scripts/evaluate_residual_spectrum.py  # Residual spectrum
python scripts/evaluate_clusters.py           # Number of beat clusters (Method)
```

R peaks come from the LUDB lead II expert annotations and, on MedalCare-XL,
from wfdb's XQRS detector. The paper drew the NSTDB noise segments at random;
`scripts/nstdb_segments.json` records where each record's segment starts, so
that the noise is the same on every run.

With the pinned versions on macOS arm64, the fits are bit-identical to the
paper's. Other versions or platforms can change float32 rounding, and 300
optimization steps amplify that, so individual fits may then differ slightly.
