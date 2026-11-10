"""Frozen feature definitions from the evaluated Colab experiment."""
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly,welch

EXTRA = ['raw_exact_flat_fraction', 'raw_exact_flat_max_s', 'raw_low_slope_fraction', 'raw_low_slope_max_s', 'raw_step_to_span', 'raw_max_slope_to_p95', 'raw_slow_power_fraction', 'raw_local_span_min_ratio', 'raw_local_span_median_ratio']
FEATURE_SPEC = dict(version='raw-artifact-v1', span_quantiles=[0.05, 0.95], low_slope_span_per_second=0.01, spectrum_fs=30, spectrum_segment=240, local_window_seconds=1, local_stride_seconds=0.5, note='Experimental features, not hard quality exclusion rules; no AF labels used to fit RF.')

def longest_run(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(lengths.max()) if len(lengths) else 0

def raw_artifact_features(raw, fs):
    """Dimensionless or time-based features, before filtering/normalization.

    No absolute amplitude rejection. Step/flat features are not physiological diagnoses.
    Quantization and sample rate can affect these features and require transfer evaluation.
    """
    x = np.asarray(raw, dtype=float)
    assert x.ndim == 1 and len(x) == round(10 * fs) and np.isfinite(x).all()
    if np.ptp(x) == 0:
        return dict(zip(EXTRA, [1.0, (len(x) - 1) / fs, 1.0, (len(x) - 1) / fs, 0.0, 0.0, 0.0, 0.0, 0.0]))
    x = x - np.median(x)
    scale = float(np.quantile(x, 0.95) - np.quantile(x, 0.05))
    scale = max(scale, float(np.ptp(x)) * 1e-06)
    z = x / scale
    diff = np.abs(np.diff(z))
    exact = np.diff(x) == 0
    slope = diff * fs
    low = slope < FEATURE_SPEC['low_slope_span_per_second']
    ratio = Fraction(30 / int(fs)).limit_denominator(10000)
    x30 = resample_poly(z - z.mean(), ratio.numerator, ratio.denominator, padtype='line')
    (f, p) = welch(x30, fs=30, nperseg=240, detrend=False)
    total = float(p[(f > 0) & (f <= 5)].sum())
    slow = float(p[(f > 0) & (f < 0.5)].sum()) / (total + 1e-15)
    (width, step) = (int(fs), max(1, int(fs / 2)))
    spans = [float(np.quantile(z[i:i + width], 0.95) - np.quantile(z[i:i + width], 0.05)) for i in range(0, len(z) - width + 1, step)]
    values = [float(exact.mean()), longest_run(exact) / fs, float(low.mean()), longest_run(low) / fs, float(diff.max()), float(min(slope.max() / (np.quantile(slope, 0.95) + 1e-12), 1000000.0)), slow, float(min(spans)), float(np.median(spans))]
    assert np.isfinite(values).all()
    return dict(zip(EXTRA, values))
