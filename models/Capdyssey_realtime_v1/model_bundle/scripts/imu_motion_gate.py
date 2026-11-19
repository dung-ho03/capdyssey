"""Experimental accelerometer gate. Thresholds require real-device validation."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    rms_threshold_g: float = 0.10
    peak_threshold_g: float = 0.50
    max_gap_s: float = 0.10
    edge_tolerance_s: float = 0.05

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


def assess_motion(accel, times_s, ppg_start_s, *, unit, config):
    """Use the same device clock for IMU timestamps and the PPG window start.

    accel: (N,3) acceleration including gravity; unit must be 'g' or 'm/s2'.
    Only samples in [ppg_start_s, ppg_start_s + 10) are used. Nominal >=25 Hz
    acquisition is recommended; absent, incomplete or gapped data never pass.
    Demeaning removes constant gravity, but rotation also contributes to motion.
    This is an exclusion rule, not activity recognition or artifact correction.
    """
    invalid = dict(state='invalid', rms_g=None, peak_g=None)
    if accel is None or times_s is None or ppg_start_s is None:
        return dict(invalid, reason='imu_missing')
    if unit not in ('g', 'm/s2'):
        return dict(invalid, reason='imu_unit_invalid')
    try:
        a = np.asarray(accel, dtype=float)
        t = np.asarray(times_s, dtype=float)
        start = float(ppg_start_s)
    except (TypeError, ValueError):
        return dict(invalid, reason='imu_invalid')
    if (a.ndim != 2 or a.shape[1] != 3 or t.ndim != 1 or len(a) != len(t)
            or len(t) < 2 or not np.isfinite(start)
            or not np.isfinite(t).all() or not np.isfinite(a).all()
            or np.any(np.diff(t) <= 0)):
        return dict(invalid, reason='imu_invalid')
    mask = (t >= start) & (t < start + 10.0)
    a, t = a[mask], t[mask]
    if (len(t) < 100 or t[0] - start > config.edge_tolerance_s + 1e-9
            or start + 10.0 - t[-1] > config.edge_tolerance_s + 1e-9
            or np.max(np.diff(t)) > config.max_gap_s + 1e-9):
        return dict(invalid, reason='imu_incomplete_or_unsynchronized')
    if unit == 'm/s2':
        a = a / 9.80665
    if np.max(np.abs(a)) < 1e-8:
        return dict(invalid, reason='imu_zero_signal')
    deviation = np.linalg.norm(a - a.mean(axis=0), axis=1)
    rms = float(np.sqrt(np.mean(deviation ** 2)))
    peak = float(deviation.max())
    if not np.isfinite(rms) or not np.isfinite(peak):
        return dict(invalid, reason='imu_invalid')
    moving = rms >= config.rms_threshold_g or peak >= config.peak_threshold_g
    return dict(state='motion' if moving else 'still', reason='motion_detected' if moving else None,
                rms_g=rms, peak_g=peak)
