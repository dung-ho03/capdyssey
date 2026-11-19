"""IMU exclusion -> original frozen PPG quality RF -> original AF model."""
import numpy as np
from imu_motion_gate import MotionConfig, assess_motion


class MotionQualityAF:
    def __init__(self, folder, motion_config=None):
        self.folder = folder
        self.motion_config = motion_config if motion_config is not None else MotionConfig()
        self._ppg_model = None

    def predict(self, raw, fs, *, accel=None, imu_times_s=None,
                ppg_start_s=None, accel_unit=None):
        # Validate the PPG window contract even when it will be excluded.
        fs = float(fs)
        x = np.asarray(raw, dtype=float)
        if not np.isfinite(fs) or not fs.is_integer() or not 25 <= fs <= 2000:
            raise ValueError('Provide the actual integer sampling rate, 25–2000 Hz.')
        if x.ndim != 1 or len(x) != 10 * int(fs):
            raise ValueError('Exactly 10 seconds of one raw PPG channel required; no padding.')
        motion = assess_motion(accel, imu_times_s, ppg_start_s,
                               unit=accel_unit, config=self.motion_config)
        if motion['state'] != 'still':
            return dict(status='excluded' if motion['state'] == 'motion' else 'remeasure',
                        reason=motion['reason'], quality_score=None, af_score=None,
                        label=None, motion=motion)
        # Lazy loading means excluded windows never invoke either learned model.
        if self._ppg_model is None:
            from quality_af_v2 import QualityAF
            self._ppg_model = QualityAF(self.folder)
        result = dict(self._ppg_model.predict(x, int(fs)))
        result['motion'] = motion
        return result
