import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from imu_motion_gate import MotionConfig, assess_motion
from quality_af_imu import MotionQualityAF


class FakePPG:
    def __init__(self, result):
        self.result, self.calls = result, 0
    def predict(self, raw, fs):
        self.calls += 1
        return self.result


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.t = 100 + np.arange(500) / 50
        self.a = np.tile([0., 0., 1.], (500, 1))
        self.x = np.sin(np.arange(1000) / 10)
        self.model = MotionQualityAF('unused')
        self.fake = FakePPG(dict(status='classified', reason=None, label='AF',
                                quality_score=.9, af_score=.8))
        self.model._ppg_model = self.fake

    def predict(self, a=None, t=None, unit='g'):
        return self.model.predict(self.x, 100, accel=self.a if a is None else a,
            imu_times_s=self.t if t is None else t, ppg_start_s=100, accel_unit=unit)

    def test_still_preserves_both_labels_and_quality_failure(self):
        for result in [dict(status='classified', label='AF', af_score=.8),
                       dict(status='classified', label='non_AF', af_score=.1),
                       dict(status='remeasure', label=None, af_score=None)]:
            self.fake.result = result
            actual = self.predict()
            for k, v in result.items(): self.assertEqual(actual[k], v)
        self.assertEqual(self.fake.calls, 3)

    def test_motion_never_calls_models(self):
        self.a[:, 0] = .3 * np.sin(2 * np.pi * 2 * (self.t - 100))
        r = self.predict()
        self.assertEqual(r['status'], 'excluded')
        self.assertIsNone(r['label'])
        self.assertIsNone(r['af_score'])
        self.assertEqual(self.fake.calls, 0)

    def test_brief_peak_excluded(self):
        self.a[250, 0] = 1
        self.assertEqual(self.predict()['status'], 'excluded')

    def test_units_equivalent(self):
        self.a[:, 0] = .2 * np.sin(self.t)
        r1, r2 = self.predict(), self.predict(self.a * 9.80665, unit='m/s2')
        self.assertAlmostEqual(r1['motion']['rms_g'], r2['motion']['rms_g'])
        self.assertEqual(r1['status'], r2['status'])

    def test_invalid_and_incomplete_never_pass(self):
        cases = [(self.a, self.t + 20), (self.a[:-10], self.t[:-10]),
                 (self.a[10:], self.t[10:]), (self.a, self.t[::-1]),
                 (self.a, np.repeat(100., 500)), (np.zeros_like(self.a), self.t),
                 (self.a[:, :2], self.t), (self.a * np.nan, self.t),
                 (np.delete(self.a, range(100, 120), axis=0),
                  np.delete(self.t, range(100, 120)))]
        for a, t in cases:
            with self.subTest(shape=a.shape):
                r = self.predict(a, t)
                self.assertEqual(r['status'], 'remeasure')
                self.assertIsNone(r['label'])
        self.assertEqual(self.fake.calls, 0)

    def test_missing_and_unknown_unit(self):
        self.assertEqual(self.model.predict(self.x, 100)['reason'], 'imu_missing')
        self.assertEqual(self.predict(unit='unknown')['reason'], 'imu_unit_invalid')
        self.assertEqual(self.fake.calls, 0)

    def test_threshold_equality_excluded(self):
        self.a[250, 0] = .2
        baseline = assess_motion(self.a, self.t, 100, unit='g', config=MotionConfig())
        self.model.motion_config = MotionConfig(peak_threshold_g=baseline['peak_g'])
        self.assertEqual(self.predict()['status'], 'excluded')

    def test_bad_config(self):
        for v in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): MotionConfig(rms_threshold_g=v)

    def test_ppg_shape_validated(self):
        with self.assertRaises(ValueError): self.model.predict(self.x[:-1], 100)


if __name__ == '__main__':
    unittest.main()
