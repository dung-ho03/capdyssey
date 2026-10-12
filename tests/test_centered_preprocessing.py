import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_af_centered import centered_window

class CenteredTests(unittest.TestCase):
    def test_large_dc_offset_invariance(self):
        for fs in (100,125):
            t=np.arange(fs*10)/fs
            raw=np.sin(2*np.pi*1.3*t)+.2*np.cos(2*np.pi*2.6*t)
            reference=centered_window(raw,fs)
            for offset in (-1000000,1000000):
                np.testing.assert_allclose(centered_window(raw+offset,fs),reference,atol=1e-6)
            self.assertEqual(reference.shape,(500,))
    def test_flat_or_nonfinite_rejected(self):
        for raw in (np.ones(1000)*1e6,np.full(1000,np.nan)):
            with self.assertRaises(ValueError):centered_window(raw,100)

if __name__=='__main__':unittest.main()
