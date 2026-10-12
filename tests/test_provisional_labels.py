import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_af_provisional import strict_labels

class ProvisionalLabelsTest(unittest.TestCase):
    def test_homogeneous_binary_and_uncovered(self):
        q=np.arange(61,dtype=float);a=np.r_[np.zeros(30),np.ones(30)]
        np.testing.assert_array_equal(strict_labels(q,a,[3,33,25,-1,55,np.nan]),[0,1,-1,-1,-1,-1])
    def test_fractional_and_missing_never_cast_to_binary(self):
        for value in (.25,.5,np.nan,2):
            a=np.zeros(30);a[8]=value
            self.assertEqual(strict_labels(np.arange(31),a,[3])[0],-1)
    def test_transition_guard(self):
        a=np.zeros(30);a[2]=1
        self.assertEqual(strict_labels(np.arange(31),a,[3])[0],-1)
    def test_bad_timing_fails(self):
        with self.assertRaises(ValueError):strict_labels([0,2,1],[0,1],[0])

if __name__=='__main__':unittest.main()
