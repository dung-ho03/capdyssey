import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from af_gpu_evidence import threshold_report

class ThresholdReportTests(unittest.TestCase):
    def test_separable_scores(self):
        r=threshold_report(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))
        self.assertEqual(r['at_sensitivity_90']['sensitivity'],1.)
        self.assertEqual(r['at_sensitivity_90']['specificity'],1.)
    def test_equal_scores_do_not_invent_separation(self):
        r=threshold_report(np.array([0,0,1,1]),np.array([.5]*4))
        self.assertEqual(r['at_05']['sensitivity'],1.)
        self.assertEqual(r['at_05']['specificity'],0.)
        self.assertEqual(r['at_specificity_80']['sensitivity'],0.)

if __name__=='__main__':unittest.main()
