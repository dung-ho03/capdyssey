import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_af_integrated import normalized_windows,start_time
from train_af_integrated import preflight,SOURCES

class IntegratedDataTests(unittest.TestCase):
    def test_no_padding_and_window_times(self):
        rows=list(normalized_windows(np.sin(np.arange(2050)/10),100))
        self.assertEqual([r[0] for r in rows],[0,10])
        self.assertTrue(all(r[1].shape==(500,) for r in rows))
        self.assertEqual(start_time('02','00:00:01'),86401)
    def test_bad_window_not_silently_relabelled(self):
        self.assertIsNone(list(normalized_windows(np.full(1000,np.nan),100))[0][1])
    def test_preflight_rejects_unknown_even_if_status_toggled(self):
        m={'sources':list(SOURCES),'window_seconds':10,'sampling_hz':50,'shards':[]}
        for s in SOURCES:
            for i in range(5):m['shards'].append({'source':s,'subject':s+str(i),'label_counts':{'-1':1 if 'wrist' in s else 0}})
        status={'datasets':{s:{'ready':True} for s in SOURCES}}
        self.assertTrue(any('unresolved' in e for e in preflight(m,status)))
        m['sources'].remove('six_rhythms')
        self.assertTrue(any('All three' in e for e in preflight(m,status)))

if __name__=='__main__':unittest.main()
