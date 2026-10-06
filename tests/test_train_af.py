import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('train_af',Path(__file__).resolve().parents[1]/'scripts/train_af.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class TrainingTests(unittest.TestCase):
    def test_subject_splits_disjoint_complete_and_stratified(self):
        records=[(f'p{i:03d}',int(i<19),None) for i in range(35)]
        splits=module.split_subjects(records,42)
        all_ids=sum(splits.values(),[])
        self.assertEqual(len(all_ids),35)
        self.assertEqual(len(set(all_ids)),35)
        labels={r[0]:r[1] for r in records}
        for ids in splits.values():self.assertEqual({labels[p] for p in ids},{0,1})
        self.assertEqual(splits,module.split_subjects(records,42))

    def test_confusion_and_auc(self):
        m=module.metrics(np.array([0,0,1,1]),np.array([0.1,0.7,0.4,0.9]))
        self.assertEqual([m[k] for k in ['tn','fp','fn','tp']],[1,1,1,1])
        self.assertEqual(m['roc_auc'],0.75)
        self.assertEqual(m['balanced_accuracy'],0.5)

    def test_nonfinite_and_flat_windows_excluded(self):
        wave=np.sin(np.arange(3750)/10)
        values=np.concatenate([wave,np.zeros(3750),np.full(3750,np.nan)])
        x,y,people,starts,audit=module.prepare([('p1',1,values)])
        self.assertEqual(x.shape,(1,1500))
        self.assertEqual(audit[0]['excluded_windows'],2)
        self.assertTrue(np.isfinite(x).all())
        self.assertAlmostEqual(float(x.std()),1.0,places=5)

if __name__=='__main__':unittest.main()
