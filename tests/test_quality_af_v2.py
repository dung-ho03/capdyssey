"""Synthetic contract checks without local model inference or third-party model loading."""
import ast
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def make_engine(self,q=.4,p=.8):
        tree=ast.parse((ROOT/'scripts/quality_af_v2.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QualityAF')
        env=dict(np=np,pd=pd,extract_features=lambda x,fs:{'basic':1.},
                 raw_artifact_features=lambda x,fs:{'extra':2.})
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'quality_af_v2','exec'),env)
        engine=env['QualityAF'].__new__(env['QualityAF'])
        engine.rf={'feature_cols':['basic','extra'],'model':types.SimpleNamespace(predict_proba=lambda x:np.array([[1-q,q]]))}
        engine.quality_threshold=.3
        calls=[]
        def predict(x):calls.append(x);return np.array([p])
        engine.af=types.SimpleNamespace(threshold=.5,predict=predict)
        return engine,calls

    def setUp(self):
        helper=types.ModuleType('prepare_af_centered')
        helper.centered_window=lambda x,fs:np.ones(500)
        self.stub=patch.dict(sys.modules,{'prepare_af_centered':helper});self.stub.start()
        self.addCleanup(self.stub.stop)
        self.wave=np.sin(np.arange(1250)*.1)

    def test_quality_rejection_never_calls_af_or_returns_non_af(self):
        e,calls=self.make_engine(q=.2)
        r=e.predict(self.wave,125)
        self.assertEqual(r['status'],'remeasure');self.assertIsNone(r['label']);self.assertIsNone(r['af_score'])
        self.assertEqual(len(calls),0)

    def test_threshold_boundaries_and_non_af(self):
        e,calls=self.make_engine(q=.3,p=.5)
        self.assertEqual(e.predict(self.wave,125)['label'],'AF')
        self.assertEqual(calls[0].shape,(1,500))
        e,_=self.make_engine(q=.3,p=.499)
        self.assertEqual(e.predict(self.wave,125)['label'],'non_AF')

    def test_invalid_data_not_a_negative_classification(self):
        e,calls=self.make_engine()
        for x in [np.zeros(1250),np.full(1250,np.nan)]:
            self.assertEqual(e.predict(x,125)['reason'],'invalid_or_flat_signal')
        self.assertEqual(len(calls),0)
        for x,fs in [(self.wave[:-1],125),(self.wave,125.1),(self.wave,np.nan),(self.wave.reshape(2,625),125)]:
            with self.assertRaises(ValueError):e.predict(x,fs)

    def test_new_notebook_and_modules_parse(self):
        for name in ['quality_af_v2.py','raw_quality_features.py','export_quality_af_v2.py']:
            ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'))
        nb=json.loads((ROOT/'notebooks/11_quality_af_v2_colab.ipynb').read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))

    def test_replay_keeps_window_position_without_dataframe_index(self):
        tree=ast.parse((ROOT/'scripts/export_quality_af_v2.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay_samples')
        scope={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'replay_samples','exec'),scope)
        records=pd.DataFrame({'source':['six']*8,'record':['a']*4+['b']*4,
            'y':[0,0,1,1,0,0,1,1],'accepted':[True,False,True,False]*2},
            index=[10,12,17,20,25,31,37,48])
        selected=scope['replay_samples'](records)
        self.assertNotIn('replay_window_index',records.columns)
        for row in selected.itertuples(index=False):
            original=records[records.record==row.record].iloc[row.replay_window_index]
            self.assertEqual(original.y,row.y)
            self.assertEqual(original.accepted,row.accepted)
            self.assertFalse(hasattr(row,'Index'))


if __name__=='__main__':unittest.main()
