"""Frozen research prototype: raw 10-second PPG -> quality gate -> AF classifier."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from but_quality_rf_runtime import FEATURES,extract_features
from raw_quality_features import EXTRA,FEATURE_SPEC,raw_artifact_features


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class QualityAF:
    """RF uses CPU; AF Predictor requires CUDA. No source/device-specific thresholds."""
    def __init__(self,folder):
        from finalize_two_sources import Predictor
        self.folder=Path(folder).resolve()
        self.config=json.loads((self.folder/'manifest.json').read_text(encoding='utf-8'))
        assert self.config['format']=='capdyssey-quality-af-v2'
        for relative,expected in self.config['sha256'].items():
            path=(self.folder/relative).resolve()
            if not path.is_relative_to(self.folder):raise ValueError('Invalid bundle path')
            if digest(path)!=expected:raise ValueError('Bundle checksum mismatch: '+relative)
        self.rf=joblib.load(self.folder/'quality_rf.joblib') # Only your own trusted bundle.
        assert self.rf['feature_cols']==FEATURES+EXTRA
        assert self.rf['extra_feature_spec']==FEATURE_SPEC
        assert list(self.rf['model'].classes_)==[0,1]
        # The selected operating point is in the manifest, not the RF training threshold.
        self.quality_threshold=float(self.config['quality_threshold'])
        self.af=Predictor(self.folder/'af_model.pt')
        assert self.af.threshold==self.config['af_threshold']

    def predict(self,raw,fs):
        from prepare_af_centered import centered_window
        fs=float(fs)
        x=np.asarray(raw,dtype=float)
        if not np.isfinite(fs) or not fs.is_integer() or not 25<=fs<=2000:
            raise ValueError('Provide the actual integer sampling rate, 25–2000 Hz.')
        fs=int(fs)
        if x.ndim!=1 or len(x)!=10*fs:
            raise ValueError('Exactly 10 seconds of one raw PPG channel required; no padding.')
        if not np.isfinite(x).all() or np.std(x)<1e-8:
            return dict(status='remeasure',reason='invalid_or_flat_signal',quality_score=None,af_score=None,label=None)
        f=pd.DataFrame([dict(**extract_features(x,fs),**raw_artifact_features(x,fs))],columns=self.rf['feature_cols'])
        q=float(self.rf['model'].predict_proba(f)[0,1])
        if not np.isfinite(q):raise ValueError('Nonfinite quality score')
        if q<self.quality_threshold:
            return dict(status='remeasure',reason='quality_gate',quality_score=q,af_score=None,label=None)
        p=float(self.af.predict(centered_window(x,fs)[None,:])[0])
        if not np.isfinite(p):raise ValueError('Nonfinite AF score')
        return dict(status='classified',reason=None,quality_score=q,af_score=p,
                    label='AF' if p>=self.af.threshold else 'non_AF')
