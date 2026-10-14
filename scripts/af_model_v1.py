"""AF research v1: raw single-channel 10-second PPG -> fixed-threshold output."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from finalize_two_sources import Predictor
from prepare_af_centered import centered_window

class AFModel:
    def __init__(self,release):
        self.release=Path(release)
        config=json.loads((self.release/'release.json').read_text(encoding='utf-8'))
        if config['version']!='af-two-source-v1':raise ValueError('Unsupported release')
        for name,expected in config['sha256'].items():
            path=self.release/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Release integrity mismatch: '+name)
        self.predictor=Predictor(self.release/'model.pt')
        if self.predictor.threshold!=config['threshold']:raise ValueError('Threshold mismatch')
        self.version=config['version']

    def predict(self,raw,sampling_hz):
        if isinstance(sampling_hz,bool) or not np.isscalar(sampling_hz):raise ValueError('Invalid sampling rate')
        fs=float(sampling_hz)
        if not np.isfinite(fs) or fs<25 or fs>2000:raise ValueError('Sampling rate outside supported preprocessing range')
        raw=np.asarray(raw,dtype=np.float64)
        if raw.ndim!=1 or abs(len(raw)-10*fs)>1e-6:raise ValueError('Exactly 10 seconds of one PPG channel required')
        if not np.isfinite(raw).all() or raw.std()<1e-8:raise ValueError('Missing or flat PPG; prediction unavailable')
        x=centered_window(raw,fs)
        score=float(self.predictor.predict(x[None])[0])
        if not np.isfinite(score):raise ValueError('Nonfinite model output')
        return {'model_version':self.version,'prediction':'AF' if score>=self.predictor.threshold else 'non-AF',
                'af_score':score,'threshold':self.predictor.threshold,'window_seconds':10,'input_sampling_hz':fs,
                'score_is_calibrated_probability':False,'quality_model_applied':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--sampling-hz',type=float,required=True);a=p.parse_args()
    print(json.dumps(AFModel(a.release).predict(np.load(a.input,allow_pickle=False),a.sampling_hz),ensure_ascii=False,indent=2))
