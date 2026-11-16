"""Optional CNN quality adapter for NEW common-split experiments; no AF imports."""
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt
from quality_only import sampling_rate, sha256


def preprocess(raw, fs):
    fs = sampling_rate(fs)
    x = np.asarray(raw, dtype=np.float64)
    if x.shape != (10*fs,) or not np.isfinite(x).all():
        raise ValueError('Expected finite, single-channel 10-second input')
    ratio = Fraction(30, fs)
    z = resample_poly(x-x.mean(), ratio.numerator, ratio.denominator) + x.mean()
    if z.std() < 1e-3:
        return np.zeros(300, dtype=np.float32)
    b, a = butter(3, [.5/15, 5/15], btype='band')
    z = filtfilt(b, a, z-z.mean())
    return ((z-z.mean())/(z.std()+1e-8)).astype(np.float32)


class QualityCNNOnly:
    def __init__(self, folder):
        import torch
        from but_quality_cnn_runtime import PPGQualityCNN
        folder = Path(folder)
        self.manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('format') != 'capdyssey-quality-cnn-v1':
            raise ValueError('Not a new common-split CNN bundle')
        for name in ['quality_cnn.pt', 'scripts/quality_cnn_only.py', 'scripts/but_quality_cnn_runtime.py']:
            path = folder/name if '/' not in name else Path(__file__).parent/Path(name).name
            if sha256(path) != self.manifest['sha256'].get(name):
                raise ValueError('CNN model or code checksum mismatch: '+name)
        self.threshold = float(self.manifest['quality_threshold'])
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= np.nextafter(1., np.inf):
            raise ValueError('Invalid threshold')
        self.model_hash = sha256(folder/'quality_cnn.pt')
        self.model = PPGQualityCNN().cpu().eval()
        self.model.load_state_dict(torch.load(folder/'quality_cnn.pt', map_location='cpu', weights_only=True))

    def predict(self, raw, fs):
        import torch
        fs = sampling_rate(fs)
        x = np.asarray(raw, dtype=float)
        if x.shape != (10*fs,):
            raise ValueError('Expected exactly 10 seconds')
        if not np.isfinite(x).all() or x.std() < 1e-8:
            return dict(status='invalid_input', reason='nonfinite_or_flat', quality_score=None,
                        predicted_quality=None, threshold=self.threshold)
        with torch.no_grad():
            score = float(self.model(torch.from_numpy(preprocess(x,fs))[None,None]).sigmoid().item())
        if not np.isfinite(score):
            raise ValueError('Nonfinite CNN score')
        label = int(score >= self.threshold)
        return dict(status='good' if label else 'bad', reason='', quality_score=score,
                    predicted_quality=label, threshold=self.threshold)
