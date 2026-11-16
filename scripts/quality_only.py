"""CPU-only PPG quality inference. Never imports or initializes the AF model."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from but_quality_rf_runtime import CONFIG, FEATURES, extract_features
from raw_quality_features import EXTRA, FEATURE_SPEC, raw_artifact_features


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sampling_rate(fs):
    fs = float(fs)
    if not np.isfinite(fs) or not fs.is_integer() or not 25 <= fs <= 2000:
        raise ValueError('fs must be the actual integer sample rate, 25–2000 Hz')
    return int(fs)


class QualityOnly:
    """Load a trusted release folder. Manifest threshold overrides training threshold.

    'good' means model-estimated HR quality, not confirmed sensor performance.
    Invalid/nonfinite/flat input is kept separate from the learned bad class.
    """
    def __init__(self, folder):
        folder = Path(folder).resolve()
        self.manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('format') not in ('capdyssey-quality-af-v2', 'capdyssey-quality-only-v1'):
            raise ValueError('Unsupported quality release')
        expected = self.manifest.get('sha256', {}).get('quality_rf.joblib')
        if not expected or sha256(folder / 'quality_rf.joblib') != expected:
            raise ValueError('Quality model checksum mismatch')
        # Deserialization is appropriate only for your trusted model release.
        self.bundle = joblib.load(folder / 'quality_rf.joblib')
        b = self.bundle
        if b.get('config') != CONFIG or b.get('feature_cols') not in (FEATURES, FEATURES + EXTRA):
            raise ValueError('Incompatible feature/preprocessing configuration')
        self.extended = b['feature_cols'] == FEATURES + EXTRA
        if self.extended and b.get('extra_feature_spec') != FEATURE_SPEC:
            raise ValueError('Incompatible raw feature specification')
        if list(b['model'].classes_) != [0, 1]:
            raise ValueError('Expected classes [0=bad, 1=good]')
        self.threshold = float(self.manifest['quality_threshold'])
        # nextafter(1,+inf) is a valid validation-selected reject-all threshold.
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= np.nextafter(1., np.inf):
            raise ValueError('Invalid quality threshold')
        for name in ('but_quality_rf_runtime.py', 'raw_quality_features.py'):
            key = 'scripts/' + name
            expected = self.manifest.get('sha256', {}).get(key)
            if expected and sha256(Path(__file__).parent / name) != expected:
                raise ValueError('Feature implementation differs from release: ' + name)
        self.model_hash = sha256(folder / 'quality_rf.joblib')

    def predict(self, raw, fs):
        fs = sampling_rate(fs)
        x = np.asarray(raw, dtype=float)
        if x.ndim != 1 or len(x) != 10 * fs:
            raise ValueError('Exactly 10 seconds of one raw PPG channel required')
        if not np.isfinite(x).all() or np.std(x) < 1e-8:
            return dict(status='invalid_input', reason='nonfinite_or_flat', quality_score=None,
                        predicted_quality=None, threshold=self.threshold)
        features = extract_features(x, fs)
        if self.extended:
            features.update(raw_artifact_features(x, fs))
        frame = pd.DataFrame([features], columns=self.bundle['feature_cols'])
        score = float(self.bundle['model'].predict_proba(frame)[0, 1])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid model score')
        label = int(score >= self.threshold)
        return dict(status='good' if label else 'bad', reason='', quality_score=score,
                    predicted_quality=label, threshold=self.threshold)
