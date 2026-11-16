"""New reproducible BUT comparison: same subjects and threshold objective for each model.

This is a NEW training protocol, not reconstruction of undocumented historical v2 training.
The existing published weights and thresholds are never overwritten.
"""
import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import wfdb
from but_quality_rf_runtime import (CONFIG, FEATURES, build_but_features, read_but_red,
    grouped_split, new_model, choose_threshold, metrics)
from quality_only import sha256
from raw_quality_features import EXTRA, FEATURE_SPEC, raw_artifact_features


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def save_rf(model, cols, threshold, out, identity):
    out.mkdir()
    bundle = dict(model=model, feature_cols=cols, config=CONFIG, threshold=threshold,
                  extra_feature_spec=FEATURE_SPEC if cols == FEATURES + EXTRA else None,
                  threshold_source='validation subjects only; maximum balanced accuracy',
                  label_meaning={0: 'bad quality for HR', 1: 'good quality for HR'})
    joblib.dump(bundle, out / 'quality_rf.joblib')
    hashes = {'quality_rf.joblib': sha256(out / 'quality_rf.joblib')}
    for name in ['but_quality_rf_runtime.py', 'raw_quality_features.py']:
        hashes['scripts/' + name] = sha256(Path(__file__).parent / name)
    write_json(out / 'manifest.json', dict(format='capdyssey-quality-only-v1',
        quality_threshold=threshold, sha256=hashes, training_identity=identity,
        sensor_transfer_validated=False, channel='BUT PPG_R; target sensor channel requires validation',
        scope='New BUT training run; not identical to historical v2 or wrist validation'))


def fit_rf_comparison(frame, split, out, identity):
    """Fit preprocessing only on train, choose threshold only on validation."""
    reports = {}
    for name, cols in [('rf_baseline', FEATURES), ('rf_extended', FEATURES + EXTRA)]:
        train, val, test = [frame.iloc[split[s]] for s in ['train', 'validation', 'test']]
        model = new_model().fit(train[cols], train.quality)
        val_score = model.predict_proba(val[cols])[:, 1]
        threshold = choose_threshold(val.quality, val_score)
        # Save the operating point before opening test features for inference.
        folder = out / name
        save_rf(model, cols, threshold, folder, identity)
        score = model.predict_proba(test[cols])[:, 1]
        reports[name] = dict(validation=metrics(val.quality, val_score, threshold),
                             test=metrics(test.quality, score, threshold))
        predictions = test[['record_id', 'subject_id', 'quality']].copy()
        predictions['quality_score'] = score
        predictions['predicted_quality'] = (score >= threshold).astype(int)
        predictions.to_csv(folder / 'test_predictions.csv', index=False)
        write_json(folder / 'metrics.json', reports[name])
    return reports


def fit_cnn(raw, frame, split, out, identity, epochs, device_name):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import average_precision_score
    from but_quality_cnn_runtime import PPGQualityCNN
    from quality_cnn_only import preprocess
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    device = torch.device(device_name)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x = np.stack([preprocess(v, 30) for v in raw])[:, None, :]
    y = frame.quality.to_numpy(dtype=np.float32)
    tr, va, te = [split[k] for k in ['train', 'validation', 'test']]
    model = PPGQualityCNN().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[tr]), torch.from_numpy(y[tr])),
                        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
    weight = torch.tensor([(y[tr] == 0).sum() / (y[tr] == 1).sum()], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    def scores(ids):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(torch.from_numpy(x[chunk]).to(device)).sigmoid().cpu().numpy()
                                   for chunk in np.array_split(ids, max(1, int(np.ceil(len(ids)/256))))])
    best, stale, history, state = -1., 0, [], None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward(); optimizer.step()
        ap = float(average_precision_score(y[va], scores(va)))
        history.append(dict(epoch=epoch, validation_auprc=ap))
        if ap > best:
            best, stale = ap, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print('CNN', epoch, 'validation AUPRC', ap, flush=True)
        if stale >= 10:
            break
    model.load_state_dict(state)
    vp = scores(va)
    threshold = choose_threshold(y[va], vp)
    folder = out / 'cnn'
    folder.mkdir()
    torch.save(state, folder / 'quality_cnn.pt')
    write_json(folder / 'manifest.json', dict(format='capdyssey-quality-cnn-v1',
        quality_threshold=threshold, sha256={'quality_cnn.pt': sha256(folder / 'quality_cnn.pt'),
          'scripts/quality_cnn_only.py': sha256(Path(__file__).parent/'quality_cnn_only.py'),
          'scripts/but_quality_cnn_runtime.py': sha256(Path(__file__).parent/'but_quality_cnn_runtime.py')},
        training_identity=identity, preprocessing='mean-remove/resample30/flat1e-3/bandpass0.5-5/zscore',
        score='uncalibrated sigmoid; not a calibrated probability', sensor_transfer_validated=False,
        torch=str(torch.__version__), device=str(device)))
    tp = scores(te)
    result = dict(validation=metrics(y[va], vp, threshold), test=metrics(y[te], tp, threshold))
    pred = frame.iloc[te][['record_id', 'subject_id', 'quality']].copy()
    pred['quality_score'] = tp
    pred['predicted_quality'] = (tp >= threshold).astype(int)
    pred.to_csv(folder / 'test_predictions.csv', index=False)
    write_json(folder / 'metrics.json', result)
    write_json(folder / 'history.json', history)
    return result


def run(raw_dir, out, include_cnn=False, epochs=60, device='cpu'):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    out.mkdir(parents=True, exist_ok=False)
    frame, audit = build_but_features(raw_dir, out / 'feature_cache')
    raw = []
    extra = []
    for rid in frame.record_id:
        rec = wfdb.rdrecord(str(raw_dir / f'{rid}/{rid}_PPG'))
        signal, _ = read_but_red(rec)
        raw.append(signal)
        extra.append(raw_artifact_features(signal, rec.fs))
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    split = grouped_split(frame, seed=42)
    split_table = frame[['record_id', 'subject_id', 'quality']].copy()
    split_table['split'] = ''
    for key, ids in split.items():
        split_table.loc[ids, 'split'] = key
    split_table.to_csv(out / 'split.csv', index=False)
    frame.to_csv(out / 'features.csv', index=False)
    identity = dict(protocol='quality-common-holdout-v1', seed=42, data=audit,
        split_sha256=sha256(out/'split.csv'), features_sha256=sha256(out/'features.csv'),
        code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*quality*.py')},
        threshold_objective='validation balanced accuracy for every model',
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__, joblib=joblib.__version__, wfdb=wfdb.__version__,
        note='Shared subjects and reader for all models. New experiment, not historical CNN/v2 reproduction. No test-based model selection.')
    write_json(out/'protocol.json', identity)
    reports = fit_rf_comparison(frame, split, out, identity)
    if include_cnn:
        reports['cnn'] = fit_cnn(raw, frame, split, out, identity, epochs, device)
    write_json(out/'comparison.json', reports)
    print('Saved new experiment to', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--include-cnn', action='store_true', help='Requires torch (optional)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = p.parse_args()
    run(a.raw_dir, a.out, a.include_cnn, a.epochs, a.device)
