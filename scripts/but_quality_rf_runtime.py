# Extracted unchanged from PPG_quality_RF_improved.ipynb (but-rf-quality-v2.2).
"""Shared training/inference functions for the corrected BUT Random Forest notebook."""
from pathlib import Path
import hashlib
import json
import math
from fractions import Fraction

import joblib
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt, find_peaks, welch, resample_poly
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score)

PIPELINE_VERSION = 'but-rf-quality-v2.2'
CONFIG = {'version': PIPELINE_VERSION, 'fs': 30, 'seconds': 10,
          'bandpass': [0.5, 5.0], 'filter_order': 3,
          'max_missing_fraction': 0.1, 'max_gap_seconds': 0.2,
          'normalization': 'filtered median/std', 'channel': 'PPG_R'}
FEATURES = ['finite_ratio', 'raw_std', 'raw_range', 'filtered_std',
            'difference_std', 'peak_count', 'peak_rate_bpm', 'interval_cv',
            'prominence_cv', 'pulse_power_ratio']


def read_but_red(record):
    values = np.asarray(record.p_signal, dtype=float)
    if not np.isclose(record.fs, 30):
        raise ValueError(f'BUT fs must be 30 Hz, got {record.fs}')
    if values.shape == (1, 300):
        return values.reshape(300), 'legacy_1x300'
    # Some wfdb versions parse the dotted "a.u." unit as part of sig_name.
    # The exact final token remains PPG_R/G/B; gains/baselines are still parsed.
    names = [str(name).split()[-1] if str(name).split() else ''
             for name in (record.sig_name or [])]
    if values.ndim == 2 and values.shape[0] == 300 and names.count('PPG_R') == 1:
        return values[:, names.index('PPG_R')], 'PPG_R'
    raise ValueError(f'Unsupported BUT shape/channel: {values.shape}, {names[:3]}')


def prepare_window(signal, fs):
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError('Select exactly one PPG channel; input must be 1-D.')
    fs = float(fs)
    if not np.isfinite(fs) or fs < 12:
        raise ValueError('fs must be finite and >=12 Hz for the fixed 0.5–5 Hz filter.')
    n = fs * CONFIG['seconds']
    if not np.isclose(n, round(n)) or len(x) != round(n):
        raise ValueError(f'Exactly 10 seconds required: expected {n} samples, got {len(x)}.')
    valid = np.isfinite(x)
    finite_ratio = float(valid.mean())
    if finite_ratio < 1 - CONFIG['max_missing_fraction']:
        raise ValueError('More than 10% of samples are missing/nonfinite.')
    if not valid[0] or not valid[-1]:
        raise ValueError('Missing endpoint samples; extrapolation is not permitted.')
    padded = np.r_[False, ~valid, False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    if len(starts) and (ends - starts).max() / fs > CONFIG['max_gap_seconds']:
        raise ValueError('Missing-data gap exceeds 0.2 seconds.')
    if not valid.all():
        idx = np.arange(len(x))
        x = np.interp(idx, idx[valid], x[valid])
    if fs != CONFIG['fs']:
        ratio = Fraction(CONFIG['fs'] / fs).limit_denominator(10000)
        if not np.isclose(float(ratio), CONFIG['fs'] / fs, rtol=1e-8):
            raise ValueError('Sampling-rate ratio cannot be represented accurately.')
        dc = x.mean()
        x = resample_poly(x-dc, ratio.numerator, ratio.denominator, padtype='line') + dc
    if x.shape != (300,) or not np.isfinite(x).all():
        raise ValueError('Resampling must produce 300 finite samples.')
    return x, finite_ratio


def extract_features(signal, fs):
    x, finite_ratio = prepare_window(signal, fs)
    fs = CONFIG['fs']
    sos = butter(CONFIG['filter_order'], CONFIG['bandpass'], btype='bandpass', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, x - x.mean())
    z = (filtered - np.median(filtered)) / (np.std(filtered) + 1e-8)
    peaks, props = find_peaks(z, distance=max(1, int(fs*60/180)), prominence=.35)
    intervals = np.diff(peaks) / fs
    interval_cv = float(intervals.std()/(intervals.mean()+1e-8)) if len(peaks) >= 3 else np.nan
    prominence = props['prominences']
    prominence_cv = float(prominence.std()/(prominence.mean()+1e-8)) if len(peaks) >= 3 else np.nan
    freqs, power = welch(z, fs=fs, nperseg=240)
    def band(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(power[mask], freqs[mask])) if mask.sum() >= 2 else 0.
    return dict(finite_ratio=finite_ratio, raw_std=float(x.std()), raw_range=float(np.ptp(x)),
        filtered_std=float(filtered.std()), difference_std=float(np.diff(z).std()),
        peak_count=int(len(peaks)), peak_rate_bpm=float(len(peaks)*6),
        interval_cv=interval_cv, prominence_cv=prominence_cv,
        pulse_power_ratio=band(.7, 3)/(band(.5, 5)+1e-8))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_but_features(root, cache_dir):
    root, cache_dir = Path(root), Path(cache_dir)
    labels = pd.read_csv(root/'quality-hr-ann.csv', dtype={'ID': str})
    if list(labels.columns) != ['ID','Quality','HR']:
        raise ValueError('Expected BUT CSV columns ID,Quality,HR.')
    if (len(labels) != 3888 or labels.ID.duplicated().any()
        or not labels.ID.str.fullmatch(r'\d{6}').all() or not labels.Quality.isin([0,1]).all()):
        raise ValueError('Expected 3,888 unique six-digit IDs and binary labels.')
    paths = [root/f'{rid}/{rid}_PPG.{ext}' for rid in labels.ID for ext in ['hea','dat']]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing waveform files: {missing[:5]}')
    fingerprint = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode())
    for path in [root/'quality-hr-ann.csv', *paths]:
        fingerprint.update(str(path.relative_to(root)).encode())
        fingerprint.update(file_sha(path).encode())
    key = fingerprint.hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir/f'{PIPELINE_VERSION}_{key[:16]}.csv'
    audit_path = cache.with_suffix('.json')
    if cache.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get('cache_sha256') == file_sha(cache) and audit.get('fingerprint') == key:
            print('Verified feature cache:', cache.name)
            return pd.read_csv(cache, dtype={'record_id':str,'subject_id':str}), audit
    rows, errors, layouts = [], [], {}
    for i, row in enumerate(labels.itertuples(index=False), 1):
        try:
            rec = wfdb.rdrecord(str(root/f'{row.ID}/{row.ID}_PPG'))
            signal, layout = read_but_red(rec)
            features = extract_features(signal, rec.fs)
            rows.append(dict(record_id=row.ID, subject_id=row.ID[:3], quality=int(row.Quality), **features))
            layouts[layout] = layouts.get(layout, 0) + 1
        except (ValueError, OSError, IndexError) as exc:
            errors.append({'record_id':row.ID, 'reason':str(exc)})
        if i % 500 == 0 or i == len(labels):
            print(f'{i}/{len(labels)} records; accepted={len(rows)}, excluded={len(errors)}', flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f'No valid BUT records: {errors[:3]}')
    frame.to_csv(cache, index=False)
    audit = dict(fingerprint=key, cache_sha256=file_sha(cache), config=CONFIG,
        total=3888, accepted=len(rows), subjects=int(frame.subject_id.nunique()), layouts=layouts, excluded=errors)
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return frame, audit


def grouped_split(frame, seed=42):
    groups = frame.subject_id.to_numpy()
    tr, rest = next(GroupShuffleSplit(n_splits=1, test_size=.30, random_state=seed).split(frame, groups=groups))
    va, te = next(GroupShuffleSplit(n_splits=1, test_size=.50, random_state=seed+1).split(frame.iloc[rest], groups=groups[rest]))
    result = {'train':tr, 'validation':rest[va], 'test':rest[te]}
    for name, ids in result.items():
        if frame.iloc[ids].quality.nunique() != 2:
            raise ValueError(f'{name} lacks both classes. Review the predeclared split protocol.')
    for left, right in [('train','validation'),('train','test'),('validation','test')]:
        assert set(groups[result[left]]).isdisjoint(groups[result[right]])
    return result


def new_model(seed=42):
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
        RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                               class_weight='balanced_subsample', random_state=seed, n_jobs=-1))


def choose_threshold(y, score):
    y, score = np.asarray(y, int), np.asarray(score, float)
    if set(np.unique(y)) != {0,1} or not np.isfinite(score).all():
        raise ValueError('Threshold selection needs both classes and finite scores.')
    thresholds = np.unique(np.r_[0., .5, score, np.nextafter(score.max(), np.inf)])
    values = np.array([balanced_accuracy_score(y, score >= t) for t in thresholds])
    best = thresholds[np.isclose(values, values.max(), atol=1e-12, rtol=0)]
    return float(best[np.lexsort((best, np.abs(best-.5)))[0]])


def metrics(y, score, threshold):
    y, score = np.asarray(y, int), np.asarray(score, float)
    prediction = (score >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    return {'n':len(y), 'threshold':float(threshold),
        'balanced_accuracy':float(balanced_accuracy_score(y,prediction)) if both else None,
        'auroc':float(roc_auc_score(y,score)) if both else None,
        'auprc_good':float(average_precision_score(y,score)) if both else None,
        'confusion_matrix':confusion_matrix(y,prediction,labels=[0,1]).tolist(),
        'classes':classification_report(y,prediction,labels=[0,1],target_names=['bad','good'],
                                         output_dict=True,zero_division=0)}


def train_holdout(frame, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    split = grouped_split(frame)
    train, val, test = [frame.iloc[split[s]] for s in ['train','validation','test']]
    model = new_model().fit(train[FEATURES], train.quality)
    threshold = choose_threshold(val.quality, model.predict_proba(val[FEATURES])[:,1])
    probabilities = model.predict_proba(test[FEATURES])[:,1]
    bundle = dict(model=model, feature_cols=FEATURES, threshold=threshold, config=CONFIG,
        label_meaning={0:'bad quality for HR estimation',1:'good quality for HR estimation'},
        seed=42, threshold_source='validation subjects only')
    joblib.dump(bundle, output/'butppg_quality_rf.joblib')
    restored = load_bundle(output/'butppg_quality_rf.joblib')
    np.testing.assert_allclose(restored['model'].predict_proba(test[FEATURES])[:,1], probabilities)
    scores = {'model':metrics(test.quality, probabilities, threshold),
              'always_bad':metrics(test.quality, np.zeros(len(test)), .5)}
    (output/'metrics.json').write_text(json.dumps(scores,indent=2), encoding='utf-8')
    split_table = frame[['record_id','subject_id','quality']].copy()
    split_table['split'] = ''
    for name, ids in split.items():
        split_table.loc[split_table.index[ids], 'split'] = name
        part = frame.iloc[ids]
        print(name, 'records=',len(part),'subjects=',part.subject_id.nunique(),'labels=',part.quality.value_counts().to_dict())
    split_table.to_csv(output/'split.csv', index=False)
    predictions = test[['record_id','subject_id','quality']].copy()
    predictions['good_score'] = probabilities
    predictions['prediction'] = (probabilities >= threshold).astype(int)
    predictions.to_csv(output/'test_predictions.csv', index=False)
    return bundle, scores, predictions


def grouped_cv(frame):
    """Outer test never chooses a threshold. Descriptive 5-fold robustness check."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    reports, predictions = [], []
    for fold, (development, test) in enumerate(splitter.split(frame,frame.quality,frame.subject_id)):
        dev = frame.iloc[development]
        tr, va = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=100+fold).split(dev,groups=dev.subject_id))
        training, validation, testing = dev.iloc[tr], dev.iloc[va], frame.iloc[test]
        for part in [training, validation, testing]:
            if part.quality.nunique() != 2:
                raise ValueError(f'Fold {fold} has a single-class subset; no silent seed search.')
        model = new_model(42+fold).fit(training[FEATURES],training.quality)
        threshold = choose_threshold(validation.quality,model.predict_proba(validation[FEATURES])[:,1])
        score = model.predict_proba(testing[FEATURES])[:,1]
        report = metrics(testing.quality,score,threshold)
        reports.append(dict(fold=fold,**{k:report[k] for k in ['n','threshold','balanced_accuracy','auroc','auprc_good']}))
        pred = testing[['record_id','subject_id','quality']].copy()
        pred['fold'], pred['good_score'], pred['threshold'] = fold, score, threshold
        predictions.append(pred)
        print('CV fold',fold,reports[-1],flush=True)
    return pd.DataFrame(reports), pd.concat(predictions,ignore_index=True)


def load_bundle(path):
    bundle = joblib.load(path)  # Load only your own trusted model file.
    if bundle.get('config') != CONFIG or bundle.get('feature_cols') != FEATURES:
        raise ValueError('Old/incompatible model: retrain with this notebook before inference.')
    return bundle


def external_windows(signal, fs):
    fs = float(fs)
    if not np.isfinite(fs) or fs < 12 or not np.isclose(fs*10, round(fs*10)):
        raise ValueError('Provide the actual finite sampling rate >=12 Hz.')
    x = np.asarray(signal,float)
    if x.ndim != 1:
        raise ValueError('External PPG must be one channel.')
    width = int(round(fs*10))
    if len(x) < width:
        raise ValueError('Recording is shorter than 10 seconds; no padding is used.')
    return [(start/fs,x[start:start+width]) for start in range(0,len(x)-width+1,width)], (len(x)%width)/fs


def predict_external(bundle, info, root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if bundle.get('config') != CONFIG:
        raise ValueError('Incompatible preprocessing config.')
    rows, errors, audits = [], [], []
    for item in info:
        filename = item['filename']
        try:
            if Path(filename).name != filename:
                raise ValueError('Use a filename only, without directories.')
            label = item.get('quality')
            if label is not None and label not in [0,1]:
                raise ValueError('quality must be 0,1 or None.')
            fs = float(item['fs'])
            table = pd.read_csv(root/filename)
            column = item.get('column','ppg')
            if column not in table:
                raise ValueError(f'Missing {column!r} column: {list(table.columns)}')
            signal = pd.to_numeric(table[column],errors='coerce').to_numpy()
            windows, tail = external_windows(signal,fs)
            if label is not None and len(windows) > 1:
                raise ValueError('For labeled evaluation, supply one 10-second window per CSV; long files use quality=None.')
            audits.append(dict(filename=filename,fs=fs,windows=len(windows),discarded_tail_seconds=tail))
            for start, raw in windows:
                try:
                    features = pd.DataFrame([extract_features(raw,fs)], columns=FEATURES)
                    score = float(bundle['model'].predict_proba(features)[0,1])
                    rows.append(dict(filename=filename,start_seconds=start,quality=label,good_score=score,
                        predicted_quality=int(score>=bundle['threshold']),status='ok',reason=''))
                except ValueError as exc:
                    rows.append(dict(filename=filename,start_seconds=start,quality=label,good_score=np.nan,
                        predicted_quality=pd.NA,status='unusable_input',reason=str(exc)))
        except (OSError,ValueError,TypeError,KeyError,pd.errors.ParserError) as exc:
            errors.append(dict(filename=filename,reason=str(exc)))
    result = pd.DataFrame(rows,columns=['filename','start_seconds','quality','good_score','predicted_quality','status','reason'])
    result.to_csv(output/'external_predictions.csv',index=False)
    (output/'external_audit.json').write_text(json.dumps({'files':audits,'file_errors':errors,
        'unusable_windows':int((result.status!='ok').sum()),'threshold':bundle['threshold']},indent=2),encoding='utf-8')
    labeled = result[(result.status=='ok') & result.quality.notna()]
    if len(labeled):
        report = metrics(labeled.quality.astype(int),labeled.good_score,bundle['threshold'])
        report['scope'] = 'User-provided independent quality labels; valid windows only. See exclusions in audit.'
        (output/'external_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    else:
        print('No independent quality labels: predictions only; no accuracy/F1 calculated.')
    if errors:
        print('File errors:', errors)
    return result
