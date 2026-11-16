"""Audit timestamped wrist CSVs and evaluate independent window quality labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from but_quality_rf_runtime import metrics
from quality_only import QualityOnly, sampling_rate, sha256

REQUIRED = ['filename', 'subject_id', 'device_id', 'session_id', 'activity', 'channel', 'fs']


def audit_window(frame, fs, channel, previous=None, jitter_fraction=.2):
    """No sorting, interpolation, padding or removal of bad rows before audit.

    Include the preceding sample to catch a gap exactly at a window boundary.
    timestamp_s must be an unwrapped device acquisition clock, not USB arrival time.
    """
    fs = sampling_rate(fs)
    if not 0 <= jitter_fraction < 1:
        raise ValueError('jitter_fraction must be in [0,1)')
    if len(frame) != 10 * fs:
        return 'incomplete_window'
    if not {'sample_index', 'timestamp_s', channel}.issubset(frame.columns):
        return 'missing_required_columns'
    check = pd.concat([previous, frame]) if previous is not None else frame
    seq = pd.to_numeric(check.sample_index, errors='coerce').to_numpy(dtype=float)
    time = pd.to_numeric(check.timestamp_s, errors='coerce').to_numpy(dtype=float)
    raw = pd.to_numeric(frame[channel], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(seq).all() or not np.equal(seq, np.floor(seq)).all():
        return 'invalid_sample_index'
    if not np.isfinite(time).all():
        return 'invalid_timestamp'
    if not np.all(np.diff(seq) == 1):
        return 'sample_gap_duplicate_or_reset'
    delta = np.diff(time)
    if np.any(delta <= 0):
        return 'nonmonotonic_timestamp'
    if np.any(np.abs(delta - 1 / fs) > jitter_fraction / fs + 1e-9):
        return 'irregular_sampling'
    # Also reject sustained drift that stays inside the per-interval tolerance.
    span = float(frame.timestamp_s.iloc[-1]) - float(frame.timestamp_s.iloc[0])
    if abs(span - (len(frame) - 1) / fs) > 1 / fs + 1e-9:
        return 'window_duration_mismatch'
    if not np.isfinite(raw).all():
        return 'nonfinite_ppg'
    return ''


def quality_summary(frame, threshold):
    labeled = frame[frame.quality.notna()]
    valid = labeled[labeled.status.isin(['good', 'bad'])]
    report = dict(total_windows=len(frame), labeled_windows=len(labeled),
                  valid_labeled_windows=len(valid), invalid_labeled_windows=len(labeled)-len(valid),
                  invalid_windows=int((frame.status == 'invalid_input').sum()),
                  unknown_label_windows=int(frame.quality.isna().sum()))
    report['metrics_valid_labeled_only'] = None
    if len(valid):
        result = metrics(valid.quality.astype(int), valid.quality_score, threshold)
        cm = np.asarray(result['confusion_matrix'])
        result['bad_pass_rate'] = float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else None
        report['metrics_valid_labeled_only'] = result
    # Acquisition/invalid inputs are deferred, never counted as correct bad predictions.
    for label, name in [(0, 'bad'), (1, 'good')]:
        subset = labeled[labeled.quality == label]
        report[name + '_input_rejection_rate'] = (
            float((subset.status == 'invalid_input').mean()) if len(subset) else None)
    return report


def run(manifest_path, data_root, model_dir, out, labels_path=None, jitter_fraction=.2):
    meta = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if not set(REQUIRED).issubset(meta.columns) or meta.empty:
        raise ValueError('Manifest requires rows and columns: ' + ','.join(REQUIRED))
    if (meta[REQUIRED] == '').any().any() or meta.filename.duplicated().any():
        raise ValueError('Missing metadata or duplicate filename')
    reserved = {'window_index', 'sample_count', 'quality', 'status', 'reason',
                'quality_score', 'predicted_quality', 'threshold'}
    if reserved.intersection(meta.columns):
        raise ValueError('Manifest contains reserved prediction/label columns')
    labels = {}
    if labels_path:
        lab = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        if not {'filename', 'window_index', 'quality'}.issubset(lab.columns):
            raise ValueError('Labels require filename,window_index,quality')
        for r in lab.itertuples(index=False):
            if not r.window_index.isdigit() or r.quality not in ('', '0', '1'):
                raise ValueError('Labels: nonnegative window_index; quality=0,1 or empty (uncertain)')
            key = (r.filename, int(r.window_index))
            if key in labels:
                raise ValueError('Duplicate window label: ' + str(key))
            labels[key] = int(r.quality) if r.quality else None
    model_format = json.loads((Path(model_dir)/'manifest.json').read_text(encoding='utf-8'))['format']
    if model_format == 'capdyssey-quality-cnn-v1':
        from quality_cnn_only import QualityCNNOnly
        engine = QualityCNNOnly(model_dir)
    else:
        engine = QualityOnly(model_dir)
    root = Path(data_root).resolve()
    rows, files, failures, seen = [], [], [], set()
    for item in meta.to_dict('records'):
        filename = item['filename']
        try:
            path = (root / filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError('CSV path leaves data root')
            fs = sampling_rate(item['fs'])
            frame = pd.read_csv(path)
            width = 10 * fs
            files.append(dict(filename=filename, sha256=sha256(path), samples=len(frame)))
            if frame.empty:
                raise ValueError('Empty CSV')
            for index, start in enumerate(range(0, len(frame), width)):
                window = frame.iloc[start:start + width]
                previous = frame.iloc[start-1:start] if start else None
                reason = audit_window(window, fs, item['channel'], previous, jitter_fraction)
                result = dict(status='invalid_input', reason=reason, quality_score=None,
                              predicted_quality=None, threshold=engine.threshold)
                if not reason:
                    result = engine.predict(pd.to_numeric(window[item['channel']]).to_numpy(), fs)
                key = (filename, index)
                seen.add(key)
                rows.append(dict(**item, window_index=index, sample_count=len(window),
                                 quality=labels.get(key), **result))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            failures.append(dict(filename=filename, reason=str(exc)))
    columns = list(meta.columns) + ['window_index', 'sample_count', 'quality', 'status', 'reason',
                                   'quality_score', 'predicted_quality', 'threshold']
    result = pd.DataFrame(rows, columns=list(dict.fromkeys(columns)))
    report = quality_summary(result, engine.threshold)
    report.update(scope='Independent user quality labels; invalid inputs excluded from classifier metrics and reported separately.',
                  sensor_transfer_validated=False, threshold=engine.threshold,
                  model_sha256=engine.model_hash, manifest_sha256=sha256(manifest_path),
                  labels_sha256=sha256(labels_path) if labels_path else None,
                  jitter_fraction=jitter_fraction, files=files, file_errors=failures,
                  unmatched_labels=[list(k) for k in sorted(set(labels)-seen)])
    report['evaluation_complete'] = not failures and not report['unmatched_labels']
    report['by_group'] = {}
    for column in ['subject_id', 'activity', 'device_id']:
        report['by_group'][column] = {str(key): quality_summary(part, engine.threshold)
                                     for key, part in result.groupby(column)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    result.to_csv(out / 'predictions.csv', index=False)
    result[['filename', 'window_index']].assign(quality='').to_csv(out / 'label_template.csv', index=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--labels', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jitter-fraction', type=float, default=.2)
    a = p.parse_args()
    report = run(a.manifest, a.data_root, a.model, a.out, a.labels, a.jitter_fraction)
    print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'by_group')}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['evaluation_complete'] else 2)
