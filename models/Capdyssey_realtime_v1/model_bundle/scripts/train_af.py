"""Small research baseline: subject-disjoint MIMIC PERform AF classification."""
import argparse
import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.stats import rankdata
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
FS = 50
SECONDS = 30

def preprocess_window(raw, sampling_hz=125, seconds=SECONDS):
    """Shared training/inference contract; legacy default is 30 seconds."""
    from fractions import Fraction
    raw = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(sampling_hz) or sampling_hz < 25 or sampling_hz > 2000:
        raise ValueError('sampling_hz must be between 25 and 2000')
    if seconds not in (10, 30):
        raise ValueError('Only 10 or 30 second model inputs are supported')
    expected = sampling_hz * seconds
    if raw.ndim != 1 or abs(len(raw) - expected) > 1e-6:
        raise ValueError(f'Input must be a 1D array containing exactly {seconds} seconds')
    if not np.isfinite(raw).all() or np.std(raw) < 1e-8:
        raise ValueError('Missing, nonfinite, or flat PPG: cannot classify')
    ratio = Fraction(str(FS / sampling_hz)).limit_denominator(10000)
    window = resample_poly(raw, ratio.numerator, ratio.denominator)
    scale = np.std(window)
    if len(window) != FS * seconds or not np.isfinite(window).all() or scale < 1e-8:
        raise ValueError('Invalid resampled PPG: cannot classify')
    return ((window - window.mean()) / scale).astype(np.float32)

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def load_records(folder):
    records = []
    sources = json.loads((ROOT / 'data/catalog.json').read_text(encoding='utf-8'))['mimic_af']['files']
    for item in sources:
        path = folder / item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Wrong file checksum: {path}')
        data = loadmat(path, simplify_cells=True)['data']
        for row in data:
            label = int(row['fix']['af_status'])
            if label not in (0, 1) or int(row['ppg']['fs']) != 125:
                raise ValueError('Unexpected label or sample rate')
            records.append((str(row['fix']['subj_id']), label, np.asarray(row['ppg']['v'])))
    ids = [r[0] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate subjects: merge records before splitting')
    return records

def split_subjects(records, seed):
    rng = np.random.default_rng(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for label in (0, 1):
        ids = sorted(r[0] for r in records if r[1] == label)
        if len(ids) < 5:
            raise ValueError('Need at least five subjects per class')
        rng.shuffle(ids)
        n = max(1, round(len(ids) * 0.2))
        splits['test'].extend(ids[:n])
        splits['val'].extend(ids[n:2*n])
        splits['train'].extend(ids[2*n:])
    assert not (set(splits['train']) & set(splits['val']) or set(splits['train']) & set(splits['test']) or set(splits['val']) & set(splits['test']))
    return splits

def prepare(records, seconds=SECONDS):
    x, y, subjects, starts = [], [], [], []
    audit = []
    for person, label, values in records:
        kept = 0
        total = len(values) // (125 * seconds)
        for i in range(total):
            raw = values[i*125*seconds:(i+1)*125*seconds]
            try:
                window = preprocess_window(raw, seconds=seconds)
            except ValueError:
                continue
            x.append(window)
            y.append(label); subjects.append(person); starts.append(i*seconds); kept += 1
        audit.append({'subject': person, 'label': label, 'total_windows': total, 'kept_windows': kept, 'excluded_windows': total-kept})
    return np.stack(x), np.array(y), np.array(subjects), np.array(starts), audit

def make_model():
    return nn.Sequential(
        nn.Conv1d(1, 16, 9, padding=4), nn.ReLU(), nn.MaxPool1d(4),
        nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
        nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 1))

def metrics(y, probabilities):
    predicted = probabilities >= 0.5
    tp = int(np.sum((y == 1) & predicted)); fn = int(np.sum((y == 1) & ~predicted))
    tn = int(np.sum((y == 0) & ~predicted)); fp = int(np.sum((y == 0) & predicted))
    sensitivity = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    pos, neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    auc = float((rankdata(probabilities)[y == 1].sum() - pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
    return {'threshold': 0.5, 'n': len(y), 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'accuracy': (tp+tn)/len(y), 'sensitivity': sensitivity, 'specificity': specificity,
            'false_positive_rate': fp/(fp+tn) if fp+tn else None,
            'balanced_accuracy': (sensitivity+specificity)/2 if pos and neg else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0, 'roc_auc': auc}

def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits.append(model(xb.to(device)).flatten().cpu())
            labels.append(yb)
    logits, labels = torch.cat(logits), torch.cat(labels)
    return labels.numpy().astype(int), torch.sigmoid(logits).numpy(), float(nn.functional.binary_cross_entropy_with_logits(logits, labels))

def train(data_dir, output, epochs=20, seed=42, batch_size=64, device_name='auto', seconds=SECONDS):
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    if seconds not in (10, 30):
        raise ValueError('Only 10 or 30 second inputs are supported')
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else device_name)
    records = load_records(data_dir)
    splits = split_subjects(records, seed)
    x, y, persons, starts, audit = prepare(records, seconds)
    write_json(output/'split.json', splits); write_json(output/'data_audit.json', audit)
    try: commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = 'unknown'
    config = {'seed': seed, 'epochs': epochs, 'batch_size': batch_size, 'device': str(device), 'sampling_hz': FS,
              'window_seconds': seconds, 'normalization': 'per-window mean/std', 'resampling': '125 to 50 Hz, scipy resample_poly 2/5',
              'quality_filter': 'nonfinite and near-zero variance only; NOT clinical signal quality',
              'model': 'small_cnn_v1' if seconds == 30 else 'small_cnn_10s_v1', 'threshold': 0.5, 'git_commit': commit,
              'datasets_used': ['mimic_perform_af'], 'three_dataset_integrated': False,
              'training_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': str(torch.__version__),
              'source': 'https://zenodo.org/records/15906524',
              'label_scope': 'AF status supplied per source record; not newly annotated window-level episodes'}
    write_json(output/'config.json', config)
    loaders, masks = {}, {}
    for name, ids in splits.items():
        mask = np.isin(persons, ids); masks[name] = mask
        if len(np.unique(y[mask])) != 2:
            raise ValueError(f'{name} lost a class after filtering')
        dataset = TensorDataset(torch.from_numpy(x[mask, None]), torch.from_numpy(y[mask].astype(np.float32)))
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=name=='train', num_workers=0)
    model = make_model().to(device)
    train_y = y[masks['train']]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(train_y==0).sum()/(train_y==1).sum()], device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, stale, history = float('inf'), 0, []
    for epoch in range(1, epochs+1):
        model.train(); total = 0.0
        for xb, yb in loaders['train']:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device)).flatten(), yb.to(device))
            loss.backward(); optimizer.step(); total += loss.item()*len(yb)
        vy, vp, val_loss = predict(model, loaders['val'], device)
        history.append({'epoch': epoch, 'train_loss': total/len(train_y), 'val_loss': val_loss, 'val_metrics': metrics(vy, vp)})
        if val_loss < best:
            best = val_loss; stale = 0
            torch.save({'state_dict': {k: v.cpu() for k,v in model.state_dict().items()}, 'config': config, 'epoch': epoch}, output/'best_model.pt')
        else: stale += 1
        write_json(output/'history.json', history)
        print(f'epoch {epoch}: train_loss={history[-1]["train_loss"]:.4f}, val_loss={val_loss:.4f}', flush=True)
        if stale >= 5: break
    checkpoint = torch.load(output/'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    vy, vp, _ = predict(model, loaders['val'], device)
    ty, tp, _ = predict(model, loaders['test'], device)
    test_people = persons[masks['test']]
    ids = np.unique(test_people)
    sy = np.array([ty[test_people==p][0] for p in ids])
    sp = np.array([tp[test_people==p].mean() for p in ids])
    result = {'best_epoch': checkpoint['epoch'], 'validation_window_metrics': metrics(vy,vp),
              'selection': 'Lowest validation BCE; fixed threshold 0.5; no test-based tuning',
              'test_window_metrics': metrics(ty,tp), 'test_subject_mean_score_metrics': metrics(sy,sp),
              'warning': 'Small 35-subject bedside dataset; held-out subjects, not external validation. No IMU or clinical diagnosis validation.'}
    write_json(output/'metrics.json', result)
    with (output/'test_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['subject','start_seconds','label','af_score'])
        writer.writerows(zip(test_people, starts[masks['test']], ty, tp))
    print(json.dumps(result, indent=2), flush=True)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/raw/mimic_af')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--seconds', type=int, choices=[10,30], default=30)
    args=parser.parse_args()
    train(args.data_dir,args.output,args.epochs,args.seed,args.batch_size,args.device,args.seconds)
