"""Classify raw PPG using the duration contract in a trusted checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_af import make_model, preprocess_window, write_json

def classify(checkpoint_path, raw, sampling_hz):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    durations={'small_cnn_v1':30, 'small_cnn_10s_v1':10}
    if config['model'] not in durations or config['sampling_hz'] != 50 or config['window_seconds'] != durations[config['model']]:
        raise ValueError('Unsupported model/preprocessing version')
    window = preprocess_window(raw, sampling_hz, seconds=config['window_seconds'])
    model = make_model()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(torch.from_numpy(window)[None, None])).item()
    return {'prediction': 'AF' if score >= config['threshold'] else 'non-AF',
            'af_score': score, 'threshold': config['threshold'],
            'input_sampling_hz': sampling_hz, 'window_seconds': config['window_seconds'],
            'warning': 'Research only. Score is not a calibrated disease probability. Non-AF does not mean healthy. No signal-quality or wrist-device validation.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Raw 1D .npy; duration must match checkpoint (10 or 30 seconds)')
    parser.add_argument('--sampling-hz', type=float, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = classify(args.checkpoint, np.load(args.input, allow_pickle=False), args.sampling_hz)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
