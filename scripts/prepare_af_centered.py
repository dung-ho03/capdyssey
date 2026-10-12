"""Rebuild exactly the previous selected windows, removing DC before resampling."""
import argparse
import json
from pathlib import Path
import numpy as np


def centered_window(raw,fs):
    from train_af import preprocess_window
    raw=np.asarray(raw,dtype=np.float64)
    return preprocess_window(raw-raw.mean(),fs,seconds=10)


def run(base):
    import google.colab
    import h5py
    from scipy.io import loadmat
    from train_af import load_records,preprocess_window
    from prepare_af_integrated import chars,start_time,digest,atomic_json
    old=base/'processed/provisional_exact01_v1';out=base/'processed/provisional_centered_v2'
    m=json.loads((old/'manifest.json').read_text())
    original=json.loads((base/'processed/integrated_10s_resume_v1/manifest.json').read_text())
    original_rows={r['path']:r for r in original['shards']}
    identity={'parent':digest(old/'manifest.json'),'code':digest(Path(__file__))}
    out.mkdir(exist_ok=True)
    checkpoint=out/'checkpoint.json'
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else {'identity':identity,'shards':[]}
    assert state['identity']==identity,'Changed inputs/code; choose a new bundle directory'
    done={r['path']:r for r in state['shards']}
    mimic={p:v for p,_,v in load_records(base/'raw/mimic_af')}
    for row in m['shards']:
        name=row['path']
        if name in done:
            assert digest(out/name)==done[name]['sha256'];continue
        assert digest(old/name)==row['sha256']
        with np.load(old/name) as d:oldx=d['x'];y=d['y'];starts=d['start_seconds']
        source=row['source'];selected=row['provenance']['original_window_indices'];parent=original_rows[name]
        if source=='longterm_wrist_v3':
            person,record=row['record'].split('_');record=int(record)
            with h5py.File(base/'raw/longterm_af_v3'/f'{person}_PPG.mat') as f:
                begin=start_time(chars(f[f['recording_startday'][()].ravel()[record]]),chars(f[f['recording_starttime'][()].ravel()[record]]))
                values=f[f['PPG_GREEN'][()].ravel()[record]][()].ravel()
            fs=100;offsets=np.rint((starts-begin)*fs).astype(int)
            raw_windows=(values[k:k+1000] for k in offsets)
        elif source=='six_rhythms':
            data=loadmat(base/'raw/six_rhythms'/(row['record']+'.mat'))
            indices=np.array(parent['provenance']['original_segment_indices'])[selected]
            fs=100;raw_windows=(data['ppgseg'][k] for k in indices)
        else:
            fs=125;values=mimic[row['record']];offsets=np.rint(starts*fs).astype(int)
            raw_windows=(values[k:k+1250] for k in offsets)
        corrected=[];maxerror=0.
        for i,raw in enumerate(raw_windows):
            reproduced=preprocess_window(raw,fs,seconds=10)
            error=float(np.max(np.abs(reproduced-oldx[i])));maxerror=max(maxerror,error)
            if error>1e-5:raise ValueError(f'Original window reconstruction mismatch: {name} {i} {error}')
            corrected.append(centered_window(raw,fs))
        with (out/(name+'.tmp')).open('wb') as f:
            np.savez_compressed(f,x=np.stack(corrected),y=y,start_seconds=starts)
        (out/(name+'.tmp')).replace(out/name)
        entry=dict(row);entry['sha256']=digest(out/name)
        entry['raw_reproduction_max_error']=maxerror
        state['shards'].append(entry);atomic_json(checkpoint,state)
        print('CENTERED',name,len(y),'original_reconstruction_error',maxerror,flush=True)
    result=dict(m);result['shards']=state['shards'];result['preprocessing']='remove mean BEFORE resample_poly; then z-score; 10s/50Hz'
    result['parent_bundle_sha256']=identity['parent'];result['preprocessing_script_sha256']=identity['code']
    assert sum(r['n'] for r in result['shards'])==sum(r['n'] for r in m['shards'])
    atomic_json(out/'manifest.json',result)
    print('CENTERED_BUNDLE_COMPLETE',sum(r['n'] for r in result['shards']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    run(p.parse_args().base)
