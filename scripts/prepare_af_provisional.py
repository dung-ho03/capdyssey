"""Explicit exploratory subset; never modify the original unlabeled bundle."""
import argparse
import json
from pathlib import Path
import numpy as np


def strict_labels(qrs, annotations, starts, seconds=10):
    """Require homogeneous exact binary labels, with one adjacent beat guard.

    qrs is absolute seconds. The extra beat on each side also rejects windows
    whose assignment changes under a one-interval indexing shift. This is a
    sensitivity guard, NOT confirmation of the source's interval convention.
    """
    qrs=np.asarray(qrs); a=np.asarray(annotations); starts=np.asarray(starts)
    if len(qrs)!=len(a)+1 or not np.isfinite(qrs).all() or not (np.diff(qrs)>0).all():
        raise ValueError('Invalid QRS/annotation dimensions or ordering')
    out=np.full(len(starts),-1,dtype=np.int8)
    left=np.searchsorted(qrs,starts-.002,side='right')-2
    right=np.searchsorted(qrs,starts+seconds+.002,side='left')+1
    valid=np.isfinite(starts)&(left>=0)&(right<=len(a))
    for label in (0,1):
        bad=np.r_[0,np.cumsum(a!=label)]
        ii=np.flatnonzero(valid)
        match=bad[right[ii]]==bad[left[ii]]
        out[ii[match]]=label
    return out


def prepare(bundle,wrist,output,cap=2000):
    import google.colab  # Real data work stays in Colab.
    import h5py
    from prepare_af_integrated import chars, start_time, digest, atomic_json
    if cap<1: raise ValueError('Positive per-subject/class cap required')
    m=json.loads((bundle/'manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    result={k:v for k,v in m.items() if k not in ('shards','readiness')}
    result.update(shards=[],provisional=True,three_dataset_training_ready=False,
        protocol='exact_binary_guarded_v1',source_manifest_sha256=digest(bundle/'manifest.json'),
        assumptions={'wrist_0':'provisional non-AF','wrist_1':'provisional AF',
        'fractional_codes':'excluded; no rhythm name inferred',
        'interval':'annotation[i] on QRS[i]..QRS[i+1]; adjacent-beat and 2ms guards',
        'cap_per_subject_per_class':cap,'seed':42})
    audit=[]; rng=np.random.default_rng(42)
    for subject in sorted({s['subject'] for s in m['shards']}):
        rows=[s for s in m['shards'] if s['subject']==subject]
        source=rows[0]['source']; ecg=None
        if source=='longterm_wrist_v3':
            with h5py.File(wrist/(subject.split(':')[1]+'_ECG.mat')) as f:
                q=f['QRSindex'][()].ravel().astype(float)
                a=f['AF_annotation'][()].ravel().astype(float)
                begin=start_time(chars(f['recording_startday']),chars(f['recording_starttime']))
                if begin is None or np.any(q<1) or np.any(q>f['ECG'].size):
                    audit.append({'subject':subject,'excluded':'QRS bounds or ECG timestamp'});continue
                ecg=(begin+(q-1)/500,a)
        candidates=[]; counts={0:0,1:0}; total=0
        for row in rows:
            path=bundle/row['path']
            if path.resolve().parent!=bundle.resolve() or digest(path)!=row['sha256']:
                raise ValueError('Shard integrity failure: '+row['path'])
            with np.load(path,allow_pickle=False) as d:
                y=strict_labels(*ecg,d['start_seconds']) if ecg else d['y'].copy()
            if ecg and not row['provenance'].get('alignment_ready',False):y[:]=-1
            total+=len(y)
            for label in (0,1):
                indexes=np.flatnonzero(y==label); counts[label]+=len(indexes)
                # Independent random priorities give uniform subject/class subsampling.
                candidates.extend((float(key),row['path'],int(index),label)
                                  for key,index in zip(rng.random(len(indexes)),indexes))
        chosen=[]
        for label in (0,1):
            chosen.extend(sorted((v for v in candidates if v[3]==label))[:cap])
        for row in rows:
            selected=[v for v in chosen if v[1]==row['path']]
            if not selected:continue
            selected.sort(key=lambda v:v[2]);idx=np.array([v[2] for v in selected]);y=np.array([v[3] for v in selected],dtype=np.int8)
            with np.load(bundle/row['path'],allow_pickle=False) as d:
                np.savez_compressed(output/row['path'],x=d['x'][idx],y=y,start_seconds=d['start_seconds'][idx])
            entry=dict(row);entry.update(n=len(y),sha256=digest(output/row['path']),
                label_counts={str(k):int((y==k).sum()) for k in (-1,0,1)},
                provenance={'original_shard_sha256':row['sha256'],'original_window_indices':idx.tolist(),
                            'label_basis':'provisional exact binary' if ecg else 'original source mapping'})
            result['shards'].append(entry)
        audit.append({'subject':subject,'prepared':total,'eligible':counts,'selected':len(chosen)})
        atomic_json(output/'audit.json',audit)
        print(subject,'eligible',counts,'selected',len(chosen),flush=True)
    result['sources']=sorted({r['source'] for r in result['shards']})
    atomic_json(output/'manifest.json',result)
    print('PROVISIONAL_BUNDLE_COMPLETE',len(result['shards']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','wrist','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cap',type=int,default=2000)
    args=p.parse_args();prepare(args.bundle,args.wrist,args.output,args.cap)
