"""Inspect verified wrist v3 files without loading continuous ECG into RAM."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from download_longterm_af import ROOT, checksum

def audit(folder, subjects):
    manifest=json.loads((ROOT/'data/longterm_af_v3_manifest.json').read_text())
    files={f['path']:f for f in manifest['files']}
    result={'source':manifest['source'],'subjects':[],
            'scope':'File integrity and structure only; not aligned/labeled 30-second training windows.'}
    for person in subjects:
        info={'subject':person}
        for kind in ('PPG','ECG'):
            name=f'{person}_{kind}.mat'; path=folder/name
            if checksum(path,'md5') != files[name]['md5']: raise ValueError('Checksum mismatch: '+name)
            with h5py.File(path,'r') as f:
                info[kind+'_keys']=[key for key in f if not key.startswith('#')]
                if kind=='PPG':
                    signals=f['PPG_GREEN']
                    if h5py.check_dtype(ref=signals.dtype) is None: raise ValueError('Unexpected PPG storage')
                    arrays=[f[ref] for ref in signals[()].ravel()]
                    info['ppg_segment_samples']=[int(a.size) for a in arrays]
                    info['ppg_hours_at_documented_100hz']=sum(a.size for a in arrays)/100/3600
                    info['ppg_nonfinite_samples']=sum(int((~np.isfinite(a[:,start:start+1000000])).sum()) for a in arrays for start in range(0,a.shape[1],1000000))
                else:
                    labels=f['AF_annotation']; counts={}
                    for start in range(0,labels.shape[1],100000):
                        values,n=np.unique(labels[:,start:start+100000],return_counts=True)
                        for value,count in zip(values,n): counts[str(float(value))]=counts.get(str(float(value)),0)+int(count)
                    info['annotation_value_counts']=counts
                    info['annotation_shape']=list(labels.shape)
                    info['qrs_shape']=list(f['QRSindex'].shape)
                    info['rr_shape']=list(f['rr'].shape)
                    if labels.size != f['rr'].size or f['QRSindex'].size != labels.size+1:
                        raise ValueError('Unexpected annotation/RR/QRS relationship')
                    info['rr_matches_qrs_difference_at_500hz']=bool(np.allclose(
                        np.diff(f['QRSindex'][()].ravel())/500, f['rr'][()].ravel(), atol=1e-8))
                    info['requires_label_codebook_confirmation']=True
                    info['label_mapping_applied']=False
        result['subjects'].append(info)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects',nargs='+',required=True)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/raw/longterm_af_v3')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value=audit(args.data_dir,args.subjects)
    args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(value,indent=2))
