"""Audit all v3 ECG annotation arrays without assigning AF/non-AF meanings."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np

def audit(folder, output):
    results=[]
    for person in range(1,40):
        path=folder/f'{person:03d}_ECG.mat'
        with h5py.File(path) as f:
            a=np.asarray(f['AF_annotation'][()]).ravel()
            q=np.asarray(f['QRSindex'][()]).ravel()
            rr=np.asarray(f['rr'][()]).ravel()
            values,counts=np.unique(a,return_counts=True)
            lengths=len(a)==len(rr)==len(q)-1
            finite=bool(np.isfinite(q).all() and np.isfinite(rr).all() and np.isfinite(a).all())
            difference=float(np.max(np.abs(np.diff(q)/500-rr))) if lengths and finite and len(rr) else None
            row={'subject':f'{person:03d}','annotation_counts':{str(float(v)):int(n) for v,n in zip(values,counts)},
                'annotation_rr_qrs_lengths':[len(a),len(rr),len(q)],'lengths_match':lengths,
                'all_finite':finite,'qrs_strictly_increasing':bool(np.all(np.diff(q)>0)),
                'rr_vs_qrs_seconds_max_error':difference,
                'qrs_within_ecg_1_based':bool(len(q) and q.min()>=1 and q.max()<=f['ECG'].size),
                'annotation_attrs':{k:str(v) for k,v in f['AF_annotation'].attrs.items()}}
            results.append(row)
            print(json.dumps(row),flush=True)
    report={'source':'https://zenodo.org/records/23187681','participants':len(results),
            'label_mapping':None,'note':'Array agreement is structural evidence, not proof of numeric rhythm meanings or boundary convention.',
            'subjects':results}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('ANNOTATION_AUDIT_COMPLETE',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.raw,args.output)
