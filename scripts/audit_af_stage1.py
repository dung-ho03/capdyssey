"""Read-only inventory of locally acquired MIMIC and six-rhythm source files."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def audit(root):
    result={'scope':'Source labels/structure/integrity; not clinical quality or training validation',
            'mimic':[], 'six_rhythms':[], 'licenses':{}}
    for name in ('mimic_perform_af_data.mat','mimic_perform_non_af_data.mat'):
        path=root/'mimic_perform_af'/name
        d=loadmat(path,simplify_cells=True)
        result['licenses'][name]=str(d['license'])
        for row in d['data']:
            result['mimic'].append({'subject':str(row['fix']['subj_id']),
                'label':int(row['fix']['af_status']), 'fs':float(row['ppg']['fs']),
                'samples':int(np.asarray(row['ppg']['v']).size)})
    six=root/'six_rhythms'
    lists={name:set((six/(name+'subjects.txt')).read_text().replace("'",'').split()) for name in ('valid','test')}
    result['six_original_split_overlap']=sorted(lists['valid']&lists['test'])
    for path in sorted((six/'valid_testDataset').glob('*.mat')):
        d=loadmat(path); labels=np.asarray(d['labels']).ravel(); x=d['ppgseg']
        if x.shape!=(len(labels),1000) or not np.isin(labels,np.arange(6)).all():
            raise ValueError('Unexpected six-rhythm format: '+path.name)
        result['six_rhythms'].append({'subject':path.stem,'original_split':[s for s,ids in lists.items() if path.name in ids],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'shape':list(x.shape),'label_counts':{str(i):int((labels==i).sum()) for i in range(6)}})
    actual={s['subject']+'.mat' for s in result['six_rhythms']}
    result['six_missing_from_local']=sorted((lists['valid']|lists['test'])-actual)
    result['six_unlisted_local']=sorted(actual-(lists['valid']|lists['test']))
    result['six_counts']={str(i):sum(s['label_counts'][str(i)] for s in result['six_rhythms']) for i in range(6)}
    result['mimic_unique_subjects']=len({s['subject'] for s in result['mimic']})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    result=audit(args.source_root)
    args.output.write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'mimic_subjects':result['mimic_unique_subjects'],
        'six_subjects':len(result['six_rhythms']),'six_counts':result['six_counts'],
        'split_overlap':result['six_original_split_overlap'],
        'missing':result['six_missing_from_local'],'unlisted':result['six_unlisted_local']}))
