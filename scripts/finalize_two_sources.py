"""Freeze validation-selected PPG candidate before evaluating the existing test split."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from af_pulse_experiment import PulseModel, extract
from train_af_rhythm import rhythm_features, at_threshold
from train_af import write_json
from prepare_af_integrated import digest

ALLOWED={'mimic_perform_af','six_rhythms'}

class Predictor:
    """Input: centered, resampled and normalized 10 s PPG, shape (N,500)."""
    def __init__(self,checkpoint):
        if not torch.cuda.is_available():raise RuntimeError('GPU required')
        self.ck=torch.load(checkpoint,map_location='cuda',weights_only=True)
        self.model=PulseModel(self.ck['n_features']).cuda().eval()
        self.model.load_state_dict(self.ck['state_dict'])
        self.threshold=float(self.ck['threshold'])

    def predict(self,x):
        x=np.asarray(x)
        if x.ndim!=2 or x.shape[1]!=500 or not np.isfinite(x).all():raise ValueError('Expected finite (N,500) input')
        if np.any(x.std(1)<1e-8):raise ValueError('Flat PPG; request remeasurement')
        probabilities=[]
        with torch.no_grad():
            for start in range(0,len(x),1024):
                filtered,pulse=extract(x[start:start+1024])
                f=torch.cat([rhythm_features(torch.as_tensor(filtered,device='cuda'))[1],torch.as_tensor(pulse,device='cuda')],dim=1)
                f=((f-self.ck['mean'])/self.ck['std']).clamp(-10,10)
                probabilities.append(self.model(f).sigmoid().cpu().numpy())
        return np.concatenate(probabilities)

def run(base):
    import google.colab
    if not torch.cuda.is_available():raise RuntimeError('Colab GPU required')
    root=base/'models/main_two_sources_v1';out=root/'candidate';out.mkdir(exist_ok=True)
    comparison=json.loads((root/'comparison.json').read_text())
    assert set(comparison['sources'])==ALLOWED
    scores={}
    for key,r in comparison['results'].items():
        assert set(r['by_source'])==ALLOWED
        assert all(v['specificity']>=.8 for v in r['by_source'].values())
        scores[key]=float(np.mean([v['sensitivity'] for v in r['by_source'].values()]))
    seed=sorted(scores,key=lambda s:(-scores[s],int(s)))[0]
    checkpoint=root/f'seed{seed}'/'best_model.pt'
    bundle=base/'processed/provisional_centered_v2'
    splitfile=base/'models/provisional_centered_v2/split.json'
    split=json.loads(splitfile.read_text());manifest=json.loads((bundle/'manifest.json').read_text())
    assert all(not set(split[a])&set(split[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    frozen={'seed':int(seed),'threshold':comparison['results'][seed]['threshold'],'checkpoint_sha256':digest(checkpoint),
            'bundle_sha256':digest(bundle/'manifest.json'),'split_sha256':digest(splitfile),
            'selection':'Highest validation macro sensitivity with both source specificities >=80%; lower seed breaks tie.',
            'sources':sorted(ALLOWED),'seconds':10,'sampling_hz':50,
            'limitations':'Existing development test split, previously inspected in earlier experiments; NOT untouched external or clinical validation. No wrist generalization claim.'}
    freeze=out/'frozen_selection.json'
    if freeze.exists():assert json.loads(freeze.read_text())==frozen,'Frozen choice changed'
    else:write_json(freeze,frozen)
    # No test shards are opened before the selection above is saved.
    target=out/'model.pt'
    if target.exists():assert digest(target)==frozen['checkpoint_sha256']
    else:target.write_bytes(checkpoint.read_bytes())
    assert digest(target)==frozen['checkpoint_sha256']
    predictor=Predictor(target)
    assert predictor.ck['identity']==json.loads((root/'identity.json').read_text())
    assert predictor.ck['identity']['bundle']==frozen['bundle_sha256'] and predictor.ck['identity']['split']==frozen['split_sha256']
    assert predictor.threshold==frozen['threshold']
    yy=[];pp=[];ss=[];persons=[]
    for row in manifest['shards']:
        if row['source'] not in ALLOWED or row['subject'] not in split['test']:continue
        assert digest(bundle/row['path'])==row['sha256']
        with np.load(bundle/row['path']) as d:
            pred=predictor.predict(d['x']);label=d['y']
            assert len(pred)==row['n'] and np.isin(label,[0,1]).all()
            yy.append(label);pp.append(pred)
        ss.extend([row['source']]*row['n']);persons.extend([row['subject']]*row['n'])
    y=np.concatenate(yy);p=np.concatenate(pp);sources=np.asarray(ss);people=np.asarray(persons)
    assert set(sources.tolist())==ALLOWED and np.isfinite(p).all()
    result={'selection':frozen,'gpu':torch.cuda.get_device_name(0),'by_source':{},'per_subject':{}}
    for s in sorted(ALLOWED):
        mask=sources==s;assert set(y[mask].tolist())=={0,1}
        result['by_source'][s]=at_threshold(y[mask],p[mask],predictor.threshold)
    for person in sorted(set(persons)):
        mask=people==person;r=at_threshold(y[mask],p[mask],predictor.threshold)
        if not np.any(y[mask]==1):r['sensitivity']=None
        if not np.any(y[mask]==0):r['specificity']=None
        result['per_subject'][person]=r
    result['research_goal_met']=all(r['sensitivity']>=.9 and r['specificity']>=.8 for r in result['by_source'].values())
    write_json(out/'test_report.json',result)
    np.savez_compressed(out/'test_predictions.npz',y=y,p=p,source=sources,subject=people)
    print('TWO_SOURCE_CANDIDATE',json.dumps({'seed':seed,'threshold':predictor.threshold,'by_source':result['by_source'],'research_goal_met':result['research_goal_met']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);run(parser.parse_args().base)
