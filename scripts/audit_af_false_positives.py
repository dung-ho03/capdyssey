"""Prespecified validation-only threshold selection; reused-test development audit."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from finalize_two_sources import Predictor,ALLOWED
from train_af_rhythm import at_threshold
from prepare_af_integrated import digest
from train_af import write_json

def threshold_for(y,p,sources,floor):
    grid=np.linspace(0,1,1001);recall=[];specificity=[]
    for s in sorted(ALLOWED):
        pos=np.sort(p[(sources==s)&(y==1)]);neg=np.sort(p[(sources==s)&(y==0)])
        assert len(pos) and len(neg)
        recall.append(1-np.searchsorted(pos,grid,side='left')/len(pos))
        specificity.append(np.searchsorted(neg,grid,side='left')/len(neg))
    valid=np.flatnonzero(np.min(specificity,axis=0)>=floor)
    assert len(valid)
    # Protect the worse source's recall; no test data are involved.
    i=valid[np.argmax(np.min(recall,axis=0)[valid])]
    return float(grid[i]),float(np.min(recall,axis=0)[i])

def run(base):
    import google.colab
    assert torch.cuda.is_available(),'Colab GPU required'
    root=base/'models/main_two_sources_v1';out=base/'models/false_positive_audit_v1';out.mkdir(exist_ok=True)
    choices=[];tradeoffs=[];validation_people={}
    for seed in [42,43]:
        with np.load(root/f'seed{seed}'/'validation_predictions.npz') as d:y=d['y'];p=d['p'];sources=d['source'];people=d['subject']
        assert set(sources.tolist())==ALLOWED
        for floor in [.8,.85,.9,.95]:
            threshold,score=threshold_for(y,p,sources,floor)
            row={'seed':seed,'specificity_floor':floor,'threshold':threshold,'worst_source_recall':score,
                 'by_source':{s:at_threshold(y[sources==s],p[sources==s],threshold) for s in sorted(ALLOWED)}}
            tradeoffs.append(row)
            if floor==.9:choices.append(row)
        validation_people[str(seed)]={s:at_threshold(y[people==s],p[people==s],choices[-1]['threshold']) for s in sorted(set(people))}
    choice=sorted(choices,key=lambda r:(-r['worst_source_recall'],r['seed']))[0]
    checkpoint=root/f'seed{choice["seed"]}'/'best_model.pt'
    frozen={'rule':'Validation per-source specificity >=90%, maximize minimum source recall; lower seed breaks tie.',
            'choice':choice,'checkpoint_sha256':digest(checkpoint),'script_sha256':digest(Path(__file__)),
            'limits':'Development revision motivated by previously observed test false positives. Reused test is NOT independent confirmation.'}
    path=out/'frozen_selection.json'
    if path.exists():assert json.loads(path.read_text())==frozen
    else:write_json(path,frozen)
    write_json(out/'validation_tradeoffs.json',tradeoffs);write_json(out/'validation_subjects.json',validation_people)
    print('VALIDATION_TRADEOFFS',json.dumps(tradeoffs),flush=True)
    print('FROZEN_CHOICE',json.dumps(choice),flush=True)
    # Only now evaluate existing test data. Do not optimize after viewing results.
    model=Predictor(checkpoint);bundle=base/'processed/provisional_centered_v2'
    manifest=json.loads((bundle/'manifest.json').read_text());splitfile=base/'models/provisional_centered_v2/split.json'
    split=json.loads(splitfile.read_text())
    assert model.ck['identity']['bundle']==digest(bundle/'manifest.json') and model.ck['identity']['split']==digest(splitfile)
    yy=[];pp=[];ss=[];subjects=[]
    for row in manifest['shards']:
        if row['source'] not in ALLOWED or row['subject'] not in split['test']:continue
        assert digest(bundle/row['path'])==row['sha256']
        with np.load(bundle/row['path']) as d:yy.append(d['y']);pp.append(model.predict(d['x']))
        ss.extend([row['source']]*row['n']);subjects.extend([row['subject']]*row['n'])
    y=np.concatenate(yy);p=np.concatenate(pp);sources=np.array(ss);people=np.array(subjects);t=choice['threshold']
    results={s:at_threshold(y[sources==s],p[sources==s],t) for s in sorted(ALLOWED)}
    per_person={s:at_threshold(y[people==s],p[people==s],t) for s in sorted(set(subjects))}
    previous=json.loads((root/'candidate/test_report.json').read_text())
    report={'selection':frozen,'previous_by_source':previous['by_source'],'new_by_source':results,'per_subject':per_person,
            'goal_met':all(r['sensitivity']>=.9 and r['specificity']>=.9 for r in results.values())}
    write_json(out/'report.json',report);np.savez_compressed(out/'predictions.npz',y=y,p=p,source=sources,subject=people)
    print('FALSE_POSITIVE_AUDIT',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);run(p.parse_args().base)
