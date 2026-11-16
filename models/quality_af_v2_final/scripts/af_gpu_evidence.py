"""GPU-only source ablations and validation-only threshold diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np


def threshold_report(y,p):
    """Fixed diagnostic grid, including all-positive/all-negative endpoints."""
    thresholds=np.linspace(0,1,1001)
    pos=np.sort(p[y==1]);neg=np.sort(p[y==0])
    if not len(pos) or not len(neg):raise ValueError('Both classes required')
    sens=1-np.searchsorted(pos,thresholds,side='left')/len(pos)
    spec=np.searchsorted(neg,thresholds,side='left')/len(neg)
    def row(i):return {'threshold':float(thresholds[i]),'sensitivity':float(sens[i]),'specificity':float(spec[i])}
    # Targets are engineering diagnostics, not clinically validated operating requirements.
    candidates=np.flatnonzero(sens>=.9)
    high_recall=row(candidates[-1]) if len(candidates) else None
    candidates=np.flatnonzero(spec>=.8)
    high_specificity=row(candidates[0]) if len(candidates) else None
    return {'at_05':row(500),'at_sensitivity_90':high_recall,'at_specificity_80':high_specificity}


def run(base):
    import google.colab
    import torch
    from torch import nn
    from train_af import make_model,metrics,write_json
    from prepare_af_integrated import digest
    if not torch.cuda.is_available():raise RuntimeError('GPU required; CPU fallback prohibited')
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    bundle=base/'processed/provisional_centered_v2'
    baseline=base/'models/provisional_centered_v2'
    out=base/'models/gpu_source_evidence_v1';out.mkdir(exist_ok=True)
    split=json.loads((baseline/'split.json').read_text())
    assert not (set(split['train'])&set(split['val']) or set(split['train'])&set(split['test']) or set(split['val'])&set(split['test']))
    manifest=json.loads((bundle/'manifest.json').read_text())
    x=[];y=[];persons=[];sources=[]
    for r in manifest['shards']:
        # Thresholds and experiment selection do not inspect the test set.
        if r['subject'] in split['test']:continue
        assert digest(bundle/r['path'])==r['sha256']
        with np.load(bundle/r['path']) as d:x.append(d['x']);y.append(d['y'])
        persons.extend([r['subject']]*r['n']);sources.extend([r['source']]*r['n'])
    x=np.concatenate(x);y=np.concatenate(y);persons=np.array(persons);sources=np.array(sources)
    assert x.shape==(len(y),500) and np.isfinite(x).all() and np.isin(y,[0,1]).all()
    train=np.isin(persons,split['train']);val=np.isin(persons,split['val'])
    all_sources=sorted(set(sources.tolist()));draws=int(train.sum())
    xx=torch.from_numpy(x[:,None]).cuda();yy=torch.from_numpy(y.astype(np.float32)).cuda()
    del x
    val_indices={s:np.flatnonzero(val&(sources==s)) for s in all_sources}
    def evaluate(model):
        model.eval();result={};probabilities={};losses={}
        with torch.no_grad():
            for source,indices in val_indices.items():
                logits=torch.cat([model(xx[torch.as_tensor(indices[i:i+512],device='cuda')]).flatten() for i in range(0,len(indices),512)])
                p=torch.sigmoid(logits).cpu().numpy();truth=y[indices]
                result[source]={'metrics':metrics(truth,p),'thresholds':threshold_report(truth,p)}
                losses[source]=float(nn.functional.binary_cross_entropy_with_logits(logits,yy[torch.as_tensor(indices,device='cuda')]).item())
                probabilities[source]=p
        return result,losses,probabilities
    model=make_model().cuda()
    checkpoint=torch.load(baseline/'best_model.pt',weights_only=True,map_location='cuda')
    assert checkpoint['config']['bundle_manifest_sha256']==digest(bundle/'manifest.json')
    model.load_state_dict(checkpoint['state_dict'])
    initial,_,_=evaluate(model)
    write_json(out/'existing_model_validation.json',initial)
    print('EXISTING_VALIDATION_THRESHOLDS',json.dumps(initial),flush=True)
    protocol={'device':torch.cuda.get_device_name(0),'seed':42,'draws_per_epoch':draws,
        'batch_size':64,'epochs_max':20,'patience':5,'learning_rate':.001,
        'split_sha256':digest(baseline/'split.json'),'bundle_sha256':digest(bundle/'manifest.json'),
        'script_sha256':digest(Path(__file__)),
        'selection':'mean validation BCE on included training sources',
        'limits':'Single-seed development evidence; validation reused for checkpoint and threshold selection. No test evaluation or clinical threshold claim.'}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        prior=json.loads(protocol_path.read_text())
        assert {k:v for k,v in prior.items() if k!='script_sha256'}=={k:v for k,v in protocol.items() if k!='script_sha256'},'Protocol mismatch'
    write_json(protocol_path,protocol)
    groups={'two_sources':[s for s in all_sources if s!='longterm_wrist_v3'],
            'wrist_only':['longterm_wrist_v3'],'three_sources':all_sources}
    comparisons={}
    for name,included in groups.items():
        folder=out/name;folder.mkdir(exist_ok=True)
        if (folder/'validation.json').exists():
            comparisons[name]=json.loads((folder/'validation.json').read_text());continue
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=make_model().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        weights=np.zeros(len(y))
        for source in included:
            for label in (0,1):
                mask=train&(sources==source)&(y==label);ids=np.unique(persons[mask])
                assert len(ids)
                for person in ids:
                    group=mask&(persons==person);weights[group]=1/(len(ids)*group.sum())
        indices=np.flatnonzero(weights>0);w=torch.as_tensor(weights[indices],dtype=torch.double)
        idx=torch.as_tensor(indices,device='cuda');best=float('inf');stale=0;history=[]
        prior_history=json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
        completed=False
        if prior_history:
            best_index=int(np.argmin([h['selection_val_bce'] for h in prior_history]))
            completed=len(prior_history)>=20 or len(prior_history)-1-best_index>=5
        for epoch in ([] if completed else range(1,21)):
            model.train();sample=torch.multinomial(w,draws,replacement=True).cuda();total=0
            for offset in range(0,draws,64):
                batch=idx[sample[offset:offset+64]]
                optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(xx[batch]).flatten(),yy[batch]);loss.backward();optimizer.step()
                total+=float(loss.item())*len(batch)
            validation,losses,_=evaluate(model);score=float(np.mean([losses[s] for s in included]))
            history.append({'epoch':epoch,'train_loss':total/draws,'selection_val_bce':score,'validation':validation})
            if score<best:
                best=score;stale=0
                torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'epoch':epoch,'sources':included,'protocol':protocol},folder/'best_model.pt')
            else:stale+=1
            write_json(folder/'history.json',history)
            print('GPU_ABLATION',name,'epoch',epoch,'val_bce',score,flush=True)
            if stale>=5:break
        # Recover only our generated v1 metadata with explicitly allowed NumPy
        # scalar types. Keep weights_only=True; future checkpoints use Python str.
        with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,np.dtypes.StrDType]):
            saved=torch.load(folder/'best_model.pt',weights_only=True,map_location='cuda')
        saved['sources']=[str(s) for s in saved['sources']]
        torch.save(saved,folder/'best_model.pt')
        model.load_state_dict(saved['state_dict'])
        result,_,probs=evaluate(model)
        record={'best_epoch':saved['epoch'],'included_sources':included,'by_source':result}
        write_json(folder/'validation.json',record);comparisons[name]=record
        arrays={}
        for source,indices in val_indices.items():arrays[source+'_y']=y[indices];arrays[source+'_p']=probs[source];arrays[source+'_subject']=persons[indices]
        np.savez_compressed(folder/'validation_predictions.npz',**arrays)
        print('GROUP_COMPLETE',name,json.dumps(record),flush=True)
    write_json(out/'comparison.json',comparisons)
    print('GPU_EVIDENCE_COMPLETE',json.dumps(comparisons),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    run(p.parse_args().base)
