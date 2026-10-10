"""Colab-only three-source trainer. Never silently drop an unavailable source."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from train_af import make_model, predict, metrics, write_json, ROOT
from prepare_af_integrated import digest

SOURCES={'mimic_perform_af','six_rhythms','longterm_wrist_v3'}

def preflight(manifest, status):
    errors=[]
    if set(manifest['sources'])!=SOURCES: errors.append('All three sources are required')
    if manifest.get('window_seconds')!=10 or manifest.get('sampling_hz')!=50:errors.append('Expected 10s/50Hz')
    for source in sorted(SOURCES):
        state=status['datasets'][source]
        if not state['ready']:errors.append(source+': '+state.get('reason','not ready'))
        rows=[s for s in manifest['shards'] if s['source']==source]
        if len({s['subject'] for s in rows})<5:errors.append(source+': at least five subjects needed for this protocol')
        if any(s['label_counts'].get('-1',0)>0 for s in rows):errors.append(source+': unresolved labels remain')
    return errors

def train(bundle,output,epochs=20):
    import google.colab  # User requires actual training/evaluation only in Colab.
    if epochs<1:raise ValueError('epochs must be positive')
    m=json.loads((bundle/'manifest.json').read_text())
    status=json.loads((ROOT/'data/af_integration_status.json').read_text())
    errors=preflight(m,status)
    if errors:raise ValueError('Integrated training not ready:\n'+'\n'.join(errors))
    xs=[];ys=[];persons=[];sources=[]
    for row in m['shards']:
        path=bundle/row['path']
        if path.resolve().parent!=bundle.resolve() or digest(path)!=row['sha256']:raise ValueError('Invalid shard path/hash')
        with np.load(path,allow_pickle=False) as d:
            x=d['x']; y=d['y']
            if x.shape!=(len(y),500) or not np.isfinite(x).all() or not np.isin(y,[0,1]).all():raise ValueError('Invalid shard data')
            xs.append(x);ys.append(y);persons.extend([row['subject']]*len(y));sources.extend([row['source']]*len(y))
    x=np.concatenate(xs);y=np.concatenate(ys);persons=np.array(persons);sources=np.array(sources)
    rng=np.random.default_rng(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    split={'train':[],'val':[],'test':[]}
    for source in sorted(SOURCES):
        if source=='mimic_perform_af':
            original=json.loads((ROOT/'models/af_mimic_v1_colab/split.json').read_text())
            original_ids={source+':'+p for ids in original.values() for p in ids}
            if original_ids!=set(persons[sources==source]):raise ValueError('MIMIC participant inventory changed')
            for name,ids in original.items():split[name].extend(source+':'+p for p in ids)
            continue
        ids=np.unique(persons[sources==source]);rng.shuffle(ids);n=max(1,round(len(ids)*.2))
        split['test'].extend(ids[:n].tolist());split['val'].extend(ids[n:2*n].tolist());split['train'].extend(ids[2*n:].tolist())
    masks={k:np.isin(persons,v) for k,v in split.items()}
    for name,mask in masks.items():
        for source in SOURCES:
            if set(y[mask&(sources==source)])!={0,1}:raise ValueError(f'{name}/{source} lacks a class; review protocol without selecting on test performance')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'split.json',split)
    # Equal mass per source, class, then subject-within-class; training rows only.
    indices=np.flatnonzero(masks['train']);weights=np.zeros(len(y))
    for source in SOURCES:
        for label in (0,1):
            mask=masks['train']&(sources==source)&(y==label);ids=np.unique(persons[mask])
            for person in ids:
                group=mask&(persons==person);weights[group]=1/(len(ids)*group.sum())
    ds=TensorDataset(torch.from_numpy(x[indices,None]),torch.from_numpy(y[indices].astype(np.float32)))
    loader=DataLoader(ds,batch_size=64,sampler=WeightedRandomSampler(weights[indices],len(indices),replacement=True))
    def evaluate(model,mask):
        dl=DataLoader(TensorDataset(torch.from_numpy(x[mask,None]),torch.from_numpy(y[mask].astype(np.float32))),batch_size=256)
        yy,pp,loss=predict(model,dl,device)
        return metrics(yy,pp),loss
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=make_model().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);criterion=nn.BCEWithLogitsLoss()
    config={'model':'small_cnn_10s_v1','window_seconds':10,'sampling_hz':50,'threshold':.5,
        'datasets_used':sorted(SOURCES),'three_dataset_integrated':True,'seed':42,'device':str(device),
        'bundle_manifest_sha256':digest(bundle/'manifest.json'),'script_sha256':digest(Path(__file__)),
        'selection':'Mean of per-source validation BCE; fixed threshold .5',
        'sampling':'Equal source/class/subject-class mass, with replacement',
        'evaluation_limit':'Source-internal held-out people; not held-out-source external validation'}
    write_json(output/'config.json',config);history=[];best=float('inf');stale=0
    for epoch in range(1,epochs+1):
        model.train();total=0
        for xb,yb in loader:
            optimizer.zero_grad();loss=criterion(model(xb.to(device)).flatten(),yb.to(device));loss.backward();optimizer.step();total+=loss.item()*len(yb)
        validation={source:evaluate(model,masks['val']&(sources==source)) for source in sorted(SOURCES)}
        score=float(np.mean([v[1] for v in validation.values()]))
        history.append({'epoch':epoch,'train_loss':total/len(indices),'val_macro_bce':score,'validation_by_source':validation})
        if score<best:
            best=score;stale=0;torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'config':config,'epoch':epoch},output/'best_model.pt')
        else:stale+=1
        write_json(output/'history.json',history);print('epoch',epoch,'validation macro BCE',score,flush=True)
        if stale>=5:break
    saved=torch.load(output/'best_model.pt',weights_only=True,map_location=device);model.load_state_dict(saved['state_dict'])
    result={'best_epoch':saved['epoch'],'test_by_source':{s:evaluate(model,masks['test']&(sources==s))[0] for s in sorted(SOURCES)},
            'test_overall':evaluate(model,masks['test'])[0]}
    write_json(output/'metrics.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=20)
    a=p.parse_args();train(a.bundle,a.output,a.epochs)
