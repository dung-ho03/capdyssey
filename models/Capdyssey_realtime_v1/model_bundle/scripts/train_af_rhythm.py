"""GPU-only 10-second rhythm candidates. Development validation, not clinical validation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_af import metrics, write_json
from prepare_af_integrated import digest


def rhythm_features(x):
    """PPG-only autocorrelation/spectrum; no ECG, labels or source IDs as input."""
    z=x-x.mean(-1,keepdim=True)
    z=z/z.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
    spectrum=torch.fft.rfft(z,n=1024).abs().square()
    ac=torch.fft.irfft(spectrum,n=1024)[:,:251]
    ac=ac/ac[:,:1].clamp_min(1e-6)
    power=torch.fft.rfft(z*torch.hann_window(500,device=x.device),n=500).abs().square()
    relative=power/power.sum(-1,keepdim=True).clamp_min(1e-6)
    # HF ratio is a descriptive proxy, not a validated quality label.
    hf=relative[:,81:].sum(-1)
    f=torch.cat([ac[:,10:251:2],torch.log(relative[:,5:81]+1e-6),
                 hf[:,None],z.abs().amax(-1,keepdim=True),
                 z.diff().square().mean(-1,keepdim=True)],dim=1)
    return z,f,hf


class RhythmModel(nn.Module):
    def __init__(self,n_features,kind):
        super().__init__();self.kind=kind
        self.features=nn.Sequential(nn.Linear(n_features,64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,32),nn.ReLU())
        layers=[nn.Conv1d(1,24,9,stride=4,padding=4),nn.GroupNorm(4,24),nn.GELU()]
        for dilation in (1,2,4,8,16):
            layers.extend([nn.Conv1d(24,24,5,padding=2*dilation,dilation=dilation),nn.GroupNorm(4,24),nn.GELU()])
        self.wave=nn.Sequential(*layers,nn.AdaptiveAvgPool1d(1),nn.Flatten()) if kind=='hybrid' else None
        self.head=nn.Sequential(nn.Dropout(.25),nn.Linear(56 if kind=='hybrid' else 32,1))

    def forward(self,x,f):
        a=self.features(f)
        if self.wave is not None:a=torch.cat([a,self.wave(x[:,None])],1)
        return self.head(a).flatten()


def at_threshold(y,p,t):
    result=metrics(y,p)
    binary=metrics(y,(p>=t).astype(float))
    binary['roc_auc']=result['roc_auc'];binary['threshold']=float(t)
    return binary


def choose_threshold(y,p,sources):
    """One common threshold; maximize macro recall with every source specificity >= .8."""
    grid=np.linspace(0,1,1001);recalls=[];specificities=[]
    for s in sorted(set(sources.tolist())):
        pos=np.sort(p[(sources==s)&(y==1)]);neg=np.sort(p[(sources==s)&(y==0)])
        assert len(pos) and len(neg)
        recalls.append(1-np.searchsorted(pos,grid,side='left')/len(pos))
        specificities.append(np.searchsorted(neg,grid,side='left')/len(neg))
    recall=np.mean(recalls,axis=0);valid=np.min(specificities,axis=0)>=.8
    ix=np.flatnonzero(valid)
    if not len(ix):return 1.0,0.0
    best=ix[np.argmax(recall[ix])]
    return float(grid[best]),float(recall[best])


def run(base):
    import google.colab
    if not torch.cuda.is_available():raise RuntimeError('Colab GPU required. No CPU fallback.')
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    bundle=base/'processed/provisional_centered_v2';old=base/'models/provisional_centered_v2'
    out=base/'models/rhythm_candidates_v1';out.mkdir(exist_ok=True)
    split=json.loads((old/'split.json').read_text())
    assert not any(set(split[a])&set(split[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    manifest=json.loads((bundle/'manifest.json').read_text())
    xs=[];ys=[];persons=[];sources=[]
    # No test data loaded while developing/selecting candidates.
    for r in manifest['shards']:
        if r['subject'] in split['test']:continue
        assert digest(bundle/r['path'])==r['sha256']
        with np.load(bundle/r['path']) as d:xs.append(d['x']);ys.append(d['y'])
        persons.extend([r['subject']]*r['n']);sources.extend([r['source']]*r['n'])
    x=torch.as_tensor(np.concatenate(xs),device='cuda');del xs
    y=np.concatenate(ys);persons=np.array(persons);sources=np.array(sources)
    train=np.isin(persons,split['train']);val=np.isin(persons,split['val'])
    assert x.shape==(len(y),500) and torch.isfinite(x).all() and np.isin(y,[0,1]).all()
    z,f,hf=rhythm_features(x);del x
    tid=torch.as_tensor(np.flatnonzero(train),device='cuda')
    mean=f[tid].mean(0);std=f[tid].std(0).clamp_min(.01);f=((f-mean)/std).clamp(-10,10)
    yy=torch.as_tensor(y.astype(np.float32),device='cuda');vid=np.flatnonzero(val)
    quality_cuts={s:np.quantile(hf[torch.as_tensor(np.flatnonzero(train&(sources==s)),device='cuda')].cpu().numpy(),[.25,.5,.75]).tolist() for s in sorted(set(sources.tolist()))}
    protocol={'gpu':torch.cuda.get_device_name(0),'seed':42,'seconds':10,'fs':50,'script_sha256':digest(Path(__file__)),
              'bundle_sha256':digest(bundle/'manifest.json'),'split_sha256':digest(old/'split.json'),
              'threshold_rule':'Common threshold maximizing macro sensitivity subject to specificity >=80% in EVERY validation source',
              'target':'Exploratory goal: sensitivity >=90% and specificity >=80% in each source; report failure explicitly',
              'limitations':'Provisional wrist exact 0/1 labels. Reused development validation. Single seed. No clinical claim.',
              'quality_cuts':quality_cuts,'epochs_max':20,'patience':5,'batch':256,'lr':.0005}
    pp=out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Changed protocol; use a new output directory'
    write_json(pp,protocol)
    weights=np.zeros(len(y))
    for s in sorted(set(sources.tolist())):
        for label in (0,1):
            m=train&(sources==s)&(y==label);ids=np.unique(persons[m]);assert len(ids)
            for person in ids:
                g=m&(persons==person);weights[g]=1/(len(ids)*g.sum())
    w=torch.as_tensor(weights[train],device='cuda');draws=int(train.sum())
    def predict(model,indices):
        model.eval();parts=[]
        with torch.no_grad():
            for start in range(0,len(indices),512):
                b=torch.as_tensor(indices[start:start+512],device='cuda')
                parts.append(model(z[b],f[b]).sigmoid().cpu().numpy())
        return np.concatenate(parts)
    results={}
    print('RHYTHM_GPU_START',protocol['gpu'],'train',draws,'validation',len(vid),flush=True)
    for kind in ('features','hybrid'):
        folder=out/kind;folder.mkdir(exist_ok=True)
        if (folder/'validation.json').exists():results[kind]=json.loads((folder/'validation.json').read_text());continue
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=RhythmModel(f.shape[1],kind).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
        history=[];best=-1.;stale=0;start_epoch=1
        if (folder/'resume.pt').exists():
            ck=torch.load(folder/'resume.pt',weights_only=True,map_location='cuda');model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
            best=ck['best'];stale=ck['stale'];history=ck['history'];start_epoch=ck['epoch']+1
            torch.set_rng_state(ck['rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        for epoch in range(start_epoch,21):
            if stale>=5:break
            model.train();sample=tid[torch.multinomial(w,draws,replacement=True)];total=0
            for offset in range(0,draws,256):
                b=sample[offset:offset+256];opt.zero_grad()
                loss=nn.functional.binary_cross_entropy_with_logits(model(z[b],f[b]),yy[b]);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(b)
            p=predict(model,vid);t,score=choose_threshold(y[val],p,sources[val])
            row={'epoch':epoch,'loss':total/draws,'threshold':t,'macro_sensitivity_at_spec80':score};history.append(row)
            if score>best:
                best=score;stale=0
                torch.save({'state_dict':model.state_dict(),'kind':kind,'n_features':f.shape[1],'mean':mean,'std':std,'threshold':t,'epoch':epoch,'protocol':protocol},folder/'best_model.pt')
            else:stale+=1
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'best':best,'stale':stale,'history':history,'epoch':epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},folder/'resume.pt')
            write_json(folder/'history.json',history);print('RHYTHM_EPOCH',kind,json.dumps(row),flush=True)
        ck=torch.load(folder/'best_model.pt',weights_only=True,map_location='cuda');model.load_state_dict(ck['state_dict'])
        p=predict(model,vid);t=ck['threshold'];by={};quality={}
        for s in sorted(set(sources.tolist())):
            m=sources[val]==s;by[s]=at_threshold(y[val][m],p[m],t)
            q=hf[torch.as_tensor(vid[m],device='cuda')].cpu().numpy();bins=np.digitize(q,quality_cuts[s]);quality[s]=[]
            for k in range(4):
                g=bins==k
                if g.any():quality[s].append({'bin':k,'coverage':float(g.mean()),'af_count':int(y[val][m][g].sum()),'metrics':at_threshold(y[val][m][g],p[m][g],t)})
        rec={'epoch':ck['epoch'],'threshold':t,'by_source':by,'quality_proxy_bins':quality,'macro_sensitivity':float(np.mean([v['sensitivity'] for v in by.values()])),
             'engineering_goal_met':all(v['sensitivity']>=.9 and v['specificity']>=.8 for v in by.values())}
        write_json(folder/'validation.json',rec);np.savez_compressed(folder/'validation_predictions.npz',y=y[val],p=p,subject=persons[val],source=sources[val])
        results[kind]=rec;print('RHYTHM_CANDIDATE_COMPLETE',kind,json.dumps({'threshold':t,'by_source':by,'goal_met':rec['engineering_goal_met']}),flush=True)
    selected=max(results,key=lambda k:results[k]['macro_sensitivity'])
    summary={'selected':selected,'status':'development_candidate','results':results,'test_evaluated':False}
    write_json(out/'comparison.json',summary)
    print('RHYTHM_COMPLETE',selected,'goal_met',results[selected]['engineering_goal_met'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    run(parser.parse_args().base)
