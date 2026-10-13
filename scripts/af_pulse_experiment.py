"""Colab GPU pulse-feature experiment; ECG is audit-only, never a model input."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.signal import butter, sosfiltfilt, find_peaks
from train_af import metrics, write_json
from train_af_rhythm import rhythm_features, choose_threshold, at_threshold
from prepare_af_integrated import digest, chars, start_time


def pulse_stats(signal,polarity=1):
    # Generic peak detector, not a validated clinical pulse detector.
    peaks,props=find_peaks(polarity*signal,distance=13,prominence=.35)
    keep=(peaks>=50)&(peaks<450);peaks=peaks[keep];prom=props['prominences'][keep]
    intervals=np.diff(peaks)/50
    valid=len(intervals)>=3
    if not valid:return np.array([0.]*10+[len(peaks)/30,1.],np.float32),peaks
    mean=intervals.mean();delta=np.diff(intervals)
    stats=[mean,intervals.std()/mean,np.sqrt(np.mean(delta**2))/mean,
           np.mean(np.abs(delta)>.05),np.median(np.abs(intervals-np.median(intervals)))/mean,
           np.quantile(intervals,.1),np.quantile(intervals,.9),
           np.mean((intervals<.3)|(intervals>2)),float(prom.mean()),float(prom.std()),len(peaks)/30,0.]
    return np.asarray(stats,np.float32),peaks


def extract(x):
    sos=butter(3,[.5,5],btype='bandpass',fs=50,output='sos')
    filtered=sosfiltfilt(sos,x,axis=1,padlen=100)
    filtered=(filtered-filtered.mean(1,keepdims=True))/np.maximum(filtered.std(1,keepdims=True),1e-8)
    feature=[]
    for wave in filtered:
        a,_=pulse_stats(wave,1);b,_=pulse_stats(wave,-1)
        # Both polarities retained; no ECG-dependent polarity selection in prediction.
        feature.append(np.r_[a,b,abs(a[1]-b[1]),abs(a[0]-b[0])])
    return filtered.astype(np.float32),np.asarray(feature,np.float32)


class PulseModel(nn.Module):
    def __init__(self,n):
        super().__init__();self.net=nn.Sequential(nn.Linear(n,96),nn.ReLU(),nn.Dropout(.3),nn.Linear(96,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
    def forward(self,f):return self.net(f).flatten()


def alignment_audit(base,out,rows,filtered,y,people,sources,starts,split):
    import h5py
    rng=np.random.default_rng(42);cases=[];summary={}
    for person in sorted(set(people[sources=='longterm_wrist_v3'].tolist())):
        with h5py.File(base/'raw/longterm_af_v3'/(person.split(':')[1]+'_ECG.mat')) as e:
            begin=start_time(chars(e['recording_startday']),chars(e['recording_starttime']))
            q=begin+(e['QRSindex'][()].ravel().astype(float)-1)/500
        for label in (0,1):
            indices=np.flatnonzero((people==person)&(y==label))
            selected=rng.choice(indices,min(12,len(indices)),replace=False)
            for idx in selected:
                t=starts[idx];qt=q[(q>=t+1)&(q<t+9)]-t;rr=np.diff(qt)
                if len(rr)<3:continue
                pp,peaks=pulse_stats(filtered[idx],1);pm,troughs=pulse_stats(filtered[idx],-1)
                curve={}
                # Bidirectional nearest-beat agreement. Maximized lag is descriptive,
                # not a clock correction; regular rhythms have periodic ambiguities.
                for shift in (0,-10,10,-30,30):
                    reference=q[(q>=t+shift)&(q<t+shift+10)]-(t+shift)
                    best=0.;best_lag=0.
                    for detected in (peaks/50,troughs/50):
                        if not len(detected) or not len(reference):continue
                        for lag in np.arange(-.5,.501,.05):
                            ref=reference+lag;ref=ref[(ref>=1)&(ref<9)]
                            if not len(ref):continue
                            d=np.abs(detected[:,None]-ref[None,:])
                            precision=np.mean(d.min(1)<.08);recall=np.mean(d.min(0)<.08)
                            score=2*precision*recall/max(precision+recall,1e-8)
                            if score>best:best=float(score);best_lag=float(lag)
                    curve[str(shift)]={'agreement':best,'lag':best_lag}
                cases.append({'person':person,'split':'train' if person in split['train'] else 'val','label':int(label),
                              'start':float(t),'ecg_rr_cv':float(rr.std()/rr.mean()),'ppg_cv':float(pp[1]),
                              'peak_failure':bool(pp[-1]),'agreement':curve})
        print('맥박 정렬 표본 검사',person,flush=True)
    for label in (0,1):
        c=[r for r in cases if r['label']==label]
        summary[str(label)]={'n':len(c),'median_ecg_rr_cv':float(np.median([r['ecg_rr_cv'] for r in c])),
                            'median_ppg_cv':float(np.median([r['ppg_cv'] for r in c])),
                            'median_agreement':{str(s):float(np.median([r['agreement'][str(s)]['agreement'] for r in c])) for s in (0,-10,10,-30,30)}}
    result={'summary':summary,'cases':cases,'limits':'Train/validation stratified sample, not all raw recordings. Lag maximization and periodic rhythm cause optimistic/ambiguous matching. Not proof of AF label meanings. ECG never supplied to classifier.'}
    write_json(out/'alignment.json',result);print('ALIGNMENT_SUMMARY',json.dumps(summary),flush=True)


def run(base):
    import google.colab
    if not torch.cuda.is_available():raise RuntimeError('Colab GPU required')
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    out=base/'models/pulse_features_v1';out.mkdir(exist_ok=True)
    bundle=base/'processed/provisional_centered_v2';m=json.loads((bundle/'manifest.json').read_text())
    audit=json.loads((base/'audits/prepared_bundle/report.json').read_text())
    assert audit['structural_checks_passed'] and audit['bundle_sha256']==digest(bundle/'manifest.json')
    splitfile=base/'models/provisional_centered_v2/split.json';split=json.loads(splitfile.read_text())
    assert not any(set(split[a])&set(split[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    identity={'bundle':digest(bundle/'manifest.json'),'split':digest(splitfile),'script':digest(Path(__file__))}
    if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==identity,'Changed experiment identity'
    write_json(out/'identity.json',identity)
    cache=out/'features.npz'
    if cache.exists():
        with np.load(cache) as d:filtered=d['filtered'];pulse=d['pulse'];y=d['y'];people=d['people'];sources=d['sources'];starts=d['starts']
    else:
        waves=[];features=[];ys=[];people=[];sources=[];starts=[]
        for i,r in enumerate(m['shards']):
            if r['subject'] in split['test']:continue
            assert digest(bundle/r['path'])==r['sha256']
            with np.load(bundle/r['path']) as d:f,p=extract(d['x']);ys.append(d['y']);starts.extend(d['start_seconds'])
            waves.append(f);features.append(p);people.extend([r['subject']]*r['n']);sources.extend([r['source']]*r['n'])
            if (i+1)%50==0:print('PPG 특징 추출',i+1,'/',len(m['shards']),flush=True)
        filtered=np.concatenate(waves);pulse=np.concatenate(features);y=np.concatenate(ys);people=np.array(people);sources=np.array(sources);starts=np.array(starts)
        with (out/'features.tmp').open('wb') as f:np.savez_compressed(f,filtered=filtered,pulse=pulse,y=y,people=people,sources=sources,starts=starts)
        (out/'features.tmp').replace(cache)
    assert np.isfinite(filtered).all() and np.isfinite(pulse).all()
    if not (out/'alignment.json').exists():alignment_audit(base,out,m['shards'],filtered,y,people,sources,starts,split)
    train=np.isin(people,split['train']);val=np.isin(people,split['val']);vid=np.flatnonzero(val)
    with torch.no_grad():
        spectral=[]
        for a in range(0,len(y),1024):spectral.append(rhythm_features(torch.as_tensor(filtered[a:a+1024],device='cuda'))[1])
        f=torch.cat([torch.cat(spectral),torch.as_tensor(pulse,device='cuda')],dim=1)
    tid=torch.as_tensor(np.flatnonzero(train),device='cuda');mean=f[tid].mean(0);std=f[tid].std(0).clamp_min(.01);f=((f-mean)/std).clamp(-10,10)
    yy=torch.as_tensor(y.astype(np.float32),device='cuda');weights=np.zeros(len(y))
    for s in sorted(set(sources.tolist())):
        for label in (0,1):
            mask=train&(sources==s)&(y==label);ids=np.unique(people[mask]);assert len(ids)
            for person in ids:
                g=mask&(people==person);weights[g]=1/(len(ids)*g.sum())
    w=torch.as_tensor(weights[train],device='cuda');draws=int(train.sum());results={}
    # Two predeclared seeds measure stability instead of endlessly tuning the same split.
    for seed in (42,43):
        folder=out/f'seed{seed}';folder.mkdir(exist_ok=True)
        if (folder/'validation.json').exists():results[str(seed)]=json.loads((folder/'validation.json').read_text());continue
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=PulseModel(f.shape[1]).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
        best=-1.;stale=0;history=[];first=1
        if (folder/'resume.pt').exists():
            ck=torch.load(folder/'resume.pt',map_location='cuda',weights_only=True);model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
            best=ck['best'];stale=ck['stale'];history=ck['history'];first=ck['epoch']+1
            torch.set_rng_state(ck['rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        def predict():
            model.eval()
            with torch.no_grad():return torch.cat([model(f[torch.as_tensor(vid[a:a+1024],device='cuda')]).sigmoid() for a in range(0,len(vid),1024)]).cpu().numpy()
        for epoch in range(first,21):
            if stale>=5:break
            model.train();sample=tid[torch.multinomial(w,draws,replacement=True)];loss_sum=0.
            for a in range(0,draws,256):
                b=sample[a:a+256];optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(f[b]),yy[b]);loss.backward();optimizer.step();loss_sum+=loss.item()*len(b)
            prob=predict();threshold,score=choose_threshold(y[val],prob,sources[val]);history.append({'epoch':epoch,'loss':loss_sum/draws,'macro_sensitivity_at_spec80':score,'threshold':threshold})
            if score>best:
                best=score;stale=0;torch.save({'state_dict':model.state_dict(),'n_features':f.shape[1],'mean':mean,'std':std,'threshold':threshold,'epoch':epoch,'identity':identity},folder/'best_model.pt')
            else:stale+=1
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'best':best,'stale':stale,'history':history,'epoch':epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},folder/'resume.pt')
            write_json(folder/'history.json',history);print('GPU_PULSE',seed,epoch,'macro_recall',round(score,4),flush=True)
        ck=torch.load(folder/'best_model.pt',weights_only=True,map_location='cuda');model.load_state_dict(ck['state_dict']);prob=predict();threshold=ck['threshold']
        result={'seed':seed,'epoch':ck['epoch'],'threshold':threshold,'gpu':torch.cuda.get_device_name(0),'by_source':{},'peak_failure_by_source_class':{}}
        for s in sorted(set(sources.tolist())):
            mask=sources[val]==s;result['by_source'][s]=at_threshold(y[val][mask],prob[mask],threshold)
            result['peak_failure_by_source_class'][s]={str(label):float(np.mean(pulse[val][mask&(y[val]==label),11])) for label in (0,1)}
        write_json(folder/'validation.json',result);np.savez_compressed(folder/'validation_predictions.npz',y=y[val],p=prob,source=sources[val],subject=people[val])
        results[str(seed)]=result;print('PULSE_VALIDATION',json.dumps(result),flush=True)
    write_json(out/'comparison.json',{'results':results,'test_used':False,'all_windows_retained':True,'limits':'Repeated development validation, provisional wrist labels, generic peak extraction. ECG only audit; no ECG features supplied to model.'})
    print('PULSE_EXPERIMENT_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);run(p.parse_args().base)
