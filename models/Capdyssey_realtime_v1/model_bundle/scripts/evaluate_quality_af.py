"""Fixed team quality gates + frozen AF model; Colab-only development evaluation."""
import argparse
import json
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly, butter, filtfilt
from scipy.io import loadmat
from scipy.special import expit
from but_quality_rf_runtime import load_bundle, extract_features, FEATURES
from but_quality_cnn_runtime import PPGQualityCNN
from finalize_two_sources import Predictor, ALLOWED
from prepare_af_centered import centered_window
from prepare_af_integrated import digest, atomic_json
from train_af import load_records


class QualityGates:
    def __init__(self, rf_path, cnn_dir):
        self.rf = load_bundle(rf_path)
        self.config = json.loads((cnn_dir/'config.json').read_text())
        c = self.config
        assert c['fs'] == 30 and c['n_samples'] == 300 and c['channel'] == 'PPG_R'
        assert c['bandpass_hz'] == [0.5, 5.0] and not c.get('smoke_test', True)
        assert np.isfinite([c['threshold'],c['platt']['slope'],c['platt']['intercept']]).all()
        assert c['platt']['slope'] > 0 and 0 <= c['threshold'] <= 1
        self.cnn = PPGQualityCNN().cuda().eval()
        self.cnn.load_state_dict(torch.load(cnn_dir/'best_model.pt',map_location='cuda',weights_only=True))
        self.thresholds = {'rf':float(self.rf['threshold']), 'cnn':float(c['threshold'])}

    def predict(self, raw, fs):
        features = pd.DataFrame([extract_features(x,fs) for x in raw],columns=FEATURES)
        rf = self.rf['model'].predict_proba(features)[:,1]
        ratio = Fraction(30, int(fs)); b,a = butter(3,[.5/15,5/15],btype='band')
        inputs=[]
        for x in raw:
            x=np.asarray(x,dtype=float)
            assert len(x)==fs*10 and np.isfinite(x).all()
            # Preserve original units before the CNN's trained flat-signal rule.
            z=resample_poly(x-x.mean(),ratio.numerator,ratio.denominator)+x.mean()
            assert z.shape==(300,)
            if z.std()<1e-3:z=np.zeros(300)
            else:
                z=filtfilt(b,a,z-z.mean());z=(z-z.mean())/(z.std()+1e-8)
            inputs.append(z)
        with torch.no_grad():
            logits=self.cnn(torch.tensor(np.asarray(inputs)[:,None,:],dtype=torch.float32,device='cuda')).cpu().numpy()
        c=self.config['platt'];cnn=expit(c['slope']*logits+c['intercept'])
        assert np.isfinite(rf).all() and np.isfinite(cnn).all()
        return rf,cnn


def summarize(y,p,accepted,threshold):
    y=np.asarray(y);p=np.asarray(p);accepted=np.asarray(accepted,dtype=bool)
    pos=y==1;neg=y==0;pred=p>=threshold
    tp=int((accepted&pos&pred).sum());fn=int((accepted&pos&~pred).sum())
    tn=int((accepted&neg&~pred).sum());fp=int((accepted&neg&pred).sum())
    def div(a,b):return float(a/b) if b else None
    return dict(n=len(y),accepted=int(accepted.sum()),coverage=div(accepted.sum(),len(y)),
        tp=tp,fn=fn,tn=tn,fp=fp,af_deferred=int((pos&~accepted).sum()),
        non_af_deferred=int((neg&~accepted).sum()),
        accepted_sensitivity=div(tp,tp+fn),accepted_specificity=div(tn,tn+fp),
        all_af_detection=div(tp,pos.sum()),all_non_af_false_alarm=div(fp,neg.sum()),
        af_deferral_rate=div((pos&~accepted).sum(),pos.sum()),
        non_af_deferral_rate=div((neg&~accepted).sum(),neg.sum()))


def run(base,rf_path,cnn_dir,out):
    import google.colab
    assert torch.cuda.is_available(),'Colab GPU required; no local/CPU fallback'
    bundle=base/'processed/provisional_centered_v2'
    split_path=base/'models/provisional_centered_v2/split.json'
    checkpoint=base/'models/main_two_sources_v1/candidate/model.pt'
    manifest=json.loads((bundle/'manifest.json').read_text())
    parent=json.loads((base/'processed/integrated_10s_resume_v1/manifest.json').read_text())
    parents={r['path']:r for r in parent['shards']}
    split=json.loads(split_path.read_text())
    for a,b in [('train','val'),('train','test'),('val','test')]:assert not set(split[a])&set(split[b])
    af=Predictor(checkpoint);q=QualityGates(rf_path,cnn_dir)
    assert af.ck['identity']['bundle']==digest(bundle/'manifest.json')
    assert af.ck['identity']['split']==digest(split_path)
    protocol=dict(af_sha256=digest(checkpoint),rf_sha256=digest(rf_path),
        cnn_sha256=digest(cnn_dir/'best_model.pt'),cnn_config_sha256=digest(cnn_dir/'config.json'),
        script_sha256=digest(Path(__file__)),bundle_sha256=digest(bundle/'manifest.json'),
        split_sha256=digest(split_path),af_threshold=af.threshold,quality_thresholds=q.thresholds,
        scope='Fixed gates, no tuning/retraining. Reused development test. Not clinical/external validation.',
        quality_scope='BUT HR-quality model transferred to other devices; no independent target quality labels.',
        wrist_included=False)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:atomic_json(out/'protocol.json',protocol)
    mimic={p:v for p,_,v in load_records(base/'raw/mimic_af')}
    frames=[]
    for row in manifest['shards']:
        if row['source'] not in ALLOWED:continue
        part=next((s for s in ['val','test'] if row['subject'] in split[s]),None)
        if part is None:continue
        assert digest(bundle/row['path'])==row['sha256']
        cache=out/(row['path']+'.csv')
        if cache.exists() and cache.with_suffix('.sha256').exists():
            assert digest(cache)==cache.with_suffix('.sha256').read_text()
            frames.append(pd.read_csv(cache));continue
        with np.load(bundle/row['path']) as d:x=d['x'];y=d['y'];starts=d['start_seconds']
        if row['source']=='mimic_perform_af':
            fs=125;values=mimic[row['record']]
            raw=[values[int(round(t*fs)):int(round(t*fs))+1250] for t in starts]
        else:
            fs=100;data=loadmat(base/'raw/six_rhythms'/(row['record']+'.mat'))
            indices=np.asarray(parents[row['path']]['provenance']['original_segment_indices'])[row['provenance']['original_window_indices']]
            raw=[data['ppgseg'][i] for i in indices]
        assert len(raw)==len(y)==row['n'] and np.isin(y,[0,1]).all()
        rf_scores=[];cnn_scores=[];af_scores=[]
        for start in range(0,len(raw),128):
            batch=raw[start:start+128]
            reproduced=np.stack([centered_window(v,fs) for v in batch])
            np.testing.assert_allclose(reproduced,x[start:start+len(batch)],atol=1e-5,rtol=1e-5)
            r,c=q.predict(batch,fs);rf_scores.extend(r);cnn_scores.extend(c)
            af_scores.extend(af.predict(x[start:start+len(batch)]))
        frame=pd.DataFrame(dict(source=row['source'],subject=row['subject'],record=row['record'],
            split=part,start_seconds=starts,y=y,p_af=af_scores,q_rf=rf_scores,q_cnn=cnn_scores))
        frame.to_csv(cache,index=False);cache.with_suffix('.sha256').write_text(digest(cache))
        frames.append(frame);print('QUALITY_AF',part,row['record'],len(frame),flush=True)
    all_rows=pd.concat(frames,ignore_index=True)
    reports=[]
    for (part,source),d in all_rows.groupby(['split','source']):
        for gate in ['none','rf','cnn']:
            accept=np.ones(len(d),bool) if gate=='none' else d['q_'+gate].to_numpy()>=q.thresholds[gate]
            reports.append(dict(split=part,source=source,gate=gate,**summarize(d.y,d.p_af,accept,af.threshold)))
    atomic_json(out/'report.json',dict(protocol=protocol,results=reports))
    pd.DataFrame(reports).to_csv(out/'comparison.csv',index=False)
    print('QUALITY_AF_COMPLETE',json.dumps(reports),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--rf',type=Path,required=True);p.add_argument('--cnn-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.base,a.rf,a.cnn_dir,a.out)
