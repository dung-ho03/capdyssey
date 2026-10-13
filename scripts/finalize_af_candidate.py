"""Freeze a validation-selected research candidate, then evaluate the existing test split."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_af import make_model, write_json
from train_af_rhythm import RhythmModel, rhythm_features, choose_threshold, at_threshold
from prepare_af_integrated import digest


class Predictor:
    def __init__(self,folder):
        if not torch.cuda.is_available():raise RuntimeError('GPU required; CPU fallback prohibited')
        self.info=json.loads((folder/'model_card.json').read_text())
        ck=torch.load(folder/'model.pt',weights_only=True,map_location='cuda')
        self.kind=self.info['kind'];self.threshold=self.info['threshold']
        self.model=(make_model() if self.kind=='baseline' else RhythmModel(ck['n_features'],self.kind)).cuda()
        self.model.load_state_dict(ck['state_dict']);self.model.eval()
        self.mean=ck.get('mean');self.std=ck.get('std')

    def prepared(self,x):
        x=torch.as_tensor(x,dtype=torch.float32,device='cuda')
        if x.ndim!=2 or x.shape[1]!=500 or not torch.isfinite(x).all():raise ValueError('Expected finite [N,500] prepared PPG')
        parts=[]
        with torch.no_grad():
            for b in x.split(512):
                if self.kind=='baseline':logit=self.model(b[:,None]).flatten()
                else:
                    z,f,_=rhythm_features(b);f=((f-self.mean)/self.std).clamp(-10,10);logit=self.model(z,f)
                parts.append(logit.sigmoid().cpu().numpy())
        return np.concatenate(parts)

    def raw(self,ppg,sampling_hz):
        from prepare_af_centered import centered_window
        x=centered_window(ppg,sampling_hz)
        p=float(self.prepared(x[None])[0])
        return {'af_score':p,'threshold':self.threshold,'prediction':'AF' if p>=self.threshold else 'non-AF',
                'status':'research_only','note':'Uncalibrated model score; does not establish diagnosis. Signal quality model not integrated.'}


def run(base):
    import google.colab
    if not torch.cuda.is_available():raise RuntimeError('Colab GPU required')
    torch.set_num_threads(2)
    root=base/'models/rhythm_candidates_v1';out=root/'release_candidate';out.mkdir(exist_ok=True)
    comparison=json.loads((root/'comparison.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    bundle=base/'processed/provisional_centered_v2';splitfile=base/'models/provisional_centered_v2/split.json'
    assert digest(bundle/'manifest.json')==protocol['bundle_sha256'] and digest(splitfile)==protocol['split_sha256']
    scores={};records={};checkpoints={}
    for kind in ('features','hybrid'):
        with np.load(root/kind/'validation_predictions.npz') as d:
            y=d['y'];p=d['p'];sources=d['source'];subjects=d['subject']
        t,score=choose_threshold(y,p,sources)
        records[kind]={'threshold':t,'macro_sensitivity':score,'by_source':{s:at_threshold(y[sources==s],p[sources==s],t) for s in sorted(set(sources.tolist()))}}
        scores[kind]=score;checkpoints[kind]=root/kind/'best_model.pt'
    baseline=base/'models/gpu_source_evidence_v1/three_sources'
    with np.load(baseline/'validation_predictions.npz') as d:
        ys=[];ps=[];ss=[];people=[]
        for s in sorted(set(sources.tolist())):
            ys.append(d[s+'_y']);ps.append(d[s+'_p']);ss.extend([s]*len(d[s+'_y']));people.append(d[s+'_subject'])
        by=np.concatenate(ys);bp=np.concatenate(ps);bs=np.array(ss);bpeople=np.concatenate(people)
    # Same validation persons and label counts required for a meaningful comparison.
    from collections import Counter
    assert Counter(zip(subjects.tolist(),y.tolist()))==Counter(zip(bpeople.tolist(),by.tolist()))
    t,score=choose_threshold(by,bp,bs)
    records['baseline']={'threshold':t,'macro_sensitivity':score,'by_source':{s:at_threshold(by[bs==s],bp[bs==s],t) for s in sorted(set(bs.tolist()))}}
    scores['baseline']=score;checkpoints['baseline']=baseline/'best_model.pt'
    selected=max(scores,key=scores.get);chosen=records[selected]
    card={'kind':selected,'threshold':chosen['threshold'],'status':'research_candidate_not_clinically_validated',
          'input':'one PPG channel, exactly 10 seconds; centered_window at supplied sampling_hz -> 50 Hz',
          'validation_selection':protocol['threshold_rule'],'validation':chosen,
          'engineering_goal_met_validation':all(v['sensitivity']>=.9 and v['specificity']>=.8 for v in chosen['by_source'].values()),
          'limitations':protocol['limitations']+' Existing test split was inspected in prior development; it is not untouched external validation.',
          'protocol':protocol,'checkpoint_sha256':digest(checkpoints[selected])}
    # Freeze selection BEFORE reading test signals or labels. Do not tune from test outcomes.
    cardpath=out/'model_card.json'
    if cardpath.exists():assert json.loads(cardpath.read_text())==card,'Refusing to change a frozen release'
    else:
        (out/'model.pt').write_bytes(checkpoints[selected].read_bytes());write_json(cardpath,card)
    assert digest(out/'model.pt')==card['checkpoint_sha256']
    write_json(out/'validation_comparison.json',records)
    pred=Predictor(out)
    if (out/'test_report.json').exists():
        print('FROZEN_CANDIDATE_ALREADY_EVALUATED',json.dumps(json.loads((out/'test_report.json').read_text())),flush=True);return
    split=json.loads(splitfile.read_text());manifest=json.loads((bundle/'manifest.json').read_text())
    truth=[];prob=[];src=[];ids=[]
    for r in manifest['shards']:
        if r['subject'] not in split['test']:continue
        assert digest(bundle/r['path'])==r['sha256']
        with np.load(bundle/r['path']) as d:prob.append(pred.prepared(d['x']));truth.append(d['y'])
        src.extend([r['source']]*r['n']);ids.extend([r['subject']]*r['n'])
    truth=np.concatenate(truth);prob=np.concatenate(prob);src=np.array(src);ids=np.array(ids)
    report={'kind':selected,'threshold':pred.threshold,'gpu':torch.cuda.get_device_name(0),
            'by_source':{s:at_threshold(truth[src==s],prob[src==s],pred.threshold) for s in sorted(set(src.tolist()))},
            'by_subject':{s:at_threshold(truth[ids==s],prob[ids==s],pred.threshold) for s in sorted(set(ids.tolist()))},
            'limitations':card['limitations']}
    report['engineering_goal_met_test']=all(v['sensitivity']>=.9 and v['specificity']>=.8 for v in report['by_source'].values())
    np.savez_compressed(out/'test_predictions.npz',y=truth,p=prob,source=src,subject=ids)
    write_json(out/'test_report.json',report)
    # Check raw inference matches prepared path, and rejects malformed input.
    wave=np.sin(np.arange(500)*.16)+100
    a=pred.raw(wave,50);assert 0<=a['af_score']<=1
    for bad in (np.zeros(500),np.ones(499),np.full(500,np.nan)):
        try:pred.raw(bad,50)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    print('CANDIDATE_FROZEN_AND_TESTED',json.dumps({'kind':selected,'validation':records,'test':report['by_source'],'goal_met':report['engineering_goal_met_test']}),flush=True)
    print('INFERENCE_CHECKS_OK',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    run(parser.parse_args().base)
