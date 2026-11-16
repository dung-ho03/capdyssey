"""Colab-only frozen-model error audit. No relabeling or test-based tuning."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np


def run(base):
    import google.colab
    import h5py
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from train_af import make_model, metrics, write_json
    from prepare_af_integrated import chars,start_time,digest
    from prepare_af_provisional import strict_labels
    torch.set_num_threads(2)
    bundle=base/'processed/provisional_exact01_v1'
    modeldir=base/'models/provisional_exact01_v1'
    out=modeldir/'error_audit_v1';out.mkdir(exist_ok=True)
    manifest=json.loads((bundle/'manifest.json').read_text())
    split=json.loads((modeldir/'split.json').read_text())
    assert not (set(split['train'])&set(split['test']) or set(split['val'])&set(split['test']))
    saved=torch.load(modeldir/'best_model.pt',weights_only=True,map_location='cpu')
    assert saved['config']['bundle_manifest_sha256']==digest(bundle/'manifest.json')
    model=make_model();model.load_state_dict(saved['state_dict']);model.eval()
    records=[]; waves=[]
    for shard in manifest['shards']:
        if shard['subject'] not in split['test']:continue
        path=bundle/shard['path']
        assert digest(path)==shard['sha256']
        with np.load(path) as d:x=d['x'];y=d['y'];starts=d['start_seconds']
        with torch.no_grad():
            probs=torch.cat([torch.sigmoid(model(torch.from_numpy(x[i:i+256,None])).flatten()) for i in range(0,len(x),256)]).numpy()
        spectrum=np.abs(np.fft.rfft(x,axis=1))**2
        hf=spectrum[:,np.fft.rfftfreq(500,1/50)>8].sum(axis=1)/np.maximum(spectrum.sum(axis=1),1e-12)
        for i in range(len(y)):
            pred=int(probs[i]>=.5);truth=int(y[i]);kind=('TP' if pred else 'FN') if truth else ('FP' if pred else 'TN')
            records.append({'source':shard['source'],'subject':shard['subject'],'record':shard['record'],
                'shard':shard['path'],'index':i,'original_window_index':shard['provenance']['original_window_indices'][i],
                'start_seconds':float(starts[i]) if np.isfinite(starts[i]) else None,
                'label':truth,'probability':float(probs[i]),'outcome':kind,
                'high_frequency_fraction':float(hf[i]),'max_abs_z':float(np.abs(x[i]).max()),
                'flat_step_fraction':float((np.abs(np.diff(x[i]))<1e-6).mean())})
            waves.append(x[i])
    with (out/'test_window_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    sources={}
    original=json.loads((modeldir/'metrics.json').read_text())
    for source in sorted({r['source'] for r in records}):
        rows=[r for r in records if r['source']==source]
        stats=metrics(np.array([r['label'] for r in rows]),np.array([r['probability'] for r in rows]))
        assert all(stats[k]==original['test_by_source'][source][k] for k in ('n','tp','fn','tn','fp'))
        sources[source]=stats
    wrist=[r for r in records if r['source']=='longterm_wrist_v3']; people={}
    for person in sorted({r['subject'] for r in wrist}):
        rows=[r for r in wrist if r['subject']==person]
        stats=metrics(np.array([r['label'] for r in rows]),np.array([r['probability'] for r in rows]))
        stats['mean_probability_by_label']={str(label):float(np.mean([r['probability'] for r in rows if r['label']==label])) for label in (0,1) if any(r['label']==label for r in rows)}
        people[person]=stats
    quality={}
    for kind in ('TP','FN','TN','FP'):
        rows=[r for r in wrist if r['outcome']==kind]
        quality[kind]={'n':len(rows),'medians':{key:float(np.median([r[key] for r in rows])) for key in ('high_frequency_fraction','max_abs_z','flat_step_fraction')}}
    alignment={};cache={}
    for person in people:
        with h5py.File(base/'raw/longterm_af_v3'/(person.split(':')[1]+'_ECG.mat')) as f:
            start=start_time(chars(f['recording_startday']),chars(f['recording_starttime']))
            q=start+(f['QRSindex'][()].ravel().astype(float)-1)/500
            a=f['AF_annotation'][()].ravel().astype(float)
        cache[person]=(start,q,a)
        rows=[r for r in wrist if r['subject']==person];t=np.array([r['start_seconds'] for r in rows]);y=np.array([r['label'] for r in rows])
        exact=strict_labels(q,a,t);assert np.array_equal(exact,y)
        alignment[person]={'reproduced_labels':len(y),'shift_sensitivity':{}}
        for shift in (-10,-1,1,10):
            shifted=strict_labels(q,a,t+shift)
            alignment[person]['shift_sensitivity'][str(shift)]={'opposite_binary':int(((shifted>=0)&(shifted!=y)).sum()),'excluded':int((shifted<0).sum())}
    # Select random examples by outcome, not just visually dramatic failures.
    rng=np.random.default_rng(20261013); chosen=[]
    for kind in ('FN','FP','TP','TN'):
        indices=[i for i,r in enumerate(records) if r['source']=='longterm_wrist_v3' and r['outcome']==kind]
        chosen.extend(rng.choice(indices,min(2,len(indices)),replace=False).tolist())
    fig,axes=plt.subplots(len(chosen),2,figsize=(16,2.6*len(chosen)))
    cases=[]
    for rownum,i in enumerate(chosen):
        r=records[i];start,q,a=cache[r['subject']];t=r['start_seconds'];left=np.searchsorted(q,t);right=np.searchsorted(q,t+10)
        rr=np.diff(q[left:right]);case=dict(r)
        case['rr_seconds']=rr.tolist();case['rr_cv']=float(rr.std()/rr.mean()) if len(rr)>1 else None
        case['nearby_annotation_counts']={str(k):int(v) for k,v in Counter(a[max(0,left-2):min(len(a),right+1)]).items()}
        with h5py.File(base/'raw/longterm_af_v3'/(r['subject'].split(':')[1]+'_ECG.mat')) as f:
            ds=f['ECG'];lo=int(round((t-start)*500));hi=lo+5000
            if ds.ndim==1:ecg=ds[lo:hi]
            elif ds.shape[0]==1:ecg=ds[0,lo:hi]
            elif ds.shape[1]==1:ecg=ds[lo:hi,0]
            else:raise ValueError('Unexpected ECG dimensions')
        axes[rownum,0].plot(np.arange(500)/50,waves[i],lw=.8)
        axes[rownum,0].set_title(f"{r['subject']} {r['outcome']} provisional y={r['label']} p(AF)={r['probability']:.3f}; start={t:.0f}s")
        axes[rownum,1].plot(np.arange(len(ecg))/500,ecg,lw=.6)
        for peak in q[left:right]-t:axes[rownum,1].axvline(peak,color='red',alpha=.3,lw=.5)
        axes[rownum,1].set_title('Aligned raw ECG; red=provided QRS; visual audit only')
        for ax in axes[rownum]:ax.set_xlabel('Seconds');ax.grid(alpha=.2)
        cases.append(case)
    fig.tight_layout();fig.savefig(out/'ppg_ecg_examples.png',dpi=130);plt.close(fig)
    result={'model_epoch':saved['epoch'],'test_metrics_reproduced':True,'by_source':sources,
        'wrist_by_subject':people,'quality_proxies_by_outcome':quality,'alignment_reproduction':alignment,
        'random_example_cases':cases,'limitations':['Provisional 0/1 mapping is not independently verified.',
        'Reproducing the same mapping does not verify physical ECG/PPG synchronization.',
        'Quality proxies are descriptive, not a clinical signal-quality classifier.',
        'Test subjects now inspected; do not tune on these and call them untouched external validation.']}
    write_json(out/'report.json',result)
    print('METRICS_REPRODUCED',json.dumps(sources),flush=True)
    print('WRIST_BY_SUBJECT',json.dumps(people),flush=True)
    print('QUALITY_PROXIES',json.dumps(quality),flush=True)
    print('ALIGNMENT_CHECKS',json.dumps(alignment),flush=True)
    print('EXAMPLE_CASES',json.dumps(cases),flush=True)
    print('ERROR_AUDIT_COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    run(p.parse_args().base)
