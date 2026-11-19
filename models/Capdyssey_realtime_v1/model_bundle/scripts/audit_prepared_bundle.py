"""Colab-only prepared data integrity, split and provisional label reconstruction."""
import argparse
import json
from pathlib import Path
import numpy as np
import h5py
from prepare_af_integrated import digest, chars, start_time, atomic_json
from prepare_af_provisional import strict_labels


def run(base):
    import google.colab
    bundle=base/'processed/provisional_centered_v2'
    m=json.loads((bundle/'manifest.json').read_text())
    parent=json.loads((base/'processed/integrated_10s_resume_v1/manifest.json').read_text())
    original_shards={r['path']:r for r in parent['shards']}
    split=json.loads((base/'models/provisional_centered_v2/split.json').read_text())
    groups=[set(split[k]) for k in ('train','val','test')]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]),'Subject split leakage'
    raw_report=json.loads((base/'audits/preprocessing_step1/report.json').read_text())
    assert len(raw_report['subjects'])==39 and not raw_report['missing_wrist_files'] and not raw_report['read_errors']
    by_person={r['subject']:r for r in raw_report['subjects']}
    out=base/'audits/prepared_bundle';out.mkdir(parents=True,exist_ok=True)
    errors=[];counts={};cache={};seen=set();total=0
    for i,r in enumerate(m['shards']):
        try:
            assert r['path'] not in seen,'Duplicate shard';seen.add(r['path'])
            path=(bundle/r['path']).resolve();assert path.parent==bundle.resolve()
            assert digest(path)==r['sha256'],'Shard hash mismatch'
            assert sum(r['subject'] in g for g in groups)==1,'Missing/nonunique split membership'
            with np.load(path,allow_pickle=False) as d:x=d['x'];y=d['y'];starts=d['start_seconds']
            assert x.shape==(r['n'],500) and y.shape==starts.shape==(r['n'],),'Shape mismatch'
            assert np.isfinite(x).all() and np.isin(y,[0,1]).all(),'Invalid samples/labels'
            if r['source']=='six_rhythms':
                # Independent source snippets have no continuous clock. Do not invent one.
                from scipy.io import loadmat
                assert np.isnan(starts).all(),'Six-rhythm snippets must not fabricate timestamps'
                original=original_shards[r['path']]
                rawpath=base/'raw/six_rhythms'/(r['record']+'.mat')
                assert digest(rawpath)==original['provenance']['source_sha256'],'Six-rhythm raw hash mismatch'
                selected=np.asarray(r['provenance']['original_window_indices'],dtype=int)
                indices=np.asarray(original['provenance']['original_segment_indices'],dtype=int)[selected]
                labels=loadmat(rawpath,variable_names=['labels'])['labels'].ravel()
                assert np.isin(labels,range(6)).all() and np.all(indices>=0) and np.all(indices<len(labels)),'Invalid source rhythm/index'
                assert np.array_equal(y,(labels[indices]==5).astype(np.int8)),'Six-rhythm source label mismatch'
            else:
                assert np.isfinite(starts).all(),'Missing timestamp in continuous recording'
            assert np.max(np.abs(x.mean(1)))<1e-4 and np.max(np.abs(x.std(1)-1))<1e-4,'Normalization mismatch'
            if r['source']=='longterm_wrist_v3':
                person=r['subject'].split(':')[1];a=by_person[person]
                assert a['length_relation_ok'] and a['all_finite'] and a['qrs_monotonic'] and a['qrs_in_bounds_one_based'],'Invalid ECG structure admitted'
                assert a['rr_error_assuming_seconds']['500']<1e-6,'500 Hz and RR mismatch'
                if person not in cache:
                    with h5py.File(base/'raw/longterm_af_v3'/f'{person}_ECG.mat') as e:
                        begin=start_time(chars(e['recording_startday']),chars(e['recording_starttime']))
                        cache[person]=(begin+(e['QRSindex'][()].ravel().astype(float)-1)/500,e['AF_annotation'][()].ravel())
                reconstructed=strict_labels(*cache[person],starts)
                assert np.array_equal(y,reconstructed),'Stored labels differ from raw annotation reconstruction'
                rec=a['ppg_records'][int(r['record'].split('_')[1])]
                assert rec['start'] is not None,'Missing PPG timestamp admitted'
                assert np.all(starts>=rec['start']-1e-6) and np.all(starts+10<=rec['start']+rec['samples']/100+1e-6),'PPG record boundary crossed'
                assert np.all(starts>=a['ecg_start']-1e-6) and np.all(starts+10<=a['ecg_start']+a['ecg_samples']/500+1e-6),'ECG record boundary crossed'
            c=counts.setdefault(r['source'],{'windows':0,'af':0,'non_af':0,'subjects':set()})
            c['windows']+=len(y);c['af']+=int((y==1).sum());c['non_af']+=int((y==0).sum());c['subjects'].add(r['subject']);total+=len(y)
        except Exception as exc:errors.append({'shard':r['path'],'error':f'{type(exc).__name__}: {exc}'})
        if (i+1)%50==0:print('전처리 검사',i+1,'/',len(m['shards']),'오류',len(errors),flush=True)
    for c in counts.values():c['subjects']=sorted(c['subjects'])
    report={'bundle_sha256':digest(bundle/'manifest.json'),'checked_shards':len(seen),'passed_windows':total,'sources':counts,'errors':errors,
            'structural_checks_passed':not errors,'official_wrist_label_mapping_confirmed':False,
            'limitations':'Exact binary provisional labels reconstructed, not clinically confirmed. No raw full-file hash refresh or full waveform reconstruction in this step. Nominal clock overlap is not proof of physical alignment.'}
    atomic_json(out/'report.json',report)
    print('전처리 검사 결과',json.dumps({k:v for k,v in report.items() if k!='sources'},ensure_ascii=False),flush=True)
    print('자료별 수량',json.dumps({s:{**c,'subjects':len(c['subjects'])} for s,c in counts.items()},ensure_ascii=False),flush=True)
    if errors:raise RuntimeError('전처리 검사 실패; 학습 재개 금지')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);run(p.parse_args().base)
