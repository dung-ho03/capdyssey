"""Colab read-only raw inventory and wrist timing audit; never assign rhythm labels."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def chars(d):
    if d.attrs.get('MATLAB_empty',0):return ''
    return ''.join(chr(int(v)) for v in d[()].ravel()).strip()


def timestamp(day,time):
    if not day or not time:return None
    h,m,s=map(int,time.split(':'))
    if int(day)<1 or not (0<=h<24 and 0<=m<60 and 0<=s<60):raise ValueError('Invalid timestamp')
    return (int(day)-1)*86400+h*3600+m*60+s


def small_metadata(f):
    result={}
    def walk(name,obj):
        if name.startswith('#refs#') or not isinstance(obj,h5py.Dataset):return
        if obj.size<=128 and h5py.check_dtype(ref=obj.dtype) is None:
            result[name]={'shape':list(obj.shape),'value':str(obj[()]),'attrs':{k:str(v) for k,v in obj.attrs.items()}}
    f.visititems(walk)
    return result


def run(base):
    import google.colab
    raw=base/'raw';wrist=raw/'longterm_af_v3'
    if not wrist.is_dir():raise RuntimeError('Drive raw directory unavailable; mount Drive first')
    out=base/'audits/preprocessing_step1';out.mkdir(parents=True,exist_ok=True)
    rows=[];errors=[]
    expected={f'{p:03d}_{kind}.mat' for p in range(1,40) for kind in ('ECG','PPG')}
    present={p.name for p in wrist.glob('*.mat')}
    inventory={name:len(list((raw/name).glob('*.mat'))) for name in ('mimic_af','six_rhythms','longterm_af_v3')}
    for person in range(1,40):
        pid=f'{person:03d}';r={'subject':pid}
        try:
            with h5py.File(wrist/f'{pid}_ECG.mat') as e,h5py.File(wrist/f'{pid}_PPG.mat') as p:
                r['ecg_metadata']=small_metadata(e);r['ppg_metadata']=small_metadata(p)
                a=e['AF_annotation'][()].ravel();q=e['QRSindex'][()].ravel().astype(float);rr=e['rr'][()].ravel().astype(float)
                r['lengths']={'annotations':len(a),'qrs':len(q),'rr':len(rr)}
                r['length_relation_ok']=len(q)==len(a)+1==len(rr)+1
                r['all_finite']=bool(np.isfinite(a).all() and np.isfinite(q).all() and np.isfinite(rr).all())
                r['qrs_monotonic']=bool(np.all(np.diff(q)>0));r['ecg_samples']=int(e['ECG'].size)
                r['qrs_in_bounds_one_based']=bool(len(q) and np.min(q)>=1 and np.max(q)<=e['ECG'].size)
                values,counts=np.unique(a,return_counts=True)
                r['raw_annotation_counts']={str(v):int(n) for v,n in zip(values,counts)}
                if r['length_relation_ok'] and r['all_finite'] and np.all(rr>0):
                    r['qrs_units_per_rr_unit_median']=float(np.median(np.diff(q)/rr))
                    r['rr_error_assuming_seconds']={str(fs):float(np.max(np.abs(np.diff(q)/fs-rr))) for fs in (100,500)}
                r['ecg_start']=timestamp(chars(e['recording_startday']),chars(e['recording_starttime']))
                records=[]
                for i,ref in enumerate(p['PPG_GREEN'][()].ravel()):
                    d=p[ref];day=chars(p[p['recording_startday'][()].ravel()[i]]);time=chars(p[p['recording_starttime'][()].ravel()[i]])
                    start=timestamp(day,time)
                    records.append({'index':i,'samples':int(d.size),'start':start,'has_timestamp':start is not None})
                r['ppg_records']=records
                # Duration/overlap is deliberately conditional, not proof of synchronization.
                if r['ecg_start'] is not None:
                    end=r['ecg_start']+r['ecg_samples']/500
                    for rec in records:
                        rec['overlap_seconds_if_ecg500_ppg100']=None if rec['start'] is None else max(0.,min(end,rec['start']+rec['samples']/100)-max(r['ecg_start'],rec['start']))
        except Exception as exc:
            r['read_error']=f'{type(exc).__name__}: {exc}';errors.append(pid)
        rows.append(r)
        (out/f'{pid}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print('원본 점검',pid,'읽기 오류' if 'read_error' in r else '기록 완료',flush=True)
    report={'inventory':inventory,'missing_wrist_files':sorted(expected-present),'unexpected_wrist_files':sorted(present-expected),
            'read_errors':errors,'subjects':rows,'training_authorized_by_this_audit':False,
            'sampling_rates_status':'500/100 are hypotheses to cross-check against metadata and documentation; RR agreement alone is not proof.',
            'scope':'Read-only inventory, metadata and interval structure. No waveform-quality scan, checksum verification, clinical label confirmation or physical synchronization proof.'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    summary={'files':inventory,'missing_files':len(expected-present),'read_errors':errors,
             'length_failures':[r['subject'] for r in rows if not r.get('length_relation_ok',False)],
             'qrs_bounds_failures':[r['subject'] for r in rows if not r.get('qrs_in_bounds_one_based',False)],
             'qrs_rr_ratios':sorted(set(round(r['qrs_units_per_rr_unit_median'],6) for r in rows if 'qrs_units_per_rr_unit_median' in r)),
             'report':str(out/'report.json'),'training_started':False}
    print('1단계 검사 결과',json.dumps(summary,ensure_ascii=False),flush=True)
    if expected-present or errors:raise RuntimeError('원본 누락/읽기 실패: 보고서 확인 후 진행')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    run(parser.parse_args().base)
