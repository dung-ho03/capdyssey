"""Prepare three sources as 10-second shards; unresolved wrist labels remain -1."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat
from train_af import load_records, preprocess_window, write_json, ROOT

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def normalized_windows(values, fs):
    """Never join records. Drop incomplete tail; report basic invalid windows."""
    values=np.asarray(values)
    if values.ndim!=1: raise ValueError('Expected one continuous channel')
    width=fs*10
    for start in range(0,len(values)-width+1,width):
        try: x=preprocess_window(values[start:start+width],fs,seconds=10)
        except ValueError: x=None
        yield start/fs,x

def chars(dataset):
    return ''.join(chr(int(v)) for v in dataset[()].ravel()).strip()

def start_time(day,time):
    h,m,s=map(int,time.split(':'))
    return (int(day)-1)*86400+h*3600+m*60+s

def prepare(mimic,six,wrist,output):
    # Refuse overwrite: keep a prior experiment reproducible.
    output.mkdir(parents=True,exist_ok=False)
    manifest={'schema':1,'window_seconds':10,'sampling_hz':50,'shards':[],
        'labels':{'-1':'unresolved; excluded from supervised training','0':'non-AF','1':'AF'},
        'quality_filter':'finite and non-flat only; no clinical signal-quality model',
        'three_dataset_training_ready':False}
    def save(source,person,record,windows,labels,starts,excluded,provenance):
        if not windows: raise ValueError('No valid windows: '+record)
        filename=f'{source}_{record}.npz'
        np.savez_compressed(output/filename,x=np.stack(windows),y=np.asarray(labels,dtype=np.int8),
                            start_seconds=np.asarray(starts,dtype=np.float64))
        entry={'source':source,'subject':source+':'+person,'record':record,'path':filename,
            'sha256':digest(output/filename),'n':len(windows),'basic_exclusions':excluded,
            'label_counts':{str(k):int(np.sum(np.asarray(labels)==k)) for k in (-1,0,1)},
            'provenance':provenance}
        manifest['shards'].append(entry)
        print(source,record,entry['label_counts'],flush=True)
    for person,label,values in load_records(mimic):
        windows=[]; starts=[]; bad=0
        for start,x in normalized_windows(values,125):
            if x is None: bad+=1;continue
            windows.append(x);starts.append(start)
        save('mimic_perform_af',person,person,windows,[label]*len(windows),starts,bad,
             {'label_scope':'source-record AF status','sampling_hz':125})
    inventory=json.loads((ROOT/'docs/verification-af-stage1-2026-10-10.json').read_text())['six_rhythms']
    for item in inventory:
        path=six/(item['subject']+'.mat')
        if digest(path)!=item['sha256']:raise ValueError('Source checksum mismatch: '+path.name)
        d=loadmat(path); labels=d['labels'].ravel(); values=d['ppgseg']
        if values.shape!=(len(labels),1000) or not np.isin(labels,range(6)).all():raise ValueError('Unexpected six-rhythm schema')
        windows=[]; kept=[]; indexes=[]; bad=0
        for index,(raw,label) in enumerate(zip(values,labels)):
            try:x=preprocess_window(raw,100,seconds=10)
            except ValueError:bad+=1;continue
            windows.append(x);kept.append(int(label==5));indexes.append(index)
        save('six_rhythms',item['subject'],item['subject'],windows,kept,[np.nan]*len(windows),bad,
             {'source_sha256':item['sha256'],'original_split':item['original_split'],
              'original_segment_indices':indexes,'original_rhythm_codes':[int(labels[i]) for i in indexes],
              'timestamps':'unknown; do not assume adjacent rows are continuous'})
    available=sorted(wrist.glob('*_PPG.mat'))
    if not available:raise ValueError('No wrist PPG files found')
    official={f['path']:f for f in json.loads((ROOT/'data/longterm_af_v3_manifest.json').read_text())['files']}
    from download_longterm_af import checksum
    for path in available:
        person=path.name[:3]; ecgpath=wrist/f'{person}_ECG.mat'
        for p in (path,ecgpath):
            if checksum(p,'md5')!=official[p.name]['md5']:raise ValueError('Wrist MD5 mismatch')
        with h5py.File(ecgpath) as e,h5py.File(path) as p:
            ecgstart=start_time(chars(e['recording_startday']),chars(e['recording_starttime']))
            ecgend=ecgstart+e['ECG'].size/500
            for i,ref in enumerate(p['PPG_GREEN'][()].ravel()):
                begin=start_time(chars(p[p['recording_startday'][()].ravel()[i]]),chars(p[p['recording_starttime'][()].ravel()[i]]))
                windows=[];starts=[];bad=0;outside=0
                for offset,x in normalized_windows(p[ref][()].ravel(),100):
                    if begin+offset<ecgstart or begin+offset+10>ecgend:outside+=1;continue
                    if x is None:bad+=1;continue
                    windows.append(x);starts.append(begin+offset)
                if not windows:continue
                save('longterm_wrist_v3',person,f'{person}_{i:03d}',windows,[-1]*len(windows),starts,bad,
                     {'ecg_overlap':'nominal header clock only','outside_ecg_windows':outside,
                      'labels':'No numeric AF annotation mapped. Codebook and interval alignment unresolved.'})
    manifest['sources']=sorted({s['source'] for s in manifest['shards']})
    manifest['readiness']=json.loads((ROOT/'data/af_integration_status.json').read_text())
    write_json(output/'manifest.json',manifest)
    print('Preparation complete; wrist windows remain UNLABELED. Not a training result.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mimic',type=Path,required=True)
    p.add_argument('--six',type=Path,required=True)
    p.add_argument('--wrist',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.mimic,a.six,a.wrist,a.output)
