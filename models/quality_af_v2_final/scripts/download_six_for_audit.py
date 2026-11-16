"""Acquire the previously audited public six-rhythm files; no training permission claim."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
ROOT=Path(__file__).resolve().parents[1]

def fetch(item,folder):
    name=item['subject']+'.mat'
    if Path(name).name!=name:raise ValueError('Invalid filename')
    path=folder/name
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    if path.exists() and sha(path)==item['sha256']:return
    url='https://raw.githubusercontent.com/zdzdliu/PPGArrhythmiaDetection/main/valid_testDataset/'+name
    part=path.with_suffix('.mat.part')
    with requests.get(url,stream=True,timeout=(30,120)) as response:
        response.raise_for_status()
        with part.open('wb') as f:
            for block in response.iter_content(1024*1024):f.write(block)
    if sha(part)!=item['sha256']:raise ValueError('Source changed or corrupt: '+name)
    part.replace(path);print('Verified',name,flush=True)

if __name__=='__main__':
    folder=ROOT/'data/raw/six_rhythms';folder.mkdir(parents=True,exist_ok=True)
    inventory=json.loads((ROOT/'docs/verification-af-stage1-2026-10-10.json').read_text())['six_rhythms']
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda item:fetch(item,folder),inventory))
    print('91 public files verified for preparation; research-use scope remains documented separately.')
