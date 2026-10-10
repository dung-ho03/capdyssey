"""Download selected participants from the pinned, noncommercial wrist AF v3 release."""
import argparse
import hashlib
import json
from pathlib import Path
import requests
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from download_data import safe_path

ROOT = Path(__file__).resolve().parents[1]

def ranged_download(item, partial):
    """Resume a large official file with four bounded, validated HTTP range requests."""
    size = item['bytes']
    start = partial.stat().st_size if partial.exists() else 0
    if start > size: raise ValueError('Partial file exceeds expected size')
    chunk_size = 8*1024*1024
    ranges = [(n,min(n+chunk_size,size)-1) for n in range(start,size,chunk_size)]
    def part(bounds):
        lo,hi=bounds
        target=partial.with_name(partial.name+f'.{lo}')
        if target.exists() and target.stat().st_size == hi-lo+1: return target
        for attempt in range(3):
            try:
                with requests.get(item['url'],headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(30,120)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {lo}-{hi}/{size}':
                        raise ValueError('Server did not honor requested byte range')
                    with target.open('wb') as stream:
                        for chunk in response.iter_content(1024*1024): stream.write(chunk)
                break
            except requests.RequestException:
                if attempt == 2: raise
                time.sleep(2**attempt)
        if target.stat().st_size != hi-lo+1: raise ValueError('Incomplete byte range')
        return target
    with ThreadPoolExecutor(max_workers=4) as pool:
        for target in pool.map(part,ranges):
            with partial.open('ab') as output, target.open('rb') as source: shutil.copyfileobj(source,output)
            target.unlink()
            print(item['path'],partial.stat().st_size,'/',size,flush=True)

def checksum(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def fetch(item, folder):
    path = safe_path(folder, item['path'])
    if path.exists() and checksum(path, 'md5') == item['md5']:
        print('Verified existing:', path.name, flush=True)
    else:
        partial = path.with_name(path.name+'.part')
        if item['bytes'] > 32*1024*1024:
            ranged_download(item,partial)
        else:
            with requests.get(item['url'],stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                with partial.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024): stream.write(chunk)
        if partial.stat().st_size != item['bytes'] or checksum(partial,'md5') != item['md5']:
            raise ValueError('Size/checksum mismatch: '+path.name)
        partial.replace(path)
        print('Downloaded and verified:',path.name,flush=True)
    return {'path':path.name,'bytes':path.stat().st_size,'md5':item['md5'], 'sha256':checksum(path,'sha256')}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', required=True, help='v3 IDs, e.g. 006 039; downloads BOTH PPG and ECG')
    parser.add_argument('--output',type=Path,default=ROOT/'data/raw/longterm_af_v3')
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'data/longterm_af_v3_manifest.json').read_text())
    ids=set(args.subjects)
    known={f['path'][:3] for f in manifest['files'] if f['path'].endswith('.mat')}
    if not ids <= known: parser.error('Unknown v3 subject ID')
    selected=[f for f in manifest['files'] if not f['path'].endswith('.mat') or f['path'][:3] in ids]
    print('License: CC BY-NC-SA 4.0; noncommercial research. Retain attribution and license.',flush=True)
    print('Download bytes:',sum(f['bytes'] for f in selected),flush=True)
    if args.plan:
        print('\n'.join(f['path'] for f in selected)); return
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for item in selected:
        results.append(fetch(item,args.output))
        (args.output/'download-status.json').write_text(json.dumps({'source':manifest['source'],'verified_files':results},indent=2))

if __name__=='__main__': main()
