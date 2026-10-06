"""Download licensed research datasets from their official distribution sites."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import threading
import time
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[1]
LOCAL = threading.local()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe file path: {name}')
    result = (root / relative).resolve()
    result.relative_to(root.resolve())
    return result

def download(url, path, expected):
    if path.is_file() and digest(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    partial = path.with_name(path.name + '.part')
    for attempt in range(3):
        try:
            with LOCAL.session.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with partial.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if digest(partial) != expected:
                raise ValueError(f'SHA256 mismatch: {path.name}')
            partial.replace(path)
            return
        except (requests.RequestException, OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

def entries(spec, folder):
    if 'files' in spec:
        return [(x['url'], safe_path(folder, x['path']), x['sha256']) for x in spec['files']]
    checksum_file = folder / 'SHA256SUMS.txt'
    download(spec['base_url'] + 'SHA256SUMS.txt', checksum_file, spec['checksums_sha256'])
    result = []
    for line in checksum_file.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        checksum, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        path = safe_path(folder, name)
        if spec.get('root_only') and '/' in name:
            continue
        if len(checksum) != 64 or any(c not in '0123456789abcdefABCDEF' for c in checksum):
            raise ValueError('Invalid SHA256 entry')
        result.append((spec['base_url'] + quote(name, safe='/'), path, checksum.lower()))
    return result

def main():
    catalog = json.loads((ROOT / 'data/catalog.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='*')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        parser.error('--workers must be between 1 and 32')
    if any(key not in catalog for key in args.datasets):
        parser.error('Unknown dataset; use --list')
    if args.list or not args.datasets:
        for key, value in catalog.items():
            print(f"{key}: {value['name']} [{value['license']}]")
        return
    for key in dict.fromkeys(args.datasets):
        spec = catalog[key]
        print(f"{key}: {spec['name']} | {spec['license']}", flush=True)
        print(spec['source'], flush=True)
        if args.plan:
            print('Explicit files' if 'files' in spec else 'Official checksum manifest; root only' if spec.get('root_only') else 'Official checksum manifest; all files')
            continue
        folder = ROOT / 'data/raw' / key
        files = entries(spec, folder)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(lambda item: download(*item), files)
            for index, _ in enumerate(results, 1):
                if index % 1000 == 0:
                    print(f'{key}: {index}/{len(files)} verified', flush=True)
        (folder / 'download-status.json').write_text(json.dumps({'dataset': key, 'files_verified': len(files), 'source': spec['source'], 'license': spec['license']}, indent=2), encoding='utf-8')
        print(f'{key}: complete, {len(files)} verified files', flush=True)

if __name__ == '__main__':
    main()
