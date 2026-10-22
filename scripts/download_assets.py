"""Download optional large course assets, verify SHA256, and restore fixed paths."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true')
    p.add_argument('--keys', help='Comma-separated asset keys, e.g. initialization,satr,annotations,D')
    args = p.parse_args()
    catalog = json.loads((ROOT/'external_assets.json').read_text(encoding='utf-8'))
    items = {r['key']:r for r in catalog['assets']}
    if args.list or not args.keys:
        for key,r in items.items(): print(f'{key:16s} {r["bytes"]/1e6:9.2f} MB  {r["url"]}')
        if not args.keys: return
    keys = list(items) if args.keys == 'all' else list(dict.fromkeys(args.keys.split(',')))
    if any(k not in items for k in keys): p.error('Unknown asset key; use --list')
    cache = ROOT/'runs/download_cache'
    cache.mkdir(parents=True,exist_ok=True)
    for key in keys:
        item = items[key]
        local = cache/item['name'] if item['kind']=='zip' else ROOT/item['target']
        local.parent.mkdir(parents=True,exist_ok=True)
        if local.is_file():
            if digest(local)!=item['sha256']: raise ValueError('Existing file differs; preserve it before downloading: '+str(local))
            print('Already verified:',local)
        else:
            partial = local.with_name(local.name+'.part')
            print('Downloading:',item['name'],flush=True)
            request=urllib.request.Request(item['url'],headers={'User-Agent':'course-asset-downloader'})
            with urllib.request.urlopen(request,timeout=120) as response, partial.open('wb') as target:
                shutil.copyfileobj(response,target,4*1024*1024)
            if partial.stat().st_size!=item['bytes'] or digest(partial)!=item['sha256']:
                raise ValueError('Downloaded asset failed size/SHA256 verification: '+str(partial))
            partial.replace(local)
        if item['kind']=='zip':
            allowed={'annotations':'data/annotations/','predictions':'results/predictions/','satr':'checkpoints/satr_export_v2/'}[key]
            with zipfile.ZipFile(local) as z:
                for member in z.infolist():
                    if member.is_dir(): continue
                    dest=(ROOT/member.filename).resolve()
                    if (not member.filename.startswith(allowed) or not dest.is_relative_to(ROOT.resolve())
                            or stat.S_ISLNK(member.external_attr>>16)):
                        raise ValueError('Unsafe or unexpected archive member: '+member.filename)
                    if dest.exists():
                        with z.open(member) as f: expected=hashlib.file_digest(f,'sha256').hexdigest()
                        if digest(dest)!=expected: raise ValueError('Refusing to overwrite different file: '+str(dest))
                        continue
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(member) as source, dest.open('wb') as target: shutil.copyfileobj(source,target,4*1024*1024)
            print('Restored verified archive:',key,flush=True)
        else: print('Verified:',item['target'],flush=True)

if __name__=='__main__': main()
