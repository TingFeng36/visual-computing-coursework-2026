"""Verify the public baseline/evidence submission; SATR-SSM implementation is excluded."""
from pathlib import Path
import argparse, hashlib, json, re, zipfile

ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--downloaded-models',action='store_true');args=p.parse_args()
    required=['README.md','proposal.pdf','report.pdf','presentation.pptx','data/README.md',
              'checkpoints/README.md','checkpoints/checkpoint_manifest.json','external_assets.json',
              'logs/train.log','logs/val.log','scripts/train.sh','scripts/evaluate.sh','scripts/run_demo.sh',
              'scripts/download_assets.py','requirements.txt','LICENSE.txt']
    assert all((ROOT/x).is_file() for x in required),'Missing teacher-required document/entry'
    for folder in ['src','data/samples','data/splits','logs/tensorboard','results/figures','results/tables','results/predictions','results/comparisons','results/demo']:
        assert any(p.is_file() for p in (ROOT/folder).rglob('*')),folder
    with zipfile.ZipFile(ROOT/'presentation.pptx') as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)])==19
    epochs={}
    for g in 'ABCDEF':
        records=[json.loads(s) for s in (ROOT/f'logs/{g}/log.txt').read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in records]==list(range(100 if g in 'EF' else 20)),g
        metrics=json.loads((ROOT/f'results/experiments/{g}/final_validation/metrics.json').read_text())
        best=max(r[k][0] for r in records for k in ('test_coco_eval_bbox','ema_test_coco_eval_bbox'))
        assert abs(metrics['AP50_95']-best)<1e-4
        epochs[g]=len(records)
    assert not (ROOT/'src/original_ultralytics').exists()
    assert not (ROOT/'checkpoints/satr_export_v2').exists()
    assert not any('satr' in x.name.lower() for x in (ROOT/'src').rglob('*.py'))
    assert digest(ROOT/'logs/train.log')==digest(ROOT/'logs/F/train.log')
    events=list((ROOT/'logs/tensorboard').rglob('events.out.tfevents.*'));assert len(events)==6
    upstream=json.loads((ROOT/'src/rfdetr_baseline/upstream_manifest.json').read_text())
    for name,expected in upstream['files'].items():assert digest(ROOT/'src/rfdetr_baseline/vendor/rf_detr'/name)==expected,name
    catalog=json.loads((ROOT/'external_assets.json').read_text());assert catalog['publication_status']=='published_and_verified'
    assert not any(a['key']=='satr' for a in catalog['assets'])
    models=json.loads((ROOT/'checkpoints/checkpoint_manifest.json').read_text())
    external={x['key']:x for x in catalog['assets']}
    for m in models:
        assert external[m['group']]['sha256']==m['sha256'] and external[m['group']]['target']==m['path']
        if args.downloaded_models:assert digest(ROOT/m['path'])==m['sha256']
    checked=0
    if (ROOT/'SHA256SUMS.txt').exists():
        for row in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
            expected,name=row.split('  ',1);file=(ROOT/name).resolve()
            assert file.is_relative_to(ROOT.resolve()) and digest(file)==expected,name
            checked+=1
    print(json.dumps({'passed':True,'submission_profile':'under-100MB; models and full predictions/annotations online',
                      'slides':19,'epochs':epochs,'tensorboard_files':6,'upstream_files':len(upstream['files']),
                      'external_trained_models':len(models),'checked_files':checked},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
