"""Restore archived split/annotation definitions with locally available original images."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def attach(source, target, copy):
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coco-root', type=Path, help='COCO2017 root with train2017/, val2017/, annotations/')
    p.add_argument('--m3fd-root', type=Path, help='Original M3FD images/train + images/test, or already converted dataset')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--copy', action='store_true')
    args = p.parse_args()
    if args.output.resolve().is_relative_to((ROOT/'data').resolve()):
        raise ValueError('Choose a new external dataset folder, keeping submitted samples unchanged')
    manifests = ROOT/'data/manifests'
    if not (ROOT/'data/annotations/coco20_voc_5k/train.json').is_file():
        raise FileNotFoundError('First restore fixed annotations: python scripts/download_assets.py --keys annotations')
    if args.coco_root:
        for dataset in ['coco80_5k', 'coco20_voc_5k']:
            target = args.output/dataset
            for split in ['train', 'valid']:
                ann_file = ROOT/f'data/annotations/{dataset}/{split}.json'
                ann = json.loads(ann_file.read_text(encoding='utf-8'))
                ann_target = target/('annotations/instances_'+('train2017' if split == 'train' else 'val2017')+'.json' if dataset == 'coco80_5k' else split+'/_annotations.coco.json')
                ann_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ann_file, ann_target)
                original_split = 'train2017' if split == 'train' else 'val2017'
                dest_split = original_split if dataset == 'coco80_5k' else split
                for image in ann['images']:
                    source = args.coco_root/original_split/image['file_name']
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    attach(source, target/dest_split/image['file_name'], args.copy)
            if dataset == 'coco20_voc_5k':
                if not (target/'test').exists():
                    if args.copy:
                        shutil.copytree(target/'valid', target/'test')
                    else:
                        (target/'test').symlink_to(target.resolve()/'valid', target_is_directory=True)
                shutil.copy2(manifests/'coco20_voc_5k/dataset_manifest.json', target/'dataset_manifest.json')
            print('Prepared', dataset, flush=True)
    if args.m3fd_root:
        manifest = json.loads((manifests/'m3fd/dataset_manifest.json').read_text(encoding='utf-8'))
        target = args.output/'m3fd'
        for split in ['train', 'valid', 'test']:
            ann_file = ROOT/f'data/annotations/m3fd/{split}.json'
            ann = json.loads(ann_file.read_text(encoding='utf-8'))
            dest = target/split
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ann_file, dest/'_annotations.coco.json')
            for image in ann['images']:
                original = Path(manifest['splits'][split]['source_images'][image['id']-1])
                source_split = original.parent.name
                candidates = [args.m3fd_root/split/image['file_name'],
                              args.m3fd_root/'images'/source_split/original.name,
                              args.m3fd_root/source_split/original.name]
                source = next((s for s in candidates if s.is_file()), None)
                if source is None:
                    raise FileNotFoundError(str(candidates))
                attach(source, dest/image['file_name'], args.copy)
        shutil.copy2(manifests/'m3fd/dataset_manifest.json', target/'dataset_manifest.json')
        print('Prepared M3FD; test is an alias of validation, not an independent test', flush=True)


if __name__ == '__main__':
    main()
