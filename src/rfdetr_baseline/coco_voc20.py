"""Paired COCO 80/20-class experiments with identical sampled images.

This is a custom VOC-overlap task built from COCO, not the Pascal VOC dataset.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

from common import sha256, write_json
from coco_tools import category_names
from data_tools import validate_dataset

# Ascending original COCO IDs, with names kept in COCO spelling.
VOC20_IDS = (1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72)


def select_images(images, count, seed):
    if count < 1 or count > len(images):
        raise ValueError(f"Requested {count} images from only {len(images)}")
    if len({item['id'] for item in images}) != len(images):
        raise ValueError("Duplicate image IDs")
    ordered = sorted(images, key=lambda item: (
        hashlib.sha256(f"{seed}:{item['id']}".encode()).digest(), item['id']))
    return sorted(ordered[:count], key=lambda item: item['id'])


def filter_annotations(data, images, mapping=None):
    result = copy.deepcopy({key: value for key, value in data.items()
                            if key not in ('images', 'annotations', 'categories')})
    ids = {item['id'] for item in images}
    result['images'] = copy.deepcopy(images)
    result['annotations'] = []
    for original in data['annotations']:
        if original['image_id'] not in ids:
            continue
        if mapping is not None and original['category_id'] not in mapping:
            continue
        item = copy.deepcopy(original)
        if mapping is not None:
            item['category_id'] = mapping[item['category_id']]
        result['annotations'].append(item)
    if mapping is None:
        result['categories'] = copy.deepcopy(data['categories'])
    else:
        by_id = {item['id']: item for item in data['categories']}
        result['categories'] = [{'id': dense, 'name': by_id[original]['name'],
                                 'supercategory': 'object'} for original, dense in mapping.items()]
    return result


def materialize(source, destination, copy_images):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def prepare(source, output, train_images=5000, seed=0, copy_images=True,
            portable=False, require_full_source=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be new/empty; existing experiments are preserved")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Choose an output separate from the original COCO directory")
    names = category_names()
    mapping = {original: dense for dense, original in enumerate(VOC20_IDS, 1)}
    selected, provenance = {}, {'seed': seed, 'splits': {}, 'source': str(source)}
    datasets = {}
    for split, count in [('train', 118287), ('val', 5000)]:
        annotation = source / 'annotations' / f'instances_{split}2017.json'
        data = json.loads(annotation.read_text(encoding='utf-8'))
        if len(data['categories']) != 80 or {c['id']: c['name'] for c in data['categories']} != names:
            raise ValueError("Source must preserve all original COCO categories and IDs")
        if require_full_source and len(data['images']) != count:
            raise ValueError(f"Source {split} must contain {count} images")
        if len({item['id'] for item in data['images']}) != len(data['images']):
            raise ValueError("Source contains duplicate image IDs")
        known = {image['id'] for image in data['images']}
        if any(a['image_id'] not in known or a['category_id'] not in names for a in data['annotations']):
            raise ValueError("Source contains invalid annotation references")
        images = select_images(data['images'], train_images, seed) if split == 'train' else sorted(data['images'], key=lambda x: x['id'])
        selected[split] = images
        datasets[split] = (filter_annotations(data, images), filter_annotations(data, images, mapping))
        counts = Counter(a['category_id'] for a in datasets[split][1]['annotations'] if not a.get('iscrowd', 0))
        missing = [names[original] for original, dense in mapping.items() if not counts[dense]]
        if missing:
            raise ValueError(f"{split}: selected images lack non-crowd examples of {missing}; increase sample size")
        provenance['splits'][split] = {'annotation_sha256': sha256(annotation),
            'image_ids': [item['id'] for item in images],
            'objects80': len(datasets[split][0]['annotations']),
            'objects20': len(datasets[split][1]['annotations']),
            'noncrowd_objects_per_dense_category': dict(counts)}
    if {i['id'] for i in selected['train']} & {i['id'] for i in selected['val']}:
        raise ValueError("Source train/validation image IDs overlap")
    native, custom = output / 'coco80_5k', output / 'coco20_voc_5k'
    manifest = {'dataset_type': 'custom_COCO_VOC_overlap20',
                'class_names': [names[i] for i in VOC20_IDS],
                'source_category_ids': list(VOC20_IDS),
                'dense_to_original': {str(d): o for o, d in mapping.items()},
                'class_mapping': {str(d-1): d for d in mapping.values()},
                'test_is_validation_alias': True, 'splits': {}}
    for source_split, target_splits in [('train', ['train']), ('val', ['valid', 'test'])]:
        original_data, filtered = datasets[source_split]
        folder = f'{source_split}2017'
        for index, image in enumerate(selected[source_split], 1):
            relative = Path(image['file_name'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError("Unsafe image file name")
            origin = source / folder / relative
            if not origin.is_file():
                raise FileNotFoundError(origin)
            materialize(origin, native / folder / relative, copy_images)
            for split in target_splits:
                materialize(native / folder / relative, custom / split / relative, portable)
            if index % 500 == 0 or index == len(selected[source_split]):
                print(f"[paired subset] {source_split}: {index}/{len(selected[source_split])}", flush=True)
        write_json(native / 'annotations' / f'instances_{source_split}2017.json', original_data)
        for split in target_splits:
            annotation = custom / split / '_annotations.coco.json'
            write_json(annotation, filtered)
            manifest['splits'][split] = {'annotation_sha256': sha256(annotation),
                'source_images': [str(source / folder / image['file_name']) for image in selected[source_split]]}
    write_json(custom / 'dataset_manifest.json', manifest)
    provenance.update(source_category_ids=list(VOC20_IDS), dense_to_original=manifest['dense_to_original'],
                      validation='all original val2017 images; no independent test',
                      sampling='SHA256(seed:image_id); same image IDs for 80/20-class tasks',
                      excluded_objects='60 non-target categories are background for the 20-class task',
                      copied_source_images=copy_images, portable=portable,
                      roots={'coco80': str(native), 'coco20': str(custom)})
    write_json(output / 'paired_subset_manifest.json', provenance)
    audit = validate_dataset(custom, verify_pixels=True)
    print('Paired subsets prepared. Validated custom task:', audit['passed'], flush=True)
    return provenance


def remap_predictions(predictions, label_space):
    if label_space not in ('original', 'voc20'):
        raise ValueError("label_space must be explicit: original or voc20")
    mapping = dict(enumerate(VOC20_IDS, 1))
    result = []
    for prediction in predictions:
        category = prediction['category_id']
        if category == 0:
            continue  # index 0 is reserved by this RF-DETR integration
        if label_space == 'voc20':
            if category not in mapping:
                raise ValueError(f"Prediction has invalid dense category {category}")
            category = mapping[category]
        if category in VOC20_IDS:
            item = dict(prediction)
            item['category_id'] = category
            result.append(item)
    return result


def score(source, predictions, label_space, output):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    output = Path(output)
    if output.exists():
        raise FileExistsError("Do not overwrite an existing score file")
    annotation = Path(source) / 'annotations/instances_val2017.json'
    gt = COCO(str(annotation))
    if len(gt.imgs) != 5000:
        raise ValueError("Common evaluation requires all 5000 original val2017 images")
    records = remap_predictions(json.loads(Path(predictions).read_text()), label_space)
    if any(p['image_id'] not in gt.imgs for p in records):
        raise ValueError("Predictions contain images outside val2017")
    if records:
        dt = gt.loadRes(records)
    else:
        dt = COCO()
        dt.dataset = {'images': list(gt.imgs.values()), 'categories': list(gt.cats.values()), 'annotations': []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.imgIds = sorted(gt.imgs)
    evaluator.params.catIds = list(VOC20_IDS)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    result = {'evaluation': 'custom COCO VOC-overlap20 on original full val2017',
        'label_space': label_space, 'category_ids': list(VOC20_IDS), 'images': 5000,
        'AP50_95': float(evaluator.stats[0]), 'AP50': float(evaluator.stats[1]),
        'AP75': float(evaluator.stats[2]), 'COCO_stats': evaluator.stats.tolist(),
        'annotation_sha256': sha256(annotation), 'predictions_sha256': sha256(predictions),
        'warning': 'Missing-image predictions count as no detections; use complete evaluate.py exports'}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--source', required=True)
    prepare_parser.add_argument('--output', required=True)
    prepare_parser.add_argument('--train-images', type=int, default=5000)
    prepare_parser.add_argument('--seed', type=int, default=0)
    prepare_parser.add_argument('--link-source', action='store_true', help='Keep reads on fs instead of copying a small local image cache')
    score_parser = commands.add_parser('score')
    score_parser.add_argument('--source', required=True)
    score_parser.add_argument('--predictions', required=True)
    score_parser.add_argument('--label-space', choices=['original', 'voc20'], required=True)
    score_parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.source, args.output, args.train_images, args.seed, not args.link_source)
        print(json.dumps({'roots': result['roots'], 'source_category_ids': result['source_category_ids']}, indent=2))
    else:
        print(json.dumps(score(args.source, args.predictions, args.label_space, args.output), indent=2))


if __name__ == '__main__':
    main()
