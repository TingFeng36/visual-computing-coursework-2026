import copy
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image

from coco_tools import category_names, inspect_coco
from coco_voc20 import VOC20_IDS, prepare, select_images, remap_predictions, score
from category_initialization import initialize_category_rows
from test_data import test_workspace


def fixture(root):
    names = category_names()
    categories = [{'id': i, 'name': name} for i, name in names.items()]
    for split, offset in [('train', 0), ('val', 1000)]:
        images, annotations = [], []
        for i in (1, 2):
            image_id = offset + i
            name = f'{image_id:012d}.jpg'
            path = root / f'{split}2017' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (32, 32)).save(path)
            images.append({'id': image_id, 'file_name': name, 'width': 32, 'height': 32})
            classes = VOC20_IDS if i == 1 else [90]
            for category in classes:
                annotations.append({'id': len(annotations)+1, 'image_id': image_id,
                    'category_id': category, 'bbox': [2, 3, 5, 7], 'area': 35, 'iscrowd': 0})
        path = root / 'annotations' / f'instances_{split}2017.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'info': {}, 'images': images, 'annotations': annotations, 'categories': categories}))


class Voc20Tests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('pycocotools'), 'COCO API is installed on AutoDL, absent in this local Python')
    def test_common_evaluation_original_and_dense_predictions_agree(self):
        with test_workspace() as temp:
            source = temp/'source'
            fixture(source)
            annotation = source/'annotations/instances_val2017.json'
            data = json.loads(annotation.read_text())
            data['images'].extend({'id': i, 'file_name': f'{i:012d}.jpg', 'width': 32, 'height': 32} for i in range(1003, 6001))
            annotation.write_text(json.dumps(data))
            original = [{'image_id': 1001, 'category_id': i, 'bbox': [2, 3, 5, 7], 'score': .9} for i in VOC20_IDS]
            dense = [dict(item, category_id=index) for index, item in enumerate(original, 1)]
            for space, predictions in [('original', original), ('voc20', dense)]:
                path = temp/f'{space}_predictions.json'
                path.write_text(json.dumps(predictions))
                result = score(source, path, space, temp/f'{space}_metrics.json')
                self.assertAlmostEqual(result['AP50_95'], 1.0)

    def test_sampling_is_repeatable_and_input_order_independent(self):
        images = [{'id': i} for i in range(100)]
        a = select_images(images, 20, 0)
        self.assertEqual(a, select_images(list(reversed(images)), 20, 0))
        self.assertNotEqual(a, select_images(images, 20, 1))

    def test_paired_images_dense_labels_and_background_are_preserved(self):
        with test_workspace() as temp:
            source, output = temp/'source', temp/'pair'
            fixture(source)
            provenance = prepare(source, output, 2, portable=True, require_full_source=False)
            native = inspect_coco(output/'coco80_5k', allow_subset=True, verify_pixels=True)
            self.assertTrue(native['passed'])
            for original_split, custom_split in [('train', 'train'), ('val', 'valid')]:
                a = json.loads((output/'coco80_5k/annotations'/f'instances_{original_split}2017.json').read_text())
                b = json.loads((output/'coco20_voc_5k'/custom_split/'_annotations.coco.json').read_text())
                self.assertEqual(a['images'], b['images'])
                self.assertEqual(len(a['categories']), 80)
                self.assertEqual(len(b['categories']), 20)
                self.assertEqual({x['category_id'] for x in b['annotations']}, set(range(1, 21)))
                self.assertEqual(len(b['images']), 2)  # no-target images are kept
                self.assertEqual(len(b['annotations']), 20)
            self.assertEqual(provenance['source_category_ids'], list(VOC20_IDS))
            with self.assertRaises(FileExistsError):
                prepare(source, output, 2, portable=True, require_full_source=False)

    def test_prediction_mapping_and_invalid_dense_category(self):
        records = [{'image_id': 1, 'category_id': 20, 'score': .9, 'bbox': [1, 2, 3, 4]},
                   {'image_id': 1, 'category_id': 0, 'score': .1, 'bbox': [1, 2, 3, 4]}]
        original = copy.deepcopy(records)
        mapped = remap_predictions(records, 'voc20')
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0]['category_id'], 72)
        self.assertEqual(records, original)
        self.assertEqual(remap_predictions([dict(records[0], category_id=90)], 'original'), [])
        with self.assertRaises(ValueError):
            remap_predictions([dict(records[0], category_id=21)], 'voc20')

    def test_classifier_rows_follow_original_category_semantics(self):
        import torch
        from torch import nn
        with test_workspace() as temp:
            model = nn.Module()
            model.class_embed = nn.Linear(2, 21)
            model.transformer = nn.Module()
            model.transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(2, 21) for _ in range(2)])
            model.bbox_embed = nn.Linear(2, 4)
            bbox = model.bbox_embed.weight.detach().clone()
            source = {}
            for key, value in model.state_dict().items():
                if key.startswith('class_embed.') or '.enc_out_class_embed.' in key:
                    shape = (91, *value.shape[1:])
                    source[key] = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).float()
            path = temp/'weights.pth'
            torch.save({'model': source}, path)
            detector = SimpleNamespace(model=SimpleNamespace(model=model))
            result = initialize_category_rows(detector, path, list(VOC20_IDS), [category_names()[i] for i in VOC20_IDS])
            rows = torch.tensor([0, *VOC20_IDS])
            for key in result['parameter_keys']:
                self.assertTrue(torch.equal(model.state_dict()[key], source[key].index_select(0, rows)))
            self.assertTrue(torch.equal(bbox, model.bbox_embed.weight))
            with self.assertRaises(ValueError):
                initialize_category_rows(detector, path, list(VOC20_IDS), ['wrong']*20)

    def test_classifier_repair_handles_stale_metadata_and_mixed_head_shapes(self):
        import torch
        from torch import nn
        with test_workspace() as temp:
            model = nn.Module()
            model.class_embed = nn.Linear(2, 21)
            model.transformer = nn.Module()
            model.transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(2, 21) for _ in range(13)])
            source = {}
            for prefix, head in [('class_embed', model.class_embed), *[
                    (f'transformer.enc_out_class_embed.{i}', h)
                    for i, h in enumerate(model.transformer.enc_out_class_embed)]]:
                source[f'{prefix}.weight'] = torch.arange(182).reshape(91, 2).float()
                source[f'{prefix}.bias'] = torch.arange(91).float()
                # Reproduce upstream resize/load: metadata says 21, tensors have 91 rows.
                head.weight.data = source[f'{prefix}.weight'].clone()
                head.bias.data = source[f'{prefix}.bias'].clone()
                self.assertEqual(head.out_features, 21)
            # Also cover the exact reported case: decoder already small, encoder still large.
            model.class_embed.weight.data = model.class_embed.weight.data[:21].clone()
            model.class_embed.bias.data = model.class_embed.bias.data[:21].clone()
            path = temp/'source.pth'
            torch.save({'model': source}, path)
            detector = SimpleNamespace(model=SimpleNamespace(model=model))
            result = initialize_category_rows(detector, path, list(VOC20_IDS), [category_names()[i] for i in VOC20_IDS])
            heads = [model.class_embed, *model.transformer.enc_out_class_embed]
            self.assertEqual(len(heads), 14)
            rows = torch.tensor([0, *VOC20_IDS])
            for head in heads:
                self.assertEqual(head.out_features, 21)
                self.assertEqual(tuple(head.weight.shape), (21, 2))
                self.assertTrue(torch.equal(head.weight, source['class_embed.weight'][rows]))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = sum(head(torch.ones(1, 2)).square().mean() for head in heads)
            loss.backward()
            optimizer.step()
            self.assertTrue(all(head.weight.grad is not None for head in heads))
            self.assertEqual(result['previous_parameter_shapes']['transformer.enc_out_class_embed.0.weight'], [91, 2])


if __name__ == '__main__':
    unittest.main()
