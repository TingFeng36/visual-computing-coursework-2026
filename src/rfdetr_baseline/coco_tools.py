"""Audit native MS COCO 2017 without renumbering its sparse category IDs."""
from __future__ import annotations
import argparse
import ast
import json
import math
import os
from pathlib import Path
import shutil
import time
from common import VENDOR, sha256, verify_upstream, write_json

SPLITS = {"train": ("train2017", "instances_train2017.json", 118287),
          "valid": ("val2017", "instances_val2017.json", 5000)}


def category_names():
    verify_upstream()
    tree = ast.parse((VENDOR / "rfdetr/util/coco_classes.py").read_text(encoding="utf-8"))
    return ast.literal_eval(tree.body[0].value)


def inspect_coco(root, splits=("train", "valid"), allow_subset=False, verify_pixels=False):
    root = Path(root).resolve()
    names = category_names()
    summary, seen = {}, set()
    for split in splits:
        started = time.monotonic()
        folder, filename, expected_count = SPLITS[split]
        annotation = root / "annotations" / filename
        print(f"[COCO check] {split}: reading {annotation}", flush=True)
        data = json.loads(annotation.read_text(encoding="utf-8"))
        categories = {c["id"]: c["name"] for c in data["categories"]}
        if categories != names or len(data["categories"]) != 80:
            raise ValueError("MS COCO requires its original 80 category IDs/names, including category 90")
        images = {item["id"]: item for item in data["images"]}
        if not images or len(images) != len(data["images"]):
            raise ValueError(f"{split}: empty or duplicate image IDs")
        if len(images) != expected_count and not allow_subset:
            raise ValueError(f"{split}: expected full MS COCO 2017 ({expected_count} images), found {len(images)}; subsets require --allow-coco-subset")
        if seen & images.keys():
            raise ValueError("COCO train/valid image IDs overlap")
        seen.update(images)
        print(f"[COCO check] {split}: checking {len(images)} image paths", flush=True)
        file_indexes = {}
        for index, image in enumerate(images.values(), 1):
            relative = Path(image["file_name"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Image file_name must stay inside the split directory")
            path = root / folder / relative
            if path.parent not in file_indexes:
                print(f"[COCO check] {split}: indexing directory {path.parent}", flush=True)
                names_in_directory = set()
                with os.scandir(path.parent) as entries:
                    for entry_index, entry in enumerate(entries, 1):
                        if entry.is_file():
                            names_in_directory.add(entry.name)
                        if entry_index % 10000 == 0:
                            print(f"[COCO check] {split}: indexed {entry_index} directory entries", flush=True)
                file_indexes[path.parent] = names_in_directory
            if path.name not in file_indexes[path.parent]:
                raise FileNotFoundError(path)
            if verify_pixels:
                from PIL import Image
                with Image.open(path) as pixels:
                    if pixels.size != (image["width"], image["height"]):
                        raise ValueError(f"Image dimensions disagree: {path}")
                    pixels.verify()
            if index % 1000 == 0 or index == len(images):
                print(f"[COCO check] {split}: images {index}/{len(images)}; elapsed {time.monotonic()-started:.1f}s", flush=True)
        print(f"[COCO check] {split}: checking {len(data['annotations'])} annotations", flush=True)
        annotation_ids = set()
        for item in data["annotations"]:
            if item["id"] in annotation_ids or item["image_id"] not in images or item["category_id"] not in names:
                raise ValueError(f"{split}: invalid annotation IDs/references/categories")
            annotation_ids.add(item["id"])
            box = item["bbox"]
            if len(box) != 4 or not all(math.isfinite(x) for x in box) or min(box[2:]) < 0:
                raise ValueError(f"{split}: invalid bounding box")
        summary[split] = {"images": len(images), "objects": len(annotation_ids),
                          "annotation_sha256": sha256(annotation), "full_2017_split": len(images) == expected_count}
        print(f"[COCO check] {split}: passed; elapsed {time.monotonic()-started:.1f}s", flush=True)
    return {"dataset_file": "coco", "dataset_dir": str(root), "classes": list(names.values()),
            "category_names": names, "num_classes": 90, "actual_categories": 80,
            "test_is_validation_alias": True, "splits": summary,
            "full_coco2017": all(s["full_2017_split"] for s in summary.values()), "passed": True}


def subset(source, output, train_images=256, val_images=64, copy_images=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if min(train_images, val_images) < 1:
        raise ValueError("Subset sizes must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Subset output must be new or empty")
    provenance = {"type": "COCO subset for execution checks, not full benchmark", "source": str(source), "splits": {}}
    for split, limit in [("train", train_images), ("valid", val_images)]:
        folder, filename, _ = SPLITS[split]
        original = source / "annotations" / filename
        data = json.loads(original.read_text(encoding="utf-8"))
        selected = sorted(data["images"], key=lambda x: x["id"])[:limit]
        ids = {image["id"] for image in selected}
        data["images"] = selected
        data["annotations"] = [a for a in data["annotations"] if a["image_id"] in ids]
        for image in selected:
            relative = Path(image["file_name"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid image file_name")
            origin, destination = source / folder / relative, output / folder / relative
            if not origin.is_file():
                raise FileNotFoundError(origin)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if copy_images:
                shutil.copy2(origin, destination)
            else:
                destination.symlink_to(origin)
        write_json(output / "annotations" / filename, data)
        provenance["splits"][split] = {"source_annotation_sha256": sha256(original), "image_ids": sorted(ids)}
    write_json(output / "subset_manifest.json", provenance)
    return inspect_coco(output, allow_subset=True, verify_pixels=True)


def synthetic(output):
    """Exercise the native sparse-label path, including the last category (90)."""
    from PIL import Image, ImageDraw
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    categories = [{"id": i, "name": name, "supercategory": "synthetic"} for i, name in category_names().items()]
    for split, count, offset in [("train", 16, 0), ("valid", 2, 100)]:
        folder, filename, _ = SPLITS[split]
        images, annotations = [], []
        for index in range(count):
            path = output / folder / f"{offset+index+1:012d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (384,384), (20,30,40))
            ImageDraw.Draw(image).rectangle((40,80,120,200), fill=(200,100,40))
            image.save(path)
            image_id = offset+index+1
            images.append({"id": image_id, "file_name": path.name, "width":384, "height":384})
            annotations.append({"id":image_id, "image_id":image_id, "category_id":90, "bbox":[40,80,80,120], "area":9600, "iscrowd":0})
        write_json(output / "annotations" / filename, {"info":{"description":"Synthetic execution check, not actual COCO"},
                   "licenses":[], "images":images, "annotations":annotations, "categories":categories})
    return inspect_coco(output, allow_subset=True, verify_pixels=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--dataset", required=True)
    check.add_argument("--val-only", action="store_true")
    check.add_argument("--allow-coco-subset", action="store_true")
    check.add_argument("--verify-pixels", action="store_true")
    small = commands.add_parser("subset")
    small.add_argument("--source", required=True)
    small.add_argument("--output", required=True)
    small.add_argument("--train-images", type=int, default=256)
    small.add_argument("--val-images", type=int, default=64)
    small.add_argument("--copy-images", action="store_true")
    synth = commands.add_parser("synthetic")
    synth.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "check":
        result = inspect_coco(args.dataset, ("valid",) if args.val_only else ("train","valid"), args.allow_coco_subset, args.verify_pixels)
    elif args.command == "subset":
        result = subset(args.source, args.output, args.train_images, args.val_images, args.copy_images)
    else:
        result = synthetic(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
