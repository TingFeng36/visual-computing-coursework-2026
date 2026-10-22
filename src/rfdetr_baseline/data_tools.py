"""Convert fixed YOLO splits into the Roboflow COCO layout expected by RF-DETR 1.3.0."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

from common import write_json, sha256

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def class_names(value):
    if isinstance(value, dict):
        converted = {int(k): str(v) for k, v in value.items()}
        if sorted(converted) != list(range(len(converted))):
            raise ValueError("YOLO names must use consecutive IDs starting at zero")
        names = [converted[i] for i in range(len(converted))]
    elif isinstance(value, list):
        names = list(map(str, value))
    else:
        raise ValueError("YAML names must be a list or a dictionary")
    if not names or len(set(names)) != len(names):
        raise ValueError("Class names must be nonempty and unique")
    return names


def resolve_entry(entry, root, yaml_dir):
    path = Path(entry).expanduser()
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()
    candidates = [root / path, yaml_dir / path]
    found = {p.resolve() for p in candidates if p.exists()}
    if len(found) != 1:
        raise ValueError(f"Missing or ambiguous relative path {entry}: {candidates}")
    return found.pop()


def collect_images(entry, root, yaml_dir):
    if isinstance(entry, list):
        images = [p for item in entry for p in collect_images(item, root, yaml_dir)]
    else:
        path = resolve_entry(entry, root, yaml_dir)
        if path.is_dir():
            images = sorted(p.resolve() for p in path.rglob("*")
                            if p.suffix.lower() in EXTENSIONS and ".ipynb_checkpoints" not in p.parts)
        elif path.suffix.lower() in EXTENSIONS:
            images = [path]
        elif path.suffix.lower() == ".txt":
            images = []
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                candidate = Path(line.strip()).expanduser()
                if candidate.is_absolute():
                    resolved = candidate.resolve()
                else:
                    options = {p.resolve() for p in (path.parent / candidate, root / candidate) if p.is_file()}
                    if len(options) != 1:
                        raise ValueError(f"Ambiguous/missing image in {path}: {line}")
                    resolved = options.pop()
                if not resolved.is_file() or resolved.suffix.lower() not in EXTENSIONS:
                    raise ValueError(f"Invalid image: {resolved}")
                images.append(resolved)
        else:
            raise ValueError(f"Unsupported split input: {path}")
    if not images or len(set(images)) != len(images):
        raise ValueError("Split must be nonempty and contain no duplicate image paths")
    return images


def label_path(image):
    parts = list(Path(image).parts)
    positions = [i for i, part in enumerate(parts) if part == "images"]
    if not positions:
        raise ValueError(f"Cannot infer label path: image path needs an 'images' component: {image}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def yolo_boxes(text, width, height, count, allow_clip=False):
    results, clipped = [], 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 5:
            raise ValueError(f"Line {line_number}: expected class cx cy w h, not polygon/OBB labels")
        values = list(map(float, tokens))
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"Line {line_number}: non-finite values")
        cls, cx, cy, bw, bh = values
        if cls != int(cls) or not 0 <= cls < count or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Line {line_number}: invalid class or normalized box")
        raw = [(cx - bw/2)*width, (cy - bh/2)*height, (cx + bw/2)*width, (cy + bh/2)*height]
        box = [max(0., raw[0]), max(0., raw[1]), min(float(width), raw[2]), min(float(height), raw[3])]
        changed = any(abs(a-b) > 1e-5 for a, b in zip(raw, box))
        if changed and not allow_clip:
            raise ValueError(f"Line {line_number}: box extends outside image; use --allow-clip explicitly")
        clipped += int(changed)
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Line {line_number}: empty box after clipping")
        results.append((int(cls) + 1, [x0, y0, x1-x0, y1-y0]))
    return results, clipped


def validate_dataset(root, verify_pixels=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    expected_categories = [{"id": i+1, "name": name, "supercategory": "object"} for i, name in enumerate(manifest["class_names"])]
    summary = {}
    seen_sources = {}
    for split in ("train", "valid", "test"):
        ann_path = root / split / "_annotations.coco.json"
        if sha256(ann_path) != manifest["splits"][split]["annotation_sha256"]:
            raise ValueError(f"{split}: annotation hash mismatch")
        annotations = json.loads(ann_path.read_text(encoding="utf-8"))
        if annotations["categories"] != expected_categories:
            raise ValueError(f"{split}: class mapping differs")
        image_index = {image["id"]: image for image in annotations["images"]}
        if not image_index or len(image_index) != len(annotations["images"]):
            raise ValueError(f"{split}: empty or duplicate image IDs")
        ann_ids, counts = set(), Counter()
        for ann in annotations["annotations"]:
            if ann["id"] in ann_ids or ann["image_id"] not in image_index:
                raise ValueError(f"{split}: invalid annotation IDs/references")
            ann_ids.add(ann["id"])
            if not 1 <= ann["category_id"] <= len(expected_categories):
                raise ValueError(f"{split}: category outside mapping")
            image = image_index[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            if not all(math.isfinite(v) for v in [x,y,w,h]) or min(w,h) <= 0 or x < -1e-5 or y < -1e-5 or x+w > image["width"]+1e-4 or y+h > image["height"]+1e-4:
                raise ValueError(f"{split}: invalid box {ann['id']}")
            counts[ann["category_id"]] += 1
        for image in image_index.values():
            path = root / split / image["file_name"]
            if not path.resolve().is_file():
                raise FileNotFoundError(path)
            if verify_pixels:
                from PIL import Image
                with Image.open(path) as img:
                    if img.size != (image["width"], image["height"]):
                        raise ValueError(f"Image size mismatch: {path}")
                    img.verify()
        sources = set(manifest["splits"][split]["source_images"])
        for previous, paths in seen_sources.items():
            if sources & paths and not ({split, previous} == {"valid", "test"} and manifest["test_is_validation_alias"]):
                raise ValueError(f"Split leakage: {previous}/{split}")
        seen_sources[split] = sources
        summary[split] = {"images": len(image_index), "objects": len(ann_ids), "objects_per_category": dict(counts),
                          "annotation_sha256": manifest["splits"][split]["annotation_sha256"]}
    return {"dataset_dir": str(root), "classes": manifest["class_names"], "splits": summary,
            "test_is_validation_alias": manifest["test_is_validation_alias"], "passed": True}


def convert(yaml_path, output, validation_as_test=False, copy_images=False, allow_clip=False):
    import yaml
    from PIL import Image
    yaml_path = Path(yaml_path).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    names = class_names(config["names"])
    root = Path(config.get("path", yaml_path.parent)).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()
    splits = {"train": collect_images(config["train"], root, yaml_path.parent),
              "valid": collect_images(config["val"], root, yaml_path.parent)}
    if config.get("test"):
        if validation_as_test:
            raise ValueError("YAML already provides test; do not request validation alias")
        splits["test"] = collect_images(config["test"], root, yaml_path.parent)
    elif validation_as_test:
        splits["test"] = list(splits["valid"])
    else:
        raise ValueError("Upstream requires a test folder. Supply YAML test or explicitly use --validation-as-test")
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        if set(splits[a]) & set(splits[b]) and not (a == "valid" and b == "test" and validation_as_test):
            raise ValueError(f"Source images overlap between {a} and {b}")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be new or empty; existing data is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_yaml": str(yaml_path), "source_yaml_sha256": sha256(yaml_path), "class_names": names,
                "class_mapping": {str(i): i+1 for i in range(len(names))},
                "test_is_validation_alias": validation_as_test, "splits": {}}
    categories = [{"id": i+1, "name": name, "supercategory": "object"} for i, name in enumerate(names)]
    for split, source_images in splits.items():
        directory = output / split
        directory.mkdir()
        images, annotations, labels_hashes, clipped_total = [], [], {}, 0
        for index, image in enumerate(source_images, 1):
            label = label_path(image)
            if not label.is_file():
                raise FileNotFoundError(f"Missing label (background images need an empty label file): {label}")
            with Image.open(image) as img:
                width, height = img.size
            boxes, clipped = yolo_boxes(label.read_text(encoding="utf-8-sig"), width, height, len(names), allow_clip)
            clipped_total += clipped
            filename = f"{index:07d}_{hashlib.sha256(str(image).encode()).hexdigest()[:10]}{image.suffix.lower()}"
            destination = directory / filename
            if copy_images:
                shutil.copy2(image, destination)
            else:
                destination.symlink_to(image)
            images.append({"id": index, "file_name": filename, "width": width, "height": height})
            labels_hashes[str(label)] = sha256(label)
            for category, bbox in boxes:
                annotations.append({"id": len(annotations)+1, "image_id": index, "category_id": category,
                                    "bbox": bbox, "area": bbox[2]*bbox[3], "iscrowd": 0})
        annotation_path = directory / "_annotations.coco.json"
        write_json(annotation_path, {"info": {"description": "Fixed YOLO split conversion for RF-DETR 1.3.0"},
                                     "licenses": [], "images": images, "annotations": annotations, "categories": categories})
        manifest["splits"][split] = {"source_images": list(map(str, source_images)), "label_sha256": labels_hashes,
                                      "annotation_sha256": sha256(annotation_path), "clipped_boxes": clipped_total}
    write_json(output / "dataset_manifest.json", manifest)
    return validate_dataset(output, verify_pixels=True)


def make_smoke_dataset(output):
    """Synthetic data checks plumbing only; it is never reported as detection accuracy."""
    from PIL import Image, ImageDraw
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    names = ["synthetic_rectangle"]
    manifest = {"synthetic": True, "class_names": names, "test_is_validation_alias": False, "splits": {}}
    for split, count in [("train", 16), ("valid", 2), ("test", 2)]:
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for i in range(count):
            filename = f"{split}_{i}.png"
            image = Image.new("RGB", (384, 384), (20+i*3, 30, 50))
            x, y, w, h = 40+i*5, 80, 80, 120
            ImageDraw.Draw(image).rectangle((x,y,x+w,y+h), fill=(220,100,40))
            image.save(directory / filename)
            images.append({"id": i+1, "file_name": filename, "width": 384, "height": 384})
            annotations.append({"id": i+1, "image_id": i+1, "category_id": 1, "bbox": [x,y,w,h], "area": w*h, "iscrowd": 0})
        annotation_path = directory / "_annotations.coco.json"
        write_json(annotation_path, {"info": {}, "licenses": [], "images": images, "annotations": annotations,
                                     "categories": [{"id": 1, "name": names[0], "supercategory": "object"}]})
        manifest["splits"][split] = {"source_images": [str(directory / item["file_name"]) for item in images],
                                      "annotation_sha256": sha256(annotation_path)}
    write_json(output / "dataset_manifest.json", manifest)
    return validate_dataset(output, verify_pixels=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--yolo-yaml", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--validation-as-test", action="store_true")
    prepare.add_argument("--copy-images", action="store_true")
    prepare.add_argument("--allow-clip", action="store_true")
    check = commands.add_parser("check")
    check.add_argument("--dataset", required=True)
    check.add_argument("--verify-pixels", action="store_true")
    synth = commands.add_parser("synthetic")
    synth.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        report = convert(args.yolo_yaml, args.output, args.validation_as_test, args.copy_images, args.allow_clip)
    elif args.command == "check":
        report = validate_dataset(args.dataset, args.verify_pixels)
    else:
        report = make_smoke_dataset(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
