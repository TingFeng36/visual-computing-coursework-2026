"""Read original M3FD classes/splits and prepare RF-DETR data without guessing names."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from PIL import Image
from common import sha256, write_json
from data_tools import class_names, collect_images, convert


def prepare(source, output, dataset_root=None, inspect_only=False, copy_images=False, allow_clip=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    names = class_names(config["names"])
    if "nc" in config and int(config["nc"]) != len(names):
        raise ValueError("Original nc differs from names; inspect m3fd.yaml before conversion")
    if not config.get("train"):
        raise ValueError("Original YAML has no train split")
    root = Path(dataset_root or config.get("path", source.parent)).expanduser()
    if not root.is_absolute():
        root = source.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"YAML dataset root missing: {root}; use --dataset-root with the verified data directory")
    validation_origin = "original val"
    if not config.get("val"):
        if not config.get("test"):
            raise ValueError("Original YAML has neither val nor test; no new random split is created")
        config["val"] = config["test"]
        validation_origin = "original test used as validation (no separate validation provided)"
    config["path"] = str(root)
    resolved = {key: collect_images(config[key],root,source.parent) for key in ("train","val")}
    test_alias = not bool(config.get("test"))
    if config.get("test"):
        test = collect_images(config["test"],root,source.parent)
        if set(test) == set(resolved["val"]):
            test_alias = True
        elif set(test) & set(resolved["val"]):
            raise ValueError("Original val/test partially overlap; cannot report independent test")
        else:
            resolved["test"] = test
    # Make all split entries absolute before placing the effective YAML elsewhere.
    for key in ("train","val","test"):
        if key in resolved:
            config[key] = [str(path) for path in resolved[key]]
    if test_alias:
        config.pop("test",None)
    samples = []
    for split in ("train","val"):
        for path in resolved[split][:3]:
            with Image.open(path) as image:
                samples.append({"split":split,"path":str(path),"mode":image.mode,"size":list(image.size)})
    original = {"source_yaml":str(source),"source_yaml_sha256":sha256(source),"dataset_root":str(root),
                "class_names":names,"source_split_counts":{k:len(v) for k,v in resolved.items()},
                "validation_origin":validation_origin,"test_is_validation_alias":test_alias,
                "sample_images":samples,"modality":"visible requested; script does not determine sensor modality"}
    print(json.dumps(original,ensure_ascii=False,indent=2))
    if inspect_only:
        return original
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Converted output exists; check existing data or use a new output")
    effective_yaml = output.parent / (output.name + ".yaml")
    serialized = yaml.safe_dump(config,allow_unicode=True,sort_keys=False)
    if effective_yaml.exists() and effective_yaml.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Effective YAML exists with different content: {effective_yaml}")
    effective_yaml.parent.mkdir(parents=True,exist_ok=True)
    if not effective_yaml.exists():
        effective_yaml.write_text(serialized,encoding="utf-8")
    result = convert(effective_yaml,output,validation_as_test=test_alias,copy_images=copy_images,allow_clip=allow_clip)
    manifest_path = output / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["original_source"] = original
    write_json(manifest_path,manifest)
    write_json(output / "original_yaml_audit.json",original)
    print("Conversion passed:",output)
    print("Independent test:",not test_alias)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",default="/root/autodl-tmp/OurMamba/datasets_M3FD/m3fd.yaml")
    parser.add_argument("--output",default="/root/autodl-tmp/OurMamba/datasets_M3FD_rfdetr")
    parser.add_argument("--dataset-root")
    parser.add_argument("--inspect-only",action="store_true")
    parser.add_argument("--copy-images",action="store_true")
    parser.add_argument("--allow-clip",action="store_true")
    args = parser.parse_args()
    prepare(args.source,args.output,args.dataset_root,args.inspect_only,args.copy_images,args.allow_clip)


if __name__ == "__main__":
    main()
