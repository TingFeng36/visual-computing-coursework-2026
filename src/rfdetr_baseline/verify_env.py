from __future__ import annotations
import argparse
from importlib import metadata
import platform
import subprocess
import sys

from common import activate_upstream, verify_upstream, write_json

CORE = {"torch": "2.4.1", "torchvision": "0.19.1", "transformers": "4.51.3", "timm": "1.0.15", "peft": "0.15.2"}


def inspect(strict=False):
    import torch
    import torchvision
    import transformers, timm, peft
    activate_upstream()
    from torch.utils.tensorboard import SummaryWriter
    from pycocotools.coco import COCO
    versions = {name: metadata.version(name) for name in CORE}
    errors = []
    if strict:
        if platform.system() != "Linux" or sys.version_info[:2] != (3, 11):
            errors.append("Required target: Linux Python 3.11")
        for name, expected in CORE.items():
            if versions[name].split("+")[0] != expected:
                errors.append(f"{name}: expected {expected}, found {versions[name]}")
        if torch.version.cuda != "12.1":
            errors.append(f"Torch CUDA build must be 12.1, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        errors.append("CUDA unavailable")
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name, "total_memory_mib": props.total_memory / 2**20,
               "capability": list(torch.cuda.get_device_capability(0)),
               "bf16_supported": torch.cuda.is_bf16_supported()}
        gpu["native_bf16_supported"] = props.major >= 8 and gpu["bf16_supported"]
        gpu["selected_precision"] = "bfloat16 autocast" if gpu["native_bf16_supported"] else "float32"
        # Exercise the actual torchvision CUDA extension, not just imports.
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
        keep = torchvision.ops.nms(boxes, torch.tensor([.9, .8], device="cuda"), .5)
        if keep.tolist() != [0]:
            errors.append("torchvision CUDA NMS check failed")
    result = {"python": sys.version, "platform": platform.platform(), "versions": versions,
              "cuda_build": torch.version.cuda, "gpu": gpu, "upstream": verify_upstream(),
              "strict_target_check": strict, "passed": not errors, "errors": errors}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", default="verification/environment.json")
    args = parser.parse_args()
    try:
        result = inspect(args.strict)
    except Exception as exc:
        write_json(args.output, {"passed": False, "strict_target_check": args.strict, "error": repr(exc)})
        raise
    write_json(args.output, result)
    print(result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
