"""Shared provenance, logging and model helpers for the untouched upstream detector."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "rf_detr"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream():
    manifest = json.loads((ROOT / "upstream_manifest.json").read_text(encoding="utf-8"))
    errors = []
    for relative, expected in manifest["files"].items():
        path = VENDOR / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(relative)
    if errors:
        raise RuntimeError(f"Upstream source hash mismatch: {errors}")
    return {key: manifest[key] for key in ("repository", "tag", "commit", "archive_sha256")}


def activate_upstream():
    verify_upstream()
    sys.path.insert(0, str(VENDOR))
    import rfdetr
    actual = Path(rfdetr.__file__).resolve()
    if not actual.is_relative_to(VENDOR.resolve()):
        raise RuntimeError(f"Wrong RF-DETR imported: {actual}")
    return rfdetr


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # grid_sample CUDA backward is not strictly deterministic.
    torch.backends.cudnn.benchmark = False


class Tee:
    def __init__(self, original, logfile):
        self.original, self.logfile = original, logfile

    def write(self, value):
        self.original.write(value)
        self.logfile.write(value)
        self.logfile.flush()

    def flush(self):
        self.original.flush()
        self.logfile.flush()

    def isatty(self):
        return False


def start_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logfile = path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = Tee(sys.stderr, logfile)
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), "Command:", sys.argv)
    return logfile


def resolve_amp(requested=True):
    """Upstream uses BF16 AMP; Turing GPUs must run this detector in FP32."""
    if not requested:
        return False
    import torch
    supported = (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
                 and torch.cuda.is_bf16_supported())
    if not supported:
        print("Native BF16 unavailable: using FP32 (AMP disabled).")
    return bool(supported)


def build_nano(weights=None, num_classes=90, device="cuda", amp=True, checkpointing=True, seed=0):
    rf = activate_upstream()
    seed_all(seed)
    if weights is not None and not Path(weights).is_file():
        raise FileNotFoundError(f"Checkpoint missing: {weights}. Run download_weights.py first.")
    model = rf.RFDETRNano(
        pretrain_weights=str(Path(weights).resolve()) if weights else None,
        num_classes=num_classes,
        device=device,
        resolution=384,
        group_detr=13,
        amp=resolve_amp(amp),
        gradient_checkpointing=checkpointing,
    )
    # Upstream loads the source head shape, so explicitly reset for custom classes.
    if model.model.model.class_embed.out_features != num_classes + 1:
        model.model.reinitialize_detection_head(num_classes + 1)
    return model
