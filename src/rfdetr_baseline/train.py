"""Train official Nano with a documented first-checkpoint safety repair."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, build_nano, resolve_amp, sha256, start_log, verify_upstream, write_json
from dataset_backend import audit_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nano_m3fd.json")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Use FP32 explicitly on either GPU generation")
    parser.add_argument("--allow-coco-subset", action="store_true", help="Explicit execution-check subset; not full COCO benchmark")
    parser.add_argument("--resume")
    parser.add_argument("--allow-non-target-env", action="store_true", help="Only for local functional verification")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("dataset_file", "roboflow")
    overrides = {"dataset_dir": args.dataset, "output_dir": args.output, "epochs": args.epochs,
                 "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps, "num_workers": args.num_workers}
    config.update({key: value for key, value in overrides.items() if value is not None})
    if args.scratch:
        config["initialization"] = "scratch"
    if args.no_amp:
        config["amp"] = False
    if config["initialization"] not in ("coco", "scratch"):
        raise ValueError("initialization must be coco or scratch")
    if config["resolution"] != 384 or config["group_detr"] != 13:
        raise ValueError("Phase 1 preserves Nano resolution=384 and group_detr=13")
    if config["multi_scale"] or config["expanded_scales"]:
        raise ValueError("This resource-controlled profile uses fixed resolution")
    for key in ("epochs", "batch_size", "grad_accum_steps"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("dataset_dir", "output_dir", "weights"):
        path = Path(config[key]).expanduser()
        config[key] = str((path if path.is_absolute() else ROOT/path).resolve())
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Output exists; choose a fresh output or explicitly --resume checkpoint.pth")
    output.mkdir(parents=True, exist_ok=True)
    start_log(output / "train.log")
    status = {"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(output / "run_status.json", status)
    try:
        from verify_env import inspect
        environment = inspect(strict=not args.allow_non_target_env)
        write_json(output / "environment.json", environment)
        if not environment["passed"]:
            raise RuntimeError(environment["errors"])
        config["amp_requested"] = config["amp"]
        config["amp"] = resolve_amp(config["amp"])
        config["training_precision"] = "bfloat16 autocast" if config["amp"] else "float32"
        dataset = audit_dataset(config["dataset_file"], config["dataset_dir"], allow_coco_subset=args.allow_coco_subset)
        if dataset["splits"]["train"]["images"] < config["batch_size"]*config["grad_accum_steps"]:
            raise ValueError("Train split is smaller than one effective batch; reduce batch/accumulation for smoke only")
        if config["run_test"] and dataset["test_is_validation_alias"]:
            raise ValueError("run_test cannot label a validation alias as independent test")
        write_json(output / "dataset_audit.json", dataset)
        num_classes = dataset["num_classes"]
        weights = config["weights"] if config["initialization"] == "coco" else None
        if weights:
            from download_weights import verify_coco_checkpoint
            config["pretrained_weights"] = verify_coco_checkpoint(weights)
            print("Source COCO checkpoint SHA256:", config["pretrained_weights"]["sha256"])
        config["upstream"] = verify_upstream()
        from training_safety import apply_checkpoint_safety
        config["runtime_patch"] = apply_checkpoint_safety()
        config["num_classes"] = num_classes
        config["test_is_validation_alias"] = dataset["test_is_validation_alias"]
        config["full_coco2017"] = dataset.get("full_coco2017")
        write_json(output / "resolved_config.json", config)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (output / "environment.freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        detector = build_nano(weights, num_classes, amp=config["amp"], checkpointing=config["gradient_checkpointing"], seed=config["seed"])
        if config.get("source_category_ids"):
            if config['dataset_file'] != 'roboflow' or not weights:
                raise ValueError('Semantic category initialization requires a custom dataset and COCO source weights')
            if not args.resume:
                from category_initialization import initialize_category_rows
                config['classifier_initialization'] = initialize_category_rows(
                    detector, weights, config['source_category_ids'], dataset['classes'])
                write_json(output / 'resolved_config.json', config)
                print('Classifier initialized by original COCO category IDs:', config['source_category_ids'], flush=True)
        options = {key: config[key] for key in ("dataset_dir", "output_dir", "epochs", "batch_size", "grad_accum_steps",
                   "lr", "lr_encoder", "lr_drop", "weight_decay", "num_workers", "checkpoint_interval", "use_ema", "run_test",
                   "multi_scale", "expanded_scales", "seed", "group_detr")}
        options.update(dataset_file=config["dataset_file"], tensorboard=True, wandb=False,
                       square_resize_div_64=True, do_benchmark=False)
        if config["dataset_file"] == "coco":
            options["coco_path"] = config["dataset_dir"]
        if args.resume:
            options["resume"] = str(Path(args.resume).resolve())
        detector.train(**options)
        required = [output / "checkpoint_best_total.pth", output / "log.txt"]
        if not all(p.is_file() for p in required):
            raise RuntimeError("Training ended without required checkpoint/metrics")
        status.update(status="completed", finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      checkpoint_sha256=sha256(required[0]))
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(output / "run_status.json", status)


if __name__ == "__main__":
    main()
