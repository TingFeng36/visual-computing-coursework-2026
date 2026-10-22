"""Evaluate checkpoint using the upstream transform, postprocessor and COCO evaluator."""
from __future__ import annotations
import argparse
from pathlib import Path
import json

from common import ROOT, build_nano, sha256, start_log, verify_upstream, write_json
from dataset_backend import audit_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-file", choices=["roboflow", "coco"], default="roboflow")
    parser.add_argument("--allow-coco-subset", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default="results/nano_valid")
    parser.add_argument("--allow-validation-alias", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new evaluation output directory")
    output.mkdir(parents=True, exist_ok=True)
    start_log(output / "val.log")
    import torch
    from torch.utils.data import DataLoader
    if args.dataset_file == "coco" and args.split != "valid":
        raise ValueError("MS COCO local metrics use val2017; test-dev has no public ground truth")
    audit = audit_dataset(args.dataset_file, args.dataset, validation_only=True, allow_coco_subset=args.allow_coco_subset)
    if args.split == "test" and audit["test_is_validation_alias"] and not args.allow_validation_alias:
        raise ValueError("Test is a validation alias, not an independent test. Use --split valid.")
    detector = build_nano(args.weights, audit["num_classes"], amp=not args.no_amp, checkpointing=False)
    from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
    from rfdetr.models import build_criterion_and_postprocessors
    from rfdetr.engine import evaluate
    from rfdetr.datasets.coco_eval import CocoEvaluator
    from rfdetr.util.misc import collate_fn
    model_args = detector.model.args
    model_args.dataset_file = args.dataset_file
    model_args.dataset_dir = str(Path(args.dataset).resolve())
    if args.dataset_file == "coco":
        model_args.coco_path = model_args.dataset_dir
    model_args.square_resize_div_64 = True
    model_args.multi_scale = False
    model_args.expanded_scales = False
    model_args.do_random_resize_via_padding = False
    model_args.fp16_eval = False
    model_args.output_dir = str(output.resolve())
    split = "val" if args.split == "valid" else "test"
    dataset = build_dataset(split, model_args, resolution=384)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    criterion, postprocessor = build_criterion_and_postprocessors(model_args)
    net = detector.model.model
    net.eval()
    # Upstream cocoDt holds the last batch, so collect every batch at its evaluator boundary.
    predictions = []
    original_update = CocoEvaluator.update
    def collect_update(evaluator, batch_predictions):
        predictions.extend(evaluator.prepare_for_coco_detection(batch_predictions))
        return original_update(evaluator, batch_predictions)
    CocoEvaluator.update = collect_update
    try:
        with torch.no_grad():
            metrics, evaluator = evaluate(net, criterion, postprocessor, loader, get_coco_api_from_dataset(dataset), torch.device("cuda"), args=model_args)
    finally:
        CocoEvaluator.update = original_update
    write_json(output / "predictions.coco.json", predictions)
    stats = metrics["coco_eval_bbox"]
    result = {"split": args.split, "independent_test": args.split == "test" and not audit["test_is_validation_alias"],
              "dataset_file": args.dataset_file, "full_coco2017": audit.get("full_coco2017"),
              "dataset_annotation_sha256": audit["splits"][args.split]["annotation_sha256"],
              "weights_sha256": sha256(args.weights), "resolution": 384,
              "AP50_95": stats[0], "AP50": stats[1], "AP75": stats[2],
              "AP_small": stats[3], "AP_medium": stats[4], "AP_large": stats[5],
              "AR_100": stats[8], "COCO_stats": stats,
              "eval_precision": "bfloat16 autocast" if model_args.amp else "float32",
              "upstream": verify_upstream(), "gpu": torch.cuda.get_device_name(0)}
    write_json(output / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
