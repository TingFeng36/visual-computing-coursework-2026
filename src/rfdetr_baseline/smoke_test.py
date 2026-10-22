"""Actual RF-DETR-Nano CUDA forward, matching, loss, backward and optimizer update."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import time

from common import build_nano, resolve_amp, start_log, verify_upstream, write_json


def run(steps=3, batch_size=1, amp=True, checkpointing=True, output="verification/gpu_smoke.json"):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU smoke test")
    amp = resolve_amp(amp)
    detector = build_nano(num_classes=2, amp=amp, checkpointing=checkpointing)
    from rfdetr.models import build_criterion_and_postprocessors
    from rfdetr.util.get_param_dicts import get_param_dict
    net = detector.model.model
    net.train()
    criterion, _ = build_criterion_and_postprocessors(detector.model.args)
    criterion.train()
    optimizer = torch.optim.AdamW(get_param_dict(detector.model.args, net), lr=1e-4)
    x = torch.rand(batch_size, 3, 384, 384, device="cuda")
    means = torch.tensor([.485,.456,.406], device="cuda").reshape(1,3,1,1)
    stds = torch.tensor([.229,.224,.225], device="cuda").reshape(1,3,1,1)
    x = (x-means)/stds
    targets = [{"labels": torch.tensor([1,2], device="cuda"),
                "boxes": torch.tensor([[.3,.4,.2,.3],[.7,.6,.15,.2]], device="cuda")} for _ in range(batch_size)]
    tracked = net.class_embed.weight
    initial = tracked.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    losses, shapes, durations = [], None, []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            predictions = net(x, targets)
            loss_dict = criterion(predictions, targets)
            loss = sum(loss_dict[key]*criterion.weight_dict[key] for key in loss_dict if key in criterion.weight_dict)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        gradients = [p.grad for p in net.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError("Missing/non-finite gradients")
        torch.nn.utils.clip_grad_norm_(net.parameters(), .1)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-started)
        losses.append(float(loss.detach()))
        shapes = {key: list(predictions[key].shape) for key in ("pred_logits", "pred_boxes")}
        print(f"Step {step+1}/{steps}: loss={losses[-1]:.6f}, seconds={durations[-1]:.3f}")
        del predictions, loss, loss_dict, gradients
    if torch.equal(initial, tracked.detach()):
        raise RuntimeError("Optimizer did not change the classification head")
    net.eval()
    criterion.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        eval_predictions = net(x)
    if eval_predictions["pred_boxes"].shape[1] != 300:
        raise RuntimeError("Expected 300 queries during evaluation")
    if shapes["pred_boxes"][1] != 300*13:
        raise RuntimeError("Expected all 13 training query groups")
    result = {"passed": True, "data": "random synthetic tensors; not an accuracy experiment",
              "torch": torch.__version__, "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "resolution": 384, "physical_batch": batch_size, "group_detr": 13,
              "amp_dtype": "bfloat16" if amp else "float32", "gradient_checkpointing": checkpointing,
              "steps": steps, "losses": losses, "step_seconds": durations,
              "train_shapes": shapes, "eval_queries": 300,
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
              "classification_weights_changed": True, "upstream": verify_upstream()}
    write_json(output, result)
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--output", default="verification/gpu_smoke.json")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch size must be positive")
    start_log(Path(args.output).with_suffix(".log"))
    try:
        run(args.steps, args.batch_size, not args.no_amp, not args.no_checkpointing, args.output)
    except BaseException as exc:
        write_json(args.output, {"passed": False, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
