#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
python verify_env.py --strict
python download_weights.py
python -m unittest discover -s tests -v
python smoke_test.py --steps 3
if [[ -e verification/synthetic_dataset ]]; then
  echo 'Synthetic dataset already exists; choose a new directory or inspect existing verification results.' >&2
  exit 1
fi
python data_tools.py synthetic --output verification/synthetic_dataset
python train.py --dataset verification/synthetic_dataset --output verification/synthetic_training --scratch --epochs 1 --batch-size 1 --grad-accum-steps 1 --num-workers 0
python evaluate.py --dataset verification/synthetic_dataset --weights verification/synthetic_training/checkpoint_best_total.pth --split test --output verification/synthetic_evaluation --num-workers 0
python predict.py --dataset verification/synthetic_dataset --weights verification/synthetic_training/checkpoint_best_total.pth --image verification/synthetic_dataset/test/test_0.png --output verification/synthetic_prediction.png
python coco_tools.py synthetic --output verification/synthetic_coco
python train.py --config configs/nano_coco.json --dataset verification/synthetic_coco --allow-coco-subset --output verification/synthetic_coco_training --scratch --epochs 1 --batch-size 1 --grad-accum-steps 1 --num-workers 0
python evaluate.py --dataset-file coco --dataset verification/synthetic_coco --allow-coco-subset --weights verification/synthetic_coco_training/checkpoint_best_total.pth --split valid --output verification/synthetic_coco_evaluation --num-workers 0
echo 'Acceptance passed. Synthetic metrics only verify execution; they are not research results.'
