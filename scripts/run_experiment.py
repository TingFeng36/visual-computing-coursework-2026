"""Public RF-DETR baseline entry point (groups A/B/C/E only)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'src/rfdetr_baseline'
DATASETS = {'A': 'coco2017_full', 'B': 'coco80_5k', 'C': 'coco20_voc_5k',
            'E': 'm3fd'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'evaluate', 'config'])
    p.add_argument('--group', required=True, choices=list(DATASETS))
    p.add_argument('--dataset', required=True, help='Prepared full dataset directory, not data/samples')
    p.add_argument('--output', required=True, help='Fresh output folder outside archived logs/results')
    p.add_argument('--weights')
    p.add_argument('--dry-run', action='store_true', help='Print commands/config; do not initialize CUDA')
    args = p.parse_args()
    output = Path(args.output).resolve()
    if output == ROOT or any(output.is_relative_to(ROOT/x) for x in ['logs', 'results', 'checkpoints', 'src', 'data']):
        raise ValueError('Use a fresh folder under runs/ or outside the submission, preserving original evidence')
    config = json.loads((ROOT/f'results/experiments/{args.group}/resolved_config.json').read_text(encoding='utf-8'))
    config.update(dataset_dir=str(Path(args.dataset).resolve()), output_dir=str(output),
                  weights=str(ROOT/'checkpoints/initialization/rf-detr-nano.pth'), amp=False)
    cfg_path = ROOT/f'runs/relocated_{args.group}/config.json'
    checkpoint = Path(args.weights).resolve() if args.weights else ROOT/f'checkpoints/{args.group}/best_model.pth'
    cmd = [sys.executable, '-u', str(PROJECT/'train.py'),
           '--config', str(cfg_path), '--no-amp']
    if args.mode == 'evaluate':
        cmd = [sys.executable, '-u', str(PROJECT/'evaluate.py')]
        cmd += ['--dataset', config['dataset_dir'], '--dataset-file', config['dataset_file'],
                '--weights', str(checkpoint), '--output', str(output), '--no-amp', '--split', 'valid']
    if args.group == 'B':
        cmd.append('--allow-coco-subset')
    print(json.dumps({'group': args.group, 'config': config, 'command': cmd,
                      'required_python': '3.11', 'dry_run': args.dry_run}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.mode == 'config':
        return
    if os.name != 'posix':
        raise RuntimeError('Training/evaluation requires Linux CUDA; saved-prediction demo runs on CPU')
    subprocess.run(cmd, cwd=PROJECT, check=True)


if __name__ == '__main__':
    main()
