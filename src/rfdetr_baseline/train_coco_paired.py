"""Visible single-GPU baseline queue; wait for existing jobs without stopping them."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
JOB_SCRIPTS = {'run_both.py', 'train.py', 'evaluate.py', 'train_coco_paired.py'}


def python_script(argv):
    if not argv or not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?(?:\.exe)?', Path(argv[0]).name):
        return None
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg in ('-c', '-m') or arg.startswith('-c') or arg.startswith('-m'):
            return None
        if arg == '--':
            return argv[index+1] if index+1 < len(argv) else None
        if arg in ('-W', '-X'):
            index += 2
        elif arg.startswith('-'):
            index += 1
        else:
            return arg
    return None


def is_project_job(argv, cwd, state, root=ROOT):
    script = python_script(argv)
    if not script or Path(script).name not in JOB_SCRIPTS or state in ('Z', 'X', 'x'):
        return False
    path = Path(script)
    if not path.is_absolute():
        path = Path(cwd) / path
    return path.resolve().parent == Path(root).resolve()


def active_jobs(root=ROOT, proc_root=Path('/proc')):
    jobs = []
    for process in proc_root.iterdir():
        if not process.name.isdecimal() or int(process.name) == os.getpid():
            continue
        try:
            argv = process.joinpath('cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
            state = process.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0]
            cwd = process.joinpath('cwd').resolve(strict=True)
            if is_project_job(argv, cwd, state, root):
                jobs.append({'pid': int(process.name), 'state': state, 'argv': argv})
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return sorted(jobs, key=lambda item: item['pid'])


def describe_jobs(jobs):
    for job in jobs[:12]:
        argv = job['argv']
        options = []
        for key in ('--config', '--output', '--launch-dir'):
            if key in argv and argv.index(key)+1 < len(argv):
                options.extend([key, argv[argv.index(key)+1]])
        print(f"[paired] occupied: PID={job['pid']} state={job['state']} "
              f"{python_script(argv)} {' '.join(options)}", flush=True)
        if job['state'] in ('T', 't'):
            print('[paired] This task is stopped/paused; it will not finish until resumed. No process will be killed.', flush=True)
    if len(jobs) > 12:
        print(f'[paired] {len(jobs)} matching processes in total', flush=True)


def wait_for_idle(wait, interval=30, scanner=active_jobs, sleeper=time.sleep):
    while True:
        jobs = scanner()
        if not jobs:
            print('[paired] No other active project training/evaluation jobs; starting next stage.', flush=True)
            return
        print(time.strftime('[paired] %Y-%m-%d %H:%M:%S'), flush=True)
        describe_jobs(jobs)
        if not wait:
            raise RuntimeError('GPU queue is occupied. Use --wait to wait for the existing tasks; no new training started.')
        print(f'[paired] Waiting {interval}s before checking again; existing tasks continue normally.', flush=True)
        sleeper(interval)


def commands():
    images = Path('/root/autodl-tmp/OurMamba/coco_paired5k')
    results = Path('/root/autodl-fs/rfdetr_runs')
    python = [sys.executable, '-u']
    jobs = []
    for kind, backend in [('coco80_5k', 'coco'), ('coco20_voc_5k', 'roboflow')]:
        config = ROOT/'configs'/f'nano_{kind}.json'
        output = results/f'{kind}_seed0'
        extra = ['--allow-coco-subset'] if backend == 'coco' else []
        jobs.append((f'{kind}: train20', python+[str(ROOT/'train.py'), '--config', str(config)]+extra, True))
        jobs.append((f'{kind}: evaluate', python+[str(ROOT/'evaluate.py'), '--dataset-file', backend,
            '--dataset', str(images/kind), '--weights', str(output/'checkpoint_best_total.pth'),
            '--output', str(output/'final_validation')]+extra, True))
    for kind, space in [('coco80_5k', 'original'), ('coco20_voc_5k', 'voc20')]:
        output = results/f'{kind}_seed0'
        jobs.append((f'{kind}: common20 score', python+[str(ROOT/'coco_voc20.py'), 'score',
            '--source', '/root/autodl-fs/datasets/coco2017', '--predictions',
            str(output/'final_validation/predictions.coco.json'), '--label-space', space,
            '--output', str(output/'voc20_common_metrics.json')], False))
    return jobs


@contextmanager
def single_queue():
    import fcntl
    directory = ROOT/'runs'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'coco_paired5k.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A paired queue is already running or waiting; do not launch another copy') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnose', action='store_true', help='Print actual blockers; do not launch any training')
    parser.add_argument('--wait', action='store_true', help='Wait for existing project jobs instead of exiting')
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise SystemExit('Run this queue on the AutoDL Linux host')
    if not 1 <= args.interval <= 60:
        raise SystemExit('interval must be between 1 and 60 seconds')
    if args.diagnose:
        jobs = active_jobs()
        describe_jobs(jobs)
        if not jobs:
            print('[paired] No blocking project process was found.', flush=True)
        return
    with single_queue():
        # An aborted old guard created no outputs. Nonempty formal results need
        # deliberate stage-specific recovery instead of blindly retraining.
        for config in ('nano_coco80_5k.json', 'nano_coco20_voc_5k.json'):
            data = json.loads((ROOT/'configs'/config).read_text())
            output = Path(data['output_dir'])
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f'Formal results already exist: {output}; inspect them before restarting this queue')
        status = {'status': 'waiting', 'pid': os.getpid(), 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        path = ROOT/'runs/coco_paired5k_status.json'
        def write_status():
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(path)
        write_status()
        try:
            for index, (name, command, needs_gpu) in enumerate(commands()):
                status.update(status='waiting' if needs_gpu else 'running', stage=index, task=name)
                write_status()
                if needs_gpu:
                    wait_for_idle(args.wait, args.interval)
                status['status'] = 'running'
                write_status()
                print(f'[paired] Starting {index+1}/6: {name}', flush=True)
                subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, check=True)
            status['status'] = 'completed'
            print('[paired] Both baselines and common-class scores completed.', flush=True)
        except BaseException as error:
            status.update(status='failed', error=str(error))
            raise
        finally:
            status['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            write_status()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f'[paired] Stopped: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
