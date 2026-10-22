"""Independent COCO/M3FD training and evaluation pipelines on selected GPU(s)."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from common import ROOT, write_json
from dataset_backend import audit_dataset


def stop(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()


def run_job(job, launch, cancelled):
    state = {"status":"running", "gpu":job["gpu"], "started":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = job["gpu"]
        for index, command in enumerate(job["commands"]):
            if cancelled.is_set():
                raise RuntimeError("Other pipeline failed or launch interrupted")
            print(f'{job["name"]}: starting {command[2]} on GPU {job["gpu"]}', flush=True)
            with (launch / f'{job["name"]}_{index}.log').open("w", encoding="utf-8") as logfile:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                                           start_new_session=os.name == "posix")
                state.update(stage=index, pid=process.pid)
                write_json(launch / f'{job["name"]}_status.json', state)
                while process.poll() is None:
                    if cancelled.wait(.5):
                        stop(process)
                        raise RuntimeError("Pipeline cancelled")
                if process.returncode:
                    raise RuntimeError(f"Stage {index} exited {process.returncode}; inspect launch logs")
        state["status"] = "completed"
    except BaseException as error:
        cancelled.set()
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(launch / f'{job["name"]}_status.json', state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sequential","parallel"], default="sequential")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--coco-config", default="configs/nano_coco.json")
    parser.add_argument("--m3fd-config", default="configs/nano_m3fd.json")
    parser.add_argument("--coco-dataset")
    parser.add_argument("--m3fd-dataset")
    parser.add_argument("--coco-output")
    parser.add_argument("--m3fd-output")
    parser.add_argument("--launch-dir", default="runs/dual_launch")
    parser.add_argument("--epochs", type=int, help="Optional override for both pipelines")
    parser.add_argument("--allow-coco-subset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= len(args.gpus) <= 2 or any(not g.isdecimal() for g in args.gpus):
        raise ValueError("Supply one or two numeric GPU IDs")
    gpus = args.gpus if len(args.gpus) == 2 else args.gpus*2
    if args.mode == "parallel" and gpus[0] == gpus[1] and not args.allow_shared_gpu:
        raise ValueError("Parallel jobs share this GPU; use --mode sequential, two GPU IDs, or explicit --allow-shared-gpu")
    launch = Path(args.launch_dir)
    if not launch.is_absolute():
        launch = ROOT / launch
    if launch.exists() and any(launch.iterdir()):
        raise FileExistsError("Use a new launch directory to preserve previous evidence")
    jobs = []
    for index, (name, expected_kind) in enumerate([("coco","coco"),("m3fd","roboflow")]):
        config_path = Path(getattr(args, f"{name}_config"))
        config = json.loads((config_path if config_path.is_absolute() else ROOT/config_path).read_text())
        if config.get("dataset_file", "roboflow") != expected_kind:
            raise ValueError(f"Wrong dataset backend in {name} config")
        for key, override in [("dataset_dir",getattr(args,f"{name}_dataset")), ("output_dir",getattr(args,f"{name}_output"))]:
            path = Path(override or config[key]).expanduser()
            config[key] = str((path if path.is_absolute() else ROOT/path).resolve())
        if args.epochs is not None:
            config["epochs"] = args.epochs
        if config["epochs"] < 1:
            raise ValueError("epochs must be positive")
        audit_dataset(expected_kind,config["dataset_dir"],allow_coco_subset=args.allow_coco_subset)
        output = Path(config["output_dir"])
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Existing training output: {output}; resume independently using train.py")
        jobs.append({"name":name,"gpu":gpus[index],"config":config})
    if jobs[0]["config"]["output_dir"] == jobs[1]["config"]["output_dir"]:
        raise ValueError("Dataset pipelines must use separate output directories")
    launch.mkdir(parents=True,exist_ok=True)
    for job in jobs:
        config_path = launch / f'{job["name"]}_config.json'
        write_json(config_path,job["config"])
        output = job["config"]["output_dir"]
        train = [sys.executable,"-u",str(ROOT/"train.py"),"--config",str(config_path)]
        evaluate = [sys.executable,"-u",str(ROOT/"evaluate.py"),"--dataset-file",job["config"].get("dataset_file","roboflow"),
                    "--dataset",job["config"]["dataset_dir"],"--weights",str(Path(output)/"checkpoint_best_total.pth"),
                    "--split","valid","--output",str(Path(output)/"final_validation")]
        if job["name"] == "coco" and args.allow_coco_subset:
            train.append("--allow-coco-subset")
            evaluate.append("--allow-coco-subset")
        job["commands"] = [train,evaluate]
    write_json(launch/"launch_plan.json",{"mode":args.mode,"dry_run":args.dry_run,"jobs":jobs})
    if args.dry_run:
        print("Dataset audits passed; plan saved:",launch/"launch_plan.json")
        return
    cancelled = threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=2 if args.mode == "parallel" else 1) as pool:
            futures = [pool.submit(run_job,job,launch,cancelled) for job in jobs]
            try:
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                cancelled.set()
                raise
    except KeyboardInterrupt:
        cancelled.set()
        raise
    print("Both training/evaluation pipelines completed:",launch)


if __name__ == "__main__":
    main()
