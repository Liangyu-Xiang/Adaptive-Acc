#!/usr/bin/env python3
"""Run full-sequence FastVGGT merge-ratio sweeps for multiple frame counts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.utils.gpu_guard import wait_for_exclusive_gpu

DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_7SCENES_ROOT = Path("/data/mmc_lyxiang/dataset/7scenes")
DEFAULT_TUM_ROOT = Path("/data/mmc_lyxiang/dataset/TUM-Dynamics")


@dataclass(frozen=True)
class Job:
    dataset: str
    frame_count: int
    ratio: float

    @property
    def tag(self) -> str:
        return f"{int(round(100 * self.ratio)):02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fastvggt_merge_rates_multiframe"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seven-scenes-root", type=Path, default=DEFAULT_7SCENES_ROOT)
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_TUM_ROOT)
    parser.add_argument("--gpus", nargs="+", default=["3", "4", "5"])
    parser.add_argument(
        "--require-exclusive-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for each physical GPU to be free and bind child jobs via CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--gpu-poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval while waiting for a GPU to become exclusive.",
    )
    parser.add_argument(
        "--gpu-max-other-memory-mib",
        type=int,
        default=512,
        help="Allowed residual non-target memory on a supposedly idle physical GPU.",
    )
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[100, 200, 300, 400, 500])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--datasets", nargs="+", choices=["7scenes", "tum_dynamics"], default=["7scenes", "tum_dynamics"])
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=["balanced", "max_size"], default="max_size")
    parser.add_argument("--attention-mode", choices=["default", "register-only-zero-shot"], default="default")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def job_dir(args: argparse.Namespace, job: Job) -> Path:
    return args.output_root / f"{job.frame_count}frames" / job.dataset / f"ratio_{job.tag}"


def command_for_job(args: argparse.Namespace, job: Job, gpu: str) -> list[str]:
    if job.dataset == "7scenes":
        script = REPO_ROOT / "scripts" / "eval_7scenes_paper.py"
        data_root = args.seven_scenes_root
        cmd = [
            sys.executable,
            str(script),
            "--data-root",
            str(data_root),
            "--checkpoint",
            str(args.checkpoint),
            "--output-dir",
            str(job_dir(args, job)),
            "--device",
            "cuda:0",
            "--attention-mode",
            args.attention_mode,
            "--seed",
            str(args.seed),
            "--num-frames",
            str(job.frame_count),
            "--sampling-unit",
            "sequence",
            "--image-resolution",
            str(args.image_resolution),
            "--resize-mode",
            args.resize_mode,
            "--merge-ratio",
            str(job.ratio),
            "--timing-repeats",
            str(args.timing_repeats),
        ]
    else:
        script = REPO_ROOT / "scripts" / "eval_tum_dynamics_paper.py"
        data_root = args.tum_root
        cmd = [
            sys.executable,
            str(script),
            "--data-root",
            str(data_root),
            "--checkpoint",
            str(args.checkpoint),
            "--output-dir",
            str(job_dir(args, job)),
            "--device",
            "cuda:0",
            "--attention-mode",
            args.attention_mode,
            "--timing-repeats",
            str(args.timing_repeats),
            "--seed",
            str(args.seed),
            "--num-frames",
            str(job.frame_count),
            "--sampling-pool",
            "full",
            "--image-resolution",
            str(args.image_resolution),
            "--resize-mode",
            args.resize_mode,
            "--merge-ratio",
            str(job.ratio),
        ]
    if args.require_exclusive_gpu:
        cmd.extend(
            [
                "--require-exclusive-gpu",
                "--exclusive-gpu-index",
                str(gpu),
                "--exclusive-gpu-max-other-memory-mib",
                str(args.gpu_max_other_memory_mib),
            ]
        )
    return cmd


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_job(args: argparse.Namespace, job: Job, gpu: str) -> dict[str, object]:
    out_dir = job_dir(args, job)
    metrics_path = out_dir / "metrics.json"
    status_path = out_dir / "status.json"
    log_path = out_dir / "run.log"
    if metrics_path.is_file():
        return {"status": "skipped", "reason": "metrics_exists", "job": job.__dict__, "gpu": gpu}

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.require_exclusive_gpu:
        wait_for_exclusive_gpu(
            gpu,
            max_other_memory_mib=args.gpu_max_other_memory_mib,
            poll_seconds=args.gpu_poll_seconds,
        )
    cmd = command_for_job(args, job, gpu)
    status = {
        "status": "running",
        "dataset": job.dataset,
        "frame_count": job.frame_count,
        "merge_ratio": job.ratio,
        "gpu": gpu,
        "visible_device": "cuda:0",
        "command": cmd,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write_status(status_path, status)
    if args.dry_run:
        return {**status, "status": "dry_run"}

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not env.get("PYTHONPATH")
        else f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
    )
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(" ".join(cmd) + "\n\n")
        log_handle.flush()
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    final_status = {
        **status,
        "status": "success" if completed.returncode == 0 and metrics_path.is_file() else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
    }
    if completed.returncode != 0:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lowered = text.lower()
            if "out of memory" in lowered or "cuda oom" in lowered:
                final_status["status"] = "oom"
        except OSError:
            pass
    write_status(status_path, final_status)
    return final_status


def worker(args: argparse.Namespace, gpu: str, jobs: list[Job]) -> list[dict[str, object]]:
    results = []
    for job in jobs:
        if args.require_exclusive_gpu:
            print(f"[GPU{gpu}] wait for exclusive access", flush=True)
        print(
            f"[GPU{gpu}] start {job.dataset} {job.frame_count}f ratio={job.ratio:.1f}",
            flush=True,
        )
        result = run_job(args, job, gpu)
        print(
            f"[GPU{gpu}] done {job.dataset} {job.frame_count}f ratio={job.ratio:.1f}: {result['status']}",
            flush=True,
        )
        results.append(result)
    return results


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        Job(dataset=dataset, frame_count=frame_count, ratio=ratio)
        for frame_count in args.frame_counts
        for dataset in args.datasets
        for ratio in args.ratios
    ]
    assignments = {gpu: [] for gpu in args.gpus}
    for index, job in enumerate(jobs):
        assignments[args.gpus[index % len(args.gpus)]].append(job)

    print(f"Scheduled {len(jobs)} jobs over GPUs {', '.join(args.gpus)}", flush=True)
    all_results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(worker, args, gpu, gpu_jobs): gpu
            for gpu, gpu_jobs in assignments.items()
        }
        for future in as_completed(futures):
            all_results.extend(future.result())

    write_status(
        args.output_root / "run_summary.json",
        {
            "output_root": str(args.output_root),
            "gpus": args.gpus,
            "frame_counts": args.frame_counts,
            "ratios": args.ratios,
            "datasets": args.datasets,
            "results": all_results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
