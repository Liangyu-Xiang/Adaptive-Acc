#!/usr/bin/env python3
"""Run register-mediated proxy analysis on 7Scenes and TUM-Dynamics sweeps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_7scenes_paper as seven_eval
from scripts import eval_tum_dynamics_paper as tum_eval


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_7SCENES_ROOT = Path("/data/mmc_lyxiang/dataset/7scenes")
DEFAULT_TUM_ROOT = Path("/data/mmc_lyxiang/dataset/TUM-Dynamics")


@dataclass(frozen=True)
class Job:
    dataset: str
    sequence: str
    frame_count: int
    manifest: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "register_proxy_dataset_sweep_gpu6")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seven-scenes-root", type=Path, default=DEFAULT_7SCENES_ROOT)
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_TUM_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=("7scenes", "tum_dynamics"), default=("7scenes", "tum_dynamics"))
    parser.add_argument("--frame-counts", nargs="+", type=int, default=(100, 200))
    parser.add_argument("--num-sequences", type=int, default=1)
    parser.add_argument("--seven-scenes-sequences", nargs="*", default=None)
    parser.add_argument("--tum-sequences", nargs="*", default=None)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--layers", default="6-18")
    parser.add_argument("--image-resolution", type=int, default=256)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--anchor-ratio", type=float, default=0.2)
    parser.add_argument("--anchor-total", type=int, default=None)
    parser.add_argument("--topk-list", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--query-chunk", type=int, default=32)
    parser.add_argument("--query-sample-total", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--eval-anchor-strategies", action="store_true")
    parser.add_argument("--save-visualization", action="store_true")
    parser.add_argument("--save-attention-stats", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count < 2:
        raise ValueError("frame count must be at least 2")
    if length < count:
        raise ValueError(f"Need {count} frames but sequence has only {length}")
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


def write_manifest(path: Path, image_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(item) for item in image_paths) + "\n", encoding="utf-8")


def choose_7scenes_jobs(args: argparse.Namespace) -> list[Job]:
    sequence_dirs = seven_eval.select_sequence_dirs(args.seven_scenes_root, args.seven_scenes_sequences)
    jobs: list[Job] = []
    for sequence_dir in sequence_dirs:
        sequence_name = f"{sequence_dir.parent.name}_{sequence_dir.name}"
        records = seven_eval.load_frame_records(sequence_dir)
        if not records:
            continue
        for frame_count in args.frame_counts:
            if len(records) < frame_count:
                continue
            indices = evenly_spaced_indices(len(records), frame_count)
            image_paths = [records[index].rgb_path for index in indices]
            manifest = args.output_root / "manifests" / f"7scenes_{sequence_name}_{frame_count}f.txt"
            output_dir = args.output_root / "7scenes" / sequence_name / f"{frame_count}frames"
            write_manifest(manifest, image_paths)
            jobs.append(Job("7scenes", sequence_name, frame_count, manifest, output_dir))
        if len({job.sequence for job in jobs if job.dataset == "7scenes"}) >= args.num_sequences:
            break
    return jobs


def choose_tum_jobs(args: argparse.Namespace) -> list[Job]:
    sequence_dirs = tum_eval.select_sequence_dirs(args.tum_root, args.tum_sequences)
    jobs: list[Job] = []
    for sequence_dir in sequence_dirs:
        sequence_name = sequence_dir.name
        records = tum_eval.load_frame_records(sequence_dir, tolerance=0.02)
        if not records:
            continue
        for frame_count in args.frame_counts:
            if len(records) < frame_count:
                continue
            indices = evenly_spaced_indices(len(records), frame_count)
            image_paths = [records[index].rgb_path for index in indices]
            manifest = args.output_root / "manifests" / f"tum_dynamics_{sequence_name}_{frame_count}f.txt"
            output_dir = args.output_root / "tum_dynamics" / sequence_name / f"{frame_count}frames"
            write_manifest(manifest, image_paths)
            jobs.append(Job("tum_dynamics", sequence_name, frame_count, manifest, output_dir))
        if len({job.sequence for job in jobs if job.dataset == "tum_dynamics"}) >= args.num_sequences:
            break
    return jobs


def command_for_job(args: argparse.Namespace, job: Job) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "analyze_register_mediated_proxy.py"),
        "--input_path",
        str(job.manifest),
        "--output_dir",
        str(job.output_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        "cuda:0",
        "--layers",
        args.layers,
        "--max_samples",
        str(args.max_samples),
        "--anchor_ratio",
        str(args.anchor_ratio),
        "--topk_list",
        args.topk_list,
        "--image-resolution",
        str(args.image_resolution),
        "--resize-mode",
        args.resize_mode,
        "--query_chunk",
        str(args.query_chunk),
        "--query_sample_total",
        str(args.query_sample_total),
        "--timing_repeats",
        str(args.timing_repeats),
    ]
    if args.anchor_total is not None:
        cmd.extend(["--anchor_total", str(args.anchor_total)])
    if args.eval_anchor_strategies:
        cmd.append("--eval_anchor_strategies")
    if args.save_visualization:
        cmd.append("--save_visualization")
    if args.save_attention_stats:
        cmd.append("--save_attention_stats")
    return cmd


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def job_to_dict(job: Job) -> dict[str, object]:
    return {
        "dataset": job.dataset,
        "sequence": job.sequence,
        "frame_count": job.frame_count,
        "manifest": str(job.manifest),
        "output_dir": str(job.output_dir),
    }


def run_job(args: argparse.Namespace, job: Job) -> dict[str, object]:
    summary_path = job.output_dir / "stage1_summary.json"
    log_path = job.output_dir / "run.log"
    status_path = job.output_dir / "status.json"
    if args.skip_existing and summary_path.is_file():
        return {"status": "skipped", "reason": "stage1_summary_exists", "job": job_to_dict(job)}
    cmd = command_for_job(args, job)
    status = {
        "status": "running",
        "dataset": job.dataset,
        "sequence": job.sequence,
        "frame_count": job.frame_count,
        "gpu": args.gpu,
        "command": cmd,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write_json(status_path, status)
    if args.dry_run:
        return {**status, "status": "dry_run"}
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) if not env.get("PYTHONPATH") else f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
    job.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    final = {
        **status,
        "status": "success" if completed.returncode == 0 and summary_path.is_file() else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "log_path": str(log_path),
        "summary_path": str(summary_path) if summary_path.is_file() else None,
    }
    if completed.returncode != 0:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        if "out of memory" in text.lower() or "cuda oom" in text.lower():
            final["status"] = "oom"
    write_json(status_path, final)
    return final


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = []
    if "7scenes" in args.datasets:
        jobs.extend(choose_7scenes_jobs(args))
    if "tum_dynamics" in args.datasets:
        jobs.extend(choose_tum_jobs(args))
    if not jobs:
        raise RuntimeError("No jobs selected; check dataset roots and frame counts")
    results = []
    for job in jobs:
        print(f"[GPU{args.gpu}] {job.dataset} {job.sequence} {job.frame_count} frames", flush=True)
        result = run_job(args, job)
        print(f"[GPU{args.gpu}] {job.dataset} {job.sequence} {job.frame_count}: {result['status']}", flush=True)
        results.append(result)
    write_json(
        args.output_root / "run_summary.json",
        {
            "gpu": args.gpu,
            "frame_counts": args.frame_counts,
            "layers": args.layers,
            "image_resolution": args.image_resolution,
            "resize_mode": args.resize_mode,
            "query_sample_total": args.query_sample_total,
            "jobs": [job_to_dict(job) for job in jobs],
            "results": results,
        },
    )
    return 0 if all(result["status"] in {"success", "skipped", "dry_run"} for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
