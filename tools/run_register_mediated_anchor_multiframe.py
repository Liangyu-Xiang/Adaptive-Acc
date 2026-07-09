#!/usr/bin/env python3
"""Run full-dataset register-mediated anchor sweeps with exclusive-GPU timing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.utils.gpu_guard import wait_for_exclusive_gpu


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_7SCENES_ROOT = Path("/data/mmc_lyxiang/dataset/7scenes")
DEFAULT_TUM_ROOT = Path("/data/mmc_lyxiang/dataset/TUM-Dynamics")
DEFAULT_BASELINE_ROOT = REPO_ROOT / "outputs" / "fastvggt_merge_rates_multiframe"


@dataclass(frozen=True)
class Job:
    dataset: str
    frame_count: int
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/register_mediated_anchor_multiframe"),
    )
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seven-scenes-root", type=Path, default=DEFAULT_7SCENES_ROOT)
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_TUM_ROOT)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
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
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[100, 200])
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["7scenes", "tum_dynamics"],
        default=["7scenes", "tum_dynamics"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["lifting", "frame_pair_gated", "hybrid"],
        default=["lifting", "frame_pair_gated"],
    )
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=["balanced", "max_size"], default="max_size")
    parser.add_argument("--attention-mode", choices=["default", "register-only-zero-shot"], default="default")
    parser.add_argument("--anchor-layers", default="all")
    parser.add_argument("--anchor-ratio", type=float, default=0.2)
    parser.add_argument("--anchor-total", type=int, default=None)
    parser.add_argument("--anchor-min-per-frame", type=int, default=4)
    parser.add_argument("--anchor-tau", type=float, default=1.0)
    parser.add_argument("--anchor-uniform-mix", type=float, default=0.2)
    parser.add_argument("--anchor-score-alpha-cross", type=float, default=1.0)
    parser.add_argument("--anchor-score-beta-intra", type=float, default=0.2)
    parser.add_argument("--anchor-topm-frames", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def job_dir(args: argparse.Namespace, job: Job) -> Path:
    return args.output_root / f"{job.frame_count}frames" / job.dataset / job.mode


def baseline_metrics_path(args: argparse.Namespace, dataset: str, frame_count: int, ratio_tag: str) -> Path:
    return args.baseline_root / f"{frame_count}frames" / dataset / f"ratio_{ratio_tag}" / "metrics.json"


def command_for_job(args: argparse.Namespace, job: Job, gpu: str) -> list[str]:
    common = [
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
        "--image-resolution",
        str(args.image_resolution),
        "--resize-mode",
        args.resize_mode,
        "--merge-ratio",
        "0.0",
        "--timing-repeats",
        str(args.timing_repeats),
        "--use-register-mediated-anchor",
        "--adaptive-anchor-layers",
        args.anchor_layers,
        "--adaptive-anchor-ratio",
        str(args.anchor_ratio),
        "--adaptive-anchor-min-per-frame",
        str(args.anchor_min_per_frame),
        "--adaptive-anchor-tau",
        str(args.anchor_tau),
        "--adaptive-anchor-uniform-mix",
        str(args.anchor_uniform_mix),
        "--adaptive-anchor-mode",
        job.mode,
        "--adaptive-anchor-score-alpha-cross",
        str(args.anchor_score_alpha_cross),
        "--adaptive-anchor-score-beta-intra",
        str(args.anchor_score_beta_intra),
        "--adaptive-anchor-topm-frames",
        str(args.anchor_topm_frames),
    ]
    if args.anchor_total is not None:
        common.extend(["--adaptive-anchor-total", str(args.anchor_total)])
    if args.require_exclusive_gpu:
        common.extend(
            [
                "--require-exclusive-gpu",
                "--exclusive-gpu-index",
                str(gpu),
                "--exclusive-gpu-max-other-memory-mib",
                str(args.gpu_max_other_memory_mib),
            ]
        )
    if job.dataset == "7scenes":
        return [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_7scenes_paper.py"),
            "--data-root",
            str(args.seven_scenes_root),
            "--sampling-unit",
            "sequence",
            *common,
        ]
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_tum_dynamics_paper.py"),
        "--data-root",
        str(args.tum_root),
        "--sampling-pool",
        "full",
        *common,
    ]


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
        "anchor_mode": job.mode,
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
            text = log_path.read_text(encoding="utf-8", errors="replace").lower()
            if "out of memory" in text or "cuda oom" in text:
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
            f"[GPU{gpu}] start {job.dataset} {job.frame_count}f mode={job.mode}",
            flush=True,
        )
        result = run_job(args, job, gpu)
        print(
            f"[GPU{gpu}] done {job.dataset} {job.frame_count}f mode={job.mode}: {result['status']}",
            flush=True,
        )
        results.append(result)
    return results


def load_overall_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["overall"]


def write_comparison_report(args: argparse.Namespace) -> Path | None:
    method_specs: list[tuple[str, Callable[[str, int], Path]]] = [
        ("VGGT-Omega", lambda dataset, frame_count: baseline_metrics_path(args, dataset, frame_count, "00")),
        ("FastVGGT+VGGT-Omega", lambda dataset, frame_count: baseline_metrics_path(args, dataset, frame_count, "90")),
    ]
    for mode in args.modes:
        label = f"Register-Mediated ({mode})"
        method_specs.append(
            (
                label,
                lambda dataset, frame_count, mode=mode: job_dir(
                    args,
                    Job(dataset=dataset, frame_count=frame_count, mode=mode),
                )
                / "metrics.json",
            )
        )
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for frame_count in sorted(args.frame_counts):
            baseline_path = method_specs[0][1](dataset, frame_count)
            if not baseline_path.is_file():
                return None
            baseline_latency = float(load_overall_metrics(baseline_path)["model_latency_ms_mean"])
            for method, getter in method_specs:
                metrics_path = getter(dataset, frame_count)
                if not metrics_path.is_file():
                    return None
                overall = load_overall_metrics(metrics_path)
                latency = float(overall["model_latency_ms_mean"])
                rows.append(
                    {
                        "dataset": dataset,
                        "frame_count": frame_count,
                        "method": method,
                        "auc_3_percent": float(overall["auc_3_percent"]),
                        "auc_30_percent": float(overall["auc_30_percent"]),
                        "delta_1_25_percent": float(overall["delta_1_25_percent"]),
                        "abs_rel": float(overall["abs_rel"]),
                        "model_latency_ms_mean": latency,
                        "speedup_vs_vggt_omega": baseline_latency / latency,
                    }
                )
    if not rows:
        return None

    csv_path = args.output_root / "comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Register-mediated anchor full-dataset comparison (100 / 200 frames)",
        "",
        f"Register-mediated runs use `layers={args.anchor_layers}`, `anchor_ratio={args.anchor_ratio}`, "
        f"`topm_frames={args.anchor_topm_frames}`, `timing_repeats={args.timing_repeats}`.",
        "Baseline and FastVGGT metrics are loaded from the existing full-dataset sweep outputs.",
        "",
    ]
    for dataset in args.datasets:
        for frame_count in sorted(args.frame_counts):
            lines.append(f"## {dataset}, {frame_count} frames")
            lines.append("")
            lines.append("| Method | AUC@3 (%) | AUC@30 (%) | delta<1.25 (%) | AbsRel | Latency (ms) | Speedup vs VGGT-Omega |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for row in rows:
                if row["dataset"] != dataset or row["frame_count"] != frame_count:
                    continue
                lines.append(
                    f"| {row['method']} | {row['auc_3_percent']:.2f} | {row['auc_30_percent']:.2f} | "
                    f"{row['delta_1_25_percent']:.2f} | {row['abs_rel']:.4f} | "
                    f"{row['model_latency_ms_mean']:.1f} | {row['speedup_vs_vggt_omega']:.3f}x |"
                )
            lines.append("")
    report_path = args.output_root / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        Job(dataset=dataset, frame_count=frame_count, mode=mode)
        for frame_count in args.frame_counts
        for dataset in args.datasets
        for mode in args.modes
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

    report_path = write_comparison_report(args)
    summary = {
        "output_root": str(args.output_root),
        "baseline_root": str(args.baseline_root),
        "gpus": args.gpus,
        "frame_counts": args.frame_counts,
        "datasets": args.datasets,
        "modes": args.modes,
        "results": all_results,
        "comparison_report": None if report_path is None else str(report_path),
    }
    write_status(args.output_root / "run_summary.json", summary)
    ok = all(result["status"] in {"success", "skipped", "dry_run"} for result in all_results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
