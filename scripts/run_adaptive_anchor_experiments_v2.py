#!/usr/bin/env python3
"""Run resumable adaptive-anchor sweeps on 7-Scenes and TUM-Dynamics."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "adaptive_anchor_experiments_v2"
DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_7SCENES_ROOT = Path("/data/mmc_lyxiang/dataset/7scenes")
DEFAULT_TUM_ROOT = Path("/data/mmc_lyxiang/dataset/TUM-Dynamics")
DEFAULT_FASTVGGT_BASELINE_ROOT = REPO_ROOT / "outputs" / "fastvggt_merge_rates_multiframe"

DEFAULT_7SCENES_SEQUENCES = ("chess/seq-03", "fire/seq-03", "office/seq-02", "redkitchen/seq-03")
DEFAULT_TUM_SEQUENCES = (
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_walking_xyz",
)

STAGE_DIRS = (
    "metadata",
    "stage0_sanity",
    "stage1_anchor_signal",
    "stage2_layer_sensitivity",
    "stage3_register_gating",
    "stage4_query_conditioned",
    "stage5_proxy_quota",
    "stage6_full_task_eval",
    "stage7_runtime",
    "summaries",
    "plots",
    "logs",
    "failed_jobs",
)


@dataclass(frozen=True)
class StrategyConfig:
    label: str
    strategy: str | None
    use_adaptive: bool
    score_mode: str = "intra"
    intra_source: str = "cached_frame_qk"
    frame_budget_mode: str = "hybrid"
    proxy_quota_ratio: float = 0.0
    topm_frames: int | None = 4
    query_eta: float = 0.1
    lambda_intra: float = 0.7
    lambda_reg: float = 0.3
    tau: float = 0.5
    uniform_mix: float = 0.05


@dataclass
class Job:
    job_id: str
    stage: str
    dataset: str
    sequences: list[str]
    frame_count: int
    strategy_label: str
    strategy: str | None
    use_adaptive: bool
    anchor_ratio: float | None
    layers: str
    layer_group: str
    topm_frames: int | None
    score_mode: str
    intra_source: str
    frame_budget_mode: str
    proxy_quota_ratio: float
    query_eta: float
    lambda_intra: float
    lambda_reg: float
    tau: float
    uniform_mix: float
    debug: bool
    profile: bool
    timing: bool
    output_dir: str
    extra_args: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seven-scenes-root", type=Path, default=DEFAULT_7SCENES_ROOT)
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_TUM_ROOT)
    parser.add_argument("--fastvggt-baseline-root", type=Path, default=DEFAULT_FASTVGGT_BASELINE_ROOT)
    parser.add_argument("--fastvggt-baseline-ratio-tag", default="90")
    parser.add_argument("--datasets", nargs="+", default=["7scenes", "tum_dynamics"], choices=("7scenes", "tum_dynamics"))
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--anchor-ratios", nargs="+", type=float, default=[0.2])
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU ids.")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", default="max_size", choices=("balanced", "max_size"))
    parser.add_argument("--merge-ratio", type=float, default=0.0)
    parser.add_argument("--conda-env", default="omega")
    parser.add_argument("--no-conda", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--run-stage0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-runtime", action="store_true")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "full_global_attention",
            "fixed_grid",
            "random",
            "proxy",
            "proxy_intra",
            "intra_current",
            "intra_cached",
            "register_gated_intra",
            "register_gated_intra_query",
            "quota_intra_proxy",
            "random_frame_intra",
            "temporal_neighbor_intra",
        ],
    )
    parser.add_argument("--seven-scenes-sequences", nargs="+", default=list(DEFAULT_7SCENES_SEQUENCES))
    parser.add_argument("--tum-sequences", nargs="+", default=list(DEFAULT_TUM_SEQUENCES))
    return parser.parse_args()


def strategy_catalog() -> dict[str, StrategyConfig]:
    return {
        "full_global_attention": StrategyConfig("full_global_attention", None, False, topm_frames=None),
        "fixed_grid": StrategyConfig("fixed_grid", "fixed_grid", True, frame_budget_mode="uniform", topm_frames=None),
        "random": StrategyConfig("random", "random", True, frame_budget_mode="uniform", topm_frames=None),
        "proxy": StrategyConfig("proxy", "proxy", True, score_mode="proxy", frame_budget_mode="uniform", topm_frames=None),
        "proxy_intra": StrategyConfig("proxy_intra", "proxy_intra", True, score_mode="linear_fusion", topm_frames=None),
        "intra_current": StrategyConfig("intra_current", "all_frame_intra", True, intra_source="current_inter_qk", topm_frames=None),
        "intra_cached": StrategyConfig("intra_cached", "all_frame_intra", True, intra_source="cached_frame_qk", topm_frames=None),
        "all_frame_intra": StrategyConfig("all_frame_intra", "all_frame_intra", True, topm_frames=None),
        "random_frame_intra": StrategyConfig("random_frame_intra", "random_frame_intra", True, frame_budget_mode="uniform"),
        "temporal_neighbor_intra": StrategyConfig("temporal_neighbor_intra", "temporal_neighbor_intra", True, frame_budget_mode="uniform"),
        "oracle_frame_intra": StrategyConfig("oracle_frame_intra", "oracle_frame_intra", True, frame_budget_mode="uniform"),
        "register_gated_intra": StrategyConfig("register_gated_intra", "register_gated_intra", True),
        "register_gated_intra_query": StrategyConfig("register_gated_intra_query", "register_gated_intra_query", True),
        "quota_intra_proxy": StrategyConfig(
            "quota_intra_proxy",
            "quota_intra_proxy",
            True,
            score_mode="quota_union",
            proxy_quota_ratio=0.25,
            topm_frames=None,
        ),
        "oracle": StrategyConfig("oracle", "oracle", True, frame_budget_mode="uniform", topm_frames=None),
    }


def prepare_output_dirs(root: Path) -> None:
    for name in STAGE_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def get_global_layer_metadata() -> dict[str, Any]:
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from vggt_omega.models.aggregator import Aggregator

        aggregator = Aggregator(global_merging=False, merging=None, merge_ratio=0.0)
        schedule = list(aggregator.inter_frame_attention_types)
        del aggregator
    except Exception as error:  # pragma: no cover - fallback only for broken local imports.
        schedule = ["global"] * 24
        for index in (2, 6, 9, 14, 20):
            schedule[index] = "register"
        return {"attention_schedule": schedule, "error": repr(error), **split_global_layers(schedule)}
    return {"attention_schedule": schedule, **split_global_layers(schedule)}


def split_global_layers(schedule: list[str]) -> dict[str, Any]:
    global_layers = [idx for idx, kind in enumerate(schedule) if kind == "global"]
    n = len(global_layers)
    first_cut = math.ceil(n / 3)
    second_cut = math.ceil(2 * n / 3)
    groups = {
        "early": global_layers[:first_cut],
        "middle": global_layers[first_cut:second_cut],
        "late": global_layers[second_cut:],
        "all": global_layers,
    }
    return {"global_layers": global_layers, "layer_groups": groups}


def layer_spec(layers: list[int]) -> str:
    if not layers:
        return "none"
    ranges: list[str] = []
    start = prev = layers[0]
    for layer in layers[1:]:
        if layer == prev + 1:
            prev = layer
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = layer
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def build_jobs(args: argparse.Namespace, metadata: dict[str, Any]) -> list[Job]:
    catalog = strategy_catalog()
    unknown = [name for name in args.strategies if name not in catalog]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Known: {sorted(catalog)}")
    all_layers = layer_spec(metadata["layer_groups"]["all"])
    jobs: list[Job] = []
    if args.run_stage0:
        stage0_layers = layer_spec(metadata["global_layers"][:1])
        stage0_strategies = [
            "full_global_attention",
            "fixed_grid",
            "random",
            "intra_cached",
            "proxy",
            "register_gated_intra",
            "register_gated_intra_query",
            "quota_intra_proxy",
            "random_frame_intra",
            "temporal_neighbor_intra",
            "oracle",
        ]
        for name in stage0_strategies:
            cfg = catalog[name]
            ratio = None if not cfg.use_adaptive else 0.2
            jobs.append(
                make_job(
                    args=args,
                    stage="stage0_sanity",
                    dataset="7scenes",
                    sequences=[args.seven_scenes_sequences[0]],
                    frame_count=20,
                    cfg=cfg,
                    anchor_ratio=ratio,
                    layers=stage0_layers,
                    layer_group="single_global",
                    debug=cfg.use_adaptive,
                    profile=cfg.use_adaptive,
                    timing=False,
                )
            )

    for dataset in args.datasets:
        sequences = args.seven_scenes_sequences if dataset == "7scenes" else args.tum_sequences
        for frame_count in args.frame_counts:
            for name in args.strategies:
                cfg = catalog[name]
                ratios = [None] if not cfg.use_adaptive else args.anchor_ratios
                for ratio in ratios:
                    jobs.append(
                        make_job(
                            args=args,
                            stage="stage6_full_task_eval",
                            dataset=dataset,
                            sequences=list(sequences),
                            frame_count=frame_count,
                            cfg=cfg,
                            anchor_ratio=ratio,
                            layers=all_layers,
                            layer_group="all",
                            debug=False,
                            profile=False,
                            timing=False,
                        )
                    )
    if args.run_runtime:
        runtime_names = ["full_global_attention", "intra_cached", "register_gated_intra", "register_gated_intra_query"]
        for dataset in args.datasets:
            sequences = args.seven_scenes_sequences[:1] if dataset == "7scenes" else args.tum_sequences[:1]
            for frame_count in args.frame_counts:
                for name in runtime_names:
                    cfg = catalog[name]
                    ratio = None if not cfg.use_adaptive else args.anchor_ratios[-1]
                    jobs.append(
                        make_job(
                            args=args,
                            stage="stage7_runtime",
                            dataset=dataset,
                            sequences=list(sequences),
                            frame_count=frame_count,
                            cfg=cfg,
                            anchor_ratio=ratio,
                            layers=all_layers,
                            layer_group="all",
                            debug=False,
                            profile=cfg.use_adaptive,
                            timing=True,
                        )
                    )
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    return jobs


def make_job(
    args: argparse.Namespace,
    stage: str,
    dataset: str,
    sequences: list[str],
    frame_count: int,
    cfg: StrategyConfig,
    anchor_ratio: float | None,
    layers: str,
    layer_group: str,
    debug: bool,
    profile: bool,
    timing: bool,
) -> Job:
    ratio_part = "full" if anchor_ratio is None else f"r{anchor_ratio:g}".replace(".", "p")
    seq_part = "seqs" + str(len(sequences))
    job_id = "__".join([stage, dataset, f"f{frame_count}", cfg.label, ratio_part, layer_group, seq_part])
    output_dir = args.output_root / stage / job_id
    return Job(
        job_id=job_id,
        stage=stage,
        dataset=dataset,
        sequences=sequences,
        frame_count=frame_count,
        strategy_label=cfg.label,
        strategy=cfg.strategy,
        use_adaptive=cfg.use_adaptive,
        anchor_ratio=anchor_ratio,
        layers=layers,
        layer_group=layer_group,
        topm_frames=cfg.topm_frames,
        score_mode=cfg.score_mode,
        intra_source=cfg.intra_source,
        frame_budget_mode=cfg.frame_budget_mode,
        proxy_quota_ratio=cfg.proxy_quota_ratio,
        query_eta=cfg.query_eta,
        lambda_intra=cfg.lambda_intra,
        lambda_reg=cfg.lambda_reg,
        tau=cfg.tau,
        uniform_mix=cfg.uniform_mix,
        debug=debug,
        profile=profile,
        timing=timing,
        output_dir=str(output_dir),
    )


def build_command(args: argparse.Namespace, job: Job) -> list[str]:
    if args.no_conda or not shutil.which("conda"):
        prefix = [sys.executable]
    else:
        prefix = ["conda", "run", "-n", args.conda_env, "python"]
    script = "scripts/eval_7scenes_paper.py" if job.dataset == "7scenes" else "scripts/eval_tum_dynamics_paper.py"
    data_root = args.seven_scenes_root if job.dataset == "7scenes" else args.tum_root
    command = [
        *prefix,
        script,
        "--data-root",
        str(data_root),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        job.output_dir,
        "--device",
        "cuda:0",
        "--seed",
        str(args.seed),
        "--num-frames",
        str(job.frame_count),
        "--image-resolution",
        str(args.image_resolution),
        "--resize-mode",
        args.resize_mode,
        "--merge-ratio",
        str(args.merge_ratio),
        "--sequences",
        *job.sequences,
    ]
    if job.dataset == "7scenes":
        command.extend(["--sampling-unit", "sequence"])
    else:
        command.extend(["--sampling-pool", "full"])
    if job.timing:
        command.extend(["--timing-repeats", str(args.timing_repeats)])
    else:
        command.append("--skip-timing")
    if job.use_adaptive:
        command.extend(
            [
                "--use-adaptive-kv-anchor",
                "--adaptive-anchor-strategy",
                str(job.strategy),
                "--adaptive-anchor-layers",
                job.layers,
                "--adaptive-anchor-ratio",
                str(job.anchor_ratio),
                "--adaptive-anchor-score-mode",
                job.score_mode,
                "--adaptive-anchor-intra-source",
                job.intra_source,
                "--adaptive-anchor-frame-budget-mode",
                job.frame_budget_mode,
                "--adaptive-anchor-frame-budget-lambda-intra",
                str(job.lambda_intra),
                "--adaptive-anchor-frame-budget-lambda-reg",
                str(job.lambda_reg),
                "--adaptive-anchor-tau",
                str(job.tau),
                "--adaptive-anchor-uniform-mix",
                str(job.uniform_mix),
                "--adaptive-anchor-proxy-quota-ratio",
                str(job.proxy_quota_ratio),
                "--adaptive-anchor-query-conditioned-eta",
                str(job.query_eta),
                "--adaptive-anchor-random-seed",
                str(args.seed),
            ]
        )
        if job.topm_frames is None:
            command.extend(["--adaptive-anchor-topm-frames", "0"])
        else:
            command.extend(["--adaptive-anchor-topm-frames", str(job.topm_frames)])
        if job.debug:
            command.extend(
                [
                    "--adaptive-anchor-debug",
                    "--adaptive-anchor-debug-dir",
                    str(Path(job.output_dir) / "debug"),
                ]
            )
        if job.profile:
            command.append("--adaptive-anchor-profile")
    command.extend(job.extra_args)
    return command


def run_jobs(args: argparse.Namespace, jobs: list[Job]) -> None:
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must name at least one GPU")
    work: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        work.put(job)
    failures: list[str] = []
    failure_lock = threading.Lock()

    def worker(gpu: str, worker_id: int) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            try:
                run_one_job(args, job, gpu, worker_id)
            except Exception as error:
                with failure_lock:
                    failures.append(f"{job.job_id}: {error}")
                if args.stop_on_failure:
                    while True:
                        try:
                            work.get_nowait()
                            work.task_done()
                        except queue.Empty:
                            break
            finally:
                work.task_done()

    threads: list[threading.Thread] = []
    for gpu in gpus:
        for worker_id in range(args.workers_per_gpu):
            thread = threading.Thread(target=worker, args=(gpu, worker_id), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    if failures and args.stop_on_failure:
        raise RuntimeError("Failures encountered:\n" + "\n".join(failures[:10]))


def run_one_job(args: argparse.Namespace, job: Job, gpu: str, worker_id: int) -> None:
    output_dir = Path(job.output_dir)
    metrics_path = output_dir / "metrics.json"
    status_path = output_dir / "job_status.json"
    if args.resume and metrics_path.is_file():
        print(f"[skip] {job.job_id}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "logs" / f"{job.job_id}.gpu{gpu}.w{worker_id}.log"
    command = build_command(args, job)
    with (output_dir / "job.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(job), handle, indent=2)
        handle.write("\n")
    with (output_dir / "command.json").open("w", encoding="utf-8") as handle:
        json.dump(command, handle, indent=2)
        handle.write("\n")
    status = {
        "job_id": job.job_id,
        "gpu": gpu,
        "worker_id": worker_id,
        "started_at": now_iso(),
        "command": command,
    }
    write_json(status_path, status)
    if args.dry_run:
        print("[dry-run]", " ".join(command))
        status.update({"returncode": 0, "ended_at": now_iso(), "dry_run": True})
        write_json(status_path, status)
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    started = time.perf_counter()
    print(f"[run gpu{gpu}] {job.job_id}")
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n\n")
        log_handle.flush()
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.perf_counter() - started
    status.update({"returncode": process.returncode, "ended_at": now_iso(), "elapsed_seconds": elapsed, "log": str(log_path)})
    write_json(status_path, status)
    if process.returncode != 0:
        failed_copy = args.output_root / "failed_jobs" / f"{job.job_id}.json"
        write_json(failed_copy, {**status, "job": asdict(job)})
        print(f"[fail gpu{gpu}] {job.job_id} rc={process.returncode}")
        return
    print(f"[done gpu{gpu}] {job.job_id} {elapsed:.1f}s")


def collect_results(args: argparse.Namespace, jobs: list[Job], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        row = asdict(job)
        row.pop("extra_args", None)
        metrics_path = Path(job.output_dir) / "metrics.json"
        status_path = Path(job.output_dir) / "job_status.json"
        status = read_json(status_path)
        row["returncode"] = status.get("returncode")
        row["elapsed_seconds"] = status.get("elapsed_seconds")
        row["total_input_frames"] = int(job.frame_count) * len(job.sequences)
        row["wall_fps"] = _safe_div(row["total_input_frames"], row["elapsed_seconds"])
        row["metrics_path"] = str(metrics_path)
        row["status_path"] = str(status_path)
        row["success"] = metrics_path.is_file() and status.get("returncode") == 0
        if metrics_path.is_file():
            metrics = read_json(metrics_path)
            protocol = metrics.get("protocol", {})
            overall = metrics.get("overall", {})
            row.update(
                {
                    "auc_3_percent": overall.get("auc_3_percent"),
                    "auc_30_percent": overall.get("auc_30_percent"),
                    "delta_1_25_percent": overall.get("delta_1_25_percent"),
                    "abs_rel": overall.get("abs_rel"),
                    "model_latency_ms_mean": overall.get("model_latency_ms_mean"),
                    "model_forward_fps": _safe_div(job.frame_count * 1000.0, overall.get("model_latency_ms_mean")),
                    "peak_allocated_gib_max": overall.get("peak_allocated_gib_max"),
                    "peak_reserved_gib_max": overall.get("peak_reserved_gib_max"),
                    "num_sequences": protocol.get("num_sequences"),
                    "num_pose_pairs": protocol.get("num_pose_pairs"),
                    "sampled_frames_path": str(Path(job.output_dir) / "sampled_frames.json"),
                    "active_layers": json.dumps(protocol.get("adaptive_anchor_active_layers", [])),
                    "attention_schedule": json.dumps(protocol.get("attention_schedule", metadata.get("attention_schedule"))),
                }
            )
        rows.append(row)
    rows.extend(collect_fastvggt_reference_rows(args, metadata))
    write_csv(args.output_root / "summaries" / "results.csv", rows)
    for stage in sorted({job.stage for job in jobs}):
        stage_rows = [row for row in rows if row["stage"] == stage]
        name = {
            "stage0_sanity": "sanity_results.csv",
            "stage1_anchor_signal": "stage1_raw.csv",
            "stage2_layer_sensitivity": "layer_sensitivity.csv",
            "stage3_register_gating": "stage3_raw.csv",
        }.get(stage, f"{stage}.csv")
        write_csv(args.output_root / stage / name, stage_rows)
    write_report(args, rows, metadata)
    return rows


def collect_fastvggt_reference_rows(args: argparse.Namespace, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ratio_tag = str(args.fastvggt_baseline_ratio_tag)
    try:
        merge_ratio = float(ratio_tag) / 100.0
    except ValueError:
        merge_ratio = None
    for dataset in args.datasets:
        for frame_count in args.frame_counts:
            metrics_path = (
                args.fastvggt_baseline_root
                / f"{int(frame_count)}frames"
                / dataset
                / f"ratio_{ratio_tag}"
                / "metrics.json"
            )
            if not metrics_path.is_file():
                continue
            metrics = read_json(metrics_path)
            protocol = metrics.get("protocol", {})
            overall = metrics.get("overall", {})
            num_sequences = protocol.get("num_sequences")
            total_input_frames = None
            if num_sequences is not None:
                total_input_frames = int(frame_count) * int(num_sequences)
            latency = overall.get("model_latency_ms_mean")
            rows.append(
                {
                    "job_id": f"reference_fastvggt__{dataset}__f{int(frame_count)}__ratio_{ratio_tag}",
                    "stage": "reference_fastvggt",
                    "dataset": dataset,
                    "sequences": [],
                    "frame_count": int(frame_count),
                    "strategy_label": f"FastVGGT+Baseline_r{ratio_tag}",
                    "strategy": "fastvggt_token_merging",
                    "use_adaptive": False,
                    "anchor_ratio": None,
                    "layers": "",
                    "layer_group": "all",
                    "topm_frames": None,
                    "score_mode": "",
                    "intra_source": "",
                    "frame_budget_mode": "",
                    "proxy_quota_ratio": "",
                    "query_eta": "",
                    "lambda_intra": "",
                    "lambda_reg": "",
                    "tau": "",
                    "uniform_mix": "",
                    "debug": False,
                    "profile": False,
                    "timing": True,
                    "output_dir": str(metrics_path.parent),
                    "returncode": 0,
                    "elapsed_seconds": None,
                    "total_input_frames": total_input_frames,
                    "wall_fps": None,
                    "metrics_path": str(metrics_path),
                    "status_path": "",
                    "success": True,
                    "auc_3_percent": overall.get("auc_3_percent"),
                    "auc_30_percent": overall.get("auc_30_percent"),
                    "delta_1_25_percent": overall.get("delta_1_25_percent"),
                    "abs_rel": overall.get("abs_rel"),
                    "model_latency_ms_mean": latency,
                    "model_forward_fps": _safe_div(int(frame_count) * 1000.0, latency),
                    "peak_allocated_gib_max": overall.get("peak_allocated_gib_max"),
                    "peak_reserved_gib_max": overall.get("peak_reserved_gib_max"),
                    "num_sequences": num_sequences,
                    "num_pose_pairs": protocol.get("num_pose_pairs"),
                    "sampled_frames_path": str(metrics_path.parent / "sampled_frames.json"),
                    "active_layers": "",
                    "attention_schedule": json.dumps(protocol.get("attention_schedule", metadata.get("attention_schedule"))),
                    "merge_ratio": protocol.get("merge_ratio", merge_ratio),
                    "reference_source": str(metrics_path),
                }
            )
    return rows


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    completed = [row for row in rows if row.get("success")]
    failed = [row for row in rows if row.get("returncode") not in (0, None)]
    report = args.output_root / "summaries" / "analysis_report.md"
    lines = [
        "# Adaptive Anchor Experiments v2",
        "",
        f"- generated_at: {now_iso()}",
        f"- git_commit: {git_commit()}",
        f"- global_layers: {metadata.get('global_layers')}",
        f"- early_layers: {metadata.get('layer_groups', {}).get('early')}",
        f"- middle_layers: {metadata.get('layer_groups', {}).get('middle')}",
        f"- late_layers: {metadata.get('layer_groups', {}).get('late')}",
        f"- completed_jobs: {len(completed)} / {len(rows)}",
        f"- failed_jobs: {len(failed)}",
        f"- fastvggt_reference_root: {args.fastvggt_baseline_root}",
        "",
        "## Notes",
        "",
        "- `wall_fps` is end-to-end job throughput: sampled frames divided by job elapsed seconds. It includes process startup, model loading, preprocessing, forward, and metric computation.",
        "- `model_latency_ms` and `model_fps` are CUDA-event model-forward timings and are populated for `stage7_runtime` jobs and prior FastVGGT+Baseline reference rows.",
        "- `reference_fastvggt` rows are loaded from prior FastVGGT+Baseline sweep outputs and were not rerun in this experiment. They use the prior sweep's sequence set, which may differ from the 4-sequence adaptive-anchor task subset.",
        "",
        "## Main Results",
        "",
    ]
    summary_rows = [
        row
        for row in completed
        if row["stage"] in {"stage6_full_task_eval", "stage7_runtime", "reference_fastvggt"}
    ]
    if summary_rows:
        lines.extend(
            [
                "| stage | dataset | seqs | frames | strategy | ratio | AUC@3 | AUC@30 | delta1.25 | AbsRel | peak GiB | elapsed s | wall fps | model latency ms | model fps |",
                "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(summary_rows, key=lambda r: (r["stage"], r["dataset"], r["frame_count"], r["strategy_label"], str(r["anchor_ratio"]))):
            lines.append(
                "| {stage} | {dataset} | {num_sequences} | {frame_count} | {strategy_label} | {anchor_ratio} | {auc3} | {auc30} | {delta} | {absrel} | {peak} | {elapsed} | {wall_fps} | {latency} | {model_fps} |".format(
                    stage=row["stage"],
                    dataset=row["dataset"],
                    num_sequences=fmt(row.get("num_sequences"), digits=0),
                    frame_count=row["frame_count"],
                    strategy_label=row["strategy_label"],
                    anchor_ratio="" if row["anchor_ratio"] is None else row["anchor_ratio"],
                    auc3=fmt(row.get("auc_3_percent")),
                    auc30=fmt(row.get("auc_30_percent")),
                    delta=fmt(row.get("delta_1_25_percent")),
                    absrel=fmt(row.get("abs_rel"), digits=4),
                    peak=fmt(row.get("peak_allocated_gib_max"), digits=2),
                    elapsed=fmt(row.get("elapsed_seconds"), digits=1),
                    wall_fps=fmt(row.get("wall_fps"), digits=3),
                    latency=fmt(row.get("model_latency_ms_mean"), digits=1),
                    model_fps=fmt(row.get("model_forward_fps"), digits=2),
                )
            )
    else:
        lines.append("No completed task-metric jobs yet.")
    lines.extend(["", "## Failed Jobs", ""])
    if failed:
        for row in failed[:50]:
            lines.append(f"- {row['job_id']} returncode={row.get('returncode')} status={row.get('status_path')}")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Re-run Commands",
            "",
            "```bash",
            f"python scripts/run_adaptive_anchor_experiments_v2.py --output-root {args.output_root} --gpus {args.gpus} --resume",
            "```",
            "",
            "ETH3D is not included because this repository currently exposes complete 7-Scenes and TUM-Dynamics paper evaluators, but no matching ETH3D paper evaluator in scripts/.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key)) for key in keys})


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=True)
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    try:
        denom = float(denominator)
        if denom <= 0.0:
            return None
        return float(numerator) / denom
    except (TypeError, ValueError):
        return None


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    args = parse_args()
    prepare_output_dirs(args.output_root)
    metadata = get_global_layer_metadata()
    metadata.update(
        {
            "created_at": now_iso(),
            "git_commit": git_commit(),
            "command": sys.argv,
            "checkpoint": str(args.checkpoint),
            "seven_scenes_root": str(args.seven_scenes_root),
            "tum_root": str(args.tum_root),
        }
    )
    write_json(args.output_root / "metadata" / "run_metadata.json", metadata)
    jobs = build_jobs(args, metadata)
    write_json(args.output_root / "metadata" / "job_manifest.json", {"jobs": [asdict(job) for job in jobs]})
    print(f"Prepared {len(jobs)} jobs under {args.output_root}")
    if jobs:
        run_jobs(args, jobs)
    collect_results(args, jobs, metadata)
    print(f"Wrote summaries to {args.output_root / 'summaries'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
