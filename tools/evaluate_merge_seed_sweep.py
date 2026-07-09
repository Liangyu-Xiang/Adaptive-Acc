#!/usr/bin/env python3
"""Evaluate whether FastVGGT merge random seeds affect camera performance.

Only the merge destination/source random seed is changed between runs. The
model, frame sampling, merge ratio, and all preprocessing settings are fixed.
This script uses a camera-only model so 100/200/300-frame runs are governed by
the global-attention merge policy rather than dense depth memory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_7scenes_paper import (
    depth_sums as depth_sums_7scenes,
    load_frame_records as load_7scenes_records,
    official_auc,
    pairwise_pose_errors,
)
from scripts.eval_tum_dynamics import evaluate_trajectory, extrinsics_to_c2w
from scripts.eval_tum_dynamics_paper import (
    depth_sums as depth_sums_tum,
    load_frame_records as load_tum_records,
)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_TUM_ROOT = Path("/data/mmc_lyxiang/dataset/TUM-Dynamics")
DEFAULT_7SCENES_ROOT = Path("/data/mmc_lyxiang/dataset/7scenes")

FIELDS = (
    "dataset",
    "sequence",
    "frame_count",
    "merge_seed",
    "status",
    "error_message",
    "sampling",
    "merge_ratio",
    "merge_source_count_total",
    "merge_source_checksum_total",
    "merge_destination_checksum_total",
    "unique_layer_source_checksums",
    "preprocess_time_sec",
    "inference_time_sec",
    "peak_memory_allocated_gb",
    "peak_memory_reserved_gb",
    "input_height",
    "input_width",
    "tum_ate_rmse_m",
    "tum_rpe_translation_rmse_m",
    "tum_rpe_rotation_rmse_deg",
    "seven_scenes_auc_3_percent",
    "seven_scenes_auc_30_percent",
    "seven_scenes_rotation_median_deg",
    "seven_scenes_translation_median_deg",
    "depth_absrel",
    "depth_delta_1_25_percent",
    "valid_depth_pixels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_TUM_ROOT)
    parser.add_argument("--seven-scenes-root", type=Path, default=DEFAULT_7SCENES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "merge_seed_sweep")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--frame-counts", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--merge-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--merge-ratio", type=float, default=0.9)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument(
        "--depth-alignment",
        choices=("per-frame-median", "per-sequence-median"),
        default="per-frame-median",
    )
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Skip dense depth head and depth metrics. The default computes depth metrics.",
    )
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--tum-sequences",
        nargs="+",
        default=["rgbd_dataset_freiburg3_walking_xyz", "rgbd_dataset_freiburg3_sitting_static"],
    )
    parser.add_argument(
        "--seven-scenes-sequences",
        nargs="+",
        default=["chess/seq-03", "fire/seq-03"],
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--sampling",
        choices=("uniform", "random"),
        default="uniform",
        help="Frame selection strategy within each sequence. Uniform is deterministic.",
    )
    parser.add_argument("--record-merge-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_vggt_omega(checkpoint: Path, device: torch.device, merge_ratio: float, enable_depth: bool) -> VGGTOmega:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    model = VGGTOmega(enable_depth=enable_depth, merge_ratio=merge_ratio)
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        state = torch.load(checkpoint, mmap=True, **kwargs)
    except TypeError:
        state = torch.load(checkpoint, **kwargs)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(state).__name__}")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if enable_depth:
        model.load_state_dict(state, strict=True)
    else:
        expected = model.state_dict()
        camera_state = {key: value for key, value in state.items() if key in expected}
        missing, unexpected = model.load_state_dict(camera_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    del state
    return model.to(device).eval()


def sample_indices(pool_size: int, count: int, *, strategy: str, seed: int) -> np.ndarray:
    if count > pool_size:
        raise ValueError(f"requested {count} frames from a pool of {pool_size}")
    if strategy == "uniform":
        indices = np.rint(np.linspace(0, pool_size - 1, count)).astype(np.int64)
        if len(np.unique(indices)) != count:
            raise RuntimeError("uniform sampling produced duplicate indices")
        return indices
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(pool_size, count, replace=False).astype(np.int64))


def as_w2c(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(c2w, dtype=np.float64))


def empty_metrics() -> dict[str, float]:
    return {
        "tum_ate_rmse_m": float("nan"),
        "tum_rpe_translation_rmse_m": float("nan"),
        "tum_rpe_rotation_rmse_deg": float("nan"),
        "seven_scenes_auc_3_percent": float("nan"),
        "seven_scenes_auc_30_percent": float("nan"),
        "seven_scenes_rotation_median_deg": float("nan"),
        "seven_scenes_translation_median_deg": float("nan"),
        "depth_absrel": float("nan"),
        "depth_delta_1_25_percent": float("nan"),
        "valid_depth_pixels": float("nan"),
    }


def collect_merge_trace(model) -> dict[str, int]:
    source_total = 0
    source_checksum = 0
    destination_checksum = 0
    layer_checksums = []
    for layer, kind in enumerate(model.aggregator.inter_frame_attention_types):
        if kind != "global":
            continue
        attn = model.aggregator.inter_frame_blocks[layer].attn
        count = int(getattr(attn, "last_merge_source_count", 0))
        src = int(getattr(attn, "last_merge_source_checksum", 0))
        dst = int(getattr(attn, "last_merge_destination_checksum", 0))
        source_total += count
        source_checksum += src
        destination_checksum += dst
        if count:
            layer_checksums.append(src)
    return {
        "merge_source_count_total": source_total,
        "merge_source_checksum_total": source_checksum,
        "merge_destination_checksum_total": destination_checksum,
        "unique_layer_source_checksums": len(set(layer_checksums)),
    }


def enable_merge_trace(model, enabled: bool) -> None:
    for block in model.aggregator.inter_frame_blocks:
        block.attn.record_merge_trace = enabled


def evaluate_tum(predictions: dict[str, torch.Tensor], records: Sequence[Any]) -> dict[str, float]:
    with torch.inference_mode():
        extrinsics, _ = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
            build_intrinsics=False,
        )
    pred_c2w = extrinsics_to_c2w(extrinsics[0])
    gt_c2w = np.stack([record.c2w for record in records])
    camera, _ = evaluate_trajectory(pred_c2w, gt_c2w)
    return {
        "tum_ate_rmse_m": float(camera["ate_rmse_m"]),
        "tum_rpe_translation_rmse_m": float(camera["rpe_translation_rmse_m"]),
        "tum_rpe_rotation_rmse_deg": float(camera["rpe_rotation_rmse_deg"]),
    }


def evaluate_7scenes(predictions: dict[str, torch.Tensor], records: Sequence[Any]) -> dict[str, float]:
    with torch.inference_mode():
        extrinsics, _ = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
            build_intrinsics=False,
        )
    pred_w2c = np.broadcast_to(np.eye(4), (len(records), 4, 4)).copy()
    pred_w2c[:, :3] = extrinsics[0].detach().float().cpu().numpy().astype(np.float64)
    gt_w2c = np.stack([as_w2c(record.c2w) for record in records])
    rotation_errors, translation_errors = pairwise_pose_errors(pred_w2c, gt_w2c)
    return {
        "seven_scenes_auc_3_percent": 100.0 * official_auc(rotation_errors, translation_errors, 3),
        "seven_scenes_auc_30_percent": 100.0 * official_auc(rotation_errors, translation_errors, 30),
        "seven_scenes_rotation_median_deg": float(np.median(rotation_errors)),
        "seven_scenes_translation_median_deg": float(np.median(translation_errors)),
    }


def evaluate_depth(
    dataset: str,
    predictions: dict[str, torch.Tensor],
    records: Sequence[Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    if "depth" not in predictions:
        return {
            "depth_absrel": float("nan"),
            "depth_delta_1_25_percent": float("nan"),
            "valid_depth_pixels": float("nan"),
        }
    predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
    if dataset == "tum":
        abs_rel_sum, delta_count, valid_count, _ = depth_sums_tum(
            predicted_depth,
            records,
            args.depth_alignment,
            args.max_depth,
        )
    elif dataset == "7scenes":
        abs_rel_sum, delta_count, valid_count, _ = depth_sums_7scenes(
            predicted_depth,
            records,
            args.depth_alignment,
            args.min_depth,
            args.max_depth,
        )
    else:
        raise ValueError(dataset)
    return {
        "depth_absrel": abs_rel_sum / valid_count,
        "depth_delta_1_25_percent": 100.0 * delta_count / valid_count,
        "valid_depth_pixels": float(valid_count),
    }


def finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "success":
            groups.setdefault((row["dataset"], row["sequence"], int(row["frame_count"])), []).append(row)
    fields = [
        "dataset",
        "sequence",
        "frame_count",
        "successes",
        "metric_name",
        "mean",
        "std",
        "min",
        "max",
        "range",
        "rel_range_percent",
    ]
    metric_by_dataset = {
        "tum": ["tum_ate_rmse_m", "tum_rpe_translation_rmse_m", "tum_rpe_rotation_rmse_deg"],
        "7scenes": [
            "seven_scenes_auc_3_percent",
            "seven_scenes_auc_30_percent",
            "seven_scenes_rotation_median_deg",
            "seven_scenes_translation_median_deg",
        ],
    }
    for dataset_metrics in metric_by_dataset.values():
        dataset_metrics.extend(["depth_absrel", "depth_delta_1_25_percent"])
    summary = []
    for (dataset, sequence, count), group in sorted(groups.items()):
        for metric in metric_by_dataset[dataset]:
            values = finite([float(row[metric]) for row in group])
            if not len(values):
                continue
            value_range = float(values.max() - values.min())
            mean = float(values.mean())
            summary.append(
                {
                    "dataset": dataset,
                    "sequence": sequence,
                    "frame_count": count,
                    "successes": len(group),
                    "metric_name": metric,
                    "mean": mean,
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "range": value_range,
                    "rel_range_percent": 100.0 * value_range / abs(mean) if abs(mean) > 1e-12 else float("nan"),
                }
            )
    with (output_dir / "summary_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def write_rows(rows: list[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "seed_sweep_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_rows = [
        {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    with (output_dir / "seed_sweep_results.json").open("w", encoding="utf-8") as handle:
        json.dump(json_rows, handle, indent=2, allow_nan=False)
        handle.write("\n")
    write_summary(rows, output_dir)


def failure_row(
    *,
    dataset: str,
    sequence: str,
    frame_count: int,
    merge_seed: int,
    args: argparse.Namespace,
    status: str,
    message: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "sequence": sequence,
        "frame_count": frame_count,
        "merge_seed": merge_seed,
        "status": status,
        "error_message": message,
        "sampling": args.sampling,
        "merge_ratio": args.merge_ratio,
        "merge_source_count_total": 0,
        "merge_source_checksum_total": 0,
        "merge_destination_checksum_total": 0,
        "unique_layer_source_checksums": 0,
        "preprocess_time_sec": float("nan"),
        "inference_time_sec": float("nan"),
        "peak_memory_allocated_gb": float("nan"),
        "peak_memory_reserved_gb": float("nan"),
        "input_height": "",
        "input_width": "",
        **empty_metrics(),
    }


def run_one(
    *,
    model,
    device: torch.device,
    dataset: str,
    sequence: str,
    records: Sequence[Any],
    frame_count: int,
    merge_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected = [records[index] for index in sample_indices(
        len(records),
        frame_count,
        strategy=args.sampling,
        seed=args.sampling_seed + frame_count,
    )]
    start = time.perf_counter()
    images_cpu = load_and_preprocess_images(
        [str(record.rgb_path) for record in selected],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    )
    preprocess_time = time.perf_counter() - start
    images = images_cpu.to(device, non_blocking=True)
    del images_cpu
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    model.aggregator.set_merge_random_seed(merge_seed)
    start = time.perf_counter()
    with torch.inference_mode():
        predictions = model(images)
    torch.cuda.synchronize(device)
    inference_time = time.perf_counter() - start

    metrics = empty_metrics()
    if dataset == "tum":
        metrics.update(evaluate_tum(predictions, selected))
    elif dataset == "7scenes":
        metrics.update(evaluate_7scenes(predictions, selected))
    else:
        raise ValueError(dataset)
    metrics.update(evaluate_depth(dataset, predictions, selected, args))

    trace = collect_merge_trace(model) if args.record_merge_trace else {
        "merge_source_count_total": 0,
        "merge_source_checksum_total": 0,
        "merge_destination_checksum_total": 0,
        "unique_layer_source_checksums": 0,
    }
    row = {
        "dataset": dataset,
        "sequence": sequence,
        "frame_count": frame_count,
        "merge_seed": merge_seed,
        "status": "success",
        "error_message": "",
        "sampling": args.sampling,
        "merge_ratio": args.merge_ratio,
        **trace,
        "preprocess_time_sec": preprocess_time,
        "inference_time_sec": inference_time,
        "peak_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "peak_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "input_height": int(images.shape[-2]),
        "input_width": int(images.shape[-1]),
        **metrics,
    }
    del images, predictions
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.merge_ratio <= 1.0:
        raise ValueError("--merge-ratio must be in [0, 1]")
    if any(count < 2 for count in args.frame_counts):
        raise ValueError("all frame counts must be >= 2")

    tum_records = {
        name: load_tum_records(args.tum_root / name, args.association_tolerance)
        for name in args.tum_sequences
    }
    seven_records = {
        name: load_7scenes_records(args.seven_scenes_root / name)
        for name in args.seven_scenes_sequences
    }
    selection_info = {
        "tum": {name: len(records) for name, records in tum_records.items()},
        "7scenes": {name: len(records) for name, records in seven_records.items()},
    }
    if args.dry_run:
        print(json.dumps(selection_info, indent=2))
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "selection_info.json").open("w", encoding="utf-8") as handle:
        json.dump(selection_info, handle, indent=2)
        handle.write("\n")

    model = load_vggt_omega(
        args.checkpoint,
        device,
        merge_ratio=args.merge_ratio,
        enable_depth=not args.camera_only,
    )
    enable_merge_trace(model, args.record_merge_trace)
    rows: list[dict[str, Any]] = []
    work = [
        *[("tum", name, records) for name, records in tum_records.items()],
        *[("7scenes", name, records) for name, records in seven_records.items()],
    ]

    for dataset, sequence, records in work:
        for frame_count in args.frame_counts:
            for merge_seed in args.merge_seeds:
                print(f"[{dataset} {sequence} N={frame_count} seed={merge_seed}]", flush=True)
                try:
                    row = run_one(
                        model=model,
                        device=device,
                        dataset=dataset,
                        sequence=sequence,
                        records=records,
                        frame_count=frame_count,
                        merge_seed=merge_seed,
                        args=args,
                    )
                    rows.append(row)
                    print(
                        f"  ok {row['inference_time_sec']:.2f}s "
                        f"mem={row['peak_memory_allocated_gb']:.2f}GB",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError as error:
                    message = str(error).replace("\n", " ")
                    rows.append(
                        failure_row(
                            dataset=dataset,
                            sequence=sequence,
                            frame_count=frame_count,
                            merge_seed=merge_seed,
                            args=args,
                            status="oom",
                            message=message,
                        )
                    )
                    print(f"  OOM: {message[:160]}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as error:
                    message = f"{type(error).__name__}: {error}".replace("\n", " ")
                    rows.append(
                        failure_row(
                            dataset=dataset,
                            sequence=sequence,
                            frame_count=frame_count,
                            merge_seed=merge_seed,
                            args=args,
                            status="error",
                            message=message,
                        )
                    )
                    print(f"  error: {message}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                finally:
                    write_rows(rows, args.output_dir)

    run_metadata = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "frame_counts": args.frame_counts,
        "merge_seeds": args.merge_seeds,
        "merge_ratio": args.merge_ratio,
        "image_resolution": args.image_resolution,
        "resize_mode": args.resize_mode,
        "sampling": args.sampling,
        "sampling_seed": args.sampling_seed,
        "record_merge_trace": args.record_merge_trace,
        "camera_only": args.camera_only,
        "depth_alignment": args.depth_alignment,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)
        handle.write("\n")
    print(f"Saved seed sweep to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
