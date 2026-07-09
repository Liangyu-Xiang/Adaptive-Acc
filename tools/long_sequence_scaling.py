#!/usr/bin/env python3
"""Profile VGGT-Omega as the number of jointly processed TUM frames grows."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Make repository modules importable even when this script is launched elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

# Reuse the repository's established TUM association, model loading, and metrics.
from scripts.eval_tum_dynamics import evaluate_trajectory, extrinsics_to_c2w
from scripts.eval_tum_dynamics_paper import (
    FrameRecord,
    depth_sums,
    load_frame_records,
    load_model,
)
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


DEFAULT_COUNTS = (2, 4, 8, 10, 16, 32, 64, 128)
RESULT_FIELDS = (
    "sequence", "frame_count", "repeat_id", "sampling_strategy", "status",
    "error_message", "inference_time_sec", "preprocessing_time_sec",
    "input_transfer_time_sec", "peak_memory_allocated_gb",
    "peak_memory_reserved_gb", "num_input_frames", "image_resolution",
    "input_height", "input_width", "model_name", "checkpoint",
    "merge_ratio",
    "depth_absrel", "depth_delta_1_25", "camera_ate",
    "rotation_error", "translation_error",
)
METRIC_FIELDS = (
    "depth_absrel", "depth_delta_1_25", "camera_ate",
    "rotation_error", "translation_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", "--data-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument(
        "--model_path", "--model-path", "--checkpoint", dest="checkpoint",
        type=Path, required=True,
    )
    parser.add_argument("--frame_counts", "--frame-counts", nargs="+", type=int,
                        default=list(DEFAULT_COUNTS))
    parser.add_argument("--output_dir", "--output-dir", type=Path,
                        default=Path("outputs/scaling_diagnostic"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_repeats", "--num-repeats", type=int, default=3)
    parser.add_argument("--save_predictions", "--save-predictions", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval", action="store_true", help="Compute camera and depth metrics.")
    mode.add_argument("--profile_only", "--profile-only", action="store_true",
                      help="Only profile latency and memory (the default).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--association_tolerance", "--association-tolerance", type=float,
                        default=0.02)
    parser.add_argument("--image_resolution", "--image-resolution", type=int, default=512)
    parser.add_argument("--merge_ratio", "--merge-ratio", type=float, default=0.9,
                        help="Global-attention token merge ratio in [0, 1].")
    parser.add_argument("--resize_mode", "--resize-mode",
                        choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--depth_alignment", "--depth-alignment",
                        choices=("per-frame-median", "per-sequence-median"),
                        default="per-frame-median")
    parser.add_argument("--max_depth", "--max-depth", type=float, default=10.0)
    parser.add_argument("--skip_warmup", "--skip-warmup", action="store_true",
                        help="Disable the one-forward warm-up (useful only for tight memory).")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    return parser.parse_args()


def normalize_requested_sequences(data_root: Path, requested: Sequence[str]) -> list[Path]:
    available = {
        path.name: path for path in data_root.iterdir()
        if path.is_dir() and (path / "rgb.txt").is_file()
    }
    selected: list[Path] = []
    for name in requested:
        candidates = (name, f"rgbd_dataset_{name}")
        match = next((available[candidate] for candidate in candidates if candidate in available), None)
        if match is None:
            raise ValueError(f"Unknown sequence {name!r}; available: {', '.join(sorted(available))}")
        selected.append(match)
    return selected


def uniform_sample(records: Sequence[FrameRecord], count: int) -> tuple[list[FrameRecord], list[int]]:
    if count > len(records):
        raise ValueError(f"only {len(records)} associated frames are available, requested {count}")
    # linspace is deterministic and includes both ends of the valid time range.
    indices = np.rint(np.linspace(0, len(records) - 1, count)).astype(int)
    if len(np.unique(indices)) != count:
        raise RuntimeError("uniform sampling produced duplicate indices")
    return [records[index] for index in indices], indices.tolist()


def empty_metrics() -> dict[str, float]:
    return {name: float("nan") for name in METRIC_FIELDS}


def evaluate_predictions(
    predictions: dict[str, torch.Tensor], records: Sequence[FrameRecord],
    depth_alignment: str, max_depth: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    with torch.inference_mode():
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
    pred_c2w = extrinsics_to_c2w(extrinsics[0])
    gt_c2w = np.stack([record.c2w for record in records])
    camera, _ = evaluate_trajectory(pred_c2w, gt_c2w)
    depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
    absrel_sum, delta_count, valid_count, _ = depth_sums(
        depth, records, depth_alignment, max_depth
    )
    metrics = {
        "depth_absrel": absrel_sum / valid_count,
        "depth_delta_1_25": delta_count / valid_count,
        "camera_ate": float(camera["ate_rmse_m"]),
        "rotation_error": float(camera["rpe_rotation_rmse_deg"]),
        "translation_error": float(camera["rpe_translation_rmse_m"]),
    }
    arrays = {
        "extrinsics_w2c": extrinsics[0].detach().float().cpu().numpy(),
        "intrinsics": intrinsics[0].detach().float().cpu().numpy(),
        "depth": depth,
        "depth_conf": predictions["depth_conf"][0, ..., 0].detach().float().cpu().numpy(),
    }
    return metrics, arrays


def prediction_arrays(predictions: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
    return {
        "extrinsics_w2c": extrinsics[0].detach().float().cpu().numpy(),
        "intrinsics": intrinsics[0].detach().float().cpu().numpy(),
        "depth": predictions["depth"][0, ..., 0].detach().float().cpu().numpy(),
        "depth_conf": predictions["depth_conf"][0, ..., 0].detach().float().cpu().numpy(),
    }


def save_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (output_dir / "scaling_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "scaling_results.json").open("w", encoding="utf-8") as handle:
        json_rows = [
            {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
             for key, value in row.items()}
            for row in rows
        ]
        json.dump(json_rows, handle, indent=2, allow_nan=False)
        handle.write("\n")


def finite_values(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def aggregate_results(rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["sequence"]), int(row["frame_count"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (sequence, frame_count), group in groups.items():
        successes = [row for row in group if row["status"] == "success"]
        item: dict[str, Any] = {
            "sequence": sequence,
            "frame_count": frame_count,
            "success_rate": len(successes) / len(group),
        }
        mappings = {
            "inference_time_sec": "time_sec",
            "peak_memory_allocated_gb": "peak_memory_allocated_gb",
            "peak_memory_reserved_gb": "peak_memory_reserved_gb",
            **{name: name for name in METRIC_FIELDS},
        }
        for source, target in mappings.items():
            values = finite_values(successes, source) if successes else np.asarray([])
            item[f"mean_{target}"] = float(np.mean(values)) if len(values) else float("nan")
            item[f"std_{target}"] = float(np.std(values)) if len(values) else float("nan")
        summary.append(item)
    summary.sort(key=lambda row: (str(row["sequence"]), int(row["frame_count"])))
    fields = list(summary[0]) if summary else ["sequence", "frame_count", "success_rate"]
    with (output_dir / "scaling_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("scaling")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "scaling.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def failure_row(base: dict[str, Any], repeat_id: int, status: str, message: str) -> dict[str, Any]:
    return {
        **base, "repeat_id": repeat_id, "status": status, "error_message": message,
        "inference_time_sec": float("nan"), "peak_memory_allocated_gb": float("nan"),
        "peak_memory_reserved_gb": float("nan"), **empty_metrics(),
    }


def main() -> int:
    args = parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")
    if args.num_repeats < 1 or any(count < 2 for count in args.frame_counts):
        raise ValueError("repeat count must be >= 1 and every frame count must be >= 2")
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise ValueError("--image-resolution must be positive and divisible by 16")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sequence_dirs = normalize_requested_sequences(args.data_root, args.sequences)
    pools = {
        path.name: load_frame_records(path, args.association_tolerance)
        for path in sequence_dirs
    }
    selections: dict[str, Any] = {}
    for name, records in pools.items():
        selections[name] = {}
        for count in args.frame_counts:
            images_cpu = None
            images = None
            warmup = None
            try:
                sampled, indices = uniform_sample(records, count)
                selections[name][str(count)] = {
                    "pool_size": len(records), "pool_indices": indices,
                    "rgb_timestamps": [record.rgb_timestamp for record in sampled],
                    "rgb_paths": [str(record.rgb_path) for record in sampled],
                }
            except ValueError as error:
                selections[name][str(count)] = {"pool_size": len(records), "error": str(error)}
    if args.dry_run:
        print(json.dumps(selections, indent=2))
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega scaling inference requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir)
    with (args.output_dir / "sampled_frames.json").open("w", encoding="utf-8") as handle:
        json.dump(selections, handle, indent=2)
        handle.write("\n")
    logger.info("Loading checkpoint %s on %s", args.checkpoint, device)
    model = load_model(args.checkpoint, device, merge_ratio=args.merge_ratio)
    rows: list[dict[str, Any]] = []

    for sequence_name, pool in pools.items():
        for count in args.frame_counts:
            preprocess_seconds = float("nan")
            transfer_seconds = float("nan")
            base: dict[str, Any] = {
                "sequence": sequence_name, "frame_count": count,
                "sampling_strategy": "uniform", "preprocessing_time_sec": preprocess_seconds,
                "input_transfer_time_sec": transfer_seconds, "num_input_frames": count,
                "image_resolution": args.image_resolution, "input_height": "", "input_width": "",
                "model_name": "VGGT-Omega-1B", "checkpoint": str(args.checkpoint),
                "merge_ratio": args.merge_ratio,
            }
            try:
                records, _ = uniform_sample(pool, count)
                started = time.perf_counter()
                images_cpu = load_and_preprocess_images(
                    [str(record.rgb_path) for record in records], mode=args.resize_mode,
                    image_resolution=args.image_resolution,
                )
                preprocess_seconds = time.perf_counter() - started
                started = time.perf_counter()
                images = images_cpu.to(device, non_blocking=True)
                torch.cuda.synchronize(device)
                transfer_seconds = time.perf_counter() - started
                base.update({
                    "preprocessing_time_sec": preprocess_seconds,
                    "input_transfer_time_sec": transfer_seconds,
                    "input_height": int(images.shape[-2]), "input_width": int(images.shape[-1]),
                })
                del images_cpu
                if not args.skip_warmup:
                    logger.info("[%s N=%d] warm-up", sequence_name, count)
                    with torch.inference_mode():
                        warmup = model(images)
                    torch.cuda.synchronize(device)
                    del warmup
                    warmup = None
            except torch.cuda.OutOfMemoryError as error:
                message = str(error).replace("\n", " ")
                logger.warning("[%s N=%d] OOM during setup/warm-up: %s", sequence_name, count, message)
                rows.extend(failure_row(base, repeat_id, "oom", message)
                            for repeat_id in range(args.num_repeats))
                save_results(args.output_dir, rows)
                aggregate_results(rows, args.output_dir)
                del images, images_cpu, warmup
                gc.collect()
                torch.cuda.empty_cache()
                continue
            except Exception as error:
                message = f"{type(error).__name__}: {error}".replace("\n", " ")
                logger.exception("[%s N=%d] setup failed", sequence_name, count)
                rows.extend(failure_row(base, repeat_id, "error", message)
                            for repeat_id in range(args.num_repeats))
                save_results(args.output_dir, rows)
                aggregate_results(rows, args.output_dir)
                del images, images_cpu, warmup
                gc.collect()
                torch.cuda.empty_cache()
                continue

            for repeat_id in range(args.num_repeats):
                predictions = None
                arrays = None
                try:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        predictions = model(images)
                    torch.cuda.synchronize(device)
                    inference_seconds = time.perf_counter() - started
                    allocated = torch.cuda.max_memory_allocated(device) / 1e9
                    reserved = torch.cuda.max_memory_reserved(device) / 1e9
                    metrics, arrays = empty_metrics(), None
                    if args.eval:
                        metrics, arrays = evaluate_predictions(
                            predictions, records, args.depth_alignment, args.max_depth
                        )
                    if args.save_predictions and repeat_id == 0:
                        if arrays is None:
                            arrays = prediction_arrays(predictions)
                        pred_dir = args.output_dir / "predictions" / sequence_name / f"N_{count}"
                        pred_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            pred_dir / "predictions.npz", **arrays,
                            rgb_timestamps=np.asarray([record.rgb_timestamp for record in records]),
                        )
                    row = {
                        **base, "repeat_id": repeat_id, "status": "success", "error_message": "",
                        "inference_time_sec": inference_seconds,
                        "peak_memory_allocated_gb": allocated,
                        "peak_memory_reserved_gb": reserved, **metrics,
                    }
                    rows.append(row)
                    logger.info(
                        "[%s N=%d repeat=%d] %.3fs, peak allocated/reserved %.2f/%.2f GB",
                        sequence_name, count, repeat_id, inference_seconds, allocated, reserved,
                    )
                except torch.cuda.OutOfMemoryError as error:
                    message = str(error).replace("\n", " ")
                    rows.append(failure_row(base, repeat_id, "oom", message))
                    logger.warning("[%s N=%d repeat=%d] OOM", sequence_name, count, repeat_id)
                except Exception as error:
                    message = f"{type(error).__name__}: {error}".replace("\n", " ")
                    rows.append(failure_row(base, repeat_id, "error", message))
                    logger.exception("[%s N=%d repeat=%d] failed", sequence_name, count, repeat_id)
                finally:
                    if predictions is not None:
                        del predictions
                    if arrays is not None:
                        del arrays
                    gc.collect()
                    save_results(args.output_dir, rows)
                    aggregate_results(rows, args.output_dir)
            del images
            gc.collect()
            torch.cuda.empty_cache()

    logger.info("Completed %d runs; results: %s", len(rows), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
