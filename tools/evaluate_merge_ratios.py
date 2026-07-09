#!/usr/bin/env python3
"""Evaluate VGGT-Omega camera and depth quality across token merge ratios."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_tum_dynamics_paper import (
    FrameRecord,
    depth_sums,
    load_frame_records,
    load_model,
    official_auc,
    pairwise_pose_errors,
    sample_records,
    to_homogeneous_w2c,
)
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-type", choices=("static", "dynamic"), required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=(0.0, 0.1, 0.25, 0.5, 0.75))
    parser.add_argument("--frame-counts", nargs="+", type=int, default=(4, 8, 64, 128, 256))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument("--depth-alignment", choices=("per-frame-median", "per-sequence-median"),
                        default="per-frame-median")
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true", help="Skip combinations already saved in metrics.json.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(frame_count < 2 for frame_count in args.frame_counts):
        raise ValueError("Every --frame-counts value must be at least 2")
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise ValueError("--image-resolution must be positive and divisible by 16")
    invalid = [ratio for ratio in args.ratios if not 0.0 <= ratio <= 1.0]
    if invalid:
        raise ValueError(f"Merge ratios must be in [0, 1], got {invalid}")


def prepare_pools(args: argparse.Namespace) -> dict[str, list[FrameRecord]]:
    pools: dict[str, list[FrameRecord]] = {}
    for sequence_name in args.sequences:
        sequence_dir = args.data_root / sequence_name
        records = load_frame_records(sequence_dir, args.association_tolerance)
        pools[sequence_name] = [
            record for record in records
            if record.rgb_path.is_file() and record.depth_path.is_file()
        ]
        skipped = len(records) - len(pools[sequence_name])
        print(
            f"{sequence_name}: {len(pools[sequence_name])} usable associated frames "
            f"({skipped} missing files skipped)",
            flush=True,
        )
    return pools


def evaluate_ratio(
    model,
    sampled: dict[str, list[FrameRecord]],
    ratio: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model.set_merge_ratio(ratio)
    all_rotation_errors: list[np.ndarray] = []
    all_translation_errors: list[np.ndarray] = []
    total_abs_rel = 0.0
    total_delta = 0
    total_valid = 0
    per_sequence: list[dict[str, object]] = []

    for sequence_name, records in sampled.items():
        images = load_and_preprocess_images(
            [str(record.rgb_path) for record in records],
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(device, non_blocking=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            predictions = model(images)
        torch.cuda.synchronize(device)
        latency_ms = 1000.0 * (time.perf_counter() - started)

        with torch.inference_mode():
            extrinsics, _ = encoding_to_camera(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
                build_intrinsics=False,
            )
        pred_w2c = to_homogeneous_w2c(extrinsics[0])
        gt_w2c = np.linalg.inv(np.stack([record.c2w for record in records]))
        rotation_errors, translation_errors = pairwise_pose_errors(pred_w2c, gt_w2c)
        predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
        abs_rel_sum, delta_count, valid_count, _ = depth_sums(
            predicted_depth,
            records,
            args.depth_alignment,
            args.max_depth,
        )

        all_rotation_errors.append(rotation_errors)
        all_translation_errors.append(translation_errors)
        total_abs_rel += abs_rel_sum
        total_delta += delta_count
        total_valid += valid_count
        row = {
            "sequence": sequence_name,
            "auc_3_percent": 100 * official_auc(rotation_errors, translation_errors, 3),
            "auc_30_percent": 100 * official_auc(rotation_errors, translation_errors, 30),
            "abs_rel": abs_rel_sum / valid_count,
            "delta_1_25_percent": 100 * delta_count / valid_count,
            "latency_ms": latency_ms,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        per_sequence.append(row)
        print(
            f"ratio={ratio:.2f} [{sequence_name}] "
            f"AUC@30={row['auc_30_percent']:.2f} AbsRel={row['abs_rel']:.4f} "
            f"delta1.25={row['delta_1_25_percent']:.2f} latency={latency_ms:.1f}ms",
            flush=True,
        )
        del images, predictions, extrinsics
        torch.cuda.empty_cache()

    rotation_errors = np.concatenate(all_rotation_errors)
    translation_errors = np.concatenate(all_translation_errors)
    return {
        "status": "success",
        "frame_count": len(next(iter(sampled.values()))),
        "merge_ratio": ratio,
        "merge_percent": 100 * ratio,
        "auc_3_percent": 100 * official_auc(rotation_errors, translation_errors, 3),
        "auc_30_percent": 100 * official_auc(rotation_errors, translation_errors, 30),
        "abs_rel": total_abs_rel / total_valid,
        "delta_1_25_percent": 100 * total_delta / total_valid,
        "latency_ms_mean": float(np.mean([row["latency_ms"] for row in per_sequence])),
        "peak_allocated_gib_max": float(max(row["peak_allocated_gib"] for row in per_sequence)),
        "per_sequence": per_sequence,
    }


def save_results(
    args: argparse.Namespace,
    sample_metadata: dict[str, object],
    results: Sequence[dict[str, object]],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene_type": args.scene_type,
        "sequences": args.sequences,
        "frame_counts": args.frame_counts,
        "seed": args.seed,
        "image_resolution": args.image_resolution,
        "resize_mode": args.resize_mode,
        "depth_alignment": args.depth_alignment,
        "samples": sample_metadata,
        "results": list(results),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    validate_args(args)
    pools = prepare_pools(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading checkpoint on {device}: {args.checkpoint}", flush=True)
    model = load_model(args.checkpoint, device, merge_ratio=args.ratios[0])
    metrics_path = args.output_dir / "metrics.json"
    if args.resume and metrics_path.is_file():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        results = saved["results"]
        sample_metadata = saved.get("samples", {})
    else:
        results = []
        sample_metadata = {}
    completed = {
        (int(result["frame_count"]), float(result["merge_ratio"]))
        for result in results
        if result["status"] == "success"
    }
    for frame_count in args.frame_counts:
        sampled, sampled_indices = sample_records(pools, frame_count, args.seed)
        sample_metadata[str(frame_count)] = {
            "sampled_indices": sampled_indices,
            "sampled_rgb_paths": {
                name: [str(record.rgb_path) for record in records]
                for name, records in sampled.items()
            },
        }
        for ratio in args.ratios:
            if (frame_count, ratio) in completed:
                print(f"frames={frame_count} ratio={ratio:.2f}: already complete", flush=True)
                continue
            try:
                result = evaluate_ratio(model, sampled, ratio, args, device)
            except torch.cuda.OutOfMemoryError as error:
                result = {
                    "status": "oom",
                    "frame_count": frame_count,
                    "merge_ratio": ratio,
                    "merge_percent": 100 * ratio,
                    "error": str(error).replace("\n", " "),
                }
                print(f"frames={frame_count} ratio={ratio:.2f}: CUDA OOM", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            results.append(result)
            save_results(args, sample_metadata, results)
    print(f"Saved {args.scene_type} results to {args.output_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
