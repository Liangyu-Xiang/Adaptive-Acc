#!/usr/bin/env python3
"""Evaluate TUM-Dynamics on externally recorded sampled RGB frames.

This is a narrow helper for cross-project comparisons where another inference
pipeline has already written the exact sampled RGB frame names in each
sequence's ``_time.json``.  It keeps the VGGT-Omega model path and metrics from
``eval_tum_dynamics_paper.py`` but associates pose/depth to those RGB frames by
nearest timestamp, matching the external inference pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from scripts.eval_tum_dynamics_paper import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_ROOT,
    FrameRecord,
    depth_sums,
    gt_row_to_c2w,
    load_model,
    official_auc,
    pairwise_pose_errors,
    read_rows,
    to_homogeneous_w2c,
)


DEFAULT_MANIFEST_ROOT = Path(
    "outputs/external_frame_persistent_spatial/tum_dynamic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VGGT-Omega frame fusion on TUM frames sampled by another pipeline."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--depth-alignment", choices=("global-median", "per-frame-median"), default="per-frame-median")
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--frame-fusion-k", type=int, default=200)
    parser.add_argument("--frame-fusion-max-group-size", type=int, default=4)
    parser.add_argument("--frame-fusion-beta", type=float, default=1.0)
    parser.add_argument("--frame-fusion-start-layer", type=int, default=-1)
    return parser.parse_args()


def _nearest_index(times: np.ndarray, timestamp: float) -> int:
    return int(np.argmin(np.abs(times - timestamp)))


def _load_external_frame_records(data_root: Path, manifest_root: Path, sequence: str) -> list[FrameRecord]:
    sequence_dir = data_root / sequence
    manifest_path = manifest_root / sequence / "_time.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing sampled-frame manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    frame_names = manifest.get("sampled_frames")
    if not isinstance(frame_names, list) or not frame_names:
        raise ValueError(f"{manifest_path} does not contain a non-empty sampled_frames list")

    gt_rows = read_rows(sequence_dir / "groundtruth.txt")
    depth_rows = read_rows(sequence_dir / "depth.txt")
    gt_times = np.asarray([row[0] for row in gt_rows], dtype=np.float64)
    depth_times = np.asarray([row[0] for row in depth_rows], dtype=np.float64)

    records: list[FrameRecord] = []
    for frame_name in frame_names:
        rgb_path = sequence_dir / "rgb" / str(frame_name)
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB frame from manifest: {rgb_path}")
        rgb_timestamp = float(Path(frame_name).stem)
        gt_index = _nearest_index(gt_times, rgb_timestamp)
        depth_index = _nearest_index(depth_times, rgb_timestamp)
        gt_timestamp, gt_data = gt_rows[gt_index]
        depth_timestamp, depth_data = depth_rows[depth_index]
        records.append(
            FrameRecord(
                rgb_timestamp=rgb_timestamp,
                rgb_path=rgb_path,
                gt_timestamp=gt_timestamp,
                c2w=gt_row_to_c2w(gt_data),
                depth_timestamp=depth_timestamp,
                depth_path=sequence_dir / depth_data[0],
            )
        )
    return records


def main() -> int:
    args = parse_args()
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be at least 1")
    if args.frame_fusion_k <= 0:
        raise ValueError("--frame-fusion-k must be positive")
    if args.frame_fusion_max_group_size <= 0:
        raise ValueError("--frame-fusion-max-group-size must be positive")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega inference requires CUDA")

    sampled = {
        sequence: _load_external_frame_records(args.data_root, args.manifest_root, sequence)
        for sequence in args.sequences
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sampled_frames.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                sequence: {
                    "rgb_timestamps": [record.rgb_timestamp for record in records],
                    "rgb_paths": [str(record.rgb_path) for record in records],
                    "gt_timestamps": [record.gt_timestamp for record in records],
                    "depth_timestamps": [record.depth_timestamp for record in records],
                    "depth_paths": [str(record.depth_path) for record in records],
                }
                for sequence, records in sampled.items()
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    model = load_model(
        args.checkpoint,
        device,
        merge_ratio=0.0,
        first_frame_token_indices=(0,),
        frame_fusion_mode="dp-medoid",
        frame_fusion_k=args.frame_fusion_k,
        frame_fusion_max_group_size=args.frame_fusion_max_group_size,
        frame_fusion_beta=args.frame_fusion_beta,
        frame_fusion_start_layer=args.frame_fusion_start_layer,
    )

    all_rotation_errors: list[np.ndarray] = []
    all_translation_errors: list[np.ndarray] = []
    total_abs_rel = 0.0
    total_delta = 0
    total_valid = 0
    per_sequence: list[dict[str, object]] = []

    for sequence, records in sampled.items():
        started = time.perf_counter()
        images = load_and_preprocess_images(
            [str(record.rgb_path) for record in records],
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(device, non_blocking=True)

        if not args.skip_timing:
            with torch.inference_mode():
                warmup_predictions = model(images)
            del warmup_predictions
            torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        timings_ms: list[float] = []
        predictions = None
        with torch.inference_mode():
            repeats = 1 if args.skip_timing else args.timing_repeats
            for _ in range(repeats):
                if not args.skip_timing:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                current_predictions = model(images)
                if not args.skip_timing:
                    end_event.record()
                    torch.cuda.synchronize(device)
                    timings_ms.append(float(start_event.elapsed_time(end_event)))
                else:
                    torch.cuda.synchronize(device)
                if predictions is not None:
                    del predictions
                predictions = current_predictions
        assert predictions is not None

        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
        model_latency_ms = None if args.skip_timing else float(np.median(timings_ms))
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
        abs_rel_sum, delta_count, valid_count, scales = depth_sums(
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
        elapsed = time.perf_counter() - started
        debug = model.aggregator.last_frame_fusion_debug or {}
        row: dict[str, object] = {
            "sequence": sequence,
            "num_frames": len(records),
            "auc_3_percent": 100 * official_auc(rotation_errors, translation_errors, 3),
            "auc_30_percent": 100 * official_auc(rotation_errors, translation_errors, 30),
            "abs_rel": abs_rel_sum / valid_count,
            "delta_1_25_percent": 100 * delta_count / valid_count,
            "valid_depth_pixels": valid_count,
            "depth_scales": scales,
            "model_latency_ms": model_latency_ms,
            "model_latency_repeats_ms": timings_ms,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "inference_seconds": elapsed,
            "fps_wall": len(records) / elapsed,
            "frame_fusion_num_groups": debug.get("num_groups"),
            "frame_fusion_num_fused_frames": debug.get("num_fused_frames"),
            "frame_fusion_partition_seconds": debug.get("partition_seconds"),
            "frame_fusion_fusion_seconds": debug.get("fusion_seconds"),
            "frame_fusion_segments": debug.get("segments"),
        }
        per_sequence.append(row)
        print(
            f"[{sequence}] AUC@3={row['auc_3_percent']:.2f}, "
            f"AUC@30={row['auc_30_percent']:.2f}, "
            f"delta1.25={row['delta_1_25_percent']:.2f}, "
            f"AbsRel={row['abs_rel']:.4f}, "
            f"latency={'skipped' if model_latency_ms is None else f'{model_latency_ms:.1f}ms'}, "
            f"fused={row['frame_fusion_num_fused_frames']}, "
            f"peak={peak_allocated_gib:.2f}GiB"
        )
        del images, predictions, extrinsics
        torch.cuda.empty_cache()

    rotation_errors = np.concatenate(all_rotation_errors)
    translation_errors = np.concatenate(all_translation_errors)
    result = {
        "protocol": {
            "source": "external_sampled_frames_time_json",
            "manifest_root": str(args.manifest_root),
            "sequences": args.sequences,
            "resize_mode": args.resize_mode,
            "image_resolution": args.image_resolution,
            "depth_alignment": args.depth_alignment,
            "max_depth_m": args.max_depth,
            "timing_repeats": args.timing_repeats,
            "skip_timing": args.skip_timing,
            "frame_fusion": {
                "mode": "dp-medoid",
                "k": args.frame_fusion_k,
                "max_group_size": args.frame_fusion_max_group_size,
                "beta": args.frame_fusion_beta,
                "start_layer": args.frame_fusion_start_layer,
                "distance": "1 - cosine_similarity",
                "weight": "nonnegative_cosine_similarity_normalized_within_segment",
                "copy_back": "expanded to original frame positions before heads",
            },
        },
        "overall": {
            "auc_3_percent": 100 * official_auc(rotation_errors, translation_errors, 3),
            "auc_30_percent": 100 * official_auc(rotation_errors, translation_errors, 30),
            "delta_1_25_percent": 100 * total_delta / total_valid,
            "abs_rel": total_abs_rel / total_valid,
            "valid_depth_pixels": total_valid,
            "model_latency_ms_mean": None
            if args.skip_timing
            else float(np.mean([float(row["model_latency_ms"]) for row in per_sequence])),
            "peak_allocated_gib_max": float(
                np.max([float(row["peak_allocated_gib"]) for row in per_sequence])
            ),
            "peak_reserved_gib_max": float(
                np.max([float(row["peak_reserved_gib"]) for row in per_sequence])
            ),
            "inference_seconds_sum": float(sum(float(row["inference_seconds"]) for row in per_sequence)),
            "fps_wall_overall": float(
                sum(int(row["num_frames"]) for row in per_sequence)
                / sum(float(row["inference_seconds"]) for row in per_sequence)
            ),
        },
        "per_sequence": per_sequence,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    np.savez_compressed(
        args.output_dir / "pose_errors.npz",
        rotation_error_deg=rotation_errors,
        translation_error_deg=translation_errors,
    )
    overall = result["overall"]
    print("\nExternal sampled-frame result:")
    print(f"  AUC@3:     {overall['auc_3_percent']:.2f}")
    print(f"  AUC@30:    {overall['auc_30_percent']:.2f}")
    print(f"  delta1.25: {overall['delta_1_25_percent']:.2f}")
    print(f"  AbsRel:    {overall['abs_rel']:.4f}")
    print(f"Saved results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
