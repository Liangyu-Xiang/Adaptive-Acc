#!/usr/bin/env python3
"""Evaluate original VGGT with sparse-vggt block-sparse global attention."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = Path("/data/mmc_lyxiang/3D/vggt")
SPARSE_VGGT_SRC = Path("/data/mmc_lyxiang/3D/sparse-vggt/src")
for path in (REPO_ROOT, VGGT_ROOT, SPARSE_VGGT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.eval_7scenes_paper as seven_eval
import scripts.eval_tum_dynamics_paper as tum_eval
from sparse_vggt.models.vggt import sparse_aggregator_from_vggt
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt_omega.utils.frame_sampling import SAMPLING_STRATEGIES


DEFAULT_VGGT_CHECKPOINT = VGGT_ROOT / "ckpt" / "model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sparse-vggt on 7scenes or TUM-Dynamics.")
    parser.add_argument("--dataset", choices=("7scenes", "tum"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_VGGT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument(
        "--sampling-strategy",
        choices=SAMPLING_STRATEGIES,
        default="uniform",
        help=(
            "Frame selection strategy. 'uniform' is the default and preserves "
            "first/last frames while sampling the middle evenly; 'random' keeps "
            "the paper-style seeded protocol."
        ),
    )
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--preprocess-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--disable-sparse", action="store_true", help="Evaluate original full-attention VGGT.")
    parser.add_argument("--sparse-ratio", type=float, default=0.1)
    parser.add_argument("--sparse-cdf-threshold", type=float, default=0.97)
    parser.add_argument("--sparse-pool-mode", choices=("avg", "max"), default="avg")
    parser.add_argument(
        "--depth-alignment",
        choices=("per-frame-median", "per-sequence-median"),
        default="per-frame-median",
    )
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument("--sampling-pool", choices=("full", "rgb_90"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sparse_vggt(args: argparse.Namespace, device: torch.device) -> VGGT:
    model = VGGT(enable_point=False, enable_track=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith("point_head.") and not key.startswith("track_head.")
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint mismatch: missing={incompatible.missing_keys[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    if not args.disable_sparse:
        model.aggregator, _ = sparse_aggregator_from_vggt(
            model.aggregator,
            sparse_ratio=args.sparse_ratio,
            cdf_threshold=args.sparse_cdf_threshold,
            pool_mode=args.sparse_pool_mode,
            verbose=True,
        )
    return model.to(device).eval()


def build_sample(args: argparse.Namespace):
    if args.dataset == "7scenes":
        sequence_dirs = seven_eval.select_sequence_dirs(args.data_root, args.sequences)
        pools = {}
        for sequence_dir in sequence_dirs:
            sequence_name = f"{sequence_dir.parent.name}/{sequence_dir.name}"
            records = seven_eval.load_frame_records(sequence_dir)
            pools[sequence_name] = records
            print(f"{sequence_name}: sampling pool has {len(records)} frames")
        sampled, sampled_indices = seven_eval.sample_records(
            pools,
            args.num_frames,
            args.seed,
            strategy=args.sampling_strategy,
        )
        selection = {
            name: {
                "sampling_strategy": args.sampling_strategy,
                "sampling_pool_size": len(pools[name]),
                "pool_indices": sampled_indices[name],
                "frame_indices": [record.index for record in records],
                "rgb_paths": [str(record.rgb_path) for record in records],
                "depth_paths": [str(record.depth_path) for record in records],
            }
            for name, records in sampled.items()
        }
        return sampled, selection

    sequence_dirs = tum_eval.select_sequence_dirs(args.data_root, args.sequences)
    pools = {}
    for sequence_dir in sequence_dirs:
        records = tum_eval.load_frame_records(sequence_dir, args.association_tolerance)
        if args.sampling_pool == "rgb_90":
            records = tum_eval.restrict_to_rgb90(records, sequence_dir, args.association_tolerance)
        pools[sequence_dir.name] = records
        print(f"{sequence_dir.name}: sampling pool has {len(records)} RGB/pose/depth frames")
    sampled, sampled_indices = tum_eval.sample_records(
        pools,
        args.num_frames,
        args.seed,
        strategy=args.sampling_strategy,
    )
    selection = {
        name: {
            "sampling_strategy": args.sampling_strategy,
            "sampling_pool_size": len(pools[name]),
            "pool_indices": sampled_indices[name],
            "rgb_timestamps": [record.rgb_timestamp for record in records],
            "rgb_paths": [str(record.rgb_path) for record in records],
        }
        for name, records in sampled.items()
    }
    return sampled, selection


def evaluate_depth(args: argparse.Namespace, predicted_depth: np.ndarray, records):
    if args.dataset == "7scenes":
        return seven_eval.depth_sums(
            predicted_depth,
            records,
            args.depth_alignment,
            args.min_depth,
            args.max_depth,
        )
    return tum_eval.depth_sums(predicted_depth, records, args.depth_alignment, args.max_depth)


def main() -> int:
    args = parse_args()
    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2")
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be at least 1")
    sampled, selection = build_sample(args)
    if args.dry_run:
        print(json.dumps(selection, indent=2))
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT inference requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sampled_frames.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
        handle.write("\n")

    model = load_sparse_vggt(args, device)
    all_rotation_errors = []
    all_translation_errors = []
    total_abs_rel = 0.0
    total_delta = 0
    total_valid = 0
    per_sequence = []

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for sequence_name, records in sampled.items():
        images = load_and_preprocess_images(
            [str(record.rgb_path) for record in records],
            mode=args.preprocess_mode,
        ).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            warmup = model(images)
        del warmup
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        timings_ms = []
        predictions = None
        for _ in range(args.timing_repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                current_predictions = model(images)
            end_event.record()
            torch.cuda.synchronize(device)
            timings_ms.append(float(start_event.elapsed_time(end_event)))
            if predictions is not None:
                del predictions
            predictions = current_predictions
        assert predictions is not None

        with torch.inference_mode():
            extrinsics, _ = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
                build_intrinsics=False,
            )
        pred_w2c = seven_eval.to_homogeneous_w2c(extrinsics[0])
        gt_w2c = np.linalg.inv(np.stack([record.c2w for record in records]))
        rotation_errors, translation_errors = seven_eval.pairwise_pose_errors(pred_w2c, gt_w2c)
        predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
        abs_rel_sum, delta_count, valid_count, scales = evaluate_depth(args, predicted_depth, records)

        all_rotation_errors.append(rotation_errors)
        all_translation_errors.append(translation_errors)
        total_abs_rel += abs_rel_sum
        total_delta += delta_count
        total_valid += valid_count
        row = {
            "sequence": sequence_name,
            "auc_3_percent": 100 * seven_eval.official_auc(rotation_errors, translation_errors, 3),
            "auc_30_percent": 100 * seven_eval.official_auc(rotation_errors, translation_errors, 30),
            "delta_1_25_percent": 100 * delta_count / valid_count,
            "abs_rel": abs_rel_sum / valid_count,
            "valid_depth_pixels": valid_count,
            "depth_scales": scales,
            "model_latency_ms": float(np.median(timings_ms)),
            "model_latency_repeats_ms": timings_ms,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        per_sequence.append(row)
        print(
            f"[{sequence_name}] AUC@3={row['auc_3_percent']:.2f}, "
            f"AUC@30={row['auc_30_percent']:.2f}, delta1.25={row['delta_1_25_percent']:.2f}, "
            f"AbsRel={row['abs_rel']:.4f}, latency={row['model_latency_ms']:.1f}ms, "
            f"peak={row['peak_allocated_gib']:.2f}GiB"
        )
        del images, predictions, extrinsics
        torch.cuda.empty_cache()

    rotation_errors = np.concatenate(all_rotation_errors)
    translation_errors = np.concatenate(all_translation_errors)
    overall = {
        "auc_3_percent": 100 * seven_eval.official_auc(rotation_errors, translation_errors, 3),
        "auc_30_percent": 100 * seven_eval.official_auc(rotation_errors, translation_errors, 30),
        "delta_1_25_percent": 100 * total_delta / total_valid,
        "abs_rel": total_abs_rel / total_valid,
        "valid_depth_pixels": total_valid,
        "model_latency_ms_mean": float(np.mean([row["model_latency_ms"] for row in per_sequence])),
        "peak_allocated_gib_max": float(np.max([row["peak_allocated_gib"] for row in per_sequence])),
    }
    result = {
        "protocol": {
            "model": "VGGT" if args.disable_sparse else "VGGT+sparse-vggt",
            "dataset": args.dataset,
            "seed": args.seed,
            "sampling_strategy": args.sampling_strategy,
            "num_frames_per_sequence": args.num_frames,
            "preprocess_mode": args.preprocess_mode,
            "sparse_attention": not args.disable_sparse,
            "sparse_ratio": None if args.disable_sparse else args.sparse_ratio,
            "sparse_cdf_threshold": None if args.disable_sparse else args.sparse_cdf_threshold,
            "sparse_pool_mode": None if args.disable_sparse else args.sparse_pool_mode,
            "timing_repeats": args.timing_repeats,
            "num_sequences": len(sampled),
            "num_pose_pairs": len(rotation_errors),
        },
        "overall": overall,
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
    print("\nVGGT result:")
    print(f"  AUC@3:     {overall['auc_3_percent']:.2f}")
    print(f"  AUC@30:    {overall['auc_30_percent']:.2f}")
    print(f"  delta1.25: {overall['delta_1_25_percent']:.2f}")
    print(f"  AbsRel:    {overall['abs_rel']:.4f}")
    print(f"  latency:   {overall['model_latency_ms_mean']:.1f}ms")
    print(f"Saved reproducible results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        raise SystemExit(main())
    finally:
        print(f"Total elapsed: {time.perf_counter() - started:.1f}s")
