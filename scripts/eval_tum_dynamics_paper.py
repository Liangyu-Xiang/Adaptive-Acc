#!/usr/bin/env python3
"""Reproduce the VGGT-Omega paper protocol on TUM-Dynamics.

Paper protocol (VGGT-Omega, Sec. 4.2): randomly sample 10 frames from each
sequence. Camera pose is evaluated over every image pair using relative
rotation/translation angular errors and AUC@3/AUC@30. Depth is evaluated with
AbsRel and delta<1.25 after resolving monocular scale ambiguity.

The pose metric follows facebookresearch/vggt's official ``evaluation`` branch.
The VGGT-Omega release does not provide its sampled frame IDs. This script uses
a fixed NumPy RandomState seed (42 by default) and records every selected frame
in ``sampled_frames.json`` so that reported results remain reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


DEFAULT_DATA_ROOT = Path("/mnt/nasdata/xly/dataset/TUM-Dynamics")
DEFAULT_CHECKPOINT = Path(
    "/mnt/nasdata/xly/3D/vggt-omega/pretrained_ckpts/vggt_omega_1b_512.pt"
)
PAPER_TARGETS = {
    "auc_3_percent": 30.2,
    "auc_30_percent": 82.3,
    "delta_1_25_percent": 97.4,
    "abs_rel": 0.041,
}


@dataclass(frozen=True)
class FrameRecord:
    rgb_timestamp: float
    rgb_path: Path
    gt_timestamp: float
    c2w: np.ndarray
    depth_timestamp: float
    depth_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce VGGT-Omega paper metrics on TUM-Dynamics."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tum_dynamics_paper"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attention-mode",
        choices=("default", "register-only-zero-shot"),
        default="default",
        help=(
            "Attention schedule to evaluate. The register-only option changes the released "
            "checkpoint at inference time; it is not the paper's separately trained ablation."
        ),
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
        help="Timed model forwards per sequence after one untimed warm-up.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument(
        "--sampling-pool",
        choices=("full", "rgb_90"),
        default="full",
        help="Frame pool to sample from. 'full' is the literal paper protocol.",
    )
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument(
        "--depth-alignment",
        choices=("per-frame-median", "per-sequence-median"),
        default="per-frame-median",
        help="Resolve scale ambiguity before computing depth metrics.",
    )
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[tuple[float, list[str]]]:
    rows: list[tuple[float, list[str]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.replace(",", " ").split()
            rows.append((float(fields[0]), fields[1:]))
    return rows


def associate_nearest(
    first: Sequence[tuple[float, list[str]]],
    second: Sequence[tuple[float, list[str]]],
    tolerance: float,
) -> list[tuple[int, int]]:
    """Unique greedy timestamp association, equivalent to the TUM tool."""
    second_times = np.asarray([row[0] for row in second], dtype=np.float64)
    candidates: list[tuple[float, int, int]] = []
    for i, (timestamp, _) in enumerate(first):
        left = int(np.searchsorted(second_times, timestamp - tolerance, side="right"))
        right = int(np.searchsorted(second_times, timestamp + tolerance, side="left"))
        candidates.extend((abs(timestamp - second_times[j]), i, j) for j in range(left, right))
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in sorted(candidates):
        if i not in used_first and j not in used_second:
            used_first.add(i)
            used_second.add(j)
            matches.append((i, j))
    return sorted(matches)


def quaternion_xyzw_to_matrix(values: Sequence[str]) -> np.ndarray:
    x, y, z, w = np.asarray(values, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def gt_row_to_c2w(fields: Sequence[str]) -> np.ndarray:
    if len(fields) != 7:
        raise ValueError(f"Expected TUM pose with 7 values, got {len(fields)}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(fields[:3], dtype=np.float64)
    pose[:3, :3] = quaternion_xyzw_to_matrix(fields[3:])
    return pose


def load_frame_records(sequence_dir: Path, tolerance: float) -> list[FrameRecord]:
    rgb_rows = read_rows(sequence_dir / "rgb.txt")
    gt_rows = read_rows(sequence_dir / "groundtruth.txt")
    depth_rows = read_rows(sequence_dir / "depth.txt")
    # Frame sampling for the camera benchmark must not depend on whether a
    # depth packet happens to be missing. TUM's depth stream has occasional
    # ~30 ms gaps, so attach the temporally nearest depth after RGB/GT
    # association instead of shrinking the camera sampling pool.
    rgb_to_gt = dict(associate_nearest(rgb_rows, gt_rows, tolerance))
    depth_times = np.asarray([row[0] for row in depth_rows], dtype=np.float64)
    records = []
    for rgb_index in sorted(rgb_to_gt):
        gt_index = rgb_to_gt[rgb_index]
        rgb_timestamp, rgb_data = rgb_rows[rgb_index]
        insertion = int(np.searchsorted(depth_times, rgb_timestamp))
        depth_index = min(
            (index for index in (insertion - 1, insertion) if 0 <= index < len(depth_rows)),
            key=lambda index: abs(depth_times[index] - rgb_timestamp),
        )
        gt_timestamp, gt_data = gt_rows[gt_index]
        depth_timestamp, depth_data = depth_rows[depth_index]
        records.append(
            FrameRecord(
                rgb_timestamp=rgb_timestamp,
                rgb_path=sequence_dir / rgb_data[0],
                gt_timestamp=gt_timestamp,
                c2w=gt_row_to_c2w(gt_data),
                depth_timestamp=depth_timestamp,
                depth_path=sequence_dir / depth_data[0],
            )
        )
    if not records:
        raise ValueError(f"{sequence_dir.name}: no RGB/pose/depth triplets could be associated")
    return records


def restrict_to_rgb90(records: list[FrameRecord], sequence_dir: Path, tolerance: float) -> list[FrameRecord]:
    rgb90_dir = sequence_dir / "rgb_90"
    if not rgb90_dir.is_dir():
        raise FileNotFoundError(f"Missing prepared subset: {rgb90_dir}")
    timestamps = sorted(float(path.stem) for path in rgb90_dir.glob("*.png"))
    record_times = np.asarray([record.rgb_timestamp for record in records])
    selected: list[FrameRecord] = []
    for timestamp in timestamps:
        index = int(np.argmin(np.abs(record_times - timestamp)))
        if abs(record_times[index] - timestamp) >= tolerance:
            raise ValueError(f"{sequence_dir.name}: cannot associate rgb_90 timestamp {timestamp}")
        selected.append(records[index])
    return selected


def select_sequence_dirs(data_root: Path, requested: Sequence[str] | None) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    sequences = sorted(path for path in data_root.iterdir() if (path / "rgb.txt").is_file())
    if requested:
        mapping = {path.name: path for path in sequences}
        unknown = sorted(set(requested) - set(mapping))
        if unknown:
            raise ValueError(f"Unknown sequence(s): {', '.join(unknown)}")
        sequences = [mapping[name] for name in requested]
    if not sequences:
        raise ValueError("No TUM-Dynamics sequences found")
    return sequences


def sample_records(
    pools: dict[str, list[FrameRecord]], num_frames: int, seed: int
) -> tuple[dict[str, list[FrameRecord]], dict[str, list[int]]]:
    # RandomState intentionally matches the official VGGT evaluation code's
    # np.random.seed(seed) + sequential np.random.choice calls.
    rng = np.random.RandomState(seed)
    sampled: dict[str, list[FrameRecord]] = {}
    sampled_indices: dict[str, list[int]] = {}
    for sequence_name, pool in pools.items():
        if len(pool) < num_frames:
            raise ValueError(f"{sequence_name}: only {len(pool)} frames, need {num_frames}")
        indices = rng.choice(len(pool), num_frames, replace=False).tolist()
        sampled_indices[sequence_name] = indices
        sampled[sequence_name] = [pool[index] for index in indices]
    return sampled, sampled_indices


def load_model(checkpoint: Path, device: torch.device) -> VGGTOmega:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    model = VGGTOmega()
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        state = torch.load(checkpoint, mmap=True, **kwargs)
    except TypeError:
        state = torch.load(checkpoint, **kwargs)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device).eval()


def to_homogeneous_w2c(extrinsics: torch.Tensor) -> np.ndarray:
    w2c = extrinsics.detach().float().cpu().numpy().astype(np.float64)
    result = np.broadcast_to(np.eye(4), (len(w2c), 4, 4)).copy()
    result[:, :3] = w2c
    return result


def pairwise_pose_errors(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for i in range(len(pred_w2c)):
        for j in range(i + 1, len(pred_w2c)):
            gt_relative = gt_w2c[i] @ np.linalg.inv(gt_w2c[j])
            pred_relative = pred_w2c[i] @ np.linalg.inv(pred_w2c[j])

            rotation_delta = gt_relative[:3, :3].T @ pred_relative[:3, :3]
            cosine = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
            rotation_errors.append(math.degrees(math.acos(float(cosine))))

            gt_t = gt_relative[:3, 3]
            pred_t = pred_relative[:3, 3]
            denominator = np.linalg.norm(gt_t) * np.linalg.norm(pred_t)
            if denominator <= 1e-15:
                translation_errors.append(1e6)
            else:
                # Translation direction is defined up to sign for essential geometry.
                cosine_t = np.clip(abs(float(np.dot(gt_t, pred_t))) / denominator, 0.0, 1.0)
                translation_errors.append(math.degrees(math.acos(cosine_t)))
    return np.asarray(rotation_errors), np.asarray(translation_errors)


def official_auc(rotation_errors: np.ndarray, translation_errors: np.ndarray, threshold: int) -> float:
    max_errors = np.maximum(rotation_errors, translation_errors)
    histogram, _ = np.histogram(max_errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(np.float64) / len(max_errors))))


def read_resized_depth(path: Path, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        raw = np.asarray(image, dtype=np.uint16)
    # TUM RGB-D depth PNG values use a factor of 5000 to represent metres.
    resized = Image.fromarray(raw).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.float32) / 5000.0


def depth_sums(
    predicted: np.ndarray,
    records: Sequence[FrameRecord],
    alignment: str,
    max_depth: float,
) -> tuple[float, int, int, list[float]]:
    height, width = predicted.shape[1:]
    ground_truth = np.stack(
        [read_resized_depth(record.depth_path, height, width) for record in records]
    )
    valid = np.isfinite(ground_truth) & (ground_truth > 0) & (ground_truth < max_depth)
    valid &= np.isfinite(predicted) & (predicted > 0)
    scales: list[float] = []
    aligned = predicted.astype(np.float64, copy=True)
    if alignment == "per-frame-median":
        for index in range(len(predicted)):
            if not np.any(valid[index]):
                scales.append(float("nan"))
                continue
            scale = float(np.median(ground_truth[index][valid[index]]) / np.median(predicted[index][valid[index]]))
            aligned[index] *= scale
            scales.append(scale)
    else:
        scale = float(np.median(ground_truth[valid]) / np.median(predicted[valid]))
        aligned *= scale
        scales = [scale] * len(predicted)

    gt_valid = ground_truth[valid].astype(np.float64)
    pred_valid = aligned[valid]
    abs_rel_sum = float(np.sum(np.abs(pred_valid - gt_valid) / gt_valid))
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta_count = int(np.count_nonzero(ratio < 1.25))
    return abs_rel_sum, delta_count, len(gt_valid), scales


def main() -> int:
    args = parse_args()
    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2")
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be at least 1")
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise ValueError("--image-resolution must be positive and divisible by 16")

    sequence_dirs = select_sequence_dirs(args.data_root, args.sequences)
    pools: dict[str, list[FrameRecord]] = {}
    for sequence_dir in sequence_dirs:
        records = load_frame_records(sequence_dir, args.association_tolerance)
        if args.sampling_pool == "rgb_90":
            records = restrict_to_rgb90(records, sequence_dir, args.association_tolerance)
        pools[sequence_dir.name] = records
        print(f"{sequence_dir.name}: sampling pool has {len(records)} RGB/pose/depth frames")

    sampled, sampled_indices = sample_records(pools, args.num_frames, args.seed)
    selection = {
        name: {
            "pool_indices": sampled_indices[name],
            "rgb_timestamps": [record.rgb_timestamp for record in records],
            "rgb_paths": [str(record.rgb_path) for record in records],
        }
        for name, records in sampled.items()
    }
    if args.dry_run:
        print(json.dumps(selection, indent=2))
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega inference requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sampled_frames.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
        handle.write("\n")

    print(f"Loading {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    if args.attention_mode == "register-only-zero-shot":
        model.aggregator.inter_frame_attention_types = ["register"] * model.aggregator.depth
    num_register_blocks = model.aggregator.inter_frame_attention_types.count("register")
    print(
        f"Attention schedule: {args.attention_mode} "
        f"({num_register_blocks}/{model.aggregator.depth} inter-frame blocks use register attention)"
    )
    all_rotation_errors: list[np.ndarray] = []
    all_translation_errors: list[np.ndarray] = []
    total_abs_rel = 0.0
    total_delta = 0
    total_valid = 0
    per_sequence: list[dict[str, object]] = []

    for sequence_name, records in sampled.items():
        started = time.perf_counter()
        images = load_and_preprocess_images(
            [str(record.rgb_path) for record in records],
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(device, non_blocking=True)
        # Warm up kernels once, then time only the model forward with CUDA
        # events. Dataset I/O, preprocessing and metric computation are not
        # included in model_latency_ms.
        with torch.inference_mode():
            _warmup_predictions = model(images)
        del _warmup_predictions
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings_ms: list[float] = []
        predictions = None
        with torch.inference_mode():
            for _ in range(args.timing_repeats):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                current_predictions = model(images)
                end_event.record()
                torch.cuda.synchronize(device)
                timings_ms.append(float(start_event.elapsed_time(end_event)))
                if predictions is not None:
                    del predictions
                predictions = current_predictions
        assert predictions is not None
        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
        model_latency_ms = float(np.median(timings_ms))
        with torch.inference_mode():
            extrinsics, _ = encoding_to_camera(
                predictions["pose_enc"], predictions["images"].shape[-2:], build_intrinsics=False
            )
        pred_w2c = to_homogeneous_w2c(extrinsics[0])
        gt_c2w = np.stack([record.c2w for record in records])
        gt_w2c = np.linalg.inv(gt_c2w)
        rotation_errors, translation_errors = pairwise_pose_errors(pred_w2c, gt_w2c)
        predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
        abs_rel_sum, delta_count, valid_count, scales = depth_sums(
            predicted_depth, records, args.depth_alignment, args.max_depth
        )
        all_rotation_errors.append(rotation_errors)
        all_translation_errors.append(translation_errors)
        total_abs_rel += abs_rel_sum
        total_delta += delta_count
        total_valid += valid_count
        elapsed = time.perf_counter() - started
        row: dict[str, object] = {
            "sequence": sequence_name,
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
        }
        per_sequence.append(row)
        print(
            f"[{sequence_name}] AUC@3={row['auc_3_percent']:.2f}, "
            f"AUC@30={row['auc_30_percent']:.2f}, delta1.25={row['delta_1_25_percent']:.2f}, "
            f"AbsRel={row['abs_rel']:.4f}, latency={model_latency_ms:.1f}ms, "
            f"peak={peak_allocated_gib:.2f}GiB"
        )
        del images, predictions, extrinsics
        torch.cuda.empty_cache()

    rotation_errors = np.concatenate(all_rotation_errors)
    translation_errors = np.concatenate(all_translation_errors)
    result = {
        "protocol": {
            "seed": args.seed,
            "attention_mode": args.attention_mode,
            "register_attention_blocks": num_register_blocks,
            "total_inter_frame_blocks": model.aggregator.depth,
            "timing_repeats": args.timing_repeats,
            "num_frames_per_sequence": args.num_frames,
            "sampling_pool": args.sampling_pool,
            "resize_mode": args.resize_mode,
            "image_resolution": args.image_resolution,
            "depth_alignment": args.depth_alignment,
            "max_depth_m": args.max_depth,
            "num_sequences": len(sampled),
            "num_pose_pairs": len(rotation_errors),
        },
        "paper_targets_1b": PAPER_TARGETS,
        "overall": {
            "auc_3_percent": 100 * official_auc(rotation_errors, translation_errors, 3),
            "auc_30_percent": 100 * official_auc(rotation_errors, translation_errors, 30),
            "delta_1_25_percent": 100 * total_delta / total_valid,
            "abs_rel": total_abs_rel / total_valid,
            "valid_depth_pixels": total_valid,
            "model_latency_ms_mean": float(
                np.mean([float(row["model_latency_ms"]) for row in per_sequence])
            ),
            "peak_allocated_gib_max": float(
                np.max([float(row["peak_allocated_gib"]) for row in per_sequence])
            ),
            "peak_reserved_gib_max": float(
                np.max([float(row["peak_reserved_gib"]) for row in per_sequence])
            ),
        },
        "per_sequence": per_sequence,
    }
    result["difference_from_paper"] = {
        key: float(result["overall"][key]) - target for key, target in PAPER_TARGETS.items()
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
    print("\nPaper reproduction result (Ours-1B target in parentheses):")
    print(f"  AUC@3:    {overall['auc_3_percent']:.2f}  ({PAPER_TARGETS['auc_3_percent']:.1f})")
    print(f"  AUC@30:   {overall['auc_30_percent']:.2f}  ({PAPER_TARGETS['auc_30_percent']:.1f})")
    print(f"  delta1.25:{overall['delta_1_25_percent']:.2f}  ({PAPER_TARGETS['delta_1_25_percent']:.1f})")
    print(f"  AbsRel:   {overall['abs_rel']:.4f} ({PAPER_TARGETS['abs_rel']:.3f})")
    print(f"Saved reproducible results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
