#!/usr/bin/env python3
"""Compare PairFusion predictions against dense VGGT-Omega per frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval_7scenes_paper as seven_eval  # noqa: E402
import scripts.eval_tum_dynamics_paper as tum_eval  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.frame_sampling import SAMPLING_STRATEGIES, sample_record_pools  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from vggt_omega.utils.reference_frame import (  # noqa: E402
    reorder_reference_first,
    resolve_first_frame_token_indices,
    resolve_reference_frame_index,
)


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_SEQUENCES = {
    "TUM-Dynamics": [
        "rgbd_dataset_freiburg3_sitting_halfsphere",
        "rgbd_dataset_freiburg3_sitting_rpy",
    ],
    "7Scenes": ["chess/seq-03", "chess/seq-05"],
}


@dataclass(frozen=True)
class PredictionBundle:
    w2c: np.ndarray
    depth: np.ndarray
    image_hw: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("TUM-Dynamics", "7Scenes"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-strategy", choices=SAMPLING_STRATEGIES, default="uniform")
    parser.add_argument("--sampling-unit", choices=("scene", "sequence"), default="sequence")
    parser.add_argument("--reference-frame-index", type=int, default=0)
    parser.add_argument("--first-frame-token-indices", default="0")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--depth-alignment", choices=("per-frame-median", "per-sequence-median"), default="per-frame-median")
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument("--frame-fusion-pair-percents", nargs="+", type=float, default=[25.0])
    parser.add_argument("--frame-fusion-start-layers", nargs="+", type=int, default=[18])
    parser.add_argument("--frame-fusion-pool-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def slugify(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def load_model(checkpoint: Path, device: torch.device, first_frame_token_indices: tuple[int, ...]) -> VGGTOmega:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    model = VGGTOmega(
        merge_ratio=0.0,
        first_frame_token_indices=first_frame_token_indices,
        frame_fusion_mode="none",
    )
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


def select_records(args: argparse.Namespace) -> tuple[dict[str, list[object]], dict[str, list[int]]]:
    requested = args.sequences or DEFAULT_SEQUENCES[args.dataset]
    if args.dataset == "7Scenes":
        sequence_dirs = seven_eval.select_sequence_dirs(args.data_root, requested)
        pools: dict[str, list[object]] = {}
        for sequence_dir in sequence_dirs:
            sequence_name = f"{sequence_dir.parent.name}/{sequence_dir.name}"
            records = seven_eval.load_frame_records(sequence_dir)
            pool_name = sequence_dir.parent.name if args.sampling_unit == "scene" else sequence_name
            pools.setdefault(pool_name, []).extend(records)
    else:
        sequence_dirs = tum_eval.select_sequence_dirs(args.data_root, requested)
        pools = {}
        for sequence_dir in sequence_dirs:
            records = tum_eval.load_frame_records(sequence_dir, args.association_tolerance)
            pools[sequence_dir.name] = records

    sampled, pool_indices = sample_record_pools(
        pools,
        args.num_frames,
        args.seed,
        strategy=args.sampling_strategy,
    )
    reference_index = resolve_reference_frame_index(args.reference_frame_index, args.num_frames)
    sampled = {
        name: reorder_reference_first(records, reference_index)
        for name, records in sampled.items()
    }
    return sampled, pool_indices


def frame_label(dataset: str, record: object) -> str:
    if dataset == "7Scenes":
        return str(record.index)
    return f"{record.rgb_timestamp:.6f}"


def frame_rgb_path(record: object) -> str:
    return str(record.rgb_path)


def extract_predictions(predictions: dict[str, torch.Tensor], dataset: str) -> PredictionBundle:
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        extrinsics, _ = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
            build_intrinsics=False,
        )
    to_w2c = seven_eval.to_homogeneous_w2c if dataset == "7Scenes" else tum_eval.to_homogeneous_w2c
    w2c = to_w2c(extrinsics[0])
    depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
    image_hw = tuple(int(value) for value in predictions["images"].shape[-2:])
    return PredictionBundle(w2c=w2c, depth=depth, image_hw=image_hw)


def run_model(model: VGGTOmega, images: torch.Tensor, dataset: str, device: torch.device) -> tuple[PredictionBundle, dict[str, object], float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        predictions = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    bundle = extract_predictions(predictions, dataset)
    debug = dict(model.aggregator.last_frame_fusion_debug or {})
    del predictions
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bundle, debug, elapsed


def rotation_angle_deg(rotation_delta: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def translation_direction_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-15:
        return float("nan")
    cosine = np.clip(abs(float(np.dot(first, second))) / denominator, 0.0, 1.0)
    return math.degrees(math.acos(cosine))


def relative_pose(w2c: np.ndarray, first: int, second: int) -> np.ndarray:
    return w2c[first] @ np.linalg.inv(w2c[second])


def compare_relative_poses(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    rot = rotation_angle_deg(first[:3, :3].T @ second[:3, :3])
    trans = translation_direction_angle_deg(first[:3, 3], second[:3, 3])
    norm = np.linalg.norm(first[:3, 3])
    norm_delta = float(np.linalg.norm(second[:3, 3] - first[:3, 3]) / max(norm, 1e-12))
    return rot, trans, norm_delta


def pairwise_frame_stats(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> dict[str, np.ndarray]:
    num_frames = pred_w2c.shape[0]
    values: list[list[float]] = [[] for _ in range(num_frames)]
    for first in range(num_frames):
        for second in range(first + 1, num_frames):
            gt_relative = relative_pose(gt_w2c, first, second)
            pred_relative = relative_pose(pred_w2c, first, second)
            rot = rotation_angle_deg(gt_relative[:3, :3].T @ pred_relative[:3, :3])
            trans = translation_direction_angle_deg(gt_relative[:3, 3], pred_relative[:3, 3])
            max_error = max(rot, trans)
            values[first].append(max_error)
            values[second].append(max_error)
    return {
        "mean": np.asarray([float(np.mean(frame_values)) for frame_values in values], dtype=np.float64),
        "p90": np.asarray([float(np.percentile(frame_values, 90)) for frame_values in values], dtype=np.float64),
        "max": np.asarray([float(np.max(frame_values)) for frame_values in values], dtype=np.float64),
    }


def pairwise_prediction_delta_stats(baseline_w2c: np.ndarray, fused_w2c: np.ndarray) -> dict[str, np.ndarray]:
    num_frames = baseline_w2c.shape[0]
    values: list[list[float]] = [[] for _ in range(num_frames)]
    for first in range(num_frames):
        for second in range(first + 1, num_frames):
            baseline_relative = relative_pose(baseline_w2c, first, second)
            fused_relative = relative_pose(fused_w2c, first, second)
            rot, trans, _ = compare_relative_poses(baseline_relative, fused_relative)
            max_error = np.nanmax([rot, trans])
            values[first].append(float(max_error))
            values[second].append(float(max_error))
    return {
        "mean": np.asarray([float(np.mean(frame_values)) for frame_values in values], dtype=np.float64),
        "p90": np.asarray([float(np.percentile(frame_values, 90)) for frame_values in values], dtype=np.float64),
        "max": np.asarray([float(np.max(frame_values)) for frame_values in values], dtype=np.float64),
    }


def ref_relative_frame_stats(
    baseline_w2c: np.ndarray,
    fused_w2c: np.ndarray,
    gt_w2c: np.ndarray,
) -> dict[str, np.ndarray]:
    num_frames = baseline_w2c.shape[0]
    camera_rot_delta = np.zeros(num_frames, dtype=np.float64)
    camera_trans_delta = np.zeros(num_frames, dtype=np.float64)
    camera_norm_delta = np.zeros(num_frames, dtype=np.float64)
    baseline_gt_max = np.zeros(num_frames, dtype=np.float64)
    fused_gt_max = np.zeros(num_frames, dtype=np.float64)
    for frame in range(num_frames):
        baseline_relative = relative_pose(baseline_w2c, frame, 0)
        fused_relative = relative_pose(fused_w2c, frame, 0)
        gt_relative = relative_pose(gt_w2c, frame, 0)
        rot, trans, norm_delta = compare_relative_poses(baseline_relative, fused_relative)
        camera_rot_delta[frame] = rot
        camera_trans_delta[frame] = 0.0 if math.isnan(trans) else trans
        camera_norm_delta[frame] = norm_delta
        baseline_rot, baseline_trans, _ = compare_relative_poses(gt_relative, baseline_relative)
        fused_rot, fused_trans, _ = compare_relative_poses(gt_relative, fused_relative)
        baseline_gt_max[frame] = np.nanmax([baseline_rot, baseline_trans])
        fused_gt_max[frame] = np.nanmax([fused_rot, fused_trans])
    return {
        "camera_ref_rot_delta_deg": camera_rot_delta,
        "camera_ref_trans_dir_delta_deg": camera_trans_delta,
        "camera_ref_trans_norm_rel_delta": camera_norm_delta,
        "baseline_ref_pose_max_error_deg": baseline_gt_max,
        "fusion_ref_pose_max_error_deg": fused_gt_max,
        "delta_ref_pose_max_error_deg": fused_gt_max - baseline_gt_max,
    }


def load_depth_ground_truth(
    dataset: str,
    records: Sequence[object],
    image_hw: tuple[int, int],
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_hw
    if dataset == "7Scenes":
        ground_truth = np.stack(
            [seven_eval.read_resized_depth(record.depth_path, height, width) for record in records]
        )
        valid = np.isfinite(ground_truth) & (ground_truth > min_depth) & (ground_truth < max_depth)
    else:
        ground_truth = np.stack(
            [tum_eval.read_resized_depth(record.depth_path, height, width) for record in records]
        )
        valid = np.isfinite(ground_truth) & (ground_truth > 0.0) & (ground_truth < max_depth)
    return ground_truth.astype(np.float64), valid


def align_depth(
    dataset: str,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid_gt: np.ndarray,
    alignment: str,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = valid_gt & np.isfinite(predicted) & (predicted > 0.0)
    aligned = predicted.astype(np.float64, copy=True)
    scales = np.full(predicted.shape[0], np.nan, dtype=np.float64)
    if alignment == "per-frame-median":
        for frame in range(predicted.shape[0]):
            if not np.any(valid[frame]):
                continue
            scale = float(np.median(ground_truth[frame][valid[frame]]) / np.median(predicted[frame][valid[frame]]))
            aligned[frame] *= scale
            scales[frame] = scale
    elif alignment == "per-sequence-median":
        if not np.any(valid):
            raise ValueError("sequence has no valid depth pixels")
        scale = float(np.median(ground_truth[valid]) / np.median(predicted[valid]))
        aligned *= scale
        scales[:] = scale
    else:
        raise ValueError(f"unknown depth alignment: {alignment}")
    if dataset == "7Scenes":
        aligned = np.clip(aligned, min_depth, max_depth)
    return aligned, valid, scales


def per_frame_depth_metrics(
    aligned: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = aligned.shape[0]
    abs_rel = np.full(num_frames, np.nan, dtype=np.float64)
    delta = np.full(num_frames, np.nan, dtype=np.float64)
    valid_count = np.zeros(num_frames, dtype=np.int64)
    for frame in range(num_frames):
        mask = valid[frame] & np.isfinite(aligned[frame])
        valid_count[frame] = int(np.count_nonzero(mask))
        if valid_count[frame] == 0:
            continue
        gt = ground_truth[frame][mask]
        pred = aligned[frame][mask]
        abs_rel[frame] = float(np.mean(np.abs(pred - gt) / gt))
        ratio = np.maximum(pred / gt, gt / pred)
        delta[frame] = float(100.0 * np.count_nonzero(ratio < 1.25) / len(gt))
    return abs_rel, delta, valid_count


def depth_deviation_metrics(
    baseline_aligned: np.ndarray,
    fused_aligned: np.ndarray,
    valid_baseline: np.ndarray,
    valid_fused: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = baseline_aligned.shape[0]
    rel = np.full(num_frames, np.nan, dtype=np.float64)
    rmse = np.full(num_frames, np.nan, dtype=np.float64)
    median_abs = np.full(num_frames, np.nan, dtype=np.float64)
    for frame in range(num_frames):
        mask = (
            valid_baseline[frame]
            & valid_fused[frame]
            & np.isfinite(baseline_aligned[frame])
            & np.isfinite(fused_aligned[frame])
            & (baseline_aligned[frame] > 0.0)
        )
        if not np.any(mask):
            continue
        baseline = baseline_aligned[frame][mask]
        fused = fused_aligned[frame][mask]
        diff = fused - baseline
        rel[frame] = float(np.mean(np.abs(diff) / np.maximum(np.abs(baseline), 1e-6)))
        rmse[frame] = float(np.sqrt(np.mean(diff * diff)))
        median_abs[frame] = float(np.median(np.abs(diff)))
    return rel, rmse, median_abs


def fusion_roles(num_frames: int, debug: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame_rows = [
        {
            "fusion_role": "reference" if frame == 0 else "unpaired",
            "paired_with": "",
            "pair_similarity": "",
        }
        for frame in range(num_frames)
    ]
    batches = debug.get("batches") or []
    first_batch = batches[0] if batches else {}
    pairs = first_batch.get("pairs") or []
    for pair in pairs:
        frame_a = int(pair["frame_a"])
        frame_b = int(pair["frame_b"])
        similarity = float(pair["similarity"])
        frame_rows[frame_a] = {
            "fusion_role": "source",
            "paired_with": frame_b,
            "pair_similarity": similarity,
        }
        frame_rows[frame_b] = {
            "fusion_role": "target",
            "paired_with": frame_a,
            "pair_similarity": similarity,
        }
    return frame_rows, pairs


def float_or_empty(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def summarize_top_rows(rows: list[dict[str, object]], key: str, top_k: int) -> list[dict[str, object]]:
    ranked = sorted(
        rows,
        key=lambda row: float(row[key]) if row[key] != "" else float("-inf"),
        reverse=True,
    )
    return ranked[:top_k]


def evaluate_config(
    *,
    args: argparse.Namespace,
    sequence_name: str,
    records: Sequence[object],
    baseline: PredictionBundle,
    fused: PredictionBundle,
    debug: dict[str, object],
    pair_percent: float,
    start_layer: int,
    output_dir: Path,
    baseline_seconds: float,
    fusion_seconds: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    gt_w2c = np.linalg.inv(np.stack([record.c2w for record in records]))
    ref_stats = ref_relative_frame_stats(baseline.w2c, fused.w2c, gt_w2c)
    baseline_pair_stats = pairwise_frame_stats(baseline.w2c, gt_w2c)
    fused_pair_stats = pairwise_frame_stats(fused.w2c, gt_w2c)
    pair_delta_stats = pairwise_prediction_delta_stats(baseline.w2c, fused.w2c)

    ground_truth_depth, valid_gt = load_depth_ground_truth(
        args.dataset,
        records,
        baseline.image_hw,
        args.min_depth,
        args.max_depth,
    )
    baseline_depth, baseline_valid, baseline_scales = align_depth(
        args.dataset,
        baseline.depth,
        ground_truth_depth,
        valid_gt,
        args.depth_alignment,
        args.min_depth,
        args.max_depth,
    )
    fused_depth, fused_valid, fused_scales = align_depth(
        args.dataset,
        fused.depth,
        ground_truth_depth,
        valid_gt,
        args.depth_alignment,
        args.min_depth,
        args.max_depth,
    )
    baseline_abs_rel, baseline_delta, valid_count = per_frame_depth_metrics(
        baseline_depth,
        ground_truth_depth,
        baseline_valid,
    )
    fused_abs_rel, fused_delta, _ = per_frame_depth_metrics(
        fused_depth,
        ground_truth_depth,
        fused_valid,
    )
    depth_rel, depth_rmse, depth_median_abs = depth_deviation_metrics(
        baseline_depth,
        fused_depth,
        baseline_valid,
        fused_valid,
    )

    roles, pairs = fusion_roles(len(records), debug)
    rows: list[dict[str, object]] = []
    for frame, record in enumerate(records):
        camera_ref_max_delta = max(
            float(ref_stats["camera_ref_rot_delta_deg"][frame]),
            float(ref_stats["camera_ref_trans_dir_delta_deg"][frame]),
        )
        row = {
            "dataset": args.dataset,
            "sequence": sequence_name,
            "pair_percent": pair_percent,
            "start_layer": start_layer,
            "frame_position": frame,
            "frame_label": frame_label(args.dataset, record),
            "rgb_path": frame_rgb_path(record),
            **roles[frame],
            "camera_ref_rot_delta_deg": ref_stats["camera_ref_rot_delta_deg"][frame],
            "camera_ref_trans_dir_delta_deg": ref_stats["camera_ref_trans_dir_delta_deg"][frame],
            "camera_ref_max_delta_deg": camera_ref_max_delta,
            "camera_ref_trans_norm_rel_delta": ref_stats["camera_ref_trans_norm_rel_delta"][frame],
            "camera_pair_pred_delta_mean_deg": pair_delta_stats["mean"][frame],
            "camera_pair_pred_delta_p90_deg": pair_delta_stats["p90"][frame],
            "camera_pair_pred_delta_max_deg": pair_delta_stats["max"][frame],
            "baseline_ref_pose_max_error_deg": ref_stats["baseline_ref_pose_max_error_deg"][frame],
            "fusion_ref_pose_max_error_deg": ref_stats["fusion_ref_pose_max_error_deg"][frame],
            "delta_ref_pose_max_error_deg": ref_stats["delta_ref_pose_max_error_deg"][frame],
            "baseline_pair_mean_max_error_deg": baseline_pair_stats["mean"][frame],
            "fusion_pair_mean_max_error_deg": fused_pair_stats["mean"][frame],
            "delta_pair_mean_max_error_deg": fused_pair_stats["mean"][frame] - baseline_pair_stats["mean"][frame],
            "baseline_pair_p90_max_error_deg": baseline_pair_stats["p90"][frame],
            "fusion_pair_p90_max_error_deg": fused_pair_stats["p90"][frame],
            "delta_pair_p90_max_error_deg": fused_pair_stats["p90"][frame] - baseline_pair_stats["p90"][frame],
            "baseline_depth_abs_rel": baseline_abs_rel[frame],
            "fusion_depth_abs_rel": fused_abs_rel[frame],
            "delta_depth_abs_rel": fused_abs_rel[frame] - baseline_abs_rel[frame],
            "baseline_depth_delta1_25_percent": baseline_delta[frame],
            "fusion_depth_delta1_25_percent": fused_delta[frame],
            "delta_depth_delta1_25_percent": fused_delta[frame] - baseline_delta[frame],
            "depth_vs_baseline_abs_rel": depth_rel[frame],
            "depth_vs_baseline_rmse_m": depth_rmse[frame],
            "depth_vs_baseline_median_abs_m": depth_median_abs[frame],
            "baseline_depth_scale": baseline_scales[frame],
            "fusion_depth_scale": fused_scales[frame],
            "valid_depth_pixels": int(valid_count[frame]),
        }
        rows.append({key: float_or_empty(value) for key, value in row.items()})

    config_dir = output_dir / f"pair{pair_percent:g}_start{start_layer}"
    config_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config_dir / "frame_deviation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": args.dataset,
        "sequence": sequence_name,
        "pair_percent": pair_percent,
        "start_layer": start_layer,
        "pool_size": args.frame_fusion_pool_size,
        "num_frames": len(records),
        "baseline_seconds": baseline_seconds,
        "fusion_seconds": fusion_seconds,
        "frame_fusion_debug": debug,
        "selected_pairs": pairs,
        "csv": str(csv_path),
        "top_by_camera_ref_max_delta": summarize_top_rows(rows, "camera_ref_max_delta_deg", args.top_k),
        "top_by_camera_pair_prediction_delta": summarize_top_rows(rows, "camera_pair_pred_delta_mean_deg", args.top_k),
        "top_by_depth_prediction_delta": summarize_top_rows(rows, "depth_vs_baseline_abs_rel", args.top_k),
        "top_by_depth_absrel_worsening": summarize_top_rows(rows, "delta_depth_abs_rel", args.top_k),
        "top_by_pose_pair_worsening": summarize_top_rows(rows, "delta_pair_mean_max_error_deg", args.top_k),
        "role_summary": {
            role: sum(1 for row in rows if row["fusion_role"] == role)
            for role in ("reference", "source", "target", "unpaired")
        },
        "means": {
            "camera_ref_max_delta_deg": float(np.mean([row["camera_ref_max_delta_deg"] for row in rows])),
            "camera_pair_pred_delta_mean_deg": float(np.mean([row["camera_pair_pred_delta_mean_deg"] for row in rows])),
            "depth_vs_baseline_abs_rel": float(np.nanmean([row["depth_vs_baseline_abs_rel"] for row in rows if row["depth_vs_baseline_abs_rel"] != ""])),
            "delta_depth_abs_rel": float(np.nanmean([row["delta_depth_abs_rel"] for row in rows if row["delta_depth_abs_rel"] != ""])),
            "delta_pair_mean_max_error_deg": float(np.mean([row["delta_pair_mean_max_error_deg"] for row in rows])),
        },
    }
    with (config_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return rows, summary


def write_markdown_summary(output_dir: Path, summaries: Sequence[dict[str, object]], top_k: int) -> None:
    lines = [
        "# Frame Fusion Prediction Deviation",
        "",
        "Dense VGGT-Omega is the baseline. Camera deltas compare relative poses against the dense prediction; depth deltas compare scale-aligned depth maps.",
        "",
        "| Sequence | Pair% | start | pairs | mean camera pair delta deg | mean depth delta | mean pose worsening deg | mean depth AbsRel worsening |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        role_summary = summary["role_summary"]
        pairs = int(role_summary["source"])
        means = summary["means"]
        lines.append(
            f"| {summary['sequence']} | {summary['pair_percent']:.0f} | {summary['start_layer']} | {pairs} | "
            f"{means['camera_pair_pred_delta_mean_deg']:.4f} | "
            f"{means['depth_vs_baseline_abs_rel']:.6f} | "
            f"{means['delta_pair_mean_max_error_deg']:.4f} | "
            f"{means['delta_depth_abs_rel']:.6f} |"
        )
    lines.append("")
    for summary in summaries:
        lines.append(f"## {summary['sequence']} pair={summary['pair_percent']:.0f}% start={summary['start_layer']}")
        lines.append("")
        lines.append(f"Top {top_k} by camera pair prediction delta:")
        lines.append("")
        lines.append("| frame | role | pair | camera pair delta mean deg | depth vs baseline | delta depth AbsRel | delta pose pair mean deg |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
        for row in summary["top_by_camera_pair_prediction_delta"][:top_k]:
            lines.append(
                f"| {row['frame_position']} | {row['fusion_role']} | {row['paired_with']} | "
                f"{float(row['camera_pair_pred_delta_mean_deg']):.4f} | "
                f"{float(row['depth_vs_baseline_abs_rel'] or 0.0):.6f} | "
                f"{float(row['delta_depth_abs_rel'] or 0.0):.6f} | "
                f"{float(row['delta_pair_mean_max_error_deg']):.4f} |"
            )
        lines.append("")
        lines.append(f"Top {top_k} by depth prediction delta:")
        lines.append("")
        lines.append("| frame | role | pair | depth vs baseline | delta depth AbsRel | camera pair delta mean deg |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: |")
        for row in summary["top_by_depth_prediction_delta"][:top_k]:
            lines.append(
                f"| {row['frame_position']} | {row['fusion_role']} | {row['paired_with']} | "
                f"{float(row['depth_vs_baseline_abs_rel'] or 0.0):.6f} | "
                f"{float(row['delta_depth_abs_rel'] or 0.0):.6f} | "
                f"{float(row['camera_pair_pred_delta_mean_deg']):.4f} |"
            )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2")
    for percent in args.frame_fusion_pair_percents:
        if not 0.0 < percent <= 100.0:
            raise ValueError("--frame-fusion-pair-percents values must be in (0, 100]")
    if args.frame_fusion_pool_size <= 0:
        raise ValueError("--frame-fusion-pool-size must be positive")
    if args.max_depth <= 0:
        raise ValueError("--max-depth must be positive")
    if args.dataset == "7Scenes" and not 0.0 < args.min_depth < args.max_depth:
        raise ValueError("7Scenes requires 0 < --min-depth < --max-depth")

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampled, pool_indices = select_records(args)
    first_frame_token_indices = resolve_first_frame_token_indices(
        args.first_frame_token_indices,
        args.num_frames,
    )
    config = {
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "sequences": args.sequences or DEFAULT_SEQUENCES[args.dataset],
        "checkpoint": str(args.checkpoint),
        "num_frames": args.num_frames,
        "seed": args.seed,
        "sampling_strategy": args.sampling_strategy,
        "sampling_unit": args.sampling_unit if args.dataset == "7Scenes" else None,
        "reference_frame_index": args.reference_frame_index,
        "first_frame_token_indices": first_frame_token_indices,
        "image_resolution": args.image_resolution,
        "resize_mode": args.resize_mode,
        "depth_alignment": args.depth_alignment,
        "min_depth": args.min_depth if args.dataset == "7Scenes" else None,
        "max_depth": args.max_depth,
        "pair_percents": args.frame_fusion_pair_percents,
        "start_layers": args.frame_fusion_start_layers,
        "pool_size": args.frame_fusion_pool_size,
        "sampled_pool_indices": pool_indices,
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"Loading {args.checkpoint}")
    model = load_model(args.checkpoint, device, first_frame_token_indices)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for sequence_name, records in sampled.items():
        print(f"[{sequence_name}] loading {len(records)} frames")
        images = load_and_preprocess_images(
            [str(record.rgb_path) for record in records],
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(device, non_blocking=True)
        model.aggregator.set_frame_fusion(mode="none")
        baseline, _, baseline_seconds = run_model(model, images, args.dataset, device)
        print(f"[{sequence_name}] baseline forward {baseline_seconds:.2f}s")
        sequence_output = args.output_dir / slugify(sequence_name)
        sequence_output.mkdir(parents=True, exist_ok=True)

        for pair_percent in args.frame_fusion_pair_percents:
            for start_layer in args.frame_fusion_start_layers:
                model.aggregator.set_frame_fusion(
                    mode="pair-top-percent",
                    start_layer=start_layer,
                    pair_percent=pair_percent,
                    pool_size=args.frame_fusion_pool_size,
                )
                fused, debug, fusion_seconds = run_model(model, images, args.dataset, device)
                print(
                    f"[{sequence_name}] pair={pair_percent:g}% start={start_layer} "
                    f"fusion forward {fusion_seconds:.2f}s"
                )
                rows, summary = evaluate_config(
                    args=args,
                    sequence_name=sequence_name,
                    records=records,
                    baseline=baseline,
                    fused=fused,
                    debug=debug,
                    pair_percent=pair_percent,
                    start_layer=start_layer,
                    output_dir=sequence_output,
                    baseline_seconds=baseline_seconds,
                    fusion_seconds=fusion_seconds,
                )
                all_rows.extend(rows)
                summaries.append(summary)
                del fused
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del baseline, images
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_rows:
        all_csv = args.output_dir / "all_frame_deviation.csv"
        with all_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "summaries": summaries}, handle, indent=2)
        handle.write("\n")
    write_markdown_summary(args.output_dir, summaries, args.top_k)
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
