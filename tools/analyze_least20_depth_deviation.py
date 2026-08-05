#!/usr/bin/env python3
"""Attribute least-20 PairFusion depth deviations to frames and token regions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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
from vggt_omega.models.aggregator import FrameFusionBatchPlan  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.reference_frame import resolve_first_frame_token_indices  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("TUM-Dynamics", "7Scenes"), required=True)
    parser.add_argument("--sampled-frames-json", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--depth-alignment", choices=("per-frame-median", "per-sequence-median"), default="per-frame-median")
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument("--first-frame-token-indices", default="0")
    parser.add_argument("--frame-fusion-pair-percent", type=float, default=25.0)
    parser.add_argument("--frame-fusion-start-layer", type=int, default=-1)
    parser.add_argument("--frame-fusion-pool-size", type=int, default=2)
    parser.add_argument("--frame-fusion-target-keep-percent", type=float, default=20.0)
    parser.add_argument("--frame-fusion-target-keep-seed", type=int, default=33)
    parser.add_argument(
        "--shared-representative",
        choices=("source", "target"),
        default="source",
        help=(
            "Which averaged shared-token duplicate is kept for compressed global attention. "
            "'source' matches the current implementation; 'target' swaps the representative "
            "and copies target shared outputs back to source."
        ),
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--save-depth-npz", action="store_true")
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


def load_sequence_records(args: argparse.Namespace) -> list[object]:
    manifest = json.loads(args.sampled_frames_json.read_text(encoding="utf-8"))
    if args.sequence not in manifest:
        raise KeyError(f"{args.sequence!r} not found in {args.sampled_frames_json}")
    entry = manifest[args.sequence]
    rgb_paths = [Path(path) for path in entry["rgb_paths"]]
    if not rgb_paths:
        raise ValueError(f"{args.sequence}: manifest has no RGB paths")

    if args.dataset == "7Scenes":
        sequence_dir = rgb_paths[0].parent
        all_records = seven_eval.load_frame_records(sequence_dir)
        by_rgb = {str(record.rgb_path): record for record in all_records}
        try:
            return [by_rgb[str(path)] for path in rgb_paths]
        except KeyError as exc:
            raise KeyError(f"{args.sequence}: sampled RGB not found in 7Scenes records: {exc}") from exc

    sequence_dir = rgb_paths[0].parent.parent
    all_records = tum_eval.load_frame_records(sequence_dir, args.association_tolerance)
    by_rgb = {str(record.rgb_path): record for record in all_records}
    try:
        return [by_rgb[str(path)] for path in rgb_paths]
    except KeyError:
        timestamps = entry.get("rgb_timestamps")
        if not timestamps:
            raise
        by_timestamp = {f"{record.rgb_timestamp:.6f}": record for record in all_records}
        return [by_timestamp[f"{float(timestamp):.6f}"] for timestamp in timestamps]


def frame_label(dataset: str, record: object) -> str:
    if dataset == "7Scenes":
        return str(record.index)
    return f"{record.rgb_timestamp:.6f}"


def read_ground_truth_depth(
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
    *,
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
    abs_rel = np.full(aligned.shape[0], np.nan, dtype=np.float64)
    delta = np.full(aligned.shape[0], np.nan, dtype=np.float64)
    valid_count = np.zeros(aligned.shape[0], dtype=np.int64)
    for frame in range(aligned.shape[0]):
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


def finite_float(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return "" if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def target_keep_mask(indices: torch.Tensor | None, num_pairs: int, patch_count: int) -> np.ndarray:
    mask = np.zeros((num_pairs, patch_count), dtype=bool)
    if indices is None or indices.numel() == 0:
        return mask
    if indices.dtype == torch.bool:
        array = indices.detach().cpu().numpy().astype(bool, copy=False)
        if array.shape != mask.shape:
            raise ValueError(f"boolean keep mask shape {array.shape} does not match {mask.shape}")
        return array
    array = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    if array.ndim != 2 or array.shape[0] != num_pairs:
        raise ValueError(f"keep indices shape {array.shape} does not match num_pairs={num_pairs}")
    for pair_index, row in enumerate(array):
        mask[pair_index, row] = True
    return mask


def target_representative_attention_indices(
    *,
    num_frames: int,
    tokens_per_frame: int,
    num_special_tokens: int,
    source_frames: torch.Tensor,
    target_frames: torch.Tensor,
    target_keep_patch_indices: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if source_frames.shape != target_frames.shape:
        raise ValueError(
            "source_frames and target_frames must have the same shape, "
            f"got {tuple(source_frames.shape)} and {tuple(target_frames.shape)}"
        )
    keep_mask = torch.zeros(
        num_frames,
        tokens_per_frame,
        device=device,
        dtype=torch.bool,
    )
    keep_mask[:, :num_special_tokens] = True
    keep_patch_frames = torch.ones(num_frames, device=device, dtype=torch.bool)
    source_frames = source_frames.to(device=device, dtype=torch.long)
    target_frames = target_frames.to(device=device, dtype=torch.long)
    if source_frames.numel() > 0:
        keep_patch_frames[source_frames] = False
    keep_mask[keep_patch_frames, num_special_tokens:] = True
    if target_keep_patch_indices is not None and target_keep_patch_indices.numel() > 0:
        patch_count = tokens_per_frame - num_special_tokens
        retained_mask = torch.as_tensor(
            target_keep_mask(
                target_keep_patch_indices,
                num_pairs=int(source_frames.numel()),
                patch_count=patch_count,
            ),
            device=device,
            dtype=torch.bool,
        )
        pair_offsets, patch_offsets = retained_mask.nonzero(as_tuple=True)
        keep_mask[source_frames[pair_offsets], num_special_tokens + patch_offsets] = True
    return keep_mask.flatten().nonzero(as_tuple=False).flatten()


def copy_target_representative_patch_outputs(
    flat_tokens: torch.Tensor,
    plan: FrameFusionBatchPlan,
    *,
    tokens_per_frame: int,
    num_special_tokens: int,
) -> torch.Tensor:
    if not plan.pairs:
        return flat_tokens
    patch_count = tokens_per_frame - num_special_tokens
    if patch_count <= 0:
        return flat_tokens
    device = flat_tokens.device
    offsets = torch.arange(patch_count, device=device, dtype=torch.long)
    source_frames = plan.source_frames.to(device=device)
    target_frames = plan.target_frames.to(device=device)
    keep_mask = torch.as_tensor(
        target_keep_mask(
            plan.target_keep_patch_indices,
            num_pairs=len(plan.pairs),
            patch_count=patch_count,
        ),
        device=device,
        dtype=torch.bool,
    )
    target_index_chunks = []
    source_index_chunks = []
    for pair_index, (source_frame, target_frame) in enumerate(zip(source_frames, target_frames)):
        copy_offsets = offsets[~keep_mask[pair_index]]
        if copy_offsets.numel() == 0:
            continue
        target_index_chunks.append(target_frame * tokens_per_frame + num_special_tokens + copy_offsets)
        source_index_chunks.append(source_frame * tokens_per_frame + num_special_tokens + copy_offsets)
    if not target_index_chunks:
        return flat_tokens
    target_indices = torch.cat(target_index_chunks, dim=0)
    source_indices = torch.cat(source_index_chunks, dim=0)
    return flat_tokens.index_copy(0, source_indices, flat_tokens.index_select(0, target_indices))


def patch_region_image_mask(patch_mask: np.ndarray, patch_grid_size: tuple[int, int], image_hw: tuple[int, int]) -> np.ndarray:
    patch_h, patch_w = patch_grid_size
    height, width = image_hw
    grid = patch_mask.reshape(patch_h, patch_w)
    rows = np.rint(np.linspace(0, height, patch_h + 1)).astype(np.int64)
    cols = np.rint(np.linspace(0, width, patch_w + 1)).astype(np.int64)
    image_mask = np.zeros((height, width), dtype=bool)
    for patch_row in range(patch_h):
        for patch_col in range(patch_w):
            if grid[patch_row, patch_col]:
                image_mask[rows[patch_row] : rows[patch_row + 1], cols[patch_col] : cols[patch_col + 1]] = True
    return image_mask


def mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def sum_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sum(values))


def run_model(
    model: VGGTOmega,
    images: torch.Tensor,
    *,
    dataset: str,
    device: torch.device,
    capture_plans: bool = False,
    shared_representative: str = "source",
) -> tuple[np.ndarray, tuple[int, int], dict[str, object], list[dict[str, object]], float]:
    if shared_representative not in {"source", "target"}:
        raise ValueError(f"unknown shared representative: {shared_representative!r}")
    captured: list[dict[str, object]] = []
    original_build = model.aggregator._build_frame_fusion_pair_plans
    original_copy = model.aggregator._copy_pair_patch_outputs

    if capture_plans or shared_representative == "target":
        def capture_build(tokens: torch.Tensor, *, patch_grid_size: tuple[int, int], source_layer: int):
            plans = original_build(tokens, patch_grid_size=patch_grid_size, source_layer=source_layer)
            captured.clear()
            patch_count = int(tokens.shape[2] - model.aggregator.patch_token_start)
            transformed_plans = []
            for plan in plans:
                keep_mask = target_keep_mask(
                    plan.target_keep_patch_indices,
                    num_pairs=len(plan.pairs),
                    patch_count=patch_count,
                )
                if shared_representative == "target":
                    attention_indices = target_representative_attention_indices(
                        num_frames=int(tokens.shape[1]),
                        tokens_per_frame=int(tokens.shape[2]),
                        num_special_tokens=int(model.aggregator.patch_token_start),
                        source_frames=plan.source_frames,
                        target_frames=plan.target_frames,
                        target_keep_patch_indices=plan.target_keep_patch_indices,
                        device=tokens.device,
                    )
                    transformed = FrameFusionBatchPlan(
                        pairs=plan.pairs,
                        source_frames=plan.source_frames,
                        target_frames=plan.target_frames,
                        attention_indices=attention_indices,
                        unique_candidate_count=plan.unique_candidate_count,
                        requested_pair_count=plan.requested_pair_count,
                        target_keep_patch_indices=plan.target_keep_patch_indices,
                    )
                else:
                    transformed = plan
                transformed_plans.append(transformed)
                captured.append(
                    {
                        "source_layer": int(source_layer),
                        "patch_grid_size": [int(patch_grid_size[0]), int(patch_grid_size[1])],
                        "patch_count": patch_count,
                        "pairs": [
                            {
                                "frame_a": int(pair.frame_a),
                                "frame_b": int(pair.frame_b),
                                "similarity": float(pair.similarity),
                            }
                            for pair in plan.pairs
                        ],
                        "target_keep_mask": keep_mask,
                    }
                )
            return transformed_plans

        model.aggregator._build_frame_fusion_pair_plans = capture_build
    if shared_representative == "target":
        model.aggregator._copy_pair_patch_outputs = copy_target_representative_patch_outputs

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            predictions = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        if capture_plans or shared_representative == "target":
            model.aggregator._build_frame_fusion_pair_plans = original_build
        if shared_representative == "target":
            model.aggregator._copy_pair_patch_outputs = original_copy
    elapsed = time.perf_counter() - started
    depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
    image_hw = tuple(int(value) for value in predictions["images"].shape[-2:])
    debug = dict(model.aggregator.last_frame_fusion_debug or {})
    del predictions
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return depth, image_hw, debug, captured, elapsed


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_region_summary(region_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in region_rows:
        grouped.setdefault((str(row["fusion_role"]), str(row["token_region"])), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (role, region), rows in sorted(grouped.items()):
        pixels = np.asarray([float(row["valid_pixels"]) for row in rows], dtype=np.float64)
        weights = pixels / max(float(np.sum(pixels)), 1.0)
        def weighted(field: str) -> float:
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            valid = np.isfinite(values)
            if not np.any(valid):
                return float("nan")
            local_weights = weights[valid]
            local_weights = local_weights / max(float(np.sum(local_weights)), 1e-12)
            return float(np.sum(values[valid] * local_weights))

        summary_rows.append(
            {
                "fusion_role": role,
                "token_region": region,
                "frames": len(rows),
                "valid_pixels": int(np.sum(pixels)),
                "pixel_weighted_depth_vs_baseline_abs_rel": weighted("depth_vs_baseline_abs_rel"),
                "pixel_weighted_delta_depth_abs_rel": weighted("delta_depth_abs_rel"),
                "mean_abs_depth_delta_m": weighted("mean_abs_depth_delta_m"),
                "mean_deviation_contribution": float(np.nanmean([float(row["deviation_contribution"]) for row in rows])),
                "mean_pixel_fraction": float(np.nanmean([float(row["pixel_fraction"]) for row in rows])),
            }
        )
    return [{key: finite_float(value) for key, value in row.items()} for row in summary_rows]


def role_summary(frame_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    roles = sorted({str(row["fusion_role"]) for row in frame_rows})
    rows: list[dict[str, object]] = []
    for role in roles:
        subset = [row for row in frame_rows if row["fusion_role"] == role]
        rows.append(
            {
                "fusion_role": role,
                "frames": len(subset),
                "mean_depth_vs_baseline_abs_rel": float(np.nanmean([float(row["depth_vs_baseline_abs_rel"]) for row in subset])),
                "mean_delta_depth_abs_rel": float(np.nanmean([float(row["delta_depth_abs_rel"]) for row in subset])),
                "mean_baseline_depth_abs_rel": float(np.nanmean([float(row["baseline_depth_abs_rel"]) for row in subset])),
                "mean_least20_depth_abs_rel": float(np.nanmean([float(row["least20_depth_abs_rel"]) for row in subset])),
            }
        )
    return [{key: finite_float(value) for key, value in row.items()} for row in rows]


def top_rows(rows: Sequence[dict[str, object]], field: str, top_k: int) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: float(row[field]) if row[field] != "" else float("-inf"),
        reverse=True,
    )[:top_k]


def main() -> int:
    args = parse_args()
    if args.frame_fusion_start_layer != -1:
        raise ValueError("token-region attribution currently expects --frame-fusion-start-layer -1")
    if args.frame_fusion_target_keep_percent <= 0.0:
        raise ValueError("--frame-fusion-target-keep-percent must be positive")

    device = torch.device(args.device)
    records = load_sequence_records(args)
    first_frame_token_indices = resolve_first_frame_token_indices(args.first_frame_token_indices, len(records))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.sequence}] loading model on {device}", flush=True)
    model = load_model(args.checkpoint, device, first_frame_token_indices)
    images = load_and_preprocess_images(
        [str(record.rgb_path) for record in records],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    ).to(device, non_blocking=True)

    print(f"[{args.sequence}] baseline forward", flush=True)
    model.aggregator.set_frame_fusion(mode="none")
    baseline_raw, image_hw, _, _, baseline_seconds = run_model(
        model,
        images,
        dataset=args.dataset,
        device=device,
    )

    print(f"[{args.sequence}] least20 forward representative={args.shared_representative}", flush=True)
    model.aggregator.set_frame_fusion(
        mode="pair-top-percent",
        start_layer=args.frame_fusion_start_layer,
        pair_percent=args.frame_fusion_pair_percent,
        pool_size=args.frame_fusion_pool_size,
        target_keep_policy="least-similar",
        target_keep_percent=args.frame_fusion_target_keep_percent,
        target_keep_seed=args.frame_fusion_target_keep_seed,
    )
    least_raw, least_hw, debug, captured, least_seconds = run_model(
        model,
        images,
        dataset=args.dataset,
        device=device,
        capture_plans=True,
        shared_representative=args.shared_representative,
    )
    if least_hw != image_hw:
        raise ValueError(f"baseline image size {image_hw} differs from least20 {least_hw}")
    if not captured:
        raise RuntimeError("did not capture any PairFusion plan")
    plan = captured[0]
    patch_grid_size = tuple(int(value) for value in plan["patch_grid_size"])
    patch_count = int(plan["patch_count"])
    pairs = list(plan["pairs"])
    keep_mask = np.asarray(plan["target_keep_mask"], dtype=bool)
    if keep_mask.shape != (len(pairs), patch_count):
        raise ValueError(f"captured keep mask shape {keep_mask.shape} does not match pairs/patch_count")

    ground_truth, valid_gt = read_ground_truth_depth(
        args.dataset,
        records,
        image_hw,
        args.min_depth,
        args.max_depth,
    )
    baseline_aligned, baseline_valid, baseline_scales = align_depth(
        dataset=args.dataset,
        predicted=baseline_raw,
        ground_truth=ground_truth,
        valid_gt=valid_gt,
        alignment=args.depth_alignment,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    least_aligned, least_valid, least_scales = align_depth(
        dataset=args.dataset,
        predicted=least_raw,
        ground_truth=ground_truth,
        valid_gt=valid_gt,
        alignment=args.depth_alignment,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    baseline_abs_rel, baseline_delta, valid_count = per_frame_depth_metrics(
        baseline_aligned,
        ground_truth,
        baseline_valid,
    )
    least_abs_rel, least_delta, _ = per_frame_depth_metrics(
        least_aligned,
        ground_truth,
        least_valid,
    )

    role_by_frame = [
        {
            "fusion_role": "reference" if frame == 0 else "unpaired",
            "paired_with": "",
            "pair_similarity": "",
            "pair_index": "",
        }
        for frame in range(len(records))
    ]
    retained_by_frame: list[np.ndarray | None] = [None for _ in records]
    shared_by_frame: list[np.ndarray | None] = [None for _ in records]
    for pair_index, pair in enumerate(pairs):
        source = int(pair["frame_a"])
        target = int(pair["frame_b"])
        retained = keep_mask[pair_index].copy()
        shared = ~retained
        for frame, role, other in ((source, "source", target), (target, "target", source)):
            role_by_frame[frame] = {
                "fusion_role": role,
                "paired_with": other,
                "pair_similarity": float(pair["similarity"]),
                "pair_index": pair_index,
            }
            retained_by_frame[frame] = retained
            shared_by_frame[frame] = shared

    height, width = image_hw
    common_valid = (
        valid_gt
        & baseline_valid
        & least_valid
        & np.isfinite(baseline_aligned)
        & np.isfinite(least_aligned)
        & (baseline_aligned > 0.0)
    )
    aligned_rel_diff = np.abs(least_aligned - baseline_aligned) / np.maximum(np.abs(baseline_aligned), 1e-6)
    raw_rel_diff = np.abs(least_raw - baseline_raw) / np.maximum(np.abs(baseline_raw), 1e-6)
    abs_depth_delta = np.abs(least_aligned - baseline_aligned)
    baseline_absrel_map = np.abs(baseline_aligned - ground_truth) / np.maximum(ground_truth, 1e-6)
    least_absrel_map = np.abs(least_aligned - ground_truth) / np.maximum(ground_truth, 1e-6)
    delta_absrel_map = least_absrel_map - baseline_absrel_map

    frame_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    for frame, record in enumerate(records):
        frame_mask = common_valid[frame]
        frame_abs_delta_sum = sum_or_zero(abs_depth_delta[frame][frame_mask])
        row: dict[str, object] = {
            "dataset": args.dataset,
            "sequence": args.sequence,
            "frame_position": frame,
            "frame_label": frame_label(args.dataset, record),
            "rgb_path": str(record.rgb_path),
            **role_by_frame[frame],
            "baseline_depth_abs_rel": baseline_abs_rel[frame],
            "least20_depth_abs_rel": least_abs_rel[frame],
            "delta_depth_abs_rel": least_abs_rel[frame] - baseline_abs_rel[frame],
            "baseline_depth_delta1_25_percent": baseline_delta[frame],
            "least20_depth_delta1_25_percent": least_delta[frame],
            "delta_depth_delta1_25_percent": least_delta[frame] - baseline_delta[frame],
            "depth_vs_baseline_abs_rel": mean_or_nan(aligned_rel_diff[frame][frame_mask]),
            "raw_depth_vs_baseline_abs_rel": mean_or_nan(raw_rel_diff[frame][frame_mask]),
            "mean_abs_depth_delta_m": mean_or_nan(abs_depth_delta[frame][frame_mask]),
            "baseline_depth_scale": baseline_scales[frame],
            "least20_depth_scale": least_scales[frame],
            "valid_depth_pixels": int(valid_count[frame]),
            "shared_depth_vs_baseline_abs_rel": "",
            "retained_depth_vs_baseline_abs_rel": "",
            "shared_delta_depth_abs_rel": "",
            "retained_delta_depth_abs_rel": "",
            "shared_abs_depth_delta_contribution": "",
            "retained_abs_depth_delta_contribution": "",
            "dominant_token_region_by_prediction_delta": "",
            "dominant_token_region_by_absrel_worsening": "",
        }
        if shared_by_frame[frame] is not None and retained_by_frame[frame] is not None:
            region_metrics: dict[str, dict[str, float]] = {}
            for region_name, patch_mask in (
                ("shared", shared_by_frame[frame]),
                ("retained", retained_by_frame[frame]),
            ):
                image_mask = patch_region_image_mask(patch_mask, patch_grid_size, image_hw)
                mask = frame_mask & image_mask
                valid_pixels = int(np.count_nonzero(mask))
                metrics = {
                    "valid_pixels": float(valid_pixels),
                    "pixel_fraction": float(valid_pixels / max(int(np.count_nonzero(frame_mask)), 1)),
                    "depth_vs_baseline_abs_rel": mean_or_nan(aligned_rel_diff[frame][mask]),
                    "delta_depth_abs_rel": mean_or_nan(delta_absrel_map[frame][mask]),
                    "mean_abs_depth_delta_m": mean_or_nan(abs_depth_delta[frame][mask]),
                    "deviation_contribution": (
                        sum_or_zero(abs_depth_delta[frame][mask]) / max(frame_abs_delta_sum, 1e-12)
                    ),
                }
                region_metrics[region_name] = metrics
                region_rows.append(
                    {
                        "dataset": args.dataset,
                        "sequence": args.sequence,
                        "frame_position": frame,
                        "frame_label": row["frame_label"],
                        **role_by_frame[frame],
                        "token_region": region_name,
                        "patch_tokens": int(np.count_nonzero(patch_mask)),
                        **metrics,
                    }
                )
            shared_metrics = region_metrics["shared"]
            retained_metrics = region_metrics["retained"]
            row.update(
                {
                    "shared_depth_vs_baseline_abs_rel": shared_metrics["depth_vs_baseline_abs_rel"],
                    "retained_depth_vs_baseline_abs_rel": retained_metrics["depth_vs_baseline_abs_rel"],
                    "shared_delta_depth_abs_rel": shared_metrics["delta_depth_abs_rel"],
                    "retained_delta_depth_abs_rel": retained_metrics["delta_depth_abs_rel"],
                    "shared_abs_depth_delta_contribution": shared_metrics["deviation_contribution"],
                    "retained_abs_depth_delta_contribution": retained_metrics["deviation_contribution"],
                    "dominant_token_region_by_prediction_delta": (
                        "shared"
                        if shared_metrics["depth_vs_baseline_abs_rel"] >= retained_metrics["depth_vs_baseline_abs_rel"]
                        else "retained"
                    ),
                    "dominant_token_region_by_absrel_worsening": (
                        "shared"
                        if shared_metrics["delta_depth_abs_rel"] >= retained_metrics["delta_depth_abs_rel"]
                        else "retained"
                    ),
                }
            )
        frame_rows.append({key: finite_float(value) for key, value in row.items()})

    sequence_dir = args.output_dir / slugify(args.sequence)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    write_csv(sequence_dir / "frame_depth_deviation.csv", frame_rows)
    write_csv(sequence_dir / "token_region_depth_deviation.csv", [{key: finite_float(value) for key, value in row.items()} for row in region_rows])
    region_summary_rows = group_region_summary(region_rows)
    role_summary_rows = role_summary(frame_rows)
    write_csv(sequence_dir / "token_region_summary.csv", region_summary_rows)
    write_csv(sequence_dir / "role_summary.csv", role_summary_rows)
    top_prediction = top_rows(frame_rows, "depth_vs_baseline_abs_rel", args.top_k)
    top_worsening = top_rows(frame_rows, "delta_depth_abs_rel", args.top_k)
    write_csv(sequence_dir / "top_frames_by_prediction_delta.csv", top_prediction)
    write_csv(sequence_dir / "top_frames_by_absrel_worsening.csv", top_worsening)

    top_prediction_roles = {
        role: sum(1 for row in top_prediction if row["fusion_role"] == role)
        for role in ("reference", "source", "target", "unpaired")
    }
    top_worsening_roles = {
        role: sum(1 for row in top_worsening if row["fusion_role"] == role)
        for role in ("reference", "source", "target", "unpaired")
    }
    top_prediction_regions = {
        region: sum(1 for row in top_prediction if row["dominant_token_region_by_prediction_delta"] == region)
        for region in ("shared", "retained")
    }
    top_worsening_regions = {
        region: sum(1 for row in top_worsening if row["dominant_token_region_by_absrel_worsening"] == region)
        for region in ("shared", "retained")
    }
    paired_rows = [row for row in frame_rows if row["fusion_role"] in {"source", "target"}]
    summary = {
        "dataset": args.dataset,
        "sequence": args.sequence,
        "num_frames": len(records),
        "checkpoint": str(args.checkpoint),
        "image_hw": list(image_hw),
        "patch_grid_size": list(patch_grid_size),
        "patch_count": patch_count,
        "baseline_seconds": baseline_seconds,
        "least20_seconds": least_seconds,
        "least20_config": {
            "mode": "pair-top-percent",
            "start_layer": args.frame_fusion_start_layer,
            "pair_percent": args.frame_fusion_pair_percent,
            "pool_size": args.frame_fusion_pool_size,
            "target_keep_policy": "least-similar",
            "target_keep_percent": args.frame_fusion_target_keep_percent,
            "target_keep_seed": args.frame_fusion_target_keep_seed,
            "shared_representative": args.shared_representative,
        },
        "frame_fusion_debug": debug,
        "captured_plan": {
            "source_layer": plan["source_layer"],
            "pairs": pairs,
            "target_keep_patch_tokens_per_pair": int(np.count_nonzero(keep_mask[0])) if len(pairs) else 0,
            "shared_patch_tokens_per_pair": int(patch_count - np.count_nonzero(keep_mask[0])) if len(pairs) else 0,
        },
        "overall": {
            "baseline_abs_rel": float(np.nansum(baseline_abs_rel * valid_count) / max(int(np.sum(valid_count)), 1)),
            "least20_abs_rel": float(np.nansum(least_abs_rel * valid_count) / max(int(np.sum(valid_count)), 1)),
            "delta_abs_rel": float(
                np.nansum((least_abs_rel - baseline_abs_rel) * valid_count) / max(int(np.sum(valid_count)), 1)
            ),
            "mean_frame_depth_vs_baseline_abs_rel": float(np.nanmean([float(row["depth_vs_baseline_abs_rel"]) for row in frame_rows])),
            "mean_paired_frame_depth_vs_baseline_abs_rel": float(
                np.nanmean([float(row["depth_vs_baseline_abs_rel"]) for row in paired_rows])
            ) if paired_rows else float("nan"),
        },
        "role_summary": role_summary_rows,
        "token_region_summary": region_summary_rows,
        "top_prediction_roles": top_prediction_roles,
        "top_worsening_roles": top_worsening_roles,
        "top_prediction_dominant_token_regions": top_prediction_regions,
        "top_worsening_dominant_token_regions": top_worsening_regions,
        "top_frames_by_prediction_delta": top_prediction,
        "top_frames_by_absrel_worsening": top_worsening,
        "outputs": {
            "frame_depth_deviation_csv": str(sequence_dir / "frame_depth_deviation.csv"),
            "token_region_depth_deviation_csv": str(sequence_dir / "token_region_depth_deviation.csv"),
            "token_region_summary_csv": str(sequence_dir / "token_region_summary.csv"),
            "role_summary_csv": str(sequence_dir / "role_summary.csv"),
        },
    }
    with (sequence_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    if args.save_depth_npz:
        np.savez_compressed(
            sequence_dir / "depth_predictions_and_deltas.npz",
            baseline_raw=baseline_raw,
            least20_raw=least_raw,
            baseline_aligned=baseline_aligned,
            least20_aligned=least_aligned,
            ground_truth=ground_truth,
            common_valid=common_valid,
        )
    print(f"[{args.sequence}] wrote {sequence_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
