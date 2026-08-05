#!/usr/bin/env python3
"""Evaluate last-layer token swaps between high-similarity VGGT-Omega frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_tum_dynamics_paper import (  # noqa: E402
    FrameRecord,
    depth_sums,
    official_auc,
    load_frame_records,
    load_model,
    pairwise_pose_errors,
    to_homogeneous_w2c,
)
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_SAMPLED_FRAMES = (
    REPO_ROOT
    / "outputs"
    / "frame-fusion-smoke__tum__300frames__K80_M5_pre0__20260730"
    / "sampled_frames.json"
)
DEFAULT_SIMILARITY_NPZ = (
    REPO_ROOT
    / "outputs"
    / "frame_similarity_matrices__tum_halfsphere_300f__layers_2_6_10_16_23"
    / "frame_similarity_matrices.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sampled-frames", type=Path, default=DEFAULT_SAMPLED_FRAMES)
    parser.add_argument("--similarity-npz", type=Path, default=DEFAULT_SIMILARITY_NPZ)
    parser.add_argument("--similarity-stage", default="layer_23")
    parser.add_argument("--threshold", type=float, default=0.76)
    parser.add_argument(
        "--pair-selection",
        choices=("greedy-matching",),
        default="greedy-matching",
        help="Select non-overlapping frame pairs in descending similarity order.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Sequence key inside sampled_frames.json. Defaults to the first key.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/last_layer_token_swap_gt_0p76"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--depth-alignment",
        choices=("per-frame-median", "per-sequence-median"),
        default="per-frame-median",
    )
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast for aggregator collection.")
    return parser.parse_args()


def load_manifest_records(sampled_frames: Path, sequence: str | None, tolerance: float) -> tuple[str, list[FrameRecord]]:
    manifest = json.loads(sampled_frames.read_text(encoding="utf-8"))
    if sequence is None:
        sequence = next(iter(manifest))
    if sequence not in manifest:
        raise ValueError(f"Sequence {sequence!r} is not in {sampled_frames}")
    rgb_paths = [Path(path) for path in manifest[sequence]["rgb_paths"]]
    if not rgb_paths:
        raise ValueError(f"No rgb_paths found for {sequence!r}")
    sequence_dir = rgb_paths[0].parents[1]
    all_records = load_frame_records(sequence_dir, tolerance)
    by_path = {record.rgb_path.resolve(): record for record in all_records}
    records: list[FrameRecord] = []
    for path in rgb_paths:
        try:
            records.append(by_path[path.resolve()])
        except KeyError as exc:
            raise ValueError(f"Manifest frame is not associated with GT/depth: {path}") from exc
    return sequence, records


def select_greedy_matching(similarity: np.ndarray, threshold: float) -> tuple[list[tuple[int, int, float]], int]:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"Expected a square similarity matrix, got {similarity.shape}")
    edges: list[tuple[float, int, int]] = []
    for i, j in zip(*np.triu_indices_from(similarity, k=1)):
        value = float(similarity[i, j])
        if value > threshold:
            edges.append((value, int(i), int(j)))
    edges.sort(reverse=True)
    used: set[int] = set()
    selected: list[tuple[int, int, float]] = []
    for value, i, j in edges:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        selected.append((i, j, value))
    return selected, len(edges)


def run_aggregator(model, images: torch.Tensor, use_amp: bool) -> tuple[list[torch.Tensor | None], int]:
    images_batched = images.unsqueeze(0)
    if use_amp and images.device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        context = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        context = nullcontext()
    with context:
        return model.aggregator(images_batched)


def swap_final_tokens(
    aggregated_tokens: list[torch.Tensor | None],
    pairs: Sequence[tuple[int, int, float]],
    patch_token_start: int,
    token_kind: str,
) -> list[torch.Tensor | None]:
    if token_kind not in {"patch", "special"}:
        raise ValueError(token_kind)
    final = aggregated_tokens[-1]
    if final is None:
        raise ValueError("Final aggregated tokens are missing")
    modified = list(aggregated_tokens)
    swapped = final.clone()
    first_indices = torch.tensor([i for i, _, _ in pairs], device=swapped.device, dtype=torch.long)
    second_indices = torch.tensor([j for _, j, _ in pairs], device=swapped.device, dtype=torch.long)
    if token_kind == "patch":
        token_slice = slice(patch_token_start, None)
    else:
        token_slice = slice(0, patch_token_start)
    first_values = swapped[:, first_indices, token_slice].clone()
    second_values = swapped[:, second_indices, token_slice].clone()
    swapped[:, first_indices, token_slice] = second_values
    swapped[:, second_indices, token_slice] = first_values
    modified[-1] = swapped
    return modified


def slice_aggregated(
    aggregated_tokens: list[torch.Tensor | None],
    frame_indices: Sequence[int],
) -> list[torch.Tensor | None]:
    index = torch.tensor(frame_indices, device=next(t for t in aggregated_tokens if t is not None).device)
    return [
        None if tokens is None else tokens.index_select(1, index)
        for tokens in aggregated_tokens
    ]


def run_camera_head(model, aggregated_tokens: list[torch.Tensor | None], patch_token_start: int, image_hw: tuple[int, int]) -> np.ndarray:
    if model.camera_head is None:
        raise RuntimeError("Model has no camera head")
    with torch.autocast(device_type="cuda", enabled=False):
        pose_enc = model.camera_head(aggregated_tokens, patch_token_start=patch_token_start)
        extrinsics, _ = encoding_to_camera(pose_enc, image_hw, build_intrinsics=False)
    return to_homogeneous_w2c(extrinsics[0])


def run_depth_head(
    model,
    aggregated_tokens: list[torch.Tensor | None],
    images_batched: torch.Tensor,
    patch_token_start: int,
) -> np.ndarray:
    if model.dense_head is None:
        raise RuntimeError("Model has no dense head")
    with torch.autocast(device_type="cuda", enabled=False):
        depth, _ = model.dense_head(
            aggregated_tokens,
            images=images_batched,
            patch_token_start=patch_token_start,
        )
    return depth[0, ..., 0].detach().float().cpu().numpy()


def pair_pose_errors(
    pred_w2c: np.ndarray,
    gt_w2c: np.ndarray,
    pairs: Sequence[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for i, j, _ in pairs:
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
            cosine_t = np.clip(abs(float(np.dot(gt_t, pred_t))) / denominator, 0.0, 1.0)
            translation_errors.append(math.degrees(math.acos(cosine_t)))
    return np.asarray(rotation_errors), np.asarray(translation_errors)


def pose_summary(rotation_errors: np.ndarray, translation_errors: np.ndarray) -> dict[str, float]:
    max_errors = np.maximum(rotation_errors, translation_errors)
    return {
        "num_pose_pairs": int(len(max_errors)),
        "auc_3_percent": 100.0 * official_auc(rotation_errors, translation_errors, 3),
        "auc_30_percent": 100.0 * official_auc(rotation_errors, translation_errors, 30),
        "rotation_error_mean_deg": float(rotation_errors.mean()),
        "translation_error_mean_deg": float(translation_errors.mean()),
        "max_error_mean_deg": float(max_errors.mean()),
        "max_error_median_deg": float(np.median(max_errors)),
    }


def depth_summary(
    predicted_depth: np.ndarray,
    records: Sequence[FrameRecord],
    alignment: str,
    max_depth: float,
) -> dict[str, float]:
    abs_rel_sum, delta_count, valid_count, _ = depth_sums(
        predicted_depth,
        records,
        alignment,
        max_depth,
    )
    return {
        "valid_depth_pixels": int(valid_count),
        "abs_rel": float(abs_rel_sum / valid_count),
        "delta_1_25_percent": float(100.0 * delta_count / valid_count),
    }


def metric_delta(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        f"delta_{key}": float(current[key] - baseline[key])
        for key in current
        if key in baseline and isinstance(current[key], (int, float))
    }


def write_pairs_csv(path: Path, pairs: Sequence[tuple[int, int, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_index", "frame_i", "frame_j", "similarity"])
        writer.writeheader()
        for index, (i, j, similarity) in enumerate(pairs):
            writer.writerow(
                {
                    "pair_index": index,
                    "frame_i": i,
                    "frame_j": j,
                    "similarity": f"{similarity:.8f}",
                }
            )


def main() -> int:
    args = parse_args()
    if args.threshold < -1.0 or args.threshold > 1.0:
        raise ValueError("--threshold must be in [-1, 1]")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires CUDA")

    sequence_name, records = load_manifest_records(
        args.sampled_frames,
        args.sequence,
        args.association_tolerance,
    )
    similarity = np.load(args.similarity_npz)[args.similarity_stage]
    if similarity.shape != (len(records), len(records)):
        raise ValueError(
            f"Similarity matrix shape {similarity.shape} does not match {len(records)} frames"
        )
    selected_pairs, all_high_similarity_edges = select_greedy_matching(similarity, args.threshold)
    if not selected_pairs:
        raise ValueError(f"No non-overlapping frame pairs found above threshold {args.threshold}")
    selected_frames = sorted({index for pair in selected_pairs for index in pair[:2]})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_pairs_csv(args.output_dir / "selected_pairs.csv", selected_pairs)

    model = load_model(args.checkpoint, device, merge_ratio=0.0)
    images = load_and_preprocess_images(
        [str(record.rgb_path) for record in records],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    ).to(device, non_blocking=True)
    images_batched = images.unsqueeze(0)
    image_hw = tuple(int(value) for value in images.shape[-2:])
    with torch.inference_mode():
        aggregated_tokens, patch_token_start = run_aggregator(model, images, use_amp=not args.no_amp)
        baseline_w2c = run_camera_head(model, aggregated_tokens, patch_token_start, image_hw)
        special_swapped = swap_final_tokens(
            aggregated_tokens,
            selected_pairs,
            patch_token_start,
            token_kind="special",
        )
        special_w2c = run_camera_head(model, special_swapped, patch_token_start, image_hw)

        selected_images = images_batched[:, selected_frames].contiguous()
        selected_records = [records[index] for index in selected_frames]
        selected_baseline_tokens = slice_aggregated(aggregated_tokens, selected_frames)
        baseline_depth = run_depth_head(model, selected_baseline_tokens, selected_images, patch_token_start)

        patch_swapped = swap_final_tokens(
            aggregated_tokens,
            selected_pairs,
            patch_token_start,
            token_kind="patch",
        )
        selected_patch_tokens = slice_aggregated(patch_swapped, selected_frames)
        patch_depth = run_depth_head(model, selected_patch_tokens, selected_images, patch_token_start)

    gt_c2w = np.stack([record.c2w for record in records])
    gt_w2c = np.linalg.inv(gt_c2w)
    selected_gt_w2c = gt_w2c[selected_frames]

    baseline_selected_rot, baseline_selected_trans = pairwise_pose_errors(
        baseline_w2c[selected_frames],
        selected_gt_w2c,
    )
    special_selected_rot, special_selected_trans = pairwise_pose_errors(
        special_w2c[selected_frames],
        selected_gt_w2c,
    )
    baseline_pair_rot, baseline_pair_trans = pair_pose_errors(baseline_w2c, gt_w2c, selected_pairs)
    special_pair_rot, special_pair_trans = pair_pose_errors(special_w2c, gt_w2c, selected_pairs)

    baseline_depth_summary = depth_summary(
        baseline_depth,
        selected_records,
        args.depth_alignment,
        args.max_depth,
    )
    patch_depth_summary = depth_summary(
        patch_depth,
        selected_records,
        args.depth_alignment,
        args.max_depth,
    )

    baseline_pose_selected = pose_summary(baseline_selected_rot, baseline_selected_trans)
    special_pose_selected = pose_summary(special_selected_rot, special_selected_trans)
    baseline_pose_pairs = pose_summary(baseline_pair_rot, baseline_pair_trans)
    special_pose_pairs = pose_summary(special_pair_rot, special_pair_trans)

    result: dict[str, object] = {
        "sequence": sequence_name,
        "num_input_frames": len(records),
        "similarity_npz": str(args.similarity_npz),
        "similarity_stage": args.similarity_stage,
        "threshold": args.threshold,
        "all_high_similarity_edges": all_high_similarity_edges,
        "pair_selection": args.pair_selection,
        "selected_pairs": len(selected_pairs),
        "selected_frames": len(selected_frames),
        "selected_frame_indices": selected_frames,
        "patch_token_start": int(patch_token_start),
        "image_shape_hw": list(image_hw),
        "model_variant": "VGGTOmega dense original, merge_ratio=0.0",
        "swap_scope": (
            "final cached aggregator layer only; greedy non-overlapping pairs are swapped simultaneously"
        ),
        "baseline": {
            "pose_selected_frames_all_pairs": baseline_pose_selected,
            "pose_selected_swap_pairs_only": baseline_pose_pairs,
            "depth_selected_frames": baseline_depth_summary,
        },
        "special_token_swap": {
            "pose_selected_frames_all_pairs": special_pose_selected,
            "pose_selected_swap_pairs_only": special_pose_pairs,
            "delta_vs_baseline": {
                "pose_selected_frames_all_pairs": metric_delta(special_pose_selected, baseline_pose_selected),
                "pose_selected_swap_pairs_only": metric_delta(special_pose_pairs, baseline_pose_pairs),
            },
            "note": "Special-token swap changes camera/register tokens used by CameraHead; depth head ignores special tokens.",
        },
        "patch_token_swap": {
            "depth_selected_frames": patch_depth_summary,
            "delta_vs_baseline": {
                "depth_selected_frames": metric_delta(patch_depth_summary, baseline_depth_summary),
            },
            "note": "Patch-token swap changes final-layer patch tokens used by DenseHead; CameraHead ignores patch tokens.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
