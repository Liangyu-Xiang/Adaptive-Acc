#!/usr/bin/env python3
"""Evaluate final-layer camera-token swaps for selected high-similarity pairs."""

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
    load_frame_records,
    load_model,
    official_auc,
    to_homogeneous_w2c,
)
from vggt_omega.models.aggregator import slice_expand_and_flatten  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from vggt_omega.utils.reference_frame import resolve_first_frame_token_indices  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_SAMPLED_FRAMES = (
    REPO_ROOT
    / "outputs"
    / "frame-fusion-smoke__tum__300frames__K80_M5_pre0__20260730"
    / "sampled_frames.json"
)
DEFAULT_PAIRS_CSV = (
    REPO_ROOT
    / "outputs"
    / "register_token_attention__tum_halfsphere_300f__layer23_gt0p76_top12_res384"
    / "selected_pairs.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sampled-frames", type=Path, default=DEFAULT_SAMPLED_FRAMES)
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selected_camera_token_swap_top12"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--patch-embed-chunk-size",
        type=int,
        default=8,
        help="Run DINO patch embedding over this many frames at a time. Use 0 to disable chunking.",
    )
    parser.add_argument("--no-amp", action="store_true")
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


def read_pairs_csv(path: Path, num_frames: int) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_i", "frame_j"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            i = int(row["frame_i"])
            j = int(row["frame_j"])
            if not (0 <= i < num_frames and 0 <= j < num_frames):
                raise ValueError(f"Pair ({i}, {j}) is outside 0..{num_frames - 1}")
            if i == j:
                raise ValueError(f"Self pair is invalid: ({i}, {j})")
            similarity = float(row.get("similarity") or "nan")
            pairs.append((i, j, similarity))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs


def run_aggregator(
    model,
    images: torch.Tensor,
    *,
    use_amp: bool,
    patch_embed_chunk_size: int,
) -> tuple[list[torch.Tensor | None], int]:
    images_batched = images.unsqueeze(0)
    if use_amp and images.device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        context = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        context = nullcontext()
    with torch.inference_mode(), context:
        if patch_embed_chunk_size > 0:
            patch_h = images.shape[-2] // model.aggregator.patch_size
            patch_w = images.shape[-1] // model.aggregator.patch_size
            rope_sin, rope_cos = model.aggregator.rope_embed(H=patch_h, W=patch_w)
            frame_rope = (
                rope_sin.to(device=images.device, dtype=torch.float32),
                rope_cos.to(device=images.device, dtype=torch.float32),
            )
            return run_aggregator_with_chunked_patch_embed(
                model,
                images_batched,
                frame_rope=frame_rope,
                patch_embed_chunk_size=patch_embed_chunk_size,
            )
        return model.aggregator(images_batched)


def run_aggregator_with_chunked_patch_embed(
    model,
    images: torch.Tensor,
    *,
    frame_rope: tuple[torch.Tensor, torch.Tensor],
    patch_embed_chunk_size: int,
) -> tuple[list[torch.Tensor | None], int]:
    aggregator = model.aggregator
    batch_size, num_frames, num_channels, height, width = images.shape
    if batch_size != 1:
        raise ValueError("Chunked patch embedding currently supports batch_size=1")
    if num_channels != 3:
        raise ValueError(f"Expected 3 input channels, got {num_channels}")
    if aggregator.frame_fusion_mode != "none":
        raise ValueError("This experiment expects frame_fusion_mode='none'")
    if getattr(aggregator, "layer_token_swap_kind", "none") != "none":
        raise ValueError("This experiment expects layer token swap to be disabled")

    normalized_images = (images - aggregator._resnet_mean) / aggregator._resnet_std
    flat_images = normalized_images.view(batch_size * num_frames, num_channels, height, width)
    patch_chunks: list[torch.Tensor] = []
    for start in range(0, flat_images.shape[0], patch_embed_chunk_size):
        end = min(flat_images.shape[0], start + patch_embed_chunk_size)
        patch_tokens = aggregator.patch_embed(flat_images[start:end])
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        patch_chunks.append(patch_tokens)
    patch_tokens = torch.cat(patch_chunks, dim=0)

    first_frame_token_indices = resolve_first_frame_token_indices(
        aggregator.first_frame_token_indices,
        num_frames,
    )
    camera_token = slice_expand_and_flatten(
        aggregator.camera_token,
        batch_size,
        num_frames,
        first_frame_token_indices=first_frame_token_indices,
    )
    register_token = slice_expand_and_flatten(
        aggregator.register_token,
        batch_size,
        num_frames,
        first_frame_token_indices=first_frame_token_indices,
    )
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
    _, num_tokens, embed_dim = tokens.shape
    patch_grid_size = (height // aggregator.patch_size, width // aggregator.patch_size)

    outputs: list[torch.Tensor | None] = []
    aggregator._register_patch_selection.clear()
    aggregator._adaptive_intra_scores.clear()
    aggregator._progressive_stage_states.clear()
    aggregator.last_progressive_attention_stats.clear()
    aggregator.last_progressive_sample_indices.clear()
    aggregator.last_adaptive_pair_scope_debug.clear()
    aggregator.last_frame_fusion_debug.clear()

    tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
    for block_idx in range(aggregator.depth):
        tokens, frame_tokens = aggregator._run_frame_block(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
            block_idx,
            frame_rope,
        )
        tokens = aggregator._run_inter_frame_attention_block(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
            block_idx,
            aggregator.inter_frame_attention_types[block_idx],
            patch_grid_size,
            frame_fusion_pair_plans=None,
        )
        if block_idx in aggregator.cached_layer_indices:
            outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
        else:
            outputs.append(None)
    return outputs, aggregator.patch_token_start


def run_camera_head(
    model,
    aggregated_tokens: list[torch.Tensor | None],
    patch_token_start: int,
    image_hw: tuple[int, int],
) -> np.ndarray:
    if model.camera_head is None:
        raise RuntimeError("Model has no camera head")
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        pose_enc = model.camera_head(aggregated_tokens, patch_token_start=patch_token_start)
        extrinsics, _ = encoding_to_camera(pose_enc, image_hw, build_intrinsics=False)
    return to_homogeneous_w2c(extrinsics[0])


def swap_final_camera_tokens(
    aggregated_tokens: list[torch.Tensor | None],
    pairs: Sequence[tuple[int, int, float]],
) -> list[torch.Tensor | None]:
    final = aggregated_tokens[-1]
    if final is None:
        raise ValueError("Final aggregated tokens are missing")
    modified = list(aggregated_tokens)
    swapped = final.clone()
    first_indices = torch.tensor([i for i, _, _ in pairs], device=swapped.device, dtype=torch.long)
    second_indices = torch.tensor([j for _, j, _ in pairs], device=swapped.device, dtype=torch.long)
    first_values = swapped[:, first_indices, 0:1].clone()
    second_values = swapped[:, second_indices, 0:1].clone()
    swapped[:, first_indices, 0:1] = second_values
    swapped[:, second_indices, 0:1] = first_values
    modified[-1] = swapped
    return modified


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
        rotation_errors.append(rotation_angle_deg(gt_relative[:3, :3], pred_relative[:3, :3]))
        translation_errors.append(translation_direction_angle_deg(gt_relative[:3, 3], pred_relative[:3, 3]))
    return np.asarray(rotation_errors), np.asarray(translation_errors)


def rotation_angle_deg(reference: np.ndarray, current: np.ndarray) -> float:
    rotation_delta = reference.T @ current
    cosine = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def translation_direction_angle_deg(reference: np.ndarray, current: np.ndarray) -> float:
    denominator = np.linalg.norm(reference) * np.linalg.norm(current)
    if denominator <= 1e-15:
        return 1e6
    cosine = np.clip(abs(float(np.dot(reference, current))) / denominator, 0.0, 1.0)
    return math.degrees(math.acos(cosine))


def predicted_relative_pose_changes(
    baseline_w2c: np.ndarray,
    swapped_w2c: np.ndarray,
    pairs: Sequence[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    rotation_changes: list[float] = []
    translation_changes: list[float] = []
    for i, j, _ in pairs:
        baseline_relative = baseline_w2c[i] @ np.linalg.inv(baseline_w2c[j])
        swapped_relative = swapped_w2c[i] @ np.linalg.inv(swapped_w2c[j])
        rotation_changes.append(rotation_angle_deg(baseline_relative[:3, :3], swapped_relative[:3, :3]))
        translation_changes.append(
            translation_direction_angle_deg(baseline_relative[:3, 3], swapped_relative[:3, 3])
        )
    return np.asarray(rotation_changes), np.asarray(translation_changes)


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
        "max_error_min_deg": float(max_errors.min()),
        "max_error_max_deg": float(max_errors.max()),
    }


def metric_delta(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        f"delta_{key}": float(current[key] - baseline[key])
        for key in current
        if key in baseline and isinstance(current[key], (int, float))
    }


def final_camera_token_pair_cosines(
    aggregated_tokens: list[torch.Tensor | None],
    pairs: Sequence[tuple[int, int, float]],
) -> np.ndarray:
    final = aggregated_tokens[-1]
    if final is None:
        raise ValueError("Final aggregated tokens are missing")
    camera_tokens = final[0, :, 0].detach().float()
    values: list[float] = []
    for i, j, _ in pairs:
        a = camera_tokens[i]
        b = camera_tokens[j]
        denominator = torch.linalg.norm(a) * torch.linalg.norm(b)
        values.append(float((torch.dot(a, b) / denominator).cpu()))
    return np.asarray(values, dtype=np.float64)


def write_pair_errors_csv(
    path: Path,
    pairs: Sequence[tuple[int, int, float]],
    camera_cosines: np.ndarray,
    baseline_rotation: np.ndarray,
    baseline_translation: np.ndarray,
    swapped_rotation: np.ndarray,
    swapped_translation: np.ndarray,
    prediction_rotation_change: np.ndarray,
    prediction_translation_change: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_index",
                "frame_i",
                "frame_j",
                "layer23_patch_similarity",
                "final_camera_token_cosine",
                "baseline_rotation_error_deg",
                "baseline_translation_error_deg",
                "baseline_max_error_deg",
                "swapped_rotation_error_deg",
                "swapped_translation_error_deg",
                "swapped_max_error_deg",
                "delta_rotation_error_deg",
                "delta_translation_error_deg",
                "delta_max_error_deg",
                "prediction_relative_rotation_change_deg",
                "prediction_relative_translation_change_deg",
            ],
        )
        writer.writeheader()
        for pair_index, (i, j, similarity) in enumerate(pairs):
            baseline_max = max(baseline_rotation[pair_index], baseline_translation[pair_index])
            swapped_max = max(swapped_rotation[pair_index], swapped_translation[pair_index])
            writer.writerow(
                {
                    "pair_index": pair_index,
                    "frame_i": i,
                    "frame_j": j,
                    "layer23_patch_similarity": f"{similarity:.8f}",
                    "final_camera_token_cosine": f"{camera_cosines[pair_index]:.8f}",
                    "baseline_rotation_error_deg": f"{baseline_rotation[pair_index]:.8f}",
                    "baseline_translation_error_deg": f"{baseline_translation[pair_index]:.8f}",
                    "baseline_max_error_deg": f"{baseline_max:.8f}",
                    "swapped_rotation_error_deg": f"{swapped_rotation[pair_index]:.8f}",
                    "swapped_translation_error_deg": f"{swapped_translation[pair_index]:.8f}",
                    "swapped_max_error_deg": f"{swapped_max:.8f}",
                    "delta_rotation_error_deg": f"{swapped_rotation[pair_index] - baseline_rotation[pair_index]:.8f}",
                    "delta_translation_error_deg": f"{swapped_translation[pair_index] - baseline_translation[pair_index]:.8f}",
                    "delta_max_error_deg": f"{swapped_max - baseline_max:.8f}",
                    "prediction_relative_rotation_change_deg": f"{prediction_rotation_change[pair_index]:.8f}",
                    "prediction_relative_translation_change_deg": f"{prediction_translation_change[pair_index]:.8f}",
                }
            )


def main() -> int:
    args = parse_args()
    if args.patch_embed_chunk_size < 0:
        raise ValueError("--patch-embed-chunk-size must be non-negative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires CUDA")

    sequence_name, records = load_manifest_records(
        args.sampled_frames,
        args.sequence,
        args.association_tolerance,
    )
    pairs = read_pairs_csv(args.pairs_csv, len(records))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device, merge_ratio=0.0)
    images = load_and_preprocess_images(
        [str(record.rgb_path) for record in records],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    ).to(device, non_blocking=True)
    image_hw = tuple(int(value) for value in images.shape[-2:])

    with torch.inference_mode():
        aggregated_tokens, patch_token_start = run_aggregator(
            model,
            images,
            use_amp=not args.no_amp,
            patch_embed_chunk_size=args.patch_embed_chunk_size,
        )
        camera_cosines = final_camera_token_pair_cosines(aggregated_tokens, pairs)
        baseline_w2c = run_camera_head(model, aggregated_tokens, patch_token_start, image_hw)
        swapped_tokens = swap_final_camera_tokens(aggregated_tokens, pairs)
        swapped_w2c = run_camera_head(model, swapped_tokens, patch_token_start, image_hw)

    gt_c2w = np.stack([record.c2w for record in records])
    gt_w2c = np.linalg.inv(gt_c2w)
    baseline_rotation, baseline_translation = pair_pose_errors(baseline_w2c, gt_w2c, pairs)
    swapped_rotation, swapped_translation = pair_pose_errors(swapped_w2c, gt_w2c, pairs)
    prediction_rotation_change, prediction_translation_change = predicted_relative_pose_changes(
        baseline_w2c,
        swapped_w2c,
        pairs,
    )

    baseline_summary = pose_summary(baseline_rotation, baseline_translation)
    swapped_summary = pose_summary(swapped_rotation, swapped_translation)
    change_summary = pose_summary(prediction_rotation_change, prediction_translation_change)
    result: dict[str, object] = {
        "sequence": sequence_name,
        "num_input_frames": len(records),
        "pairs_csv": str(args.pairs_csv),
        "selected_pairs": len(pairs),
        "selected_frame_indices": sorted({index for pair in pairs for index in pair[:2]}),
        "checkpoint": str(args.checkpoint),
        "model_variant": "VGGTOmega dense original, merge_ratio=0.0, frame_fusion_mode=none",
        "image_shape_hw": list(image_hw),
        "patch_token_start": int(patch_token_start),
        "swap_scope": (
            "Only final cached aggregator output camera token offset 0 is swapped inside each selected pair; "
            "register tokens and patch tokens stay unchanged. CameraHead is run on all 300 frames."
        ),
        "final_camera_token_cosine": {
            "mean": float(camera_cosines.mean()),
            "min": float(camera_cosines.min()),
            "max": float(camera_cosines.max()),
        },
        "baseline_pairs_only": baseline_summary,
        "camera_token_swap_pairs_only": swapped_summary,
        "delta_vs_baseline": metric_delta(swapped_summary, baseline_summary),
        "prediction_change_after_swap": {
            "relative_pose_rotation_change_mean_deg": float(prediction_rotation_change.mean()),
            "relative_pose_rotation_change_max_deg": float(prediction_rotation_change.max()),
            "relative_pose_translation_change_mean_deg": float(prediction_translation_change.mean()),
            "relative_pose_translation_change_max_deg": float(prediction_translation_change.max()),
            "relative_pose_max_change_mean_deg": float(
                np.maximum(prediction_rotation_change, prediction_translation_change).mean()
            ),
            "relative_pose_max_change_max_deg": float(
                np.maximum(prediction_rotation_change, prediction_translation_change).max()
            ),
        },
    }
    write_pair_errors_csv(
        args.output_dir / "pair_camera_token_swap_errors.csv",
        pairs,
        camera_cosines,
        baseline_rotation,
        baseline_translation,
        swapped_rotation,
        swapped_translation,
        prediction_rotation_change,
        prediction_translation_change,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
