#!/usr/bin/env python3
"""Evaluate PairFusion with a dense middle-layer gap and repartition."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval_7scenes_paper as seven_eval  # noqa: E402
import scripts.eval_tum_dynamics_paper as tum_eval  # noqa: E402
import vggt_omega.models.aggregator as aggregator_mod  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.models.aggregator import FrameFusionPair, slice_expand_and_flatten  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
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
    parser.add_argument(
        "--experiment-mode",
        choices=("baseline", "continuous", "dense-gap", "tail-dense"),
        default="dense-gap",
        help=(
            "baseline disables frame fusion; continuous keeps PairFusion active "
            "throughout; dense-gap disables PairFusion for the requested middle layers; "
            "tail-dense disables PairFusion only for the final inter-frame/global block."
        ),
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument(
        "--depth-alignment",
        choices=("per-frame-median", "per-sequence-median"),
        default="per-frame-median",
    )
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument("--first-frame-token-indices", default="0")
    parser.add_argument("--frame-fusion-pair-percent", type=float, default=25.0)
    parser.add_argument("--frame-fusion-pool-size", type=int, default=2)
    parser.add_argument(
        "--frame-fusion-target-keep-policy",
        choices=("least-similar", "similarity-threshold", "random-grid", "none"),
        default="least-similar",
    )
    parser.add_argument("--frame-fusion-target-keep-grid-size", type=int, default=4)
    parser.add_argument("--frame-fusion-target-keep-percent", type=float, default=20.0)
    parser.add_argument("--frame-fusion-target-keep-threshold", type=float, default=0.0)
    parser.add_argument("--frame-fusion-target-keep-seed", type=int, default=33)
    parser.add_argument(
        "--pair-selection-method",
        choices=("model-default", "upper-tri"),
        default="model-default",
    )
    parser.add_argument("--early-fusion-end-layer", type=int, default=9)
    parser.add_argument("--dense-gap-start-layer", type=int, default=10)
    parser.add_argument("--dense-gap-end-layer", type=int, default=17)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--save-depth-npz", action="store_true")
    return parser.parse_args()


def parse_indices(spec: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in spec.split(",") if part.strip())
    if not values:
        raise ValueError("index list is empty")
    return values


def slugify(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def finite_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_model(
    checkpoint: Path,
    device: torch.device,
    *,
    experiment_mode: str,
    first_frame_token_indices: tuple[int, ...],
    frame_fusion_pair_percent: float,
    frame_fusion_pool_size: int,
    frame_fusion_target_keep_policy: str,
    frame_fusion_target_keep_grid_size: int,
    frame_fusion_target_keep_percent: float,
    frame_fusion_target_keep_threshold: float,
    frame_fusion_target_keep_seed: int,
) -> VGGTOmega:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    frame_fusion_mode = "none" if experiment_mode == "baseline" else "pair-top-percent"
    model = VGGTOmega(
        merge_ratio=0.0,
        first_frame_token_indices=first_frame_token_indices,
        frame_fusion_mode=frame_fusion_mode,
        frame_fusion_start_layer=-1,
        frame_fusion_pair_percent=frame_fusion_pair_percent,
        frame_fusion_pool_size=frame_fusion_pool_size,
        frame_fusion_target_keep_policy=frame_fusion_target_keep_policy,
        frame_fusion_target_keep_grid_size=frame_fusion_target_keep_grid_size,
        frame_fusion_target_keep_percent=frame_fusion_target_keep_percent,
        frame_fusion_target_keep_threshold=frame_fusion_target_keep_threshold,
        frame_fusion_target_keep_seed=frame_fusion_target_keep_seed,
        frame_fusion_recompute_each_global=False,
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


def install_dense_gap_forward(
    aggregator: object,
    *,
    early_fusion_end_layer: int,
    dense_gap_start_layer: int,
    dense_gap_end_layer: int,
) -> None:
    if dense_gap_start_layer != early_fusion_end_layer + 1:
        raise ValueError("dense gap must start immediately after early fusion end layer")
    if dense_gap_end_layer < dense_gap_start_layer:
        raise ValueError("dense gap end layer must be >= start layer")
    if dense_gap_end_layer >= aggregator.depth:
        raise ValueError("dense gap end layer must be inside the model depth")
    if aggregator.frame_fusion_mode != "pair-top-percent":
        raise ValueError("dense-gap ablation expects pair-top-percent frame fusion")

    def scheduled_forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        original_num_frames = num_frames
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        first_frame_token_indices = resolve_first_frame_token_indices(
            self.first_frame_token_indices,
            num_frames,
        )
        camera_token = slice_expand_and_flatten(
            self.camera_token,
            batch_size,
            num_frames,
            first_frame_token_indices=first_frame_token_indices,
        )
        register_token = slice_expand_and_flatten(
            self.register_token,
            batch_size,
            num_frames,
            first_frame_token_indices=first_frame_token_indices,
        )

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs: list[torch.Tensor | None] = []
        self._register_patch_selection.clear()
        self._adaptive_intra_scores.clear()
        self._progressive_stage_states.clear()
        self.last_progressive_attention_stats.clear()
        self.last_progressive_sample_indices.clear()
        self.last_adaptive_pair_scope_debug.clear()
        self.last_frame_fusion_debug.clear()
        self._frame_fusion_debug_layers.clear()

        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
        debug_layers: list[dict[str, object]] = []

        def build_pair_plans(current_tokens: torch.Tensor, source_layer: int, stage: str):
            plans = self._build_frame_fusion_pair_plans(
                current_tokens,
                patch_grid_size=patch_grid_size,
                source_layer=source_layer,
            )
            debug = copy.deepcopy(self.last_frame_fusion_debug)
            debug["schedule_stage"] = stage
            debug["source_layer"] = source_layer
            debug_layers.append(debug)
            return plans

        early_pair_plans = build_pair_plans(tokens, -1, "early")
        resumed_pair_plans = None
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)

        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            if block_idx <= early_fusion_end_layer:
                current_pair_plans = early_pair_plans
            elif block_idx <= dense_gap_end_layer:
                current_pair_plans = None
            else:
                current_pair_plans = resumed_pair_plans
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
                patch_grid_size,
                frame_fusion_pair_plans=current_pair_plans,
            )
            layer_token_swap_active = (
                self.layer_token_swap_layer == block_idx
                and self.layer_token_swap_kind != "none"
            )
            if layer_token_swap_active:
                tokens = self._apply_layer_token_swap(
                    tokens,
                    kind=self.layer_token_swap_kind,
                    pairs=self.layer_token_swap_pairs,
                )
            if block_idx in self.cached_layer_indices:
                if layer_token_swap_active:
                    frame_tokens = self._apply_layer_token_swap(
                        frame_tokens,
                        kind=self.layer_token_swap_kind,
                        pairs=self.layer_token_swap_pairs,
                    )
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

            if block_idx == dense_gap_end_layer and block_idx + 1 < self.depth:
                resumed_pair_plans = build_pair_plans(
                    tokens,
                    dense_gap_end_layer,
                    "resume_after_dense_gap",
                )

        last_debug = copy.deepcopy(debug_layers[-1]) if debug_layers else {}
        last_debug["scheduled_dense_gap"] = True
        last_debug["fusion_schedule"] = {
            "early_fusion_blocks": list(range(0, early_fusion_end_layer + 1)),
            "dense_gap_blocks": list(range(dense_gap_start_layer, dense_gap_end_layer + 1)),
            "resumed_fusion_blocks": list(range(dense_gap_end_layer + 1, self.depth)),
            "resume_plan_source_layer": (
                dense_gap_end_layer if dense_gap_end_layer + 1 < self.depth else None
            ),
            "original_num_frames": original_num_frames,
        }
        last_debug["layers"] = debug_layers
        last_debug["num_recomputed_layers"] = len(debug_layers)
        last_debug["recomputed_source_layers"] = [
            layer_debug.get("source_layer") for layer_debug in debug_layers
        ]
        self.last_frame_fusion_debug = last_debug
        self._frame_fusion_debug_layers = debug_layers
        return outputs, self.patch_token_start

    aggregator.forward = types.MethodType(scheduled_forward, aggregator)


def install_upper_tri_pair_selection() -> None:
    def select_upper_tri_pairs_from_normalized_representations(
        normalized_frame_representations: torch.Tensor,
        *,
        pair_percent: float,
        exclude_frames: tuple[int, ...] | list[int] = (),
    ) -> tuple[list[FrameFusionPair], int, int]:
        if normalized_frame_representations.ndim != 2:
            raise ValueError(
                "normalized_frame_representations must have shape [frames, channels], "
                f"got {tuple(normalized_frame_representations.shape)}"
            )
        num_frames = int(normalized_frame_representations.shape[0])
        pair_percent = float(pair_percent)
        if not 0.0 < pair_percent <= 100.0:
            raise ValueError(f"pair_percent must be in (0, 100], got {pair_percent}")
        excluded = {int(frame) for frame in exclude_frames}
        invalid_excluded = sorted(frame for frame in excluded if frame < 0 or frame >= num_frames)
        if invalid_excluded:
            raise ValueError(f"exclude_frames contains out-of-range indices: {invalid_excluded}")
        if num_frames - len(excluded) < 2:
            return [], 0, 0

        reps = normalized_frame_representations.detach().float()
        similarity = torch.matmul(reps, reps.T).clamp(-1.0, 1.0)
        candidates: list[FrameFusionPair] = []
        for frame_a in range(num_frames - 1):
            if frame_a in excluded:
                continue
            for frame_b in range(frame_a + 1, num_frames):
                if frame_b in excluded:
                    continue
                score = float(similarity[frame_a, frame_b].detach().cpu().item())
                if math.isfinite(score):
                    candidates.append(
                        FrameFusionPair(
                            frame_a=frame_a,
                            frame_b=frame_b,
                            similarity=score,
                        )
                    )
        return aggregator_mod._select_top_percent_disjoint_frame_pairs(
            candidates,
            pair_percent=pair_percent,
        )

    aggregator_mod.select_frame_fusion_pairs_from_normalized_representations = (
        select_upper_tri_pairs_from_normalized_representations
    )


def run_model(
    model: VGGTOmega,
    images: torch.Tensor,
    device: torch.device,
    *,
    skip_timing: bool,
    timing_repeats: int,
) -> tuple[dict[str, torch.Tensor], list[float], float, float, float]:
    if not skip_timing:
        with torch.inference_mode():
            warmup = model(images)
        del warmup
        torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    timings_ms: list[float] = []
    predictions = None
    started = time.perf_counter()
    repeats = 1 if skip_timing else timing_repeats
    with torch.inference_mode():
        for _ in range(repeats):
            if not skip_timing:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            current_predictions = model(images)
            if not skip_timing:
                end_event.record()
                torch.cuda.synchronize(device)
                timings_ms.append(float(start_event.elapsed_time(end_event)))
            else:
                torch.cuda.synchronize(device)
            if predictions is not None:
                del predictions
            predictions = current_predictions
    assert predictions is not None
    elapsed = time.perf_counter() - started
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
    return predictions, timings_ms, peak_allocated_gib, peak_reserved_gib, elapsed


def evaluate_predictions(
    dataset: str,
    predictions: dict[str, torch.Tensor],
    records: Sequence[object],
    *,
    depth_alignment: str,
    min_depth: float,
    max_depth: float,
) -> tuple[dict[str, object], np.ndarray]:
    with torch.inference_mode():
        extrinsics, _ = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
            build_intrinsics=False,
        )
    if dataset == "7Scenes":
        pred_w2c = seven_eval.to_homogeneous_w2c(extrinsics[0])
        rotation_errors, translation_errors = seven_eval.pairwise_pose_errors(
            pred_w2c,
            np.linalg.inv(np.stack([record.c2w for record in records])),
        )
        official_auc = seven_eval.official_auc
        depth_sums = seven_eval.depth_sums
        predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
        abs_rel_sum, delta_count, valid_count, scales = depth_sums(
            predicted_depth,
            records,
            depth_alignment,
            min_depth,
            max_depth,
        )
    else:
        pred_w2c = tum_eval.to_homogeneous_w2c(extrinsics[0])
        rotation_errors, translation_errors = tum_eval.pairwise_pose_errors(
            pred_w2c,
            np.linalg.inv(np.stack([record.c2w for record in records])),
        )
        official_auc = tum_eval.official_auc
        depth_sums = tum_eval.depth_sums
        predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
        abs_rel_sum, delta_count, valid_count, scales = depth_sums(
            predicted_depth,
            records,
            depth_alignment,
            max_depth,
        )

    metrics = {
        "auc_3_percent": 100.0 * official_auc(rotation_errors, translation_errors, 3),
        "auc_30_percent": 100.0 * official_auc(rotation_errors, translation_errors, 30),
        "delta_1_25_percent": 100.0 * delta_count / valid_count,
        "abs_rel": abs_rel_sum / valid_count,
        "valid_depth_pixels": valid_count,
        "depth_scales": scales,
        "rotation_error_mean_deg": float(np.mean(rotation_errors)),
        "translation_error_mean_deg": float(np.mean(translation_errors)),
    }
    return metrics, predicted_depth


def sequence_debug_summary(debug: dict[str, object]) -> dict[str, object]:
    layers = debug.get("layers") or []
    result: dict[str, object] = {
        "scheduled_dense_gap": debug.get("scheduled_dense_gap"),
        "fusion_schedule": debug.get("fusion_schedule"),
        "num_recomputed_layers": debug.get("num_recomputed_layers"),
        "recomputed_source_layers": debug.get("recomputed_source_layers"),
    }
    layer_summaries = []
    for layer in layers:
        batches = layer.get("batches") or []
        first_batch = batches[0] if batches else {}
        layer_summaries.append(
            {
                "stage": layer.get("schedule_stage"),
                "source_layer": layer.get("source_layer"),
                "selected_pairs": layer.get("avg_selected_pairs", first_batch.get("selected_pairs")),
                "attention_tokens": layer.get("avg_attention_tokens", first_batch.get("attention_tokens")),
                "patch_retention_vs_full": layer.get(
                    "avg_patch_token_retention_vs_full",
                    layer.get("patch_token_retention_vs_full"),
                ),
                "target_keep_patch_tokens_per_pair": first_batch.get(
                    "target_keep_patch_tokens_per_pair"
                ),
            }
        )
    result["layers"] = layer_summaries
    return result


def main() -> int:
    args = parse_args()
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise ValueError("--image-resolution must be positive and divisible by 16")
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be at least 1")
    records = load_sequence_records(args)
    if len(records) != 300:
        raise ValueError(f"{args.sequence}: expected 300 sampled frames, got {len(records)}")
    first_frame_token_indices = resolve_first_frame_token_indices(
        parse_indices(args.first_frame_token_indices),
        len(records),
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega inference requires CUDA")
    if device.index is not None:
        torch.cuda.set_device(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_slug = slugify(args.sequence)
    output_json = args.output_dir / f"{args.dataset}_{sequence_slug}_metrics.json"
    output_depth = args.output_dir / f"{args.dataset}_{sequence_slug}_depth.npz"

    print(f"Loading {args.checkpoint}")
    if args.pair_selection_method == "upper-tri":
        install_upper_tri_pair_selection()
    model = load_model(
        args.checkpoint,
        device,
        experiment_mode=args.experiment_mode,
        first_frame_token_indices=first_frame_token_indices,
        frame_fusion_pair_percent=args.frame_fusion_pair_percent,
        frame_fusion_pool_size=args.frame_fusion_pool_size,
        frame_fusion_target_keep_policy=args.frame_fusion_target_keep_policy,
        frame_fusion_target_keep_grid_size=args.frame_fusion_target_keep_grid_size,
        frame_fusion_target_keep_percent=args.frame_fusion_target_keep_percent,
        frame_fusion_target_keep_threshold=args.frame_fusion_target_keep_threshold,
        frame_fusion_target_keep_seed=args.frame_fusion_target_keep_seed,
    )
    if args.experiment_mode == "dense-gap":
        install_dense_gap_forward(
            model.aggregator,
            early_fusion_end_layer=args.early_fusion_end_layer,
            dense_gap_start_layer=args.dense_gap_start_layer,
            dense_gap_end_layer=args.dense_gap_end_layer,
        )
    elif args.experiment_mode == "tail-dense":
        install_dense_gap_forward(
            model.aggregator,
            early_fusion_end_layer=model.aggregator.depth - 2,
            dense_gap_start_layer=model.aggregator.depth - 1,
            dense_gap_end_layer=model.aggregator.depth - 1,
        )

    images = load_and_preprocess_images(
        [str(record.rgb_path) for record in records],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    ).to(device, non_blocking=True)
    predictions, timings_ms, peak_allocated_gib, peak_reserved_gib, inference_seconds = run_model(
        model,
        images,
        device,
        skip_timing=args.skip_timing,
        timing_repeats=args.timing_repeats,
    )
    metrics, predicted_depth = evaluate_predictions(
        args.dataset,
        predictions,
        records,
        depth_alignment=args.depth_alignment,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    model_latency_ms = None if args.skip_timing else float(np.median(timings_ms))
    row = {
        "sequence": args.sequence,
        **metrics,
        "model_latency_ms": model_latency_ms,
        "model_latency_repeats_ms": timings_ms,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "inference_seconds": inference_seconds,
        "frame_fusion_debug": sequence_debug_summary(model.aggregator.last_frame_fusion_debug),
    }
    if args.experiment_mode == "tail-dense":
        schedule = {
            "early_fusion_end_layer": model.aggregator.depth - 2,
            "dense_gap_start_layer": model.aggregator.depth - 1,
            "dense_gap_end_layer": model.aggregator.depth - 1,
            "resume_fusion_start_layer": None,
            "tail_dense_block": model.aggregator.depth - 1,
        }
    else:
        schedule = {
            "early_fusion_end_layer": args.early_fusion_end_layer,
            "dense_gap_start_layer": args.dense_gap_start_layer,
            "dense_gap_end_layer": args.dense_gap_end_layer,
            "resume_fusion_start_layer": (
                args.dense_gap_end_layer + 1
                if args.experiment_mode == "dense-gap"
                else None
            ),
            "tail_dense_block": None,
        }
    result = {
        "protocol": {
            "ablation": f"least20_pairfusion_{args.experiment_mode}",
            "experiment_mode": args.experiment_mode,
            "dataset": args.dataset,
            "sequence": args.sequence,
            "num_frames": len(records),
            "image_resolution": args.image_resolution,
            "resize_mode": args.resize_mode,
            "depth_alignment": args.depth_alignment,
            "checkpoint": str(args.checkpoint),
            "first_frame_token_indices": list(first_frame_token_indices),
            "frame_fusion": {
                "pair_selection_method": args.pair_selection_method,
                "pair_percent": args.frame_fusion_pair_percent,
                "pool_size": args.frame_fusion_pool_size,
                "target_keep_policy": args.frame_fusion_target_keep_policy,
                "target_keep_grid_size": args.frame_fusion_target_keep_grid_size,
                "target_keep_percent": args.frame_fusion_target_keep_percent,
                "target_keep_threshold": args.frame_fusion_target_keep_threshold,
                "target_keep_seed": args.frame_fusion_target_keep_seed,
            },
            "schedule": schedule,
            "skip_timing": args.skip_timing,
            "timing_repeats": args.timing_repeats,
        },
        "overall": {
            key: row[key]
            for key in (
                "auc_3_percent",
                "auc_30_percent",
                "delta_1_25_percent",
                "abs_rel",
                "valid_depth_pixels",
                "model_latency_ms",
                "peak_allocated_gib",
                "peak_reserved_gib",
                "inference_seconds",
            )
        },
        "per_sequence": [row],
    }
    output_json.write_text(json.dumps(finite_json(result), indent=2) + "\n", encoding="utf-8")
    if args.save_depth_npz:
        np.savez_compressed(output_depth, depth=predicted_depth.astype(np.float32))

    latency_text = "skipped" if args.skip_timing else f"{model_latency_ms:.1f}ms"
    print(
        f"[{args.sequence}] AUC@3={row['auc_3_percent']:.2f}, "
        f"AUC@30={row['auc_30_percent']:.2f}, "
        f"delta1.25={row['delta_1_25_percent']:.2f}, "
        f"AbsRel={row['abs_rel']:.4f}, peak={peak_allocated_gib:.2f}GiB, "
        f"latency={latency_text}, inference_seconds={inference_seconds:.1f}"
    )
    print(f"Wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
