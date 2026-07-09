#!/usr/bin/env python3
"""Oracle experiment: merge only patches covered by the register rollout.

Coverage masks are read from analyze_register_path_coverage.py outputs. Their
calibration requires a full-attention pass and is excluded from timed inference;
therefore this measures a quality/compute upper bound, not an online speedup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_tum_dynamics_paper import (
    FrameRecord,
    depth_sums,
    gt_row_to_c2w,
    official_auc,
    pairwise_pose_errors,
    read_rows,
    to_homogeneous_w2c,
)
from tools.analyze_token_evolution import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "token_evolution_3frame",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coverage-k", type=int, choices=(5, 20, 50), default=20)
    parser.add_argument(
        "--coverage-rule",
        choices=("all-other-frames", "any-other-frame"),
        default="all-other-frames",
    )
    parser.add_argument(
        "--mask-side",
        choices=("query", "source"),
        default="source",
        help="Select query tokens or source tokens whose outgoing top-1 edges are covered.",
    )
    parser.add_argument("--source-min-direct-edges", type=int, default=1)
    parser.add_argument("--merge-kv-only", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="Optional global block indices to merge; default uses all global blocks.",
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "token_evolution_3frame" / "coverage_guided_merge.json",
    )
    return parser.parse_args()


def build_masks(
    coverage_path: Path,
    patch_start: int,
    tokens_per_frame: int,
    coverage_k: int,
    rule: str,
    mask_side: str,
    source_min_direct_edges: int,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    data = np.load(coverage_path)
    layers = data["global_layers"]
    thresholds = data["coverage_k"]
    threshold_index = int(np.flatnonzero(thresholds == coverage_k)[0])
    covered = data["direct_top1_covered"][..., threshold_index]
    direct_top1 = data["direct_top1_patch_index"]
    # [layer, batch, query frame, source frame, query patch]
    masks: dict[int, torch.Tensor] = {}
    fractions: dict[int, float] = {}
    for layer_index, layer in enumerate(layers):
        layer_covered = covered[layer_index]
        batch, frames, _, patches = layer_covered.shape
        eligible = np.zeros((batch, frames, patches), dtype=bool)
        if mask_side == "query":
            for query_frame in range(frames):
                other = [
                    layer_covered[:, query_frame, source]
                    for source in range(frames)
                    if source != query_frame
                ]
                stacked = np.stack(other, axis=0)
                eligible[:, query_frame] = (
                    np.all(stacked, axis=0) if rule == "all-other-frames" else np.any(stacked, axis=0)
                )
        else:
            for batch_index in range(batch):
                for source_frame in range(frames):
                    counts = np.zeros(patches, dtype=np.int32)
                    covered_counts = np.zeros(patches, dtype=np.int32)
                    for query_frame in range(frames):
                        if query_frame == source_frame:
                            continue
                        indices = direct_top1[layer_index, batch_index, query_frame, source_frame]
                        np.add.at(counts, indices, 1)
                        np.add.at(
                            covered_counts,
                            indices,
                            layer_covered[batch_index, query_frame, source_frame].astype(np.int32),
                        )
                    eligible[batch_index, source_frame] = (
                        (counts >= source_min_direct_edges) & (covered_counts == counts)
                    )
        full = np.zeros((batch, frames, tokens_per_frame), dtype=bool)
        full[:, :, patch_start:] = eligible
        masks[int(layer)] = torch.from_numpy(full.reshape(batch, frames * tokens_per_frame))
        # The current bipartite merger keeps the first frame as destinations.
        fractions[int(layer)] = float(np.mean(eligible[:, 1:]))
    return masks, fractions


def configure_baseline(model) -> None:
    model.aggregator.global_merging = False
    model.aggregator.merging = None
    for block in model.aggregator.inter_frame_blocks:
        block.attn.disable_global_merging = False
        block.attn.merge_kv_only = False
        if hasattr(block.attn, "merge_eligible_mask"):
            delattr(block.attn, "merge_eligible_mask")


def configure_guided(
    model,
    masks: dict[int, torch.Tensor],
    device: torch.device,
    selected_layers: set[int] | None,
    merge_kv_only: bool,
) -> None:
    model.aggregator.global_merging = True
    model.aggregator.merging = 0
    model.set_merge_ratio(1.0)
    for layer, block in enumerate(model.aggregator.inter_frame_blocks):
        enabled = layer in masks and (selected_layers is None or layer in selected_layers)
        block.attn.disable_global_merging = not enabled
        block.attn.merge_kv_only = merge_kv_only
        if enabled:
            block.attn.merge_eligible_mask = masks[layer].to(device)
        elif hasattr(block.attn, "merge_eligible_mask"):
            delattr(block.attn, "merge_eligible_mask")


def timed_forward(model, images: torch.Tensor, device: torch.device, repeats: int):
    with torch.inference_mode():
        warmup = model(images)
    del warmup
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    prediction = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            current = model(images)
        end.record()
        torch.cuda.synchronize(device)
        timings.append(float(start.elapsed_time(end)))
        if prediction is not None:
            del prediction
        prediction = current
    return prediction, timings, torch.cuda.max_memory_allocated(device) / (1024**3)


def prediction_difference(reference, candidate) -> dict[str, float]:
    result = {}
    for key in ("pose_enc", "depth", "depth_conf"):
        delta = (reference[key].float() - candidate[key].float()).abs()
        result[f"{key}_mean_abs"] = float(delta.mean())
        result[f"{key}_max_abs"] = float(delta.max())
    return result


def task_metrics(predictions, records, image_shape) -> dict[str, float]:
    with torch.inference_mode():
        extrinsics, _ = encoding_to_camera(
            predictions["pose_enc"], image_shape, build_intrinsics=False
        )
    pred_w2c = to_homogeneous_w2c(extrinsics[0])
    gt_w2c = np.linalg.inv(np.stack([record.c2w for record in records]))
    rotation, translation = pairwise_pose_errors(pred_w2c, gt_w2c)
    predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
    abs_rel_sum, delta_count, valid_count, _ = depth_sums(
        predicted_depth, records, "per-frame-median", 10.0
    )
    return {
        "auc_3_percent": 100 * official_auc(rotation, translation, 3),
        "auc_30_percent": 100 * official_auc(rotation, translation, 30),
        "delta_1_25_percent": 100 * delta_count / valid_count,
        "abs_rel": abs_rel_sum / valid_count,
    }


def records_for_rgb_paths(paths: list[str]) -> list[FrameRecord]:
    """Attach nearest GT/depth packets to the exact RGB files used by the pilot."""
    sequence_dir = Path(paths[0]).parents[1]
    gt_rows = read_rows(sequence_dir / "groundtruth.txt")
    depth_rows = read_rows(sequence_dir / "depth.txt")
    gt_times = np.asarray([row[0] for row in gt_rows])
    depth_times = np.asarray([row[0] for row in depth_rows])
    records = []
    for value in paths:
        rgb_path = Path(value)
        rgb_timestamp = float(rgb_path.stem)
        gt_index = int(np.argmin(np.abs(gt_times - rgb_timestamp)))
        depth_index = int(np.argmin(np.abs(depth_times - rgb_timestamp)))
        gt_timestamp, gt_fields = gt_rows[gt_index]
        depth_timestamp, depth_fields = depth_rows[depth_index]
        records.append(
            FrameRecord(
                rgb_timestamp=rgb_timestamp,
                rgb_path=rgb_path,
                gt_timestamp=gt_timestamp,
                c2w=gt_row_to_c2w(gt_fields),
                depth_timestamp=depth_timestamp,
                depth_path=sequence_dir / depth_fields[0],
            )
        )
    return records


def main() -> int:
    args = parse_args()
    summary = json.loads((args.analysis_dir / "summary.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    results = {
        "config": {
            "coverage_k": args.coverage_k,
            "coverage_rule": args.coverage_rule,
            "timing_repeats": args.timing_repeats,
            "merge_ratio_cap": 1.0,
            "oracle_calibration_excluded_from_timing": True,
            "selected_layers": args.layers,
            "mask_side": args.mask_side,
            "source_min_direct_edges": args.source_min_direct_edges,
            "merge_kv_only": args.merge_kv_only,
        },
        "sequences": [],
    }
    for sequence, paths in summary["frame_selections"].items():
        images = load_and_preprocess_images(
            paths, mode=args.resize_mode, image_resolution=args.image_resolution
        ).to(device)
        tokens_per_frame = 1 + 16 + (images.shape[-2] // 16) * (images.shape[-1] // 16)
        masks, fractions = build_masks(
            args.analysis_dir / sequence / "register_path_coverage" / "register_path_coverage_metrics.npz",
            model.aggregator.patch_token_start,
            tokens_per_frame,
            args.coverage_k,
            args.coverage_rule,
            args.mask_side,
            args.source_min_direct_edges,
        )

        configure_baseline(model)
        baseline, baseline_times, baseline_peak = timed_forward(
            model, images, device, args.timing_repeats
        )
        selected_layers = None if args.layers is None else set(args.layers)
        configure_guided(model, masks, device, selected_layers, args.merge_kv_only)
        guided, guided_times, guided_peak = timed_forward(
            model, images, device, args.timing_repeats
        )

        records = records_for_rgb_paths(paths)
        baseline_metrics = task_metrics(baseline, records, baseline["images"].shape[-2:])
        guided_metrics = task_metrics(guided, records, guided["images"].shape[-2:])
        baseline_latency = float(np.median(baseline_times))
        guided_latency = float(np.median(guided_times))
        results["sequences"].append(
            {
                "sequence": sequence,
                "eligible_fraction_nonfirst_patch_by_layer": fractions,
                "eligible_fraction_nonfirst_patch_mean": float(
                    np.mean(
                        [
                            fraction
                            for layer, fraction in fractions.items()
                            if selected_layers is None or layer in selected_layers
                        ]
                    )
                ),
                "baseline": {
                    "latency_ms": baseline_latency,
                    "timing_repeats_ms": baseline_times,
                    "peak_allocated_gib": baseline_peak,
                    **baseline_metrics,
                },
                "coverage_guided_merge": {
                    "latency_ms": guided_latency,
                    "timing_repeats_ms": guided_times,
                    "peak_allocated_gib": guided_peak,
                    **guided_metrics,
                },
                "latency_speedup": baseline_latency / guided_latency,
                "prediction_difference": prediction_difference(baseline, guided),
            }
        )
        del images, baseline, guided
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
