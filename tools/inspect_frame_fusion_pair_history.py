#!/usr/bin/env python3
"""Inspect pair-top-percent frame-fusion pair history across recomputed layers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval_tum_dynamics_paper as tum_eval  # noqa: E402
import vggt_omega.models.aggregator as aggregator_module  # noqa: E402
from vggt_omega.models.aggregator import FrameFusionPair  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled-frames", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preprocess-mode", default="balanced")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--pair-percent", type=float, default=25.0)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--target-keep-percent", type=float, default=20.0)
    parser.add_argument(
        "--selection-semantics",
        choices=("current", "legacy-nearest-dedup"),
        default="current",
        help="Use current repository pair selection or the pre-upper-triangle nearest-neighbor dedup semantics.",
    )
    return parser.parse_args()


def install_legacy_nearest_dedup_selector() -> None:
    def legacy_select(
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
        eligible_frames = [frame for frame in range(num_frames) if frame not in excluded]
        if len(eligible_frames) < 2:
            return [], 0, 0

        reps = normalized_frame_representations.detach().float().cpu()
        sim = torch.matmul(reps, reps.T).clamp(-1.0, 1.0)
        sim.fill_diagonal_(float("-inf"))
        if excluded:
            excluded_index = torch.tensor(sorted(excluded), dtype=torch.long)
            sim[excluded_index, :] = float("-inf")
            sim[:, excluded_index] = float("-inf")
        nearest = sim.argmax(dim=1)
        candidates_by_pair: dict[tuple[int, int], float] = {}
        for frame_index in eligible_frames:
            neighbor = int(nearest[frame_index].item())
            frame_a, frame_b = sorted((frame_index, neighbor))
            score = float(sim[frame_index, neighbor].item())
            previous = candidates_by_pair.get((frame_a, frame_b))
            if previous is None or score > previous:
                candidates_by_pair[(frame_a, frame_b)] = score
        candidates = [
            FrameFusionPair(frame_a=frame_a, frame_b=frame_b, similarity=score)
            for (frame_a, frame_b), score in candidates_by_pair.items()
        ]
        return aggregator_module._select_top_percent_disjoint_frame_pairs(
            candidates,
            pair_percent=pair_percent,
        )

    aggregator_module.select_frame_fusion_pairs_from_normalized_representations = legacy_select


def load_rgb_paths(sampled_frames: Path, sequence: str) -> list[str]:
    data = json.loads(sampled_frames.read_text(encoding="utf-8"))
    if sequence not in data:
        raise KeyError(f"Sequence {sequence!r} not found in {sampled_frames}")
    paths = data[sequence].get("rgb_paths")
    if not paths:
        raise ValueError(f"Sequence {sequence!r} has no rgb_paths in {sampled_frames}")
    return [str(path) for path in paths]


def summarize_pair_history(debug: dict[str, object]) -> dict[str, object]:
    layers = debug.get("layers") or []
    layer_rows: list[dict[str, object]] = []
    pair_sets: list[frozenset[tuple[int, int]]] = []
    ordered_pair_sets: list[tuple[tuple[int, int], ...]] = []
    for layer_debug in layers:
        batches = layer_debug.get("batches") or []
        batch = batches[0] if batches else {}
        pairs = [
            (int(pair["frame_a"]), int(pair["frame_b"]))
            for pair in batch.get("pairs", [])
        ]
        similarities = [
            float(pair["similarity"])
            for pair in batch.get("pairs", [])
        ]
        pair_set = frozenset(pairs)
        pair_sets.append(pair_set)
        ordered_pair_sets.append(tuple(pairs))
        layer_rows.append(
            {
                "source_layer": int(layer_debug.get("source_layer", -999)),
                "selected_pairs": len(pairs),
                "unique_candidate_pairs": int(batch.get("unique_candidate_pairs") or 0),
                "requested_pairs": int(batch.get("requested_pairs") or 0),
                "first_pairs": [
                    {
                        "frame_a": pair[0],
                        "frame_b": pair[1],
                        "similarity": similarities[index],
                    }
                    for index, pair in enumerate(pairs[:8])
                ],
                "pairs": [
                    {"frame_a": frame_a, "frame_b": frame_b, "similarity": similarities[index]}
                    for index, (frame_a, frame_b) in enumerate(pairs)
                ],
            }
        )

    changed_layers: list[int] = []
    count_changed_layers: list[int] = []
    adjacent_jaccard: list[float] = []
    entered_counts: list[int] = []
    dropped_counts: list[int] = []
    for index in range(1, len(pair_sets)):
        previous = pair_sets[index - 1]
        current = pair_sets[index]
        if current != previous:
            changed_layers.append(layer_rows[index]["source_layer"])
        if len(current) != len(previous):
            count_changed_layers.append(layer_rows[index]["source_layer"])
        union = previous | current
        adjacent_jaccard.append(len(previous & current) / max(len(union), 1))
        entered_counts.append(len(current - previous))
        dropped_counts.append(len(previous - current))

    union_pairs = set().union(*pair_sets) if pair_sets else set()
    common_pairs = set.intersection(*map(set, pair_sets)) if pair_sets else set()
    return {
        "num_layers": len(layer_rows),
        "source_layers": [row["source_layer"] for row in layer_rows],
        "pair_count_by_layer": [row["selected_pairs"] for row in layer_rows],
        "unique_pair_sets": len(set(ordered_pair_sets)),
        "unique_pair_identity_sets": len(set(pair_sets)),
        "pair_identity_changed": len(set(pair_sets)) > 1,
        "pair_order_or_identity_changed": len(set(ordered_pair_sets)) > 1,
        "adjacent_changed_layers": changed_layers,
        "adjacent_changed_count": len(changed_layers),
        "adjacent_count_changed_layers": count_changed_layers,
        "adjacent_count_changed_count": len(count_changed_layers),
        "adjacent_jaccard_mean": float(sum(adjacent_jaccard) / len(adjacent_jaccard)) if adjacent_jaccard else 1.0,
        "adjacent_jaccard_min": float(min(adjacent_jaccard)) if adjacent_jaccard else 1.0,
        "adjacent_entered_pairs_mean": float(sum(entered_counts) / len(entered_counts)) if entered_counts else 0.0,
        "adjacent_dropped_pairs_mean": float(sum(dropped_counts) / len(dropped_counts)) if dropped_counts else 0.0,
        "union_pair_count": len(union_pairs),
        "common_pair_count_all_layers": len(common_pairs),
        "layers": layer_rows,
    }


def main() -> None:
    args = parse_args()
    if args.selection_semantics == "legacy-nearest-dedup":
        install_legacy_nearest_dedup_selector()

    device = torch.device(args.device)
    rgb_paths = load_rgb_paths(args.sampled_frames, args.sequence)
    model = tum_eval.load_model(
        args.checkpoint,
        device,
        merge_ratio=0.0,
        first_frame_token_indices=(0,),
        frame_fusion_mode="pair-top-percent",
        frame_fusion_start_layer=-1,
        frame_fusion_pair_percent=args.pair_percent,
        frame_fusion_pool_size=args.pool_size,
        frame_fusion_target_keep_policy="least-similar",
        frame_fusion_target_keep_percent=args.target_keep_percent,
        frame_fusion_recompute_each_global=True,
    )
    images = load_and_preprocess_images(
        rgb_paths,
        mode=args.preprocess_mode,
        image_resolution=args.image_resolution,
        patch_size=model.aggregator.patch_size,
    ).unsqueeze(0).to(device)

    started = time.perf_counter()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        _outputs, _patch_token_start = model.aggregator(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    debug = model.aggregator.last_frame_fusion_debug
    summary = summarize_pair_history(debug)
    payload = {
        "sequence": args.sequence,
        "sampled_frames": str(args.sampled_frames),
        "selection_semantics": args.selection_semantics,
        "num_frames": len(rgb_paths),
        "pair_percent": args.pair_percent,
        "pool_size": args.pool_size,
        "target_keep_percent": args.target_keep_percent,
        "recompute_each_global": True,
        "forward_seconds": forward_seconds,
        "summary": {key: value for key, value in summary.items() if key != "layers"},
        "layers": summary["layers"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
