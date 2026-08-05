#!/usr/bin/env python3
"""Calibrate a global token-similarity threshold for pair frame fusion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.models.aggregator import (  # noqa: E402
    pooled_frame_representations,
    select_frame_fusion_pairs_from_normalized_representations,
)
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-sequence",
        action="append",
        required=True,
        help="Calibration input as sampled_frames.json::sequence_key. Repeat for multiple sequences.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--pair-percent", type=float, default=25.0)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--first-frame-token-indices", default="0")
    parser.add_argument(
        "--feature-stage",
        choices=("aggregator-last", "fusion-start"),
        default="aggregator-last",
        help=(
            "Feature stage used for pair selection and per-token similarities. "
            "fusion-start matches frame_fusion_start_layer=-1."
        ),
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def parse_manifest_sequence(spec: str) -> tuple[Path, str]:
    try:
        manifest_text, sequence = spec.split("::", 1)
    except ValueError as exc:
        raise ValueError(
            "--manifest-sequence must use sampled_frames.json::sequence_key format"
        ) from exc
    manifest = Path(manifest_text)
    if not manifest.is_file():
        raise FileNotFoundError(f"sampled_frames.json does not exist: {manifest}")
    if not sequence:
        raise ValueError(f"Missing sequence key in manifest spec: {spec}")
    return manifest, sequence


def load_rgb_paths(manifest: Path, sequence: str) -> list[str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if sequence not in data:
        raise KeyError(f"Sequence {sequence!r} not found in {manifest}")
    paths = data[sequence].get("rgb_paths")
    if not paths:
        raise ValueError(f"Sequence {sequence!r} has no rgb_paths in {manifest}")
    return [str(path) for path in paths]


def load_model(checkpoint: Path, device: torch.device, first_frame_token_indices: str) -> VGGTOmega:
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


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0}
    quantile_points = [0.0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    summary: dict[str, float | int] = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    for point in quantile_points:
        summary[f"q{int(point * 100):02d}"] = float(np.quantile(values, point))
    return summary


def run_sequence(
    *,
    model: VGGTOmega,
    manifest: Path,
    sequence: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    rgb_paths = load_rgb_paths(manifest, sequence)
    images = load_and_preprocess_images(
        rgb_paths,
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
        patch_size=model.aggregator.patch_size,
    ).unsqueeze(0).to(device)
    patch_grid_size = (
        images.shape[-2] // model.aggregator.patch_size,
        images.shape[-1] // model.aggregator.patch_size,
    )

    started = time.perf_counter()
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        context = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        context = nullcontext()
    aggregated_tokens = None
    with torch.inference_mode(), context:
        if args.feature_stage == "aggregator-last":
            aggregated_tokens, patch_token_start = model.aggregator(images)
            decoder_before_tokens = aggregated_tokens[-1][0]
            patch_tokens = decoder_before_tokens[:, patch_token_start:]
        else:
            aggregator = model.aggregator
            normalized_images = (images - aggregator._resnet_mean) / aggregator._resnet_std
            batch_size, num_frames, num_channels, height, width = normalized_images.shape
            flat_images = normalized_images.view(batch_size * num_frames, num_channels, height, width)
            patch_tokens = aggregator.patch_embed(flat_images)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            patch_tokens = patch_tokens.view(num_frames, patch_tokens.shape[1], patch_tokens.shape[2])
        frame_representations = pooled_frame_representations(
            patch_tokens.unsqueeze(0),
            patch_grid_size=patch_grid_size,
            pool_size=args.pool_size,
        )[0]
        normalized = F.normalize(frame_representations.float(), p=2, dim=-1)
        selected_pairs, unique_candidate_count, requested_pair_count = (
            select_frame_fusion_pairs_from_normalized_representations(
                normalized,
                pair_percent=args.pair_percent,
                exclude_frames=(0,),
            )
        )
        if selected_pairs:
            source_frames = torch.tensor(
                [pair.frame_a for pair in selected_pairs],
                device=device,
                dtype=torch.long,
            )
            target_frames = torch.tensor(
                [pair.frame_b for pair in selected_pairs],
                device=device,
                dtype=torch.long,
            )
            token_similarities = F.cosine_similarity(
                patch_tokens.index_select(0, source_frames).float(),
                patch_tokens.index_select(0, target_frames).float(),
                dim=-1,
                eps=1e-8,
            )
            values = token_similarities.flatten().detach().cpu().numpy().astype(np.float32)
        else:
            values = np.empty((0,), dtype=np.float32)
    seconds = time.perf_counter() - started

    pair_rows = np.array(
        [
            (pair.frame_a, pair.frame_b, pair.similarity)
            for pair in selected_pairs
        ],
        dtype=np.float32,
    )
    sequence_slug = slugify(sequence)
    np.savez_compressed(
        args.output_dir / f"{sequence_slug}_token_similarities.npz",
        token_similarities=values,
        selected_pairs=pair_rows,
    )
    summary = {
        "sequence": sequence,
        "sampled_frames": str(manifest),
        "num_frames": len(rgb_paths),
        "patch_grid_size": list(patch_grid_size),
        "patch_tokens_per_frame": int(patch_tokens.shape[1]),
        "unique_candidate_pairs": unique_candidate_count,
        "requested_pairs": requested_pair_count,
        "selected_pairs": len(selected_pairs),
        "candidate_token_similarities": summarize_values(values),
        "seconds": seconds,
        "first_pairs": [
            {
                "frame_a": pair.frame_a,
                "frame_b": pair.frame_b,
                "similarity": pair.similarity,
            }
            for pair in selected_pairs[:8]
        ],
    }
    del images
    if aggregated_tokens is not None:
        del aggregated_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values, summary


def write_markdown(path: Path, result: dict[str, object]) -> None:
    sequence_rows = result["sequences"]
    lines = [
        "# Frame-Fusion Token Threshold Calibration",
        "",
        f"- quantile: {result['quantile']}",
        f"- calibrated_threshold: {result['calibrated_threshold']:.8f}",
        f"- total_candidate_token_similarities: {result['candidate_token_similarities']['count']}",
        "",
        "| sequence | selected pairs | token sims | q20 | mean | min | max | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sequence_rows:
        stats = row["candidate_token_similarities"]
        lines.append(
            "| {sequence} | {pairs} | {count} | {q20:.8f} | {mean:.8f} | {minv:.8f} | {maxv:.8f} | {seconds:.2f} |".format(
                sequence=row["sequence"],
                pairs=row["selected_pairs"],
                count=stats["count"],
                q20=stats.get("q20", float("nan")),
                mean=stats.get("mean", float("nan")),
                minv=stats.get("q00", float("nan")),
                maxv=stats.get("q100", float("nan")),
                seconds=row["seconds"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.quantile <= 1.0:
        raise ValueError("--quantile must be in [0, 1]")
    if not 0.0 < args.pair_percent <= 100.0:
        raise ValueError("--pair-percent must be in (0, 100]")
    if args.pool_size <= 0:
        raise ValueError("--pool-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, args.first_frame_token_indices)
    all_values: list[np.ndarray] = []
    sequence_summaries = []
    for spec in args.manifest_sequence:
        manifest, sequence = parse_manifest_sequence(spec)
        values, summary = run_sequence(
            model=model,
            manifest=manifest,
            sequence=sequence,
            args=args,
            device=device,
        )
        all_values.append(values)
        sequence_summaries.append(summary)
        stats = summary["candidate_token_similarities"]
        print(
            f"[{sequence}] pairs={summary['selected_pairs']}, "
            f"token_sims={stats['count']}, q20={stats.get('q20', float('nan')):.8f}"
        )

    combined = np.concatenate(all_values) if all_values else np.empty((0,), dtype=np.float32)
    if combined.size == 0:
        raise RuntimeError("No token similarities were collected")
    threshold = float(np.quantile(combined, args.quantile))
    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "image_resolution": args.image_resolution,
        "resize_mode": args.resize_mode,
        "pair_percent": args.pair_percent,
        "pool_size": args.pool_size,
        "exclude_frames": [0],
        "candidate_pairs": "nearest_neighbor_unique_undirected_frame_pairs",
        "overlap_policy": "greedy_similarity_ordered_disjoint_pairs",
        "token_feature": args.feature_stage,
        "quantile": args.quantile,
        "calibrated_threshold": threshold,
        "candidate_token_similarities": summarize_values(combined),
        "sequences": sequence_summaries,
    }
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(args.output_dir / "all_token_similarities.npz", token_similarities=combined)
    write_markdown(args.output_dir / "calibration_summary.md", result)
    print(f"calibrated_threshold={threshold:.8f}")


if __name__ == "__main__":
    main()
