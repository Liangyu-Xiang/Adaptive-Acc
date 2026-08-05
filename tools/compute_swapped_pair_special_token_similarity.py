#!/usr/bin/env python3
"""Compute per-special-token cosine similarity for token-swap experiment pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_selected_camera_token_swap import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SAMPLED_FRAMES,
    load_manifest_records,
    read_pairs_csv,
    run_aggregator,
)
from scripts.eval_tum_dynamics_paper import load_model  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402


DEFAULT_PAIRS_CSV = (
    REPO_ROOT
    / "outputs"
    / "last_layer_token_swap__tum_halfsphere_300f__layer23_gt0p76"
    / "selected_pairs.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sampled-frames", type=Path, default=DEFAULT_SAMPLED_FRAMES)
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--sequence", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/special_token_similarity__last_layer_swap_pairs_layer23_gt0p76"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument("--patch-embed-chunk-size", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def token_labels(patch_token_start: int) -> list[str]:
    if patch_token_start < 1:
        raise ValueError(f"patch_token_start must include the camera token, got {patch_token_start}")
    return ["camera"] + [f"register_{index:02d}" for index in range(patch_token_start - 1)]


def compute_special_cosines(
    final_tokens: torch.Tensor,
    pairs: list[tuple[int, int, float]],
    patch_token_start: int,
) -> np.ndarray:
    if final_tokens.ndim != 4 or final_tokens.shape[0] != 1:
        raise ValueError(f"Expected final tokens with shape [1, F, T, D], got {tuple(final_tokens.shape)}")
    special = final_tokens[0, :, :patch_token_start].detach().float()
    special = special / torch.linalg.norm(special, dim=-1, keepdim=True).clamp_min(1e-12)
    first = torch.tensor([i for i, _, _ in pairs], device=special.device, dtype=torch.long)
    second = torch.tensor([j for _, j, _ in pairs], device=special.device, dtype=torch.long)
    return (special[first] * special[second]).sum(dim=-1).cpu().numpy().astype(np.float64)


def summarize_by_token(cosines: np.ndarray, labels: list[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for token_index, label in enumerate(labels):
        values = cosines[:, token_index]
        rows.append(
            {
                "token_index": token_index,
                "token_label": label,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p01": float(np.percentile(values, 1)),
                "p05": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(values.max()),
            }
        )
    return rows


def write_pair_csv(
    path: Path,
    pairs: list[tuple[int, int, float]],
    cosines: np.ndarray,
    labels: list[str],
) -> None:
    fieldnames = ["pair_index", "frame_i", "frame_j", "layer23_patch_similarity", *labels]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair_index, (i, j, similarity) in enumerate(pairs):
            row: dict[str, str | int] = {
                "pair_index": pair_index,
                "frame_i": i,
                "frame_j": j,
                "layer23_patch_similarity": f"{similarity:.8f}",
            }
            for token_index, label in enumerate(labels):
                row[label] = f"{cosines[pair_index, token_index]:.8f}"
            writer.writerow(row)


def write_token_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "token_index",
        "token_label",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p50",
        "p95",
        "p99",
        "max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row[key] if isinstance(row[key], str) else f"{float(row[key]):.8f}"
                    for key in fieldnames
                }
            )


def main() -> int:
    args = parse_args()
    if args.patch_embed_chunk_size < 0:
        raise ValueError("--patch-embed-chunk-size must be non-negative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This computation requires CUDA")

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
    with torch.inference_mode():
        aggregated_tokens, patch_token_start = run_aggregator(
            model,
            images,
            use_amp=not args.no_amp,
            patch_embed_chunk_size=args.patch_embed_chunk_size,
        )
        final_tokens = aggregated_tokens[-1]
        if final_tokens is None:
            raise ValueError("Final aggregated tokens are missing")
        labels = token_labels(patch_token_start)
        cosines = compute_special_cosines(final_tokens, pairs, patch_token_start)

    per_token_rows = summarize_by_token(cosines, labels)
    write_pair_csv(args.output_dir / "per_pair_special_token_cosine.csv", pairs, cosines, labels)
    write_token_summary_csv(args.output_dir / "per_token_similarity_summary.csv", per_token_rows)

    all_values = cosines.reshape(-1)
    result: dict[str, object] = {
        "sequence": sequence_name,
        "num_input_frames": len(records),
        "pairs_csv": str(args.pairs_csv),
        "selected_pairs": len(pairs),
        "checkpoint": str(args.checkpoint),
        "model_variant": "VGGTOmega dense original, merge_ratio=0.0, frame_fusion_mode=none",
        "image_shape_hw": [int(images.shape[-2]), int(images.shape[-1])],
        "patch_token_start": int(patch_token_start),
        "special_tokens": labels,
        "similarity_scope": (
            "Cosine similarity between same special-token offsets of each swapped frame pair "
            "at the final cached aggregator layer."
        ),
        "all_special_token_cosine": {
            "count": int(all_values.size),
            "mean": float(all_values.mean()),
            "std": float(all_values.std(ddof=0)),
            "min": float(all_values.min()),
            "p01": float(np.percentile(all_values, 1)),
            "p05": float(np.percentile(all_values, 5)),
            "p50": float(np.percentile(all_values, 50)),
            "p95": float(np.percentile(all_values, 95)),
            "p99": float(np.percentile(all_values, 99)),
            "max": float(all_values.max()),
        },
        "camera_token_cosine": per_token_rows[0],
        "register_token_cosine_mean_over_registers": {
            "mean": float(cosines[:, 1:].mean()),
            "std": float(cosines[:, 1:].std(ddof=0)),
            "min": float(cosines[:, 1:].min()),
            "p05": float(np.percentile(cosines[:, 1:], 5)),
            "p50": float(np.percentile(cosines[:, 1:], 50)),
            "p95": float(np.percentile(cosines[:, 1:], 95)),
            "max": float(cosines[:, 1:].max()),
        },
        "per_token_summary": per_token_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "special_token_cosine.npz",
        cosines=cosines,
        labels=np.asarray(labels),
        pairs=np.asarray([(i, j, s) for i, j, s in pairs], dtype=np.float64),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
