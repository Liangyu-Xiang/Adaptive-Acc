#!/usr/bin/env python3
"""Measure same-position patch-token cosine similarity by sequence.

The measured representation is the patch embedding immediately before
inter-frame block 0, which is the feature stage used by
``frame_fusion_start_layer=-1``.  Statistics are reported for all undirected
frame pairs (excluding reference frame 0) and for the old Top-25% disjoint
pair selection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser.add_argument("--manifest-sequence", action="append", required=False)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--pair-percent", type=float, default=25.0)
    parser.add_argument("--pair-chunk-size", type=int, default=32)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def parse_manifest_sequence(spec: str) -> tuple[Path, str]:
    manifest_text, separator, sequence = spec.partition("::")
    if not separator or not sequence:
        raise ValueError(f"Invalid manifest sequence: {spec!r}")
    manifest = Path(manifest_text)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return manifest, sequence


def load_rgb_paths(manifest: Path, sequence: str) -> list[str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    try:
        paths = data[sequence]["rgb_paths"]
    except KeyError as exc:
        raise KeyError(f"Missing sequence/rgb_paths for {sequence!r} in {manifest}") from exc
    if not paths:
        raise ValueError(f"Sequence {sequence!r} has no frames")
    return [str(path) for path in paths]


def load_model(checkpoint: Path, device: torch.device) -> VGGTOmega:
    model = VGGTOmega(
        global_merging=False,
        merging=None,
        merge_ratio=0.0,
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


def summarize(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0}
    result: dict[str, float | int] = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }
    for q in (0, 1, 5, 10, 20, 25, 50, 75, 90, 95, 99, 100):
        result[f"q{q:02d}"] = float(np.percentile(values, q))
    for threshold in (0.85, 0.90, 0.95):
        result[f"fraction_ge_{threshold:.2f}"] = float(np.mean(values >= threshold))
    return result


def pair_values(
    normalized_tokens: torch.Tensor,
    pairs: list[tuple[int, int]],
    *,
    chunk_size: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]
        first = torch.tensor([pair[0] for pair in chunk], device=normalized_tokens.device)
        second = torch.tensor([pair[1] for pair in chunk], device=normalized_tokens.device)
        similarity = (normalized_tokens.index_select(0, first) * normalized_tokens.index_select(0, second)).sum(-1)
        values.append(similarity.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(values) if values else np.empty((0,), dtype=np.float32)


def run_sequence(
    model: VGGTOmega,
    manifest: Path,
    sequence: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    rgb_paths = load_rgb_paths(manifest, sequence)
    images = load_and_preprocess_images(
        rgb_paths,
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
        patch_size=model.aggregator.patch_size,
    ).unsqueeze(0).to(device)
    aggregator = model.aggregator
    started = time.perf_counter()
    with torch.inference_mode(), (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    ):
        normalized_images = (images - aggregator._resnet_mean) / aggregator._resnet_std
        flat_images = normalized_images.flatten(0, 1)
        patch_tokens = aggregator.patch_embed(flat_images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        patch_tokens = patch_tokens.view(len(rgb_paths), patch_tokens.shape[1], patch_tokens.shape[2])
        normalized_tokens = F.normalize(patch_tokens.float(), p=2, dim=-1)
        patch_grid_size = (
            images.shape[-2] // aggregator.patch_size,
            images.shape[-1] // aggregator.patch_size,
        )
        frame_representations = pooled_frame_representations(
            patch_tokens.unsqueeze(0),
            patch_grid_size=patch_grid_size,
            pool_size=args.pool_size,
        )[0]
        normalized_frames = F.normalize(frame_representations.float(), p=2, dim=-1)
        selected, unique_count, requested_count = select_frame_fusion_pairs_from_normalized_representations(
            normalized_frames,
            pair_percent=args.pair_percent,
            exclude_frames=(0,),
            disjoint=True,
        )

        candidate_pairs = [(i, j) for i in range(1, len(rgb_paths)) for j in range(i + 1, len(rgb_paths))]
        selected_pairs = [(int(pair.frame_a), int(pair.frame_b)) for pair in selected]
        all_values = pair_values(normalized_tokens, candidate_pairs, chunk_size=args.pair_chunk_size)
        selected_values = pair_values(normalized_tokens, selected_pairs, chunk_size=args.pair_chunk_size)
    seconds = time.perf_counter() - started

    result = {
        "sequence": sequence,
        "manifest": str(manifest),
        "num_frames": len(rgb_paths),
        "patch_grid_size": list(patch_grid_size),
        "patch_tokens_per_frame": int(normalized_tokens.shape[1]),
        "feature_stage": "patch_embed_before_inter_frame_block_0",
        "reference_frame_excluded": 0,
        "pair_percent": args.pair_percent,
        "all_candidate_pairs": len(candidate_pairs),
        "selected_pairs": len(selected_pairs),
        "unique_candidate_pairs": int(unique_count),
        "requested_pairs": int(requested_count),
        "all_frame_pairs": summarize(all_values),
        "top25_disjoint_frame_pairs": summarize(selected_values),
        "seconds": seconds,
    }
    output_path = args.output_dir / "sequences" / f"{slugify(sequence)}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[{sequence}] all_mean={result['all_frame_pairs']['mean']:.6f} "
        f"top25_mean={result['top25_disjoint_frame_pairs']['mean']:.6f} "
        f"seconds={seconds:.1f}",
        flush=True,
    )
    del images, patch_tokens, normalized_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate(output_dir: Path) -> None:
    paths = sorted((output_dir / "sequences").glob("*.json"))
    if not paths:
        raise RuntimeError(f"No per-sequence JSON files under {output_dir / 'sequences'}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows.sort(key=lambda row: row["sequence"])
    fields = [
        "sequence", "num_frames", "all_candidate_pairs", "selected_pairs",
        "all_frame_pairs.mean", "all_frame_pairs.q05", "all_frame_pairs.q20",
        "all_frame_pairs.q50", "all_frame_pairs.q95", "all_frame_pairs.fraction_ge_0.90",
        "all_frame_pairs.fraction_ge_0.95", "top25_disjoint_frame_pairs.mean",
        "top25_disjoint_frame_pairs.q05", "top25_disjoint_frame_pairs.q20",
        "top25_disjoint_frame_pairs.q50", "top25_disjoint_frame_pairs.q95",
        "top25_disjoint_frame_pairs.fraction_ge_0.90",
        "top25_disjoint_frame_pairs.fraction_ge_0.95",
    ]
    def value(row: dict[str, object], field: str) -> object:
        if "." not in field:
            return row[field]
        parent, child = field.split(".", 1)
        return row[parent][child]  # type: ignore[index]

    import csv
    with (output_dir / "sequence_token_similarity_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in rows:
            writer.writerow([value(row, field) for field in fields])

    lines = [
        "# Same-Position Token Similarity by Sequence", "",
        "Feature stage: patch embedding before inter-frame block 0; frame 0 excluded.", "",
        "| sequence | all mean | all q20 | all q50 | all >=0.90 | top25 mean | top25 q20 | top25 q50 | top25 >=0.90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        all_stats = row["all_frame_pairs"]
        top_stats = row["top25_disjoint_frame_pairs"]
        lines.append(
            f"| {row['sequence']} | {all_stats['mean']:.6f} | {all_stats['q20']:.6f} | "
            f"{all_stats['q50']:.6f} | {all_stats['fraction_ge_0.90']:.4f} | "
            f"{top_stats['mean']:.6f} | {top_stats['q20']:.6f} | {top_stats['q50']:.6f} | "
            f"{top_stats['fraction_ge_0.90']:.4f} |"
        )
    (output_dir / "sequence_token_similarity_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = [row["sequence"] for row in rows]
    all_means = [row["all_frame_pairs"]["mean"] for row in rows]
    top_means = [row["top25_disjoint_frame_pairs"]["mean"] for row in rows]
    figure, axis = plt.subplots(figsize=(max(12, len(rows) * 0.55), 6))
    x = np.arange(len(rows))
    axis.bar(x - 0.19, all_means, width=0.38, label="all frame pairs")
    axis.bar(x + 0.19, top_means, width=0.38, label="Top-25% disjoint pairs")
    axis.set_ylabel("mean same-position cosine similarity")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x, labels, rotation=65, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "sequence_token_similarity_means.png", dpi=180)
    plt.close(figure)
    (output_dir / "summary.json").write_text(
        json.dumps({"feature_stage": "patch_embed_before_inter_frame_block_0", "reference_frame_excluded": 0, "sequences": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"aggregated {len(rows)} sequences into {output_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate(args.output_dir)
        return
    if not args.manifest_sequence:
        raise ValueError("--manifest-sequence is required unless --aggregate is used")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    for spec in args.manifest_sequence:
        manifest, sequence = parse_manifest_sequence(spec)
        run_sequence(model, manifest, sequence, args, device)


if __name__ == "__main__":
    main()
