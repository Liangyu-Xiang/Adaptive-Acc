#!/usr/bin/env python3
"""Visualize per-token similarity for representative top frame pairs.

The script runs VGGT-Omega without frame fusion or token swaps, selects high
frame-similarity pairs before decoder heads, remembers only frame-pair indices,
and overlays per-token similarities from cached aggregator layers on the inputs.
"""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval_7scenes_paper as seven_eval  # noqa: E402
import scripts.eval_tum_dynamics_paper as tum_eval  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.models.aggregator import (  # noqa: E402
    pooled_frame_representations,
    select_frame_fusion_pairs,
)
from vggt_omega.utils.frame_pair_selection import (  # noqa: E402
    RankedFramePair,
    ranked_undirected_frame_pairs,
    representative_frame_pairs,
    select_top_percent_disjoint_frame_pairs,
)
from vggt_omega.utils.frame_sampling import SAMPLING_STRATEGIES, sample_record_pools  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
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
class PairRow:
    pair_index: int
    frame_a: int
    frame_b: int
    frame_pair_rank: int
    frame_pair_percentile: float
    representative_bucket: str
    pair_source_layer: int
    token_feature_layer: int
    frame_similarity: float
    patch_similarity_mean: float
    patch_similarity_min: float
    patch_similarity_p05: float
    patch_similarity_p50: float
    patch_similarity_p95: float
    patch_similarity_max: float
    special_similarity_mean: float


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
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--pair-percent",
        type=float,
        default=50.0,
        help="Keep this top percent of ranked frame pairs before redundancy filtering.",
    )
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument(
        "--pair-source-layer",
        type=int,
        default=-1,
        help="Cached aggregator layer used to compute/select frame pairs. -1 means outputs[-1].",
    )
    parser.add_argument(
        "--token-feature-layer",
        type=int,
        default=None,
        help=(
            "Single cached aggregator layer used for per-token similarity. "
            "If omitted, all default cached layers are used."
        ),
    )
    parser.add_argument(
        "--token-feature-layers",
        nargs="+",
        default=None,
        help=(
            "One or more cached aggregator layers for per-token similarity, "
            "e.g. 4 11 17 23 or 4,11,17,23. Overrides --token-feature-layer."
        ),
    )
    parser.add_argument(
        "--selected-pairs-csv",
        type=Path,
        default=None,
        help="Reuse an existing selected_pairs.csv so token similarity is recomputed on the same frame pairs.",
    )
    parser.add_argument("--exclude-reference-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pair-selection",
        choices=(
            "all-undirected-top-percent-greedy-disjoint",
            "nearest-dedup-greedy",
            "all-undirected-top-percent",
        ),
        default="all-undirected-top-percent-greedy-disjoint",
        help=(
            "The default ranks all undirected pairs, keeps the top percent, then "
            "greedily removes pairs that reuse a frame. nearest-dedup-greedy "
            "matches PairFusion pair selection. all-undirected-top-percent keeps "
            "the raw top percent without overlap filtering."
        ),
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--analyze-all-selected-pairs",
        action="store_true",
        help="Compute token similarity for every remembered pair instead of a representative subset.",
    )
    parser.add_argument(
        "--representative-pair-count",
        type=int,
        default=8,
        help="Number of representative remembered pairs to analyze and visualize by default.",
    )
    parser.add_argument(
        "--representative-per-bucket",
        type=int,
        default=2,
        help="Representative pairs to pick near each requested top-similarity bucket.",
    )
    parser.add_argument(
        "--representative-buckets",
        nargs="+",
        type=float,
        default=[1.0, 10.0, 25.0, 50.0],
        help="Top-similarity percentile buckets used for representative pair sampling.",
    )
    parser.add_argument("--max-visualized-pairs", type=int, default=None)
    parser.add_argument("--colormap", default="turbo")
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    return parser.parse_args()


def slugify(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def load_model(checkpoint: Path, device: torch.device, first_frame_token_indices: tuple[int, ...]) -> VGGTOmega:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    model = VGGTOmega(
        global_merging=False,
        merging=None,
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
        pools = {
            sequence_dir.name: tum_eval.load_frame_records(sequence_dir, args.association_tolerance)
            for sequence_dir in sequence_dirs
        }
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


def run_aggregator(
    model: VGGTOmega,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[list[torch.Tensor | None], int, float]:
    model.aggregator.set_frame_fusion(mode="none")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        outputs, patch_token_start = model.aggregator(images.unsqueeze(0))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return outputs, int(patch_token_start), elapsed


def normalize_layer_index(layer: int, depth: int) -> int:
    if layer == -1:
        return depth - 1
    if layer < 0 or layer >= depth:
        raise ValueError(f"Layer index must be -1 or in [0, {depth - 1}], got {layer}")
    return layer


def parse_layer_values(values: Sequence[str | int]) -> list[int]:
    layers: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                layers.append(int(part))
    if not layers:
        raise ValueError("At least one token feature layer must be requested")
    return layers


def resolve_token_feature_layers(args: argparse.Namespace, model: VGGTOmega) -> list[int]:
    if args.token_feature_layers is not None:
        requested = parse_layer_values(args.token_feature_layers)
    elif args.token_feature_layer is not None:
        requested = [int(args.token_feature_layer)]
    else:
        requested = sorted(int(layer) for layer in model.aggregator.cached_layer_indices)

    normalized: list[int] = []
    for layer in requested:
        normalized_layer = normalize_layer_index(layer, model.aggregator.depth)
        if normalized_layer not in normalized:
            normalized.append(normalized_layer)
    return normalized


def cached_tokens_for_layer(
    outputs: Sequence[torch.Tensor | None],
    layer: int,
    *,
    name: str,
) -> tuple[torch.Tensor, int]:
    normalized_layer = normalize_layer_index(layer, len(outputs))
    tokens = outputs[normalized_layer]
    if tokens is None:
        cached_layers = [index for index, value in enumerate(outputs) if value is not None]
        raise ValueError(
            f"{name} layer {layer} resolves to layer {normalized_layer}, but it was not cached. "
            f"Cached layers are {cached_layers}."
        )
    return tokens.detach().float(), normalized_layer


def similarity_matrix_from_tokens(
    tokens: torch.Tensor,
    *,
    patch_token_start: int,
    patch_grid_size: tuple[int, int],
    pool_size: int,
) -> np.ndarray:
    patch_tokens = tokens[:, :, patch_token_start:]
    representations = pooled_frame_representations(
        patch_tokens,
        patch_grid_size=patch_grid_size,
        pool_size=pool_size,
    )
    normalized = F.normalize(representations.float(), p=2, dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2)).clamp(-1.0, 1.0)
    return similarity[0].detach().cpu().numpy().astype(np.float32)


def select_pairs(
    similarity: np.ndarray,
    *,
    pair_percent: float,
    exclude_reference: bool,
    pair_selection: str,
    max_pairs: int | None,
) -> tuple[list[RankedFramePair], int, int]:
    exclude_frames = (0,) if exclude_reference else ()
    if pair_selection == "all-undirected-top-percent-greedy-disjoint":
        return select_top_percent_disjoint_frame_pairs(
            similarity,
            top_percent=pair_percent,
            exclude_frames=exclude_frames,
            max_pairs=max_pairs,
        )
    if pair_selection == "nearest-dedup-greedy":
        pairs, unique_count, requested_count = select_frame_fusion_pairs(
            torch.from_numpy(similarity),
            pair_percent=pair_percent,
            exclude_frames=exclude_frames,
        )
        ranked = ranked_undirected_frame_pairs(similarity, exclude_frames=exclude_frames)
        by_pair = {(pair.frame_a, pair.frame_b): pair for pair in ranked}
        selected = []
        for index, pair in enumerate(pairs):
            key = tuple(sorted((pair.frame_a, pair.frame_b)))
            ranked_pair = by_pair.get(key)
            if ranked_pair is None:
                percentile = 100.0 * float(index + 1) / max(len(pairs), 1)
                ranked_pair = RankedFramePair(
                    frame_a=key[0],
                    frame_b=key[1],
                    similarity=pair.similarity,
                    rank=index + 1,
                    percentile=percentile,
                )
            selected.append(ranked_pair)
    else:
        candidates = ranked_undirected_frame_pairs(similarity, exclude_frames=exclude_frames)
        unique_count = len(candidates)
        requested_count = min(unique_count, max(1, int(math.ceil(unique_count * pair_percent / 100.0))))
        selected = candidates[:requested_count]
    if max_pairs is not None:
        selected = selected[: max(0, int(max_pairs))]
    return selected, unique_count, requested_count


def load_selected_pairs_csv(path: Path) -> list[RankedFramePair]:
    pairs: list[RankedFramePair] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if {"frame_a", "frame_b"}.issubset(fieldnames):
            frame_a_key, frame_b_key = "frame_a", "frame_b"
            similarity_key = "frame_similarity" if "frame_similarity" in fieldnames else "similarity"
        elif {"frame_i", "frame_j"}.issubset(fieldnames):
            frame_a_key, frame_b_key = "frame_i", "frame_j"
            similarity_key = "similarity" if "similarity" in fieldnames else "frame_similarity"
        else:
            raise ValueError(f"{path} must contain frame_a/frame_b or frame_i/frame_j columns")
        required_fields = {frame_a_key, frame_b_key}
        if similarity_key not in fieldnames:
            required_fields.add(similarity_key)
        missing = required_fields.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if "pair_index" in (reader.fieldnames or []):
        rows.sort(key=lambda row: int(row["pair_index"]))
    total = max(len(rows), 1)
    for index, row in enumerate(rows):
        rank = int(row["frame_pair_rank"]) if row.get("frame_pair_rank") else index + 1
        percentile = (
            float(row["frame_pair_percentile"])
            if row.get("frame_pair_percentile")
            else 100.0 * float(rank) / float(total)
        )
        pairs.append(
            RankedFramePair(
                frame_a=int(row[frame_a_key]),
                frame_b=int(row[frame_b_key]),
                similarity=float(row[similarity_key]),
                rank=rank,
                percentile=percentile,
                representative_bucket=row.get("representative_bucket", ""),
            )
        )
    return pairs


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    values: np.ndarray,
    *,
    colormap: str,
    alpha: float,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    heatmap = Image.fromarray(values.astype(np.float32)).resize((width, height), Image.Resampling.BICUBIC)
    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    normalized = np.clip((heatmap_array - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    rgba = plt.get_cmap(colormap)(normalized)
    colored = (rgba[..., :3] * 255.0).astype(np.float32)
    blended = (1.0 - alpha) * image.astype(np.float32) + alpha * colored
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def plot_pair_overlay(
    path: Path,
    *,
    image_a: np.ndarray,
    image_b: np.ndarray,
    patch_similarity: np.ndarray,
    frame_a: int,
    frame_b: int,
    frame_similarity: float,
    patch_mean: float,
    colormap: str,
    alpha: float,
    vmin: float,
    vmax: float,
) -> None:
    overlay_a = overlay_heatmap(
        image_a,
        patch_similarity,
        colormap=colormap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )
    overlay_b = overlay_heatmap(
        image_b,
        patch_similarity,
        colormap=colormap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(overlay_a)
    axes[0].set_title(f"frame {frame_a}")
    image = axes[1].imshow(patch_similarity, cmap=colormap, vmin=vmin, vmax=vmax)
    axes[1].set_title("patch-token cosine")
    axes[2].imshow(overlay_b)
    axes[2].set_title(f"frame {frame_b}")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"pair {frame_a}-{frame_b}: frame sim={frame_similarity:.4f}, patch mean={patch_mean:.4f}")
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="cosine similarity")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_pair_layer_overlays(
    path: Path,
    *,
    image_a: np.ndarray,
    image_b: np.ndarray,
    layer_patch_similarities: Sequence[tuple[int, np.ndarray, float]],
    frame_a: int,
    frame_b: int,
    frame_similarity: float,
    colormap: str,
    alpha: float,
    vmin: float,
    vmax: float,
) -> None:
    if not layer_patch_similarities:
        return

    rows = sorted(layer_patch_similarities, key=lambda item: item[0])
    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12, max(3.2, 3.2 * len(rows))),
        constrained_layout=True,
        squeeze=False,
    )
    heatmap_image = None
    for row_index, (layer, patch_similarity, patch_mean) in enumerate(rows):
        overlay_a = overlay_heatmap(
            image_a,
            patch_similarity,
            colormap=colormap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )
        overlay_b = overlay_heatmap(
            image_b,
            patch_similarity,
            colormap=colormap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )
        axes[row_index, 0].imshow(overlay_a)
        heatmap_image = axes[row_index, 1].imshow(
            patch_similarity,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[row_index, 2].imshow(overlay_b)
        axes[row_index, 0].set_ylabel(f"layer {layer}", rotation=0, labelpad=34, va="center")
        axes[row_index, 1].set_title(f"patch mean={patch_mean:.4f}", fontsize=10)
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])

    axes[0, 0].set_title(f"frame {frame_a}")
    axes[0, 1].set_title(f"patch-token cosine\n{axes[0, 1].get_title()}", fontsize=10)
    axes[0, 2].set_title(f"frame {frame_b}")
    figure.suptitle(f"pair {frame_a}-{frame_b}: frame sim={frame_similarity:.4f}")
    if heatmap_image is not None:
        figure.colorbar(
            heatmap_image,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.02,
            label="cosine similarity",
        )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def token_similarity_rows(
    *,
    pair_index: int,
    pair: RankedFramePair,
    pair_source_layer: int,
    token_feature_layer: int,
    token_similarity: np.ndarray,
    patch_token_start: int,
    patch_grid_size: tuple[int, int],
) -> list[dict[str, object]]:
    patch_h, patch_w = patch_grid_size
    rows: list[dict[str, object]] = []
    for token_offset, value in enumerate(token_similarity):
        if token_offset < patch_token_start:
            token_kind = "special"
            patch_index = ""
            patch_row = ""
            patch_col = ""
        else:
            token_kind = "patch"
            patch_index_int = token_offset - patch_token_start
            patch_index = patch_index_int
            patch_row = patch_index_int // patch_w
            patch_col = patch_index_int % patch_w
        rows.append(
            {
                "pair_index": pair_index,
                "frame_a": pair.frame_a,
                "frame_b": pair.frame_b,
                "frame_pair_rank": pair.rank,
                "frame_pair_percentile": f"{pair.percentile:.8f}",
                "representative_bucket": pair.representative_bucket,
                "pair_source_layer": pair_source_layer,
                "token_feature_layer": token_feature_layer,
                "frame_similarity": f"{pair.similarity:.8f}",
                "token_offset": token_offset,
                "token_kind": token_kind,
                "patch_index": patch_index,
                "patch_row": patch_row,
                "patch_col": patch_col,
                "token_cosine_similarity": f"{float(value):.8f}",
            }
        )
    return rows


def summarize_pair(
    *,
    pair_index: int,
    pair: RankedFramePair,
    pair_source_layer: int,
    token_feature_layer: int,
    token_similarity: np.ndarray,
    patch_token_start: int,
) -> PairRow:
    patch_values = token_similarity[patch_token_start:]
    special_values = token_similarity[:patch_token_start]
    return PairRow(
        pair_index=pair_index,
        frame_a=pair.frame_a,
        frame_b=pair.frame_b,
        frame_pair_rank=pair.rank,
        frame_pair_percentile=pair.percentile,
        representative_bucket=pair.representative_bucket,
        pair_source_layer=pair_source_layer,
        token_feature_layer=token_feature_layer,
        frame_similarity=pair.similarity,
        patch_similarity_mean=float(np.mean(patch_values)),
        patch_similarity_min=float(np.min(patch_values)),
        patch_similarity_p05=float(np.percentile(patch_values, 5)),
        patch_similarity_p50=float(np.percentile(patch_values, 50)),
        patch_similarity_p95=float(np.percentile(patch_values, 95)),
        patch_similarity_max=float(np.max(patch_values)),
        special_similarity_mean=float(np.mean(special_values)) if len(special_values) else float("nan"),
    )


def write_selected_pairs_csv(
    path: Path,
    pairs: Sequence[RankedFramePair],
    records: Sequence[object],
    dataset: str,
) -> None:
    fieldnames = [
        "pair_index",
        "frame_a",
        "frame_b",
        "frame_pair_rank",
        "frame_pair_percentile",
        "representative_bucket",
        "frame_a_label",
        "frame_b_label",
        "frame_a_rgb_path",
        "frame_b_rgb_path",
        "frame_similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair_index, pair in enumerate(pairs):
            record_a = records[pair.frame_a]
            record_b = records[pair.frame_b]
            writer.writerow(
                {
                    "pair_index": pair_index,
                    "frame_a": pair.frame_a,
                    "frame_b": pair.frame_b,
                    "frame_pair_rank": pair.rank,
                    "frame_pair_percentile": f"{pair.percentile:.8f}",
                    "representative_bucket": pair.representative_bucket,
                    "frame_a_label": frame_label(dataset, record_a),
                    "frame_b_label": frame_label(dataset, record_b),
                    "frame_a_rgb_path": str(record_a.rgb_path),
                    "frame_b_rgb_path": str(record_b.rgb_path),
                    "frame_similarity": f"{pair.similarity:.8f}",
                }
            )


def write_pair_summary_csv(path: Path, pairs: Sequence[PairRow], records: Sequence[object], dataset: str) -> None:
    fieldnames = [
        "pair_index",
        "frame_a",
        "frame_b",
        "frame_pair_rank",
        "frame_pair_percentile",
        "representative_bucket",
        "pair_source_layer",
        "token_feature_layer",
        "frame_a_label",
        "frame_b_label",
        "frame_a_rgb_path",
        "frame_b_rgb_path",
        "frame_similarity",
        "patch_similarity_mean",
        "patch_similarity_min",
        "patch_similarity_p05",
        "patch_similarity_p50",
        "patch_similarity_p95",
        "patch_similarity_max",
        "special_similarity_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            record_a = records[pair.frame_a]
            record_b = records[pair.frame_b]
            writer.writerow(
                {
                    "pair_index": pair.pair_index,
                    "frame_a": pair.frame_a,
                    "frame_b": pair.frame_b,
                    "frame_pair_rank": pair.frame_pair_rank,
                    "frame_pair_percentile": f"{pair.frame_pair_percentile:.8f}",
                    "representative_bucket": pair.representative_bucket,
                    "pair_source_layer": pair.pair_source_layer,
                    "token_feature_layer": pair.token_feature_layer,
                    "frame_a_label": frame_label(dataset, record_a),
                    "frame_b_label": frame_label(dataset, record_b),
                    "frame_a_rgb_path": str(record_a.rgb_path),
                    "frame_b_rgb_path": str(record_b.rgb_path),
                    "frame_similarity": f"{pair.frame_similarity:.8f}",
                    "patch_similarity_mean": f"{pair.patch_similarity_mean:.8f}",
                    "patch_similarity_min": f"{pair.patch_similarity_min:.8f}",
                    "patch_similarity_p05": f"{pair.patch_similarity_p05:.8f}",
                    "patch_similarity_p50": f"{pair.patch_similarity_p50:.8f}",
                    "patch_similarity_p95": f"{pair.patch_similarity_p95:.8f}",
                    "patch_similarity_max": f"{pair.patch_similarity_max:.8f}",
                    "special_similarity_mean": f"{pair.special_similarity_mean:.8f}",
                }
            )


def evaluate_sequence(
    *,
    args: argparse.Namespace,
    model: VGGTOmega,
    device: torch.device,
    token_feature_layers: Sequence[int],
    sequence_name: str,
    records: Sequence[object],
) -> dict[str, object]:
    output_dir = args.output_dir / slugify(sequence_name)
    pair_plot_dir = output_dir / "pair_overlays"
    combined_pair_plot_dir = pair_plot_dir / "combined"
    pair_plot_dir.mkdir(parents=True, exist_ok=True)
    combined_pair_plot_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [str(record.rgb_path) for record in records]
    images = load_and_preprocess_images(
        image_paths,
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    ).to(device, non_blocking=True)
    aggregated_tokens, patch_token_start, forward_seconds = run_aggregator(model, images, device)
    pair_source_tokens, pair_source_layer = cached_tokens_for_layer(
        aggregated_tokens,
        args.pair_source_layer,
        name="pair source",
    )
    token_features_by_layer = {
        layer: cached_tokens_for_layer(aggregated_tokens, layer, name="token feature")[0]
        for layer in token_feature_layers
    }
    first_token_feature_layer = token_feature_layers[0]
    first_token_feature_tokens = token_features_by_layer[first_token_feature_layer]
    _, num_frames, num_tokens, hidden_dim = first_token_feature_tokens.shape
    for layer, token_feature_tokens in token_features_by_layer.items():
        if pair_source_tokens.shape[:3] != token_feature_tokens.shape[:3]:
            raise ValueError(
                "Pair source and token feature tensors must share [batch, frames, tokens], got "
                f"{tuple(pair_source_tokens.shape[:3])} and {tuple(token_feature_tokens.shape[:3])} "
                f"for token feature layer {layer}"
            )
    patch_count = num_tokens - patch_token_start
    height, width = images.shape[-2:]
    patch_grid_size = (height // model.aggregator.patch_size, width // model.aggregator.patch_size)
    if patch_count != patch_grid_size[0] * patch_grid_size[1]:
        raise ValueError(
            "Patch count does not match image grid: "
            f"{patch_count} vs {patch_grid_size[0]}x{patch_grid_size[1]}"
        )
    frame_similarity = similarity_matrix_from_tokens(
        pair_source_tokens,
        patch_token_start=patch_token_start,
        patch_grid_size=patch_grid_size,
        pool_size=args.pool_size,
    )
    np.savez_compressed(
        output_dir / "frame_similarity.npz",
        pair_source=frame_similarity,
        pair_source_layer=np.asarray(pair_source_layer, dtype=np.int64),
        token_feature_layers=np.asarray(token_feature_layers, dtype=np.int64),
    )
    if args.selected_pairs_csv is None:
        remembered_pairs, unique_candidate_count, requested_pair_count = select_pairs(
            frame_similarity,
            pair_percent=args.pair_percent,
            exclude_reference=args.exclude_reference_frame,
            pair_selection=args.pair_selection,
            max_pairs=args.max_pairs,
        )
    else:
        remembered_pairs = load_selected_pairs_csv(args.selected_pairs_csv)
        if args.max_pairs is not None:
            remembered_pairs = remembered_pairs[: max(0, int(args.max_pairs))]
        unique_candidate_count = len(remembered_pairs)
        requested_pair_count = len(remembered_pairs)
    if args.analyze_all_selected_pairs:
        analysis_pairs = list(remembered_pairs)
    else:
        analysis_pairs = representative_frame_pairs(
            remembered_pairs,
            buckets=args.representative_buckets,
            per_bucket=args.representative_per_bucket,
            max_pairs=args.representative_pair_count,
        )
    write_selected_pairs_csv(output_dir / "selected_pairs_all.csv", remembered_pairs, records, args.dataset)
    write_selected_pairs_csv(output_dir / "selected_pairs.csv", analysis_pairs, records, args.dataset)
    image_arrays = [tensor_image_to_uint8(images[index]) for index in range(num_frames)]
    token_rows: list[dict[str, object]] = []
    pair_rows: list[PairRow] = []
    max_visualized = len(analysis_pairs) if args.max_visualized_pairs is None else max(0, args.max_visualized_pairs)
    combined_patch_similarities: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for token_feature_layer in token_feature_layers:
        feature_frame_tokens = token_features_by_layer[token_feature_layer][0]
        layer_plot_dir = pair_plot_dir / f"layer_{token_feature_layer:02d}"
        layer_plot_dir.mkdir(parents=True, exist_ok=True)
        for pair_index, pair in enumerate(analysis_pairs):
            token_similarity = F.cosine_similarity(
                feature_frame_tokens[pair.frame_a].float(),
                feature_frame_tokens[pair.frame_b].float(),
                dim=-1,
                eps=1e-8,
            ).detach().cpu().numpy().astype(np.float32)
            pair_row = summarize_pair(
                pair_index=pair_index,
                pair=pair,
                pair_source_layer=pair_source_layer,
                token_feature_layer=token_feature_layer,
                token_similarity=token_similarity,
                patch_token_start=patch_token_start,
            )
            pair_rows.append(pair_row)
            token_rows.extend(
                token_similarity_rows(
                    pair_index=pair_index,
                    pair=pair,
                    pair_source_layer=pair_source_layer,
                    token_feature_layer=token_feature_layer,
                    token_similarity=token_similarity,
                    patch_token_start=patch_token_start,
                    patch_grid_size=patch_grid_size,
                )
            )
            if pair_index < max_visualized:
                patch_similarity = token_similarity[patch_token_start:].reshape(patch_grid_size)
                combined_patch_similarities.setdefault(pair_index, []).append(
                    (token_feature_layer, patch_similarity.copy(), pair_row.patch_similarity_mean)
                )
                plot_pair_overlay(
                    layer_plot_dir / f"pair_{pair_index:03d}_f{pair.frame_a:03d}_f{pair.frame_b:03d}_sim{pair.similarity:.4f}.png",
                    image_a=image_arrays[pair.frame_a],
                    image_b=image_arrays[pair.frame_b],
                    patch_similarity=patch_similarity,
                    frame_a=pair.frame_a,
                    frame_b=pair.frame_b,
                    frame_similarity=pair.similarity,
                    patch_mean=pair_row.patch_similarity_mean,
                    colormap=args.colormap,
                    alpha=args.overlay_alpha,
                    vmin=args.vmin,
                    vmax=args.vmax,
                )
    combined_overlay_count = 0
    for pair_index, layer_patch_similarities in sorted(combined_patch_similarities.items()):
        pair = analysis_pairs[pair_index]
        plot_pair_layer_overlays(
            combined_pair_plot_dir / f"pair_{pair_index:03d}_f{pair.frame_a:03d}_f{pair.frame_b:03d}_sim{pair.similarity:.4f}_layers.png",
            image_a=image_arrays[pair.frame_a],
            image_b=image_arrays[pair.frame_b],
            layer_patch_similarities=layer_patch_similarities,
            frame_a=pair.frame_a,
            frame_b=pair.frame_b,
            frame_similarity=pair.similarity,
            colormap=args.colormap,
            alpha=args.overlay_alpha,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        combined_overlay_count += 1
    write_pair_summary_csv(output_dir / "pair_layer_summaries.csv", pair_rows, records, args.dataset)
    with (output_dir / "token_similarities.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "pair_index",
            "frame_a",
            "frame_b",
            "frame_pair_rank",
            "frame_pair_percentile",
            "representative_bucket",
            "pair_source_layer",
            "token_feature_layer",
            "frame_similarity",
            "token_offset",
            "token_kind",
            "patch_index",
            "patch_row",
            "patch_col",
            "token_cosine_similarity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(token_rows)
    summary = {
        "dataset": args.dataset,
        "sequence": sequence_name,
        "num_frames": int(num_frames),
        "num_tokens": int(num_tokens),
        "patch_token_start": int(patch_token_start),
        "pair_source_layer": int(pair_source_layer),
        "token_feature_layers": [int(layer) for layer in token_feature_layers],
        "patch_tokens": int(patch_count),
        "hidden_dim": int(hidden_dim),
        "patch_grid_size": list(patch_grid_size),
        "pair_percent": args.pair_percent,
        "pair_selection": args.pair_selection,
        "exclude_reference_frame": args.exclude_reference_frame,
        "selected_pairs_csv_input": str(args.selected_pairs_csv) if args.selected_pairs_csv is not None else None,
        "unique_candidate_pairs": unique_candidate_count,
        "requested_pairs": requested_pair_count,
        "remembered_pairs": len(remembered_pairs),
        "analyzed_pairs": len(analysis_pairs),
        "visualized_pairs_per_layer": min(max_visualized, len(analysis_pairs)),
        "combined_overlay_pairs": combined_overlay_count,
        "forward_seconds": forward_seconds,
        "pair_overlay_dir": str(pair_plot_dir),
        "combined_pair_overlay_dir": str(combined_pair_plot_dir),
        "selected_pairs_csv": str(output_dir / "selected_pairs.csv"),
        "selected_pairs_all_csv": str(output_dir / "selected_pairs_all.csv"),
        "pair_layer_summaries_csv": str(output_dir / "pair_layer_summaries.csv"),
        "token_similarities_csv": str(output_dir / "token_similarities.csv"),
        "frame_similarity_npz": str(output_dir / "frame_similarity.npz"),
        "representative_pairs": [
            {
                "pair_index": index,
                "frame_a": pair.frame_a,
                "frame_b": pair.frame_b,
                "frame_pair_rank": pair.rank,
                "frame_pair_percentile": pair.percentile,
                "representative_bucket": pair.representative_bucket,
                "frame_similarity": pair.similarity,
            }
            for index, pair in enumerate(analysis_pairs)
        ],
        "top_pair_layer_summaries": [
            {
                "pair_index": row.pair_index,
                "frame_a": row.frame_a,
                "frame_b": row.frame_b,
                "frame_pair_rank": row.frame_pair_rank,
                "frame_pair_percentile": row.frame_pair_percentile,
                "representative_bucket": row.representative_bucket,
                "pair_source_layer": row.pair_source_layer,
                "token_feature_layer": row.token_feature_layer,
                "frame_similarity": row.frame_similarity,
                "patch_similarity_mean": row.patch_similarity_mean,
                "special_similarity_mean": row.special_similarity_mean,
            }
            for row in pair_rows[:20]
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# {sequence_name} Token Similarity\n\n")
        handle.write(
            f"Pair source layer: {pair_source_layer}; token feature layers: "
            f"{', '.join(str(layer) for layer in token_feature_layers)}.\n\n"
        )
        handle.write(
            f"Remembered {len(remembered_pairs)} pairs from {unique_candidate_count} candidates "
            f"with pair_percent={args.pair_percent:g}, selection={args.pair_selection}; "
            f"analyzed {len(analysis_pairs)} representative pairs.\n\n"
        )
        handle.write("| pair | bucket | rank pct | frames | layer | frame sim | patch mean | special mean | overlay |\n")
        handle.write("| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |\n")
        for row in pair_rows[:20]:
            overlay = (
                pair_plot_dir
                / f"layer_{row.token_feature_layer:02d}"
                / f"pair_{row.pair_index:03d}_f{row.frame_a:03d}_f{row.frame_b:03d}_sim{row.frame_similarity:.4f}.png"
            )
            combined_overlay = (
                combined_pair_plot_dir
                / f"pair_{row.pair_index:03d}_f{row.frame_a:03d}_f{row.frame_b:03d}_sim{row.frame_similarity:.4f}_layers.png"
            )
            handle.write(
                f"| {row.pair_index} | {row.representative_bucket or ''} | "
                f"{row.frame_pair_percentile:.2f} | {row.frame_a}-{row.frame_b} | "
                f"{row.token_feature_layer} | {row.frame_similarity:.4f} | {row.patch_similarity_mean:.4f} | "
                f"{row.special_similarity_mean:.4f} | "
                f"{combined_overlay.name if combined_overlay.exists() else (overlay.name if overlay.exists() else '')} |\n"
            )
    del images, aggregated_tokens, pair_source_tokens, token_features_by_layer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> int:
    args = parse_args()
    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2")
    if not 0.0 < args.pair_percent <= 100.0:
        raise ValueError("--pair-percent must be in (0, 100]")
    if args.pool_size <= 0:
        raise ValueError("--pool-size must be positive")
    if args.selected_pairs_csv is not None and not args.selected_pairs_csv.is_file():
        raise FileNotFoundError(f"--selected-pairs-csv does not exist: {args.selected_pairs_csv}")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1]")
    if args.vmax <= args.vmin:
        raise ValueError("--vmax must be greater than --vmin")
    if args.representative_pair_count < 0:
        raise ValueError("--representative-pair-count must be non-negative")
    if args.representative_per_bucket <= 0:
        raise ValueError("--representative-per-bucket must be positive")
    invalid_buckets = [bucket for bucket in args.representative_buckets if bucket <= 0.0 or bucket > 100.0]
    if invalid_buckets:
        raise ValueError(f"--representative-buckets must be in (0, 100], got {invalid_buckets}")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampled, sampled_indices = select_records(args)
    first_frame_token_indices = resolve_first_frame_token_indices(
        args.first_frame_token_indices,
        args.num_frames,
    )
    print(f"Loading {args.checkpoint}")
    model = load_model(args.checkpoint, device, first_frame_token_indices)
    token_feature_layers = resolve_token_feature_layers(args, model)
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
        "pair_percent": args.pair_percent,
        "pool_size": args.pool_size,
        "pair_source_layer": args.pair_source_layer,
        "token_feature_layer": args.token_feature_layer,
        "token_feature_layers": token_feature_layers,
        "selected_pairs_csv": str(args.selected_pairs_csv) if args.selected_pairs_csv is not None else None,
        "pair_selection": args.pair_selection,
        "analyze_all_selected_pairs": args.analyze_all_selected_pairs,
        "representative_pair_count": args.representative_pair_count,
        "representative_per_bucket": args.representative_per_bucket,
        "representative_buckets": args.representative_buckets,
        "exclude_reference_frame": args.exclude_reference_frame,
        "sampled_indices_before_reference_reorder": sampled_indices,
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    summaries = []
    for sequence_name, records in sampled.items():
        print(f"[{sequence_name}] {len(records)} frames")
        summaries.append(
            evaluate_sequence(
                args=args,
                model=model,
                device=device,
                token_feature_layers=token_feature_layers,
                sequence_name=sequence_name,
                records=records,
            )
        )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "sequences": summaries}, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
