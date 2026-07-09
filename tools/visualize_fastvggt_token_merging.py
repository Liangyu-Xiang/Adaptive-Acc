#!/usr/bin/env python3
"""Visualize FastVGGT/VGGT-Omega bipartite token merging on image patch grids.

The script runs one inference pass with token merging enabled, records the
source->destination token indices selected by each global inter-frame attention
block, and maps patch-token merge status back onto the resized input images.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_7scenes_paper import load_frame_records, load_model, sample_records
from vggt_omega.utils.load_fn import (
    _balanced_target_shape,
    _crop_to_supported_aspect_ratio,
    _load_rgb_image,
    _max_size_target_shape,
    load_and_preprocess_images,
)


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_DATA_ROOT = Path("/data/mmc_lyxiang/dataset/7scenes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sequence", default="chess/seq-03")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fastvggt_token_merging_visualization"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--merge-ratio", type=float, default=0.9)
    parser.add_argument("--merging-start-layer", type=int, default=0)
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="Global inter-frame layers to visualize. Default: all layers that actually merge.",
    )
    parser.add_argument("--max-arrow-pairs", type=int, default=250)
    return parser.parse_args()


def preprocess_display_image(path: Path, mode: str, image_resolution: int, patch_size: int) -> Image.Image:
    image = _crop_to_supported_aspect_ratio(_load_rgb_image(path))
    width, height = image.size
    aspect_ratio = height / max(width, 1)
    if mode == "balanced":
        target_h, target_w = _balanced_target_shape(aspect_ratio, image_resolution, patch_size)
    else:
        target_h, target_w = _max_size_target_shape(aspect_ratio, image_resolution, patch_size)
    return image.resize((target_w, target_h), Image.Resampling.BICUBIC)


def token_info(index: int, tokens_per_frame: int, patch_start: int, patch_count: int) -> dict[str, int | str]:
    frame = index // tokens_per_frame
    within = index % tokens_per_frame
    if within == 0:
        return {"type": "camera", "frame": frame, "patch": -1, "row": -1, "col": -1}
    if within < patch_start:
        return {"type": "register", "frame": frame, "patch": -1, "row": -1, "col": -1}
    patch = within - patch_start
    return {"type": "patch", "frame": frame, "patch": patch, "row": -1, "col": -1}


def patch_global_index(frame: int, patch: int, tokens_per_frame: int, patch_start: int) -> int:
    return frame * tokens_per_frame + patch_start + patch


def write_layer_csv(
    path: Path,
    sources: np.ndarray,
    destinations: np.ndarray,
    tokens_per_frame: int,
    patch_start: int,
    grid_h: int,
    grid_w: int,
) -> None:
    patch_count = grid_h * grid_w
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "source_index",
            "source_frame",
            "source_patch",
            "source_row",
            "source_col",
            "destination_index",
            "destination_type",
            "destination_frame",
            "destination_patch",
            "destination_row",
            "destination_col",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for src, dst in zip(sources.tolist(), destinations.tolist()):
            src_info = token_info(src, tokens_per_frame, patch_start, patch_count)
            dst_info = token_info(dst, tokens_per_frame, patch_start, patch_count)
            if src_info["type"] != "patch":
                continue
            src_patch = int(src_info["patch"])
            row = {
                "source_index": src,
                "source_frame": src_info["frame"],
                "source_patch": src_patch,
                "source_row": src_patch // grid_w,
                "source_col": src_patch % grid_w,
                "destination_index": dst,
                "destination_type": dst_info["type"],
                "destination_frame": dst_info["frame"],
                "destination_patch": dst_info["patch"],
                "destination_row": -1,
                "destination_col": -1,
            }
            if dst_info["type"] == "patch":
                dst_patch = int(dst_info["patch"])
                row["destination_row"] = dst_patch // grid_w
                row["destination_col"] = dst_patch % grid_w
            writer.writerow(row)


def make_status_maps(
    sources: np.ndarray,
    destinations: np.ndarray,
    num_frames: int,
    tokens_per_frame: int,
    patch_start: int,
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return status grid: 0=unmerged, 1=destination patch, 2=merged source."""
    patch_count = grid_h * grid_w
    status = np.zeros((num_frames, patch_count), dtype=np.uint8)
    incoming_patch_destinations: set[int] = set()
    for dst in destinations.tolist():
        info = token_info(int(dst), tokens_per_frame, patch_start, patch_count)
        if info["type"] == "patch":
            incoming_patch_destinations.add(int(dst))
    for idx in incoming_patch_destinations:
        frame = idx // tokens_per_frame
        patch = idx % tokens_per_frame - patch_start
        if 0 <= frame < num_frames and 0 <= patch < patch_count:
            status[frame, patch] = 1
    for src in sources.tolist():
        info = token_info(int(src), tokens_per_frame, patch_start, patch_count)
        if info["type"] == "patch":
            frame = int(info["frame"])
            patch = int(info["patch"])
            if 0 <= frame < num_frames and 0 <= patch < patch_count:
                status[frame, patch] = 2
    pairs = [(int(src), int(dst)) for src, dst in zip(sources.tolist(), destinations.tolist())]
    return status.reshape(num_frames, grid_h, grid_w), pairs


def plot_status_montage(
    output: Path,
    layer: int,
    images: Sequence[Image.Image],
    status: np.ndarray,
) -> None:
    num_frames, grid_h, grid_w = status.shape
    cols = min(5, num_frames)
    rows = int(np.ceil(num_frames / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows), dpi=160, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    colors = {
        0: np.array([0.15, 0.35, 1.0, 0.34]),  # unmerged
        1: np.array([0.1, 0.85, 0.25, 0.58]),  # destination
        2: np.array([1.0, 0.12, 0.08, 0.62]),  # merged source
    }
    for frame, ax in enumerate(axes[:num_frames]):
        image = images[frame]
        ax.imshow(image)
        overlay = np.zeros((grid_h, grid_w, 4), dtype=np.float32)
        for key, value in colors.items():
            overlay[status[frame] == key] = value
        overlay = np.kron(
            overlay,
            np.ones((image.height // grid_h, image.width // grid_w, 1), dtype=np.float32),
        )
        overlay = overlay[: image.height, : image.width]
        ax.imshow(overlay)
        ax.set_title(f"frame {frame}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[num_frames:]:
        ax.axis("off")
    fig.suptitle(
        f"FastVGGT token merging L{layer:02d}: blue=unmerged, green=kept destination, red=merged source",
        fontsize=13,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_arrow_sample(
    output: Path,
    layer: int,
    images: Sequence[Image.Image],
    pairs: Sequence[tuple[int, int]],
    num_frames: int,
    tokens_per_frame: int,
    patch_start: int,
    grid_h: int,
    grid_w: int,
    max_pairs: int,
) -> None:
    patch_count = grid_h * grid_w
    patch_pairs = []
    for src, dst in pairs:
        src_info = token_info(src, tokens_per_frame, patch_start, patch_count)
        dst_info = token_info(dst, tokens_per_frame, patch_start, patch_count)
        if src_info["type"] == "patch" and dst_info["type"] == "patch":
            patch_pairs.append((src_info, dst_info))
    if not patch_pairs:
        return
    rng = np.random.default_rng(0)
    if len(patch_pairs) > max_pairs:
        chosen = rng.choice(len(patch_pairs), size=max_pairs, replace=False)
        patch_pairs = [patch_pairs[int(i)] for i in chosen]

    cols = min(5, num_frames)
    rows = int(np.ceil(num_frames / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows), dpi=160, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for frame, ax in enumerate(axes[:num_frames]):
        ax.imshow(images[frame])
        ax.set_title(f"frame {frame}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[num_frames:]:
        ax.axis("off")

    for src_info, dst_info in patch_pairs:
        sf, df = int(src_info["frame"]), int(dst_info["frame"])
        if not (0 <= sf < num_frames and 0 <= df < num_frames):
            continue
        # Draw within the source frame only if destination is in the same frame;
        # cross-frame links are marked by source red point and destination green point.
        for info, color, marker in ((src_info, "red", "x"), (dst_info, "lime", "o")):
            f = int(info["frame"])
            patch = int(info["patch"])
            r, c = divmod(patch, grid_w)
            image = images[f]
            x = (c + 0.5) * image.width / grid_w
            y = (r + 0.5) * image.height / grid_h
            axes[f].scatter([x], [y], s=14, c=color, marker=marker, linewidths=0.6)
        if sf == df:
            s_patch = int(src_info["patch"])
            d_patch = int(dst_info["patch"])
            sr, sc = divmod(s_patch, grid_w)
            dr, dc = divmod(d_patch, grid_w)
            image = images[sf]
            sx = (sc + 0.5) * image.width / grid_w
            sy = (sr + 0.5) * image.height / grid_h
            dx = (dc + 0.5) * image.width / grid_w
            dy = (dr + 0.5) * image.height / grid_h
            axes[sf].annotate("", xy=(dx, dy), xytext=(sx, sy), arrowprops={"arrowstyle": "->", "color": "yellow", "lw": 0.5, "alpha": 0.55})

    fig.suptitle(f"Sampled source→destination patch pairs L{layer:02d}: red source, green destination", fontsize=13)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.merge_ratio <= 1.0:
        raise ValueError("--merge-ratio must be in (0, 1] for merge visualization")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    sequence_dir = args.data_root / args.sequence
    records = load_frame_records(sequence_dir)
    sampled, sampled_indices = sample_records({args.sequence: records}, args.num_frames, args.seed)
    records = sampled[args.sequence]
    image_paths = [record.rgb_path for record in records]
    display_images = [
        preprocess_display_image(path, args.resize_mode, args.image_resolution, patch_size=16)
        for path in image_paths
    ]
    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    ).to(device, non_blocking=True)
    grid_h = images.shape[-2] // 16
    grid_w = images.shape[-1] // 16

    model = load_model(
        args.checkpoint,
        device,
        merge_ratio=args.merge_ratio,
        sparse_attention=False,
        sparse_ratio=None,
        sparse_cdf_threshold=None,
        sparse_pool_mode="avg",
    )
    model.aggregator.merging = args.merging_start_layer
    for block in model.aggregator.inter_frame_blocks:
        block.attn.record_merge_trace = True
    model.eval()

    with torch.inference_mode():
        _ = model(images)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_start = model.aggregator.patch_token_start
    patch_count = grid_h * grid_w
    tokens_per_frame = patch_start + patch_count
    available_layers = []
    for layer, kind in enumerate(model.aggregator.inter_frame_attention_types):
        if kind != "global":
            continue
        attn = model.aggregator.inter_frame_blocks[layer].attn
        if hasattr(attn, "last_merge_source_indices") and hasattr(attn, "last_merge_destination_indices"):
            available_layers.append(layer)
    target_layers = args.layers if args.layers is not None else available_layers

    summary_rows = []
    for layer in target_layers:
        attn = model.aggregator.inter_frame_blocks[layer].attn
        if not (hasattr(attn, "last_merge_source_indices") and hasattr(attn, "last_merge_destination_indices")):
            print(f"Layer {layer}: no merge trace, skipped")
            continue
        sources = attn.last_merge_source_indices.numpy().reshape(-1)
        destinations = attn.last_merge_destination_indices.numpy().reshape(-1)
        status, pairs = make_status_maps(
            sources,
            destinations,
            args.num_frames,
            tokens_per_frame,
            patch_start,
            grid_h,
            grid_w,
        )
        write_layer_csv(
            args.output_dir / f"merge_pairs_L{layer:02d}.csv",
            sources,
            destinations,
            tokens_per_frame,
            patch_start,
            grid_h,
            grid_w,
        )
        np.savez_compressed(
            args.output_dir / f"merge_status_L{layer:02d}.npz",
            status=status,
            sources=sources,
            destinations=destinations,
        )
        plot_status_montage(
            args.output_dir / f"merge_status_on_images_L{layer:02d}.png",
            layer,
            display_images,
            status,
        )
        plot_arrow_sample(
            args.output_dir / f"merge_pairs_sample_L{layer:02d}.png",
            layer,
            display_images,
            pairs,
            args.num_frames,
            tokens_per_frame,
            patch_start,
            grid_h,
            grid_w,
            args.max_arrow_pairs,
        )
        merged_patch_count = int((status == 2).sum())
        destination_patch_count = int((status == 1).sum())
        unmerged_plain_count = int((status == 0).sum())
        summary_rows.append(
            {
                "layer": layer,
                "merged_patch_tokens": merged_patch_count,
                "destination_patch_tokens": destination_patch_count,
                "plain_unmerged_patch_tokens": unmerged_plain_count,
                "total_patch_tokens": int(status.size),
                "source_tokens_recorded": int(sources.size),
            }
        )

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "layer",
            "merged_patch_tokens",
            "destination_patch_tokens",
            "plain_unmerged_patch_tokens",
            "total_patch_tokens",
            "source_tokens_recorded",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "sequence": args.sequence,
        "sampled_indices": sampled_indices[args.sequence],
        "image_paths": [str(path) for path in image_paths],
        "num_frames": args.num_frames,
        "merge_ratio": args.merge_ratio,
        "merging_start_layer": args.merging_start_layer,
        "patch_grid": [grid_h, grid_w],
        "patch_token_start": patch_start,
        "tokens_per_frame": tokens_per_frame,
        "available_merge_layers": available_layers,
        "visualized_layers": [row["layer"] for row in summary_rows],
        "legend": {
            "blue": "patch token not selected as merged source",
            "green": "patch token kept and used as destination for at least one merged source",
            "red": "patch token selected as source and merged into another token",
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved FastVGGT token merging visualizations to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
