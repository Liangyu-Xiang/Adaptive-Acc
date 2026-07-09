#!/usr/bin/env python3
"""Visualize averaged intra-frame attention matrices for VGGT-Omega.

This mirrors ``visualize_global_attention_matrices.py`` but hooks the per-frame
attention blocks. For each dataset, it samples the same kind of 10-frame clips
from 3 sequences, recomputes head-mean attention probabilities inside each
frame block, averages over frames and selected sequences, and saves one heatmap
per selected layer.

The saved matrix orientation is rows=key tokens and columns=query tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_attention_information_flow import qkv_from_block
from tools.analyze_token_evolution import load_model
from tools.visualize_global_attention_matrices import (
    DEFAULT_7SCENES_ROOT,
    DEFAULT_CHECKPOINT,
    DEFAULT_TUM_ROOT,
    DatasetSelection,
    choose_7scenes,
    choose_tum,
    make_bin_index_from_ranks,
)
from vggt_omega.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seven-scenes-root", type=Path, default=DEFAULT_7SCENES_ROOT)
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_TUM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/intra_frame_attention_matrices_10f_3seq"))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--num-sequences", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--query-chunk", type=int, default=1024)
    parser.add_argument("--display-bins", type=int, default=1024)
    parser.add_argument(
        "--layers",
        default="global",
        help=(
            "'global' visualizes frame blocks whose same-index inter-frame block is global; "
            "'all' visualizes every frame block; otherwise use comma-separated 0-based "
            "indices/ranges such as '12-17,23'."
        ),
    )
    parser.add_argument("--datasets", nargs="+", choices=("7scenes", "tum_dynamics"), default=("7scenes", "tum_dynamics"))
    return parser.parse_args()


def parse_layer_spec(spec: str, model) -> list[int]:
    if spec == "global":
        return [
            layer
            for layer, kind in enumerate(model.aggregator.inter_frame_attention_types)
            if kind == "global"
        ]
    if spec == "all":
        return list(range(model.aggregator.depth))

    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid layer range {part!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= model.aggregator.depth)
    if invalid:
        raise ValueError(f"Layer indices out of range 0..{model.aggregator.depth - 1}: {invalid}")
    return sorted(layers)


@dataclass
class FrameLayout:
    token_count: int
    patch_token_start: int
    patch_grid_size: tuple[int, int]
    display_bins: int
    edges: np.ndarray
    counts: np.ndarray
    full_display_bins: int
    full_edges: np.ndarray
    full_valid_pair_counts: np.ndarray


class DownsampledIntraFrameAttentionCollector:
    def __init__(
        self,
        model,
        layers: list[int],
        num_frames: int,
        query_chunk: int,
        display_bins: int,
        patch_grid_size: tuple[int, int],
    ) -> None:
        self.aggregator = model.aggregator
        self.layers = layers
        self.num_frames = num_frames
        self.query_chunk = query_chunk
        self.display_bins_requested = display_bins
        self.patch_grid_size = patch_grid_size
        self.handles = []
        self.sequence_results: dict[int, np.ndarray] = {}
        self.sequence_frame_marked_results: dict[int, np.ndarray] = {}
        self.layout: FrameLayout | None = None

    def __enter__(self):
        for layer in self.layers:
            self.handles.append(
                self.aggregator.frame_blocks[layer].register_forward_pre_hook(
                    self._hook(layer)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _ensure_layout(self, token_count: int, device: torch.device) -> torch.Tensor:
        if self.layout is None:
            ranks = np.arange(token_count, dtype=np.int64)
            bin_index, edges, counts = make_bin_index_from_ranks(
                ranks,
                self.display_bins_requested,
                device,
            )
            full_token_count = self.num_frames * token_count
            full_ranks = np.arange(full_token_count, dtype=np.int64)
            full_bin_index, full_edges, _ = make_bin_index_from_ranks(
                full_ranks,
                self.display_bins_requested,
                device,
            )
            full_bin_index_np = full_bin_index.detach().cpu().numpy()
            full_valid_pair_counts = np.zeros((len(full_edges) - 1, len(full_edges) - 1), dtype=np.float64)
            for frame_idx in range(self.num_frames):
                start = frame_idx * token_count
                end = start + token_count
                frame_counts = np.bincount(
                    full_bin_index_np[start:end],
                    minlength=len(full_edges) - 1,
                ).astype(np.float64)
                full_valid_pair_counts += frame_counts[:, None] * frame_counts[None, :]
            self.layout = FrameLayout(
                token_count=token_count,
                patch_token_start=self.aggregator.patch_token_start,
                patch_grid_size=self.patch_grid_size,
                display_bins=len(counts),
                edges=edges,
                counts=counts,
                full_display_bins=len(full_edges) - 1,
                full_edges=full_edges,
                full_valid_pair_counts=full_valid_pair_counts,
            )
            self._bin_index = bin_index
            self._full_bin_index = full_bin_index
        elif self.layout.token_count != token_count:
            raise RuntimeError(f"Token count changed from {self.layout.token_count} to {token_count}")
        return self._bin_index

    def _hook(self, layer: int):
        def hook(block, inputs):
            x = inputs[0]
            rope = inputs[1] if len(inputs) > 1 else None
            batch_frames, token_count, _ = x.shape
            bin_index = self._ensure_layout(token_count, x.device)
            assert self.layout is not None

            q, k, _ = qkv_from_block(block, x, rope)
            key_t = k.float().transpose(-2, -1)
            layer_sum = torch.zeros(
                (self.layout.display_bins, self.layout.display_bins),
                device=x.device,
                dtype=torch.float32,
            )
            frame_marked_sum = torch.zeros(
                (self.layout.full_display_bins, self.layout.full_display_bins),
                device=x.device,
                dtype=torch.float32,
            )
            for start in range(0, token_count, self.query_chunk):
                end = min(start + self.query_chunk, token_count)
                logits = torch.matmul(q[:, :, start:end].float(), key_t) * block.attn.scale
                probabilities_by_frame = logits.softmax(dim=-1).mean(dim=1)  # [batch_frame, query, key]
                probabilities = probabilities_by_frame.mean(dim=0)  # [query, key]
                key_query = probabilities.transpose(0, 1).contiguous()  # [key, query]
                key_reduced = torch.zeros(
                    (self.layout.display_bins, end - start),
                    device=x.device,
                    dtype=torch.float32,
                )
                key_reduced.scatter_add_(
                    0,
                    bin_index[:, None].expand(-1, end - start),
                    key_query,
                )
                query_bins = bin_index[start:end]
                layer_sum.scatter_add_(
                    1,
                    query_bins[None, :].expand(self.layout.display_bins, -1),
                    key_reduced,
                )
                for batch_frame in range(batch_frames):
                    frame_idx = batch_frame % self.num_frames
                    frame_start = frame_idx * token_count
                    full_key_bins = self._full_bin_index[frame_start : frame_start + token_count]
                    full_query_bins = self._full_bin_index[frame_start + start : frame_start + end]
                    frame_key_query = probabilities_by_frame[batch_frame].transpose(0, 1).contiguous()
                    frame_key_reduced = torch.zeros(
                        (self.layout.full_display_bins, end - start),
                        device=x.device,
                        dtype=torch.float32,
                    )
                    frame_key_reduced.scatter_add_(
                        0,
                        full_key_bins[:, None].expand(-1, end - start),
                        frame_key_query,
                    )
                    frame_marked_sum.scatter_add_(
                        1,
                        full_query_bins[None, :].expand(self.layout.full_display_bins, -1),
                        frame_key_reduced,
                    )
                    del frame_key_query, frame_key_reduced
                del logits, probabilities_by_frame, probabilities, key_query, key_reduced
            self.sequence_results[layer] = layer_sum.cpu().numpy()
            self.sequence_frame_marked_results[layer] = frame_marked_sum.cpu().numpy()
            del q, k, key_t, layer_sum, frame_marked_sum

        return hook


def bin_for_token(edges: np.ndarray, token_index: int) -> int:
    return int(np.searchsorted(edges, token_index, side="left"))


def plot_layer_matrix(
    output_path: Path,
    matrix: np.ndarray,
    layout: FrameLayout,
    layer: int,
    dataset: str,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.percentile(finite, 99.5))
    vmin = 0.0
    figure, axis = plt.subplots(figsize=(8.5, 8))
    image = axis.imshow(matrix, cmap="magma", origin="upper", vmin=vmin, vmax=vmax, aspect="auto")

    patch_start = layout.patch_token_start
    camera_end = bin_for_token(layout.edges, 1)
    patch_boundary = bin_for_token(layout.edges, patch_start)
    for boundary, color, width, alpha in (
        (camera_end, "lime", 0.6, 0.75),
        (patch_boundary, "cyan", 0.8, 0.85),
    ):
        axis.axvline(boundary - 0.5, color=color, linewidth=width, alpha=alpha)
        axis.axhline(boundary - 0.5, color=color, linewidth=width, alpha=alpha)

    patch_h, patch_w = layout.patch_grid_size
    for row in range(1, patch_h):
        boundary = bin_for_token(layout.edges, patch_start + row * patch_w)
        axis.axvline(boundary - 0.5, color="white", linewidth=0.2, alpha=0.2)
        axis.axhline(boundary - 0.5, color="white", linewidth=0.2, alpha=0.2)

    token_centers = [
        (0.0, "cam"),
        ((camera_end + patch_boundary - 1) / 2, "reg"),
        ((patch_boundary + layout.display_bins - 1) / 2, "patch"),
    ]
    axis.set_xticks([center for center, _ in token_centers])
    axis.set_xticklabels([label for _, label in token_centers], rotation=90)
    axis.set_yticks([center for center, _ in token_centers])
    axis.set_yticklabels([label for _, label in token_centers])
    axis.set_xlabel("Query token index within one frame (columns; downsampled)")
    axis.set_ylabel("Key token index within one frame (rows; downsampled)")
    axis.set_title(f"{dataset}: intra-frame block {layer:02d}")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Mean attention probability")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_frame_marked_layer_matrix(
    output_path: Path,
    matrix: np.ndarray,
    layout: FrameLayout,
    num_frames: int,
    layer: int,
    dataset: str,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.percentile(finite, 99.5))
    vmin = 0.0
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color=(0.92, 0.92, 0.92, 1.0))
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    token_count = layout.token_count
    frame_centers: list[float] = []
    frame_labels: list[str] = []
    for frame_idx in range(1, num_frames):
        token_boundary = frame_idx * token_count
        bin_boundary = int(np.searchsorted(layout.full_edges, token_boundary, side="left"))
        axis.axvline(bin_boundary - 0.5, color="cyan", linewidth=0.35, alpha=0.75)
        axis.axhline(bin_boundary - 0.5, color="cyan", linewidth=0.35, alpha=0.75)
    for frame_idx in range(num_frames):
        start = int(np.searchsorted(layout.full_edges, frame_idx * token_count, side="left"))
        end = int(np.searchsorted(layout.full_edges, (frame_idx + 1) * token_count, side="left"))
        frame_centers.append((start + end - 1) / 2)
        frame_labels.append(f"F{frame_idx}")

    axis.set_xticks(frame_centers)
    axis.set_xticklabels(frame_labels, rotation=90)
    axis.set_yticks(frame_centers)
    axis.set_yticklabels(frame_labels)
    axis.set_xlabel("Query token index (columns; frame-marked)")
    axis.set_ylabel("Key token index (rows; frame-marked)")
    axis.set_title(f"{dataset}: intra-frame block {layer:02d}, frames marked")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Mean attention probability")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_dataset(args: argparse.Namespace, model, selection: DatasetSelection, layers: list[int]) -> None:
    output_dir = args.output_dir / selection.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulators: dict[int, np.ndarray] = {}
    frame_marked_accumulators: dict[int, np.ndarray] = {}
    layout: FrameLayout | None = None
    patch_grid_size: tuple[int, int] | None = None

    for sequence_name in selection.sequences:
        image_tensor = load_and_preprocess_images(
            selection.image_paths[sequence_name],
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(args.device, non_blocking=True)
        current_grid = (
            image_tensor.shape[-2] // model.aggregator.patch_size,
            image_tensor.shape[-1] // model.aggregator.patch_size,
        )
        if patch_grid_size is None:
            patch_grid_size = current_grid
        elif patch_grid_size != current_grid:
            raise RuntimeError(f"Patch grid changed from {patch_grid_size} to {current_grid}")

        with DownsampledIntraFrameAttentionCollector(
            model,
            layers,
            args.num_frames,
            args.query_chunk,
            args.display_bins,
            patch_grid_size,
        ) as collector:
            with torch.inference_mode():
                _ = model(image_tensor)
        if layout is None:
            assert collector.layout is not None
            layout = collector.layout
        elif collector.layout is None or collector.layout.token_count != layout.token_count:
            raise RuntimeError("Collector did not produce a compatible layout")
        for layer, matrix_sum in collector.sequence_results.items():
            accumulators.setdefault(layer, np.zeros_like(matrix_sum, dtype=np.float64))
            accumulators[layer] += matrix_sum.astype(np.float64)
        for layer, matrix_sum in collector.sequence_frame_marked_results.items():
            frame_marked_accumulators.setdefault(layer, np.zeros_like(matrix_sum, dtype=np.float64))
            frame_marked_accumulators[layer] += matrix_sum.astype(np.float64)
        del image_tensor
        torch.cuda.empty_cache()
        print(f"{selection.dataset}: collected {sequence_name}", flush=True)

    assert layout is not None
    counts = layout.counts.astype(np.float64)
    denominator = counts[:, None] * counts[None, :] * len(selection.sequences)
    frame_marked_denominator = layout.full_valid_pair_counts * len(selection.sequences)
    averaged: dict[str, np.ndarray] = {}
    frame_marked_averaged: dict[str, np.ndarray] = {}
    for layer in sorted(accumulators):
        matrix = (accumulators[layer] / denominator).astype(np.float32)
        averaged[f"layer_{layer:02d}"] = matrix
        plot_layer_matrix(
            output_dir / f"intra_frame_attention_L{layer:02d}.png",
            matrix,
            layout,
            layer,
            selection.dataset,
        )
        frame_marked = np.full_like(frame_marked_accumulators[layer], np.nan, dtype=np.float64)
        valid = frame_marked_denominator > 0
        frame_marked[valid] = frame_marked_accumulators[layer][valid] / frame_marked_denominator[valid]
        frame_marked = frame_marked.astype(np.float32)
        frame_marked_averaged[f"layer_{layer:02d}"] = frame_marked
        plot_frame_marked_layer_matrix(
            output_dir / f"intra_frame_attention_L{layer:02d}_frame_marked.png",
            frame_marked,
            layout,
            args.num_frames,
            layer,
            selection.dataset,
        )

    np.savez_compressed(output_dir / "downsampled_intra_frame_attention_matrices.npz", **averaged)
    np.savez_compressed(
        output_dir / "downsampled_intra_frame_attention_frame_marked_matrices.npz",
        **frame_marked_averaged,
    )
    metadata = {
        "dataset": selection.dataset,
        "sequences": selection.sequences,
        "sampled_indices": selection.sampled_indices,
        "image_paths": selection.image_paths,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "resize_mode": args.resize_mode,
        "image_resolution": args.image_resolution,
        "matrix_orientation": "rows are key-token bins, columns are query-token bins",
        "attention": "softmax(QK^T/sqrt(d)); averaged over heads, frames, and selected sequences",
        "frame_marked_attention": (
            "Frame-marked matrices place each frame's intra-frame attention in the corresponding "
            "diagonal block; off-frame blocks are NaN because frame attention has no cross-frame keys."
        ),
        "selected_layers": layers,
        "layer_selection": args.layers,
        "token_count": layout.token_count,
        "patch_token_start": layout.patch_token_start,
        "patch_grid_size": list(layout.patch_grid_size),
        "display_bins": layout.display_bins,
        "edges": layout.edges.tolist(),
        "counts": layout.counts.tolist(),
        "full_display_bins": layout.full_display_bins,
        "full_edges": layout.full_edges.tolist(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_index(output_dir: Path, datasets: list[str]) -> None:
    lines = ["# Intra-frame attention matrix visualizations", ""]
    for dataset in datasets:
        metadata_path = output_dir / dataset / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- sequences: {', '.join(metadata['sequences'])}",
                f"- orientation: {metadata['matrix_orientation']}",
                f"- layer selection: {metadata['layer_selection']}",
                "",
            ]
        )
        for layer in metadata["selected_layers"]:
            image = f"{dataset}/intra_frame_attention_L{layer:02d}.png"
            frame_marked = f"{dataset}/intra_frame_attention_L{layer:02d}_frame_marked.png"
            lines.extend(
                [
                    f"### Frame block {layer:02d}",
                    "",
                    f"![{image}]({image})",
                    "",
                    f"![{frame_marked}]({frame_marked})",
                    "",
                ]
            )
    (output_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.device = device
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    layers = parse_layer_spec(args.layers, model)
    selections = []
    if "7scenes" in args.datasets:
        selections.append(choose_7scenes(args))
    if "tum_dynamics" in args.datasets:
        selections.append(choose_tum(args))
    for selection in selections:
        run_dataset(args, model, selection, layers)
    write_index(args.output_dir, [selection.dataset for selection in selections])
    print(f"Saved intra-frame attention visualizations to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
