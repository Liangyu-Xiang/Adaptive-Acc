#!/usr/bin/env python3
"""Visualize averaged global-attention matrices for VGGT-Omega.

For each dataset, this script samples 10 frames from 3 sequences, recomputes
head-mean attention probabilities at every global inter-frame block, averages
the matrices over the 3 sequences, and saves one heatmap per global block.

The saved matrix orientation is rows=key tokens and columns=query tokens.
"""

from __future__ import annotations

import argparse
import importlib
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
from vggt_omega.utils.load_fn import load_and_preprocess_images


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_7SCENES_ROOT = Path("/data/mmc_lyxiang/dataset/7scenes")
DEFAULT_TUM_ROOT = Path("/data/mmc_lyxiang/dataset/TUM-Dynamics")


@dataclass
class DatasetSelection:
    dataset: str
    sequences: list[str]
    image_paths: dict[str, list[str]]
    sampled_indices: dict[str, list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seven-scenes-root", type=Path, default=DEFAULT_7SCENES_ROOT)
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_TUM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/global_attention_matrices_10f_3seq"))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--num-sequences", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--query-chunk", type=int, default=64)
    parser.add_argument("--display-bins", type=int, default=1024)
    parser.add_argument(
        "--token-order",
        choices=("original", "special-first"),
        default="original",
        help=(
            "original keeps frame0 all tokens, frame1 all tokens, ...; "
            "special-first puts all frames' camera/register tokens first, then patch tokens by frame."
        ),
    )
    parser.add_argument("--datasets", nargs="+", choices=("7scenes", "tum_dynamics"), default=("7scenes", "tum_dynamics"))
    return parser.parse_args()


def choose_7scenes(args: argparse.Namespace) -> DatasetSelection:
    module = importlib.import_module("scripts.eval_7scenes_paper")
    sequence_dirs = module.select_sequence_dirs(args.seven_scenes_root, None)[: args.num_sequences]
    pools = {
        f"{path.parent.name}/{path.name}": module.load_frame_records(path)
        for path in sequence_dirs
    }
    sampled, sampled_indices = module.sample_records(pools, args.num_frames, args.seed)
    return DatasetSelection(
        dataset="7scenes",
        sequences=list(sampled),
        image_paths={
            name: [str(record.rgb_path) for record in records]
            for name, records in sampled.items()
        },
        sampled_indices=sampled_indices,
    )


def choose_tum(args: argparse.Namespace) -> DatasetSelection:
    module = importlib.import_module("scripts.eval_tum_dynamics_paper")
    sequence_dirs = module.select_sequence_dirs(args.tum_root, None)[: args.num_sequences]
    pools = {
        path.name: module.load_frame_records(path, tolerance=0.02)
        for path in sequence_dirs
    }
    sampled, sampled_indices = module.sample_records(pools, args.num_frames, args.seed)
    return DatasetSelection(
        dataset="tum_dynamics",
        sequences=list(sampled),
        image_paths={
            name: [str(record.rgb_path) for record in records]
            for name, records in sampled.items()
        },
        sampled_indices=sampled_indices,
    )


def make_bin_index_from_ranks(ranks: np.ndarray, bins: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    num_tokens = len(ranks)
    bins = min(bins, num_tokens)
    edges = np.linspace(0, num_tokens, bins + 1, dtype=np.int64)
    counts = np.diff(edges).astype(np.float32)
    rank_to_bin = np.empty(num_tokens, dtype=np.int64)
    for bin_index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        rank_to_bin[start:end] = bin_index
    index = rank_to_bin[ranks]
    return torch.from_numpy(index).to(device), edges, counts


def original_to_ordered_ranks(
    total_tokens: int,
    num_frames: int,
    patch_token_start: int,
    token_order: str,
) -> np.ndarray:
    tokens_per_frame = total_tokens // num_frames
    patch_count = tokens_per_frame - patch_token_start
    ranks = np.empty(total_tokens, dtype=np.int64)
    for original_index in range(total_tokens):
        frame = original_index // tokens_per_frame
        token = original_index % tokens_per_frame
        if token_order == "original":
            rank = original_index
        elif token_order == "special-first":
            if token < patch_token_start:
                rank = frame * patch_token_start + token
            else:
                rank = num_frames * patch_token_start + frame * patch_count + (token - patch_token_start)
        else:
            raise ValueError(token_order)
        ranks[original_index] = rank
    return ranks


class DownsampledGlobalAttentionCollector:
    def __init__(
        self,
        model,
        num_frames: int,
        query_chunk: int,
        display_bins: int,
        token_order: str,
    ) -> None:
        self.aggregator = model.aggregator
        self.num_frames = num_frames
        self.query_chunk = query_chunk
        self.display_bins_requested = display_bins
        self.token_order = token_order
        self.global_layers = [
            layer
            for layer, kind in enumerate(self.aggregator.inter_frame_attention_types)
            if kind == "global"
        ]
        self.handles = []
        self.sequence_results: dict[int, np.ndarray] = {}
        self.token_count: int | None = None
        self.tokens_per_frame: int | None = None
        self.display_bins: int | None = None
        self.edges: np.ndarray | None = None
        self.counts: np.ndarray | None = None

    def __enter__(self):
        for layer in self.global_layers:
            self.handles.append(
                self.aggregator.inter_frame_blocks[layer].register_forward_pre_hook(
                    self._hook(layer)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _ensure_layout(self, total_tokens: int, device: torch.device) -> torch.Tensor:
        if self.token_count is None:
            self.token_count = total_tokens
            self.tokens_per_frame = total_tokens // self.num_frames
            ranks = original_to_ordered_ranks(
                total_tokens,
                self.num_frames,
                self.aggregator.patch_token_start,
                self.token_order,
            )
            bin_index, edges, counts = make_bin_index_from_ranks(ranks, self.display_bins_requested, device)
            self.display_bins = len(counts)
            self.edges = edges
            self.counts = counts
            self.original_to_ordered_ranks = ranks
            self._bin_index = bin_index
        elif self.token_count != total_tokens:
            raise RuntimeError(f"Token count changed from {self.token_count} to {total_tokens}")
        return self._bin_index

    def _hook(self, layer: int):
        def hook(block, inputs):
            x = inputs[0]
            batch_size, total_tokens, _ = x.shape
            if batch_size != 1:
                raise RuntimeError(f"Expected batch size 1, got {batch_size}")
            bin_index = self._ensure_layout(total_tokens, x.device)
            assert self.display_bins is not None
            q, k, _ = qkv_from_block(block, x, None)
            key_t = k.float().transpose(-2, -1)
            layer_sum = torch.zeros(
                (self.display_bins, self.display_bins), device=x.device, dtype=torch.float32
            )
            for start in range(0, total_tokens, self.query_chunk):
                end = min(start + self.query_chunk, total_tokens)
                logits = torch.matmul(q[:, :, start:end].float(), key_t) * block.attn.scale
                probabilities = logits.softmax(dim=-1).mean(dim=1)[0]  # [query, key]
                key_query = probabilities.transpose(0, 1).contiguous()  # [key, query]
                key_reduced = torch.zeros(
                    (self.display_bins, end - start), device=x.device, dtype=torch.float32
                )
                key_reduced.scatter_add_(
                    0,
                    bin_index[:, None].expand(-1, end - start),
                    key_query,
                )
                query_bins = bin_index[start:end]
                layer_sum.scatter_add_(
                    1,
                    query_bins[None, :].expand(self.display_bins, -1),
                    key_reduced,
                )
                del logits, probabilities, key_query, key_reduced
            self.sequence_results[layer] = layer_sum.cpu().numpy()
            del q, k, key_t, layer_sum

        return hook


def plot_layer_matrix(
    output_path: Path,
    matrix: np.ndarray,
    edges: np.ndarray,
    tokens_per_frame: int,
    patch_token_start: int,
    num_frames: int,
    layer: int,
    dataset: str,
    token_order: str,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.percentile(finite, 99.5))
    vmin = 0.0
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="magma", origin="upper", vmin=vmin, vmax=vmax, aspect="auto")
    frame_centers: list[float] = []
    frame_labels: list[str] = []
    if token_order == "original":
        for frame in range(1, num_frames):
            token_boundary = frame * tokens_per_frame
            bin_boundary = int(np.searchsorted(edges, token_boundary, side="left"))
            axis.axvline(bin_boundary - 0.5, color="cyan", linewidth=0.35, alpha=0.6)
            axis.axhline(bin_boundary - 0.5, color="cyan", linewidth=0.35, alpha=0.6)
        for frame in range(num_frames):
            start = int(np.searchsorted(edges, frame * tokens_per_frame, side="left"))
            end = int(np.searchsorted(edges, (frame + 1) * tokens_per_frame, side="left"))
            frame_centers.append((start + end - 1) / 2)
            frame_labels.append(f"F{frame}")
    else:
        patch_count = tokens_per_frame - patch_token_start
        special_end = num_frames * patch_token_start
        special_boundary = int(np.searchsorted(edges, special_end, side="left"))
        axis.axvline(special_boundary - 0.5, color="lime", linewidth=0.8, alpha=0.85)
        axis.axhline(special_boundary - 0.5, color="lime", linewidth=0.8, alpha=0.85)
        frame_centers.append((special_boundary - 1) / 2)
        frame_labels.append("cam/reg")
        for frame in range(1, num_frames):
            token_boundary = special_end + frame * patch_count
            bin_boundary = int(np.searchsorted(edges, token_boundary, side="left"))
            axis.axvline(bin_boundary - 0.5, color="cyan", linewidth=0.35, alpha=0.6)
            axis.axhline(bin_boundary - 0.5, color="cyan", linewidth=0.35, alpha=0.6)
        for frame in range(num_frames):
            start = int(np.searchsorted(edges, special_end + frame * patch_count, side="left"))
            end = int(np.searchsorted(edges, special_end + (frame + 1) * patch_count, side="left"))
            frame_centers.append((start + end - 1) / 2)
            frame_labels.append(f"P{frame}")
    axis.set_xticks(frame_centers)
    axis.set_xticklabels(frame_labels, rotation=90)
    axis.set_yticks(frame_centers)
    axis.set_yticklabels(frame_labels)
    axis.set_xlabel("Query token index (columns; downsampled)")
    axis.set_ylabel("Key token index (rows; downsampled)")
    title_suffix = "original order" if token_order == "original" else "cam/register first, patches by frame"
    axis.set_title(f"{dataset}: global block {layer:02d}, {title_suffix}")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Mean attention probability")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_dataset(args: argparse.Namespace, model, selection: DatasetSelection) -> None:
    output_dir = args.output_dir / selection.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulators: dict[int, np.ndarray] = {}
    layout: dict[str, object] | None = None
    for sequence_name in selection.sequences:
        image_tensor = load_and_preprocess_images(
            selection.image_paths[sequence_name],
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(args.device, non_blocking=True)
        with DownsampledGlobalAttentionCollector(
            model, args.num_frames, args.query_chunk, args.display_bins, args.token_order
        ) as collector:
            with torch.inference_mode():
                _ = model(image_tensor)
        if layout is None:
            assert collector.edges is not None and collector.counts is not None
            layout = {
                "token_count": collector.token_count,
                "tokens_per_frame": collector.tokens_per_frame,
                "display_bins": collector.display_bins,
                "edges": collector.edges.tolist(),
                "counts": collector.counts.tolist(),
                "global_layers": collector.global_layers,
                "patch_token_start": model.aggregator.patch_token_start,
                "token_order": args.token_order,
            }
        for layer, matrix_sum in collector.sequence_results.items():
            accumulators.setdefault(layer, np.zeros_like(matrix_sum, dtype=np.float64))
            accumulators[layer] += matrix_sum.astype(np.float64)
        del image_tensor
        torch.cuda.empty_cache()
        print(f"{selection.dataset}: collected {sequence_name}", flush=True)

    assert layout is not None
    counts = np.asarray(layout["counts"], dtype=np.float64)
    denominator = counts[:, None] * counts[None, :] * len(selection.sequences)
    averaged: dict[str, np.ndarray] = {}
    for layer in sorted(accumulators):
        # accumulators are key-bin × query-bin sums. Divide by token-pair counts
        # and sequence count to get mean attention probability per bin pair.
        matrix = (accumulators[layer] / denominator).astype(np.float32)
        averaged[f"layer_{layer:02d}"] = matrix
        plot_layer_matrix(
            output_dir / f"global_attention_L{layer:02d}.png",
            matrix,
            np.asarray(layout["edges"], dtype=np.int64),
            int(layout["tokens_per_frame"]),
            int(layout["patch_token_start"]),
            args.num_frames,
            layer,
            selection.dataset,
            args.token_order,
        )
    np.savez_compressed(output_dir / "downsampled_global_attention_matrices.npz", **averaged)
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
        "token_order": args.token_order,
        "attention": "softmax(QK^T/sqrt(d)); averaged over heads and selected sequences",
        **layout,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_index(output_dir: Path, datasets: list[str]) -> None:
    lines = ["# Global attention matrix visualizations", ""]
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
                "",
            ]
        )
        for layer in metadata["global_layers"]:
            image = f"{dataset}/global_attention_L{layer:02d}.png"
            lines.extend([f"### Global block {layer:02d}", "", f"![{image}]({image})", ""])
    (output_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.device = device
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)
    selections = []
    if "7scenes" in args.datasets:
        selections.append(choose_7scenes(args))
    if "tum_dynamics" in args.datasets:
        selections.append(choose_tum(args))
    for selection in selections:
        run_dataset(args, model, selection)
    write_index(args.output_dir, [selection.dataset for selection in selections])
    print(f"Saved global attention visualizations to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
