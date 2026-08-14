#!/usr/bin/env python3
"""Trace frame-register injection and subsequent cross-frame global attention.

Attention probabilities are recomputed from read-only block inputs because the
model uses scaled_dot_product_attention and does not return its attention
matrix. The forward outputs are not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_token_evolution import load_model, read_sequence_images
from vggt_omega.utils.load_fn import load_and_preprocess_images


TOKEN_TYPES = ("camera", "register", "patch")


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
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--correspondences", type=int, default=20)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--sequence", default=None,
                        help="Direct sequence mode; avoids a precomputed summary.json.")
    parser.add_argument("--frame-gap", type=int, default=None,
                        help="In direct sequence mode, select exactly ten frames at this fixed source-frame gap.")
    parser.add_argument(
        "--export-full-attention-matrices", action="store_true",
        help=("Write one full-resolution, head-mean global-attention TIFF per global "
              "layer. Pixels are never token-pooled; this is expensive for 10 frames."),
    )
    return parser.parse_args()


def qkv_from_block(block, x: torch.Tensor, rope=None):
    normalized = block.norm1(x)
    batch, tokens, hidden = normalized.shape
    heads = block.attn.num_heads
    qkv = block.attn.qkv(normalized).reshape(batch, tokens, 3, heads, hidden // heads)
    q, k, v = torch.unbind(qkv, dim=2)
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    if block.attn.use_qk_norm:
        q = block.attn.q_norm(q)
        k = block.attn.k_norm(k)
    if rope is not None:
        q, k = block.attn.apply_rope(q, k, rope)
    return q, k, v


class AttentionFlowCollector:
    def __init__(self, model, num_frames: int, query_chunk: int, top_k: int) -> None:
        self.aggregator = model.aggregator
        self.num_frames = num_frames
        self.query_chunk = query_chunk
        self.top_k = top_k
        self.global_layers = [
            layer
            for layer, kind in enumerate(self.aggregator.inter_frame_attention_types)
            if kind == "global"
        ]
        self.frame_results: dict[int, dict[str, np.ndarray]] = {}
        self.global_results: dict[int, dict[str, np.ndarray]] = {}
        self.handles = []


    def __enter__(self):
        for layer in self.global_layers:
            self.handles.append(
                self.aggregator.frame_blocks[layer].register_forward_pre_hook(
                    self._frame_hook(layer)
                )
            )
            self.handles.append(
                self.aggregator.inter_frame_blocks[layer].register_forward_pre_hook(
                    self._global_hook(layer)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _frame_hook(self, layer: int):
        def hook(block, inputs):
            x = inputs[0]
            rope = inputs[1] if len(inputs) > 1 else None
            batch_frames, num_tokens, _ = x.shape
            patch_start = self.aggregator.patch_token_start
            q, k, v = qkv_from_block(block, x, rope)
            register_q = q[:, :, 1:patch_start].float()
            logits = torch.matmul(register_q, k.float().transpose(-2, -1)) * block.attn.scale
            probabilities = logits.softmax(dim=-1)
            patch_probabilities = probabilities[..., patch_start:]
            patch_weights = patch_probabilities.mean(dim=1)
            patch_value_norm = v[..., patch_start:, :].float().norm(dim=-1)
            value_weighted = (patch_probabilities * patch_value_norm[:, :, None, :]).mean(dim=1)
            batch_size = batch_frames // self.num_frames
            result = {
                "register_to_patch_weight": patch_weights.view(
                    batch_size, self.num_frames, patch_start - 1, num_tokens - patch_start
                ).cpu().numpy(),
                "register_to_patch_value_weighted": value_weighted.view(
                    batch_size, self.num_frames, patch_start - 1, num_tokens - patch_start
                ).cpu().numpy(),
                "register_patch_attention_mass": patch_weights.sum(dim=-1).view(
                    batch_size, self.num_frames, patch_start - 1
                ).cpu().numpy(),
            }
            self.frame_results[layer] = result
            del q, k, v, register_q, logits, probabilities, patch_probabilities

        return hook

    def _global_hook(self, layer: int):
        def hook(block, inputs):
            x = inputs[0]
            batch_size, total_tokens, _ = x.shape
            tokens_per_frame = total_tokens // self.num_frames
            patch_start = self.aggregator.patch_token_start
            patch_count = tokens_per_frame - patch_start
            q, k, _ = qkv_from_block(block, x, None)
            k_t = k.float().transpose(-2, -1)

            type_mass = np.zeros(
                (batch_size, self.num_frames, len(TOKEN_TYPES), self.num_frames, len(TOKEN_TYPES)),
                dtype=np.float32,
            )
            patch_sent = np.zeros(
                (batch_size, self.num_frames, self.num_frames, patch_count), dtype=np.float32
            )
            patch_received = np.zeros_like(patch_sent)
            top_indices = np.zeros(
                (batch_size, self.num_frames, self.num_frames, patch_count, self.top_k),
                dtype=np.int16,
            )
            top_weights = np.zeros(
                (batch_size, self.num_frames, self.num_frames, patch_count, self.top_k),
                dtype=np.float16,
            )

            def flat_slice(frame: int, token_type: str) -> slice:
                start = frame * tokens_per_frame
                if token_type == "camera":
                    return slice(start, start + 1)
                if token_type == "register":
                    return slice(start + 1, start + patch_start)
                return slice(start + patch_start, start + tokens_per_frame)

            for source_frame in range(self.num_frames):
                for query_type_index, query_type in enumerate(TOKEN_TYPES):
                    query_slice = flat_slice(source_frame, query_type)
                    query_indices = torch.arange(query_slice.start, query_slice.stop, device=x.device)
                    for query_offset in range(0, len(query_indices), self.query_chunk):
                        chunk = query_indices[query_offset : query_offset + self.query_chunk]
                        logits = torch.matmul(q[:, :, chunk].float(), k_t) * block.attn.scale
                        probabilities = logits.softmax(dim=-1)
                        for target_frame in range(self.num_frames):
                            for key_type_index, key_type in enumerate(TOKEN_TYPES):
                                key_slice = flat_slice(target_frame, key_type)
                                mass = probabilities[..., key_slice].sum(dim=-1).mean(dim=(1, 2))
                                type_mass[
                                    :, source_frame, query_type_index, target_frame, key_type_index
                                ] += mass.cpu().numpy() * (len(chunk) / len(query_indices))

                            if query_type == "patch":
                                target_patch_slice = flat_slice(target_frame, "patch")
                                target_probabilities = probabilities[..., target_patch_slice].mean(dim=1)
                                start = query_offset
                                end = query_offset + len(chunk)
                                patch_sent[:, source_frame, target_frame, start:end] = (
                                    target_probabilities.sum(dim=-1).cpu().numpy()
                                )
                                patch_received[:, source_frame, target_frame] += (
                                    target_probabilities.sum(dim=1).cpu().numpy()
                                )
                                values, indices = target_probabilities.topk(self.top_k, dim=-1)
                                top_indices[:, source_frame, target_frame, start:end] = (
                                    indices.cpu().numpy().astype(np.int16)
                                )
                                top_weights[:, source_frame, target_frame, start:end] = (
                                    values.cpu().numpy().astype(np.float16)
                                )
                        del logits, probabilities

            self.global_results[layer] = {
                "token_type_attention_mass": type_mass,
                "patch_to_frame_patch_mass": patch_sent,
                "patch_received_attention": patch_received,
                "cross_frame_topk_patch_indices": top_indices,
                "cross_frame_topk_patch_weights": top_weights,
            }
            del q, k, k_t

        return hook


class FullAttentionMatrixCollector:
    """Read-only full-resolution, head-mean global-attention image exporter."""
    def __init__(self, model, target: Path, query_chunk: int):
        self.aggregator, self.target, self.query_chunk = model.aggregator, target, query_chunk
        self.global_layers = [i for i, kind in enumerate(self.aggregator.inter_frame_attention_types) if kind == "global"]
        self.handles = []

    def __enter__(self):
        self.target.mkdir(parents=True, exist_ok=True)
        for layer in self.global_layers:
            self.handles.append(self.aggregator.inter_frame_blocks[layer].register_forward_pre_hook(self._hook(layer)))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def _hook(self, layer: int):
        def collect(block, inputs):
            q, k, _ = qkv_from_block(block, inputs[0], None)
            n = q.shape[-2]
            matrix = np.memmap(self.target / f"global_{layer:02d}_mean_heads.u16", mode="w+", dtype=np.uint16, shape=(n, n))
            kt = k.float().transpose(-2, -1)
            for start in range(0, n, self.query_chunk):
                end = min(start + self.query_chunk, n)
                probabilities = (torch.matmul(q[:, :, start:end].float(), kt) * block.attn.scale).softmax(dim=-1).mean(dim=1)[0]
                matrix[start:end] = (probabilities.clamp(0, 1).cpu().numpy() * 65535).round().astype(np.uint16)
            matrix.flush()
            Image.fromarray(matrix, mode="I;16").save(self.target / f"global_{layer:02d}_mean_heads.tiff", compression="tiff_lzw")
            del matrix, q, k, kt
        return collect


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values[np.isfinite(values)], (2, 98))
    if high <= low:
        high = low + 1e-8
    return float(low), float(high)


def plot_register_injection(
    target: Path,
    layer: int,
    result: dict[str, np.ndarray],
    grid: tuple[int, int],
) -> None:
    weight = result["register_to_patch_weight"][0].mean(axis=1)
    contribution = result["register_to_patch_value_weighted"][0].mean(axis=1)
    figure, axes = plt.subplots(len(weight), 2, figsize=(8, 2.8 * len(weight)), squeeze=False)
    for column, (values, label) in enumerate(
        ((weight, "Mean register→patch attention"), (contribution, "Attention × value norm"))
    ):
        vmin, vmax = robust_limits(values)
        for frame, axis in enumerate(axes[:, column]):
            rendered = axis.imshow(values[frame].reshape(grid), cmap="magma", vmin=vmin, vmax=vmax)
            axis.set_title(f"F{frame}: {label}")
            axis.axis("off")
        figure.colorbar(rendered, ax=axes[:, column].tolist(), fraction=0.025, pad=0.02)
    figure.suptitle(f"Frame block {layer}: patch information read by registers")
    figure.savefig(target / f"frame_register_injection_L{layer:02d}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_global_patch_flow(
    target: Path,
    layer: int,
    result: dict[str, np.ndarray],
    grid: tuple[int, int],
) -> None:
    sent = result["patch_to_frame_patch_mass"][0]
    frames = sent.shape[0]
    vmin, vmax = robust_limits(sent)
    figure, axes = plt.subplots(frames, frames, figsize=(3 * frames, 2.7 * frames), squeeze=False)
    for source in range(frames):
        for destination in range(frames):
            axis = axes[source, destination]
            rendered = axis.imshow(
                sent[source, destination].reshape(grid), cmap="viridis", vmin=vmin, vmax=vmax
            )
            axis.set_title(
                f"query F{source} → key F{destination}\nmean mass={sent[source, destination].mean():.3f}"
            )
            axis.axis("off")
    figure.colorbar(rendered, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    figure.suptitle(f"Global block {layer}: per-patch attention mass to each target frame")
    figure.savefig(target / f"global_patch_flow_L{layer:02d}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def patch_center(index: int, grid: tuple[int, int], image_shape: tuple[int, int]) -> tuple[float, float]:
    grid_h, grid_w = grid
    height, width = image_shape
    row, column = divmod(index, grid_w)
    return (column + 0.5) * width / grid_w, (row + 0.5) * height / grid_h


def plot_top_correspondences(
    target: Path,
    layer: int,
    result: dict[str, np.ndarray],
    images: np.ndarray,
    grid: tuple[int, int],
    count: int,
) -> None:
    pairs = [(source, destination) for source in range(len(images)) for destination in range(source + 1, len(images))]
    figure, axes = plt.subplots(len(pairs), 1, figsize=(12, 3.3 * len(pairs)), squeeze=False)
    indices = result["cross_frame_topk_patch_indices"][0]
    weights = result["cross_frame_topk_patch_weights"][0].astype(np.float32)
    height, width = images.shape[1:3]
    for row, (source, destination) in enumerate(pairs):
        axis = axes[row, 0]
        canvas = np.concatenate((images[source], images[destination]), axis=1)
        axis.imshow(canvas)
        strongest = weights[source, destination, :, 0]
        selected = np.argsort(strongest)[-count:]
        norm = plt.Normalize(strongest[selected].min(), strongest[selected].max() + 1e-12)
        for query_index in selected:
            key_index = int(indices[source, destination, query_index, 0])
            x0, y0 = patch_center(int(query_index), grid, (height, width))
            x1, y1 = patch_center(key_index, grid, (height, width))
            color = plt.cm.plasma(norm(strongest[query_index]))
            axis.plot([x0, x1 + width], [y0, y1], color=color, linewidth=0.8, alpha=0.75)
            axis.scatter([x0, x1 + width], [y0, y1], color=[color], s=7)
        axis.axvline(width - 0.5, color="white", linewidth=1)
        axis.set_title(f"F{source} query → F{destination} key: top {count} head-mean links")
        axis.axis("off")
    figure.suptitle(f"Global block {layer}: strongest cross-frame patch correspondences")
    figure.savefig(target / f"global_top_correspondences_L{layer:02d}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_results(
    sequence_dir: Path,
    collector: AttentionFlowCollector,
    images: np.ndarray,
    grid: tuple[int, int],
    correspondence_count: int,
) -> None:
    target = sequence_dir / "attention_flow"
    target.mkdir(parents=True, exist_ok=True)
    layers = collector.global_layers
    frame_keys = tuple(next(iter(collector.frame_results.values())))
    global_keys = tuple(next(iter(collector.global_results.values())))
    arrays = {"global_layers": np.asarray(layers), "token_types": np.asarray(TOKEN_TYPES)}
    for key in frame_keys:
        arrays[key] = np.stack([collector.frame_results[layer][key] for layer in layers])
    for key in global_keys:
        arrays[key] = np.stack([collector.global_results[layer][key] for layer in layers])
    np.savez_compressed(target / "attention_flow_metrics.npz", **arrays)

    rows = []
    for layer in layers:
        frame_result = collector.frame_results[layer]
        global_result = collector.global_results[layer]
        patch_mass = frame_result["register_patch_attention_mass"][0]
        sent = global_result["patch_to_frame_patch_mass"][0]
        for frame in range(sent.shape[0]):
            rows.append(
                {
                    "global_block": layer,
                    "frame": frame,
                    "register_to_patch_mass_mean": float(patch_mass[frame].mean()),
                    "same_frame_patch_attention_mass": float(sent[frame, frame].mean()),
                    "cross_frame_patch_attention_mass": float(
                        np.delete(sent[frame], frame, axis=0).sum(axis=0).mean()
                    ),
                }
            )
        plot_register_injection(target, layer, frame_result, grid)
        plot_global_patch_flow(target, layer, global_result, grid)
        plot_top_correspondences(
            target, layer, global_result, images, grid, correspondence_count
        )
    with (target / "attention_flow_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "global_layers": layers,
        "num_frames": len(images),
        "patch_grid": list(grid),
        "top_k": collector.top_k,
        "interpretation": {
            "register_to_patch_weight": "head-mean register-query attention probability to each patch key",
            "register_to_patch_value_weighted": "attention probability multiplied by pre-projection value-vector norm",
            "patch_to_frame_patch_mass": "for each patch query, attention probability summed over patch keys in each target frame",
            "cross_frame_topk": "top target-frame patch keys from head-mean attention probabilities",
        },
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    if args.sequence is not None:
        if args.data_root is None or args.frame_gap is None:
            raise ValueError("--sequence requires --data-root and --frame-gap")
        sequence_dir = args.data_root / args.sequence
        rgb_file = sequence_dir / "rgb.txt"
        if rgb_file.is_file():
            source = read_sequence_images(sequence_dir, "full")
        else:
            source = sorted(sequence_dir.glob("*.color.png"))
        if len(source) < 1 + 9 * args.frame_gap:
            raise ValueError("insufficient sequence frames for requested gap")
        selections = {args.sequence: [str(path) for path in source[0 : 1 + 9 * args.frame_gap : args.frame_gap]]}
    else:
        summary = json.loads((args.analysis_dir / "summary.json").read_text(encoding="utf-8"))
        selections = summary["frame_selections"]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Attention-flow analysis requires CUDA")
    model = load_model(args.checkpoint, device)
    for sequence, paths in selections.items():
        image_tensor = load_and_preprocess_images(
            paths, mode=args.resize_mode, image_resolution=args.image_resolution
        )
        display_images = image_tensor.permute(0, 2, 3, 1).numpy()
        images = image_tensor.to(device)
        grid = (
            images.shape[-2] // model.aggregator.patch_size,
            images.shape[-1] // model.aggregator.patch_size,
        )
        full_target = args.analysis_dir / sequence / "attention_full_matrices"
        with AttentionFlowCollector(model, len(paths), args.query_chunk, args.top_k) as collector:
            with (
                FullAttentionMatrixCollector(model, full_target, args.query_chunk)
                if args.export_full_attention_matrices else __import__("contextlib").nullcontext()
            ):
                with torch.inference_mode():
                    predictions = model(images)
        save_results(
            args.analysis_dir / sequence,
            collector,
            display_images,
            grid,
            args.correspondences,
        )
        print(f"Saved attention flow for {sequence}")
        del images, image_tensor, predictions
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
