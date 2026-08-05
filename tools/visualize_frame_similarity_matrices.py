#!/usr/bin/env python3
"""Visualize frame-to-frame feature similarity matrices for VGGT-Omega.

The default input-stage matrix uses the preprocessed RGB tensor before it enters
the model. Transformer-layer matrices use patch tokens after the requested
aggregator layer has finished its frame and inter-frame blocks. For every stage,
features are average-pooled with a 2x2 spatial window, L2-normalized per pooled
position, and compared by cosine similarity at corresponding spatial positions.
The final frame-pair value is the mean cosine similarity over pooled positions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nnf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_token_evolution import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_LAYERS = (2, 6, 10, 16, 23)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-paths", nargs="+", type=Path, default=None)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of input frames. Used when --image-paths is omitted.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Evenly sample this many frames from the resolved input list.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=[str(layer) for layer in DEFAULT_LAYERS],
        help="0-based aggregator layer indices, e.g. 2 6 10 16 23 or 2,6,10,16,23.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/frame_similarity_matrices"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--position-chunk",
        type=int,
        default=64,
        help="Number of pooled spatial positions processed per cosine-similarity chunk.",
    )
    parser.add_argument(
        "--input-feature",
        choices=("rgb", "patch_embed"),
        default="rgb",
        help=(
            "Feature source for the input-stage matrix. 'rgb' is the preprocessed "
            "model input tensor; 'patch_embed' is the patch-token tensor before layer 0."
        ),
    )
    parser.add_argument(
        "--color-scale",
        choices=("fixed", "auto"),
        default="auto",
        help="Use fixed [-1, 1] cosine limits or robust per-matrix limits.",
    )
    parser.add_argument("--annotate-max-frames", type=int, default=16)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast for feature collection.")
    return parser.parse_args()


def parse_layers(values: Iterable[str]) -> list[int]:
    layers: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                layers.append(int(part))
    if not layers:
        raise ValueError("At least one layer must be requested")
    return layers


def natural_key(path: Path) -> list[object]:
    pieces = re.split(r"(\d+)", path.name)
    return [int(piece) if piece.isdigit() else piece.lower() for piece in pieces]


def resolve_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.image_paths:
        paths = list(args.image_paths)
    elif args.input_dir is not None:
        if not args.input_dir.is_dir():
            raise FileNotFoundError(args.input_dir)
        paths = [
            path
            for path in args.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        paths.sort(key=natural_key)
    else:
        raise ValueError("Provide either --image-paths or --input-dir")

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input image not found: {missing[0]}")
    if args.num_frames is not None:
        paths = evenly_spaced(paths, args.num_frames)
    if len(paths) < 2:
        raise ValueError("At least two frames are required for a similarity matrix")
    return paths


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count < 2:
        raise ValueError("--num-frames must be at least 2")
    if len(paths) < count:
        raise ValueError(f"Requested {count} frames from a pool of {len(paths)}")
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def pooled_cosine_frame_similarity(
    features: torch.Tensor,
    *,
    pool_size: int,
    eps: float,
    position_chunk: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return a batch-averaged frame similarity matrix from [B, F, C, H, W]."""

    if features.ndim != 5:
        raise ValueError(f"Expected [B, F, C, H, W] features, got {tuple(features.shape)}")
    if pool_size <= 0:
        raise ValueError("--pool-size must be positive")
    if position_chunk <= 0:
        raise ValueError("--position-chunk must be positive")

    batch_size, num_frames, channels, height, width = features.shape
    if height < pool_size or width < pool_size:
        raise ValueError(
            f"Feature grid {height}x{width} is smaller than pool size {pool_size}"
        )

    pooled = nnf.avg_pool2d(
        features.reshape(batch_size * num_frames, channels, height, width).float(),
        kernel_size=pool_size,
        stride=pool_size,
    )
    _, _, pooled_height, pooled_width = pooled.shape
    num_positions = pooled_height * pooled_width
    vectors = pooled.reshape(batch_size, num_frames, channels, num_positions).permute(0, 1, 3, 2)
    normalized = nnf.normalize(vectors, p=2, dim=-1, eps=eps)

    similarity_sum = torch.zeros(
        batch_size,
        num_frames,
        num_frames,
        device=features.device,
        dtype=torch.float32,
    )
    for start in range(0, num_positions, position_chunk):
        end = min(start + position_chunk, num_positions)
        chunk = normalized[:, :, start:end].float()
        similarity_sum += torch.einsum("bfpc,bgpc->bfgp", chunk, chunk).sum(dim=-1)

    matrix = (similarity_sum / max(num_positions, 1)).mean(dim=0)
    layout = {
        "batch_size": int(batch_size),
        "num_frames": int(num_frames),
        "channels": int(channels),
        "height": int(height),
        "width": int(width),
        "pooled_height": int(pooled_height),
        "pooled_width": int(pooled_width),
        "pooled_positions": int(num_positions),
    }
    return matrix.detach().cpu().numpy().astype(np.float32), layout


def patch_features_to_bfchw(
    tokens: torch.Tensor,
    *,
    num_frames: int,
    patch_token_start: int,
    patch_grid_size: tuple[int, int],
) -> torch.Tensor:
    """Convert flattened frame tokens to patch features shaped [B, F, C, H, W]."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected a 3D token tensor, got {tuple(tokens.shape)}")
    first_dim, num_tokens, hidden_dim = tokens.shape
    if first_dim % num_frames == 0:
        batch_size = first_dim // num_frames
        frame_tokens = tokens.view(batch_size, num_frames, num_tokens, hidden_dim)
    else:
        batch_size, total_tokens, hidden_dim = tokens.shape
        if total_tokens % num_frames:
            raise ValueError(
                f"Cannot split {total_tokens} total tokens across {num_frames} frames"
            )
        num_tokens = total_tokens // num_frames
        frame_tokens = tokens.view(batch_size, num_frames, num_tokens, hidden_dim)

    patch_count = num_tokens - patch_token_start
    patch_height, patch_width = patch_grid_size
    expected_patch_count = patch_height * patch_width
    if patch_count != expected_patch_count:
        raise ValueError(
            "Patch token count does not match patch grid: "
            f"{patch_count} versus {patch_height}x{patch_width}"
        )
    patches = frame_tokens[:, :, patch_token_start:]
    return patches.reshape(batch_size, num_frames, patch_height, patch_width, hidden_dim).permute(0, 1, 4, 2, 3)


class FrameSimilarityCollector:
    """Collect requested layer matrices without retaining all activations."""

    def __init__(
        self,
        model,
        *,
        num_frames: int,
        layers: Iterable[int],
        patch_grid_size: tuple[int, int],
        pool_size: int,
        eps: float,
        position_chunk: int,
        capture_patch_input: bool,
    ) -> None:
        self.aggregator = model.aggregator
        self.num_frames = int(num_frames)
        self.layers = set(int(layer) for layer in layers)
        self.patch_grid_size = patch_grid_size
        self.pool_size = int(pool_size)
        self.eps = float(eps)
        self.position_chunk = int(position_chunk)
        self.capture_patch_input = bool(capture_patch_input)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.matrices: dict[str, np.ndarray] = {}
        self.layouts: dict[str, dict[str, int]] = {}
        self.layer_attention_types: dict[int, str] = {}
        self._pending_frame_tokens: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "FrameSimilarityCollector":
        if self.capture_patch_input:
            self.handles.append(
                self.aggregator.frame_blocks[0].register_forward_pre_hook(self._input_hook)
            )
        for layer in sorted(self.layers):
            self.handles.append(
                self.aggregator.frame_blocks[layer].register_forward_hook(self._frame_hook(layer))
            )
            self.handles.append(
                self.aggregator.inter_frame_blocks[layer].register_forward_hook(self._inter_frame_hook(layer))
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._pending_frame_tokens.clear()

    def _store_matrix(self, key: str, features: torch.Tensor) -> None:
        matrix, layout = pooled_cosine_frame_similarity(
            features,
            pool_size=self.pool_size,
            eps=self.eps,
            position_chunk=self.position_chunk,
        )
        self.matrices[key] = matrix
        self.layouts[key] = layout

    def _input_hook(self, _module, inputs) -> None:
        tokens = inputs[0].detach()
        features = patch_features_to_bfchw(
            tokens,
            num_frames=self.num_frames,
            patch_token_start=self.aggregator.patch_token_start,
            patch_grid_size=self.patch_grid_size,
        )
        self._store_matrix("input_patch_embed", features)

    def _frame_hook(self, layer: int):
        def hook(_module, _inputs, output: torch.Tensor) -> None:
            if self.aggregator.inter_frame_attention_types[layer] == "register":
                self._pending_frame_tokens[layer] = output.detach()

        return hook

    def _inter_frame_hook(self, layer: int):
        def hook(_module, _inputs, output: torch.Tensor) -> None:
            attention_type = self.aggregator.inter_frame_attention_types[layer]
            self.layer_attention_types[layer] = attention_type
            key = f"layer_{layer:02d}"
            if attention_type == "global":
                tokens = output.detach()
            elif attention_type == "register":
                tokens = self._pending_frame_tokens.pop(layer, None)
                if tokens is None:
                    raise RuntimeError(f"Missing frame-block patch tokens for register layer {layer}")
            else:
                raise ValueError(f"Unsupported inter-frame attention type {attention_type!r}")
            features = patch_features_to_bfchw(
                tokens,
                num_frames=self.num_frames,
                patch_token_start=self.aggregator.patch_token_start,
                patch_grid_size=self.patch_grid_size,
            )
            self._store_matrix(key, features)

        return hook


def stage_summary(matrix: np.ndarray) -> dict[str, float]:
    finite = matrix[np.isfinite(matrix)]
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    finite_off_diagonal = off_diagonal[np.isfinite(off_diagonal)]
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "off_diagonal_mean": float(finite_off_diagonal.mean()) if finite_off_diagonal.size else float("nan"),
        "adjacent_mean": float(np.diag(matrix, k=1).mean()) if matrix.shape[0] > 1 else float("nan"),
    }


def color_limits(matrix: np.ndarray, mode: str) -> tuple[float, float]:
    if mode == "fixed":
        return -1.0, 1.0
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (2, 98))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        pad = max(abs(float(high)) * 0.01, 1e-6)
        low = float(low) - pad
        high = float(high) + pad
    return float(low), float(high)


def plot_similarity_matrix(
    matrix: np.ndarray,
    output_path: Path,
    *,
    title: str,
    frame_labels: list[str],
    color_scale: str,
    annotate_max_frames: int,
) -> None:
    vmin, vmax = color_limits(matrix, color_scale)
    figure_size = max(5.0, min(12.0, matrix.shape[0] * 0.38))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
    tick_step = max(1, math.ceil(matrix.shape[0] / 20))
    ticks = np.arange(0, matrix.shape[0], tick_step)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels([frame_labels[index] for index in ticks], rotation=90)
    axis.set_yticklabels([frame_labels[index] for index in ticks])
    axis.set_xlabel("Frame")
    axis.set_ylabel("Frame")
    axis.set_title(title)
    if matrix.shape[0] <= annotate_max_frames:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text_color = "white" if value < (vmin + vmax) * 0.5 else "black"
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Mean cosine similarity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    *,
    matrices: dict[str, np.ndarray],
    layouts: dict[str, dict[str, int]],
    metadata: dict[str, object],
    frame_labels: list[str],
    color_scale: str,
    annotate_max_frames: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "frame_similarity_matrices.npz", **matrices)

    stage_rows: list[dict[str, object]] = []
    lines = ["# Frame similarity matrices", ""]
    lines.extend(
        [
            f"- frames: {len(frame_labels)}",
            f"- pool size: {metadata['pool_size']}",
            "- similarity: mean cosine over corresponding pooled spatial positions",
            "",
        ]
    )

    for key, matrix in matrices.items():
        csv_path = output_dir / f"{key}_similarity.csv"
        png_path = output_dir / f"{key}_similarity.png"
        np.savetxt(csv_path, matrix, delimiter=",", fmt="%.6f")
        summary = stage_summary(matrix)
        layout = layouts[key]
        title = f"{key}: frame similarity"
        plot_similarity_matrix(
            matrix,
            png_path,
            title=title,
            frame_labels=frame_labels,
            color_scale=color_scale,
            annotate_max_frames=annotate_max_frames,
        )
        stage_rows.append(
            {
                "stage": key,
                "shape": list(matrix.shape),
                "layout": layout,
                "summary": summary,
                "csv": csv_path.name,
                "png": png_path.name,
            }
        )
        lines.extend(
            [
                f"## {key}",
                "",
                f"- off-diagonal mean: {summary['off_diagonal_mean']:.6f}",
                f"- adjacent mean: {summary['adjacent_mean']:.6f}",
                f"- feature grid: {layout['height']}x{layout['width']} -> {layout['pooled_height']}x{layout['pooled_width']}",
                "",
                f"![{key}]({png_path.name})",
                "",
            ]
        )

    full_metadata = dict(metadata)
    full_metadata["stages"] = stage_rows
    (output_dir / "metadata.json").write_text(json.dumps(full_metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def run_aggregator_for_collection(
    model,
    images: torch.Tensor,
    *,
    use_amp: bool,
) -> None:
    if use_amp and images.device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        context = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        context = nullcontext()
    with context:
        _ = model.aggregator(images.unsqueeze(0))


def main() -> int:
    args = parse_args()
    layers = parse_layers(args.layers)
    image_paths = resolve_image_paths(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    invalid_layers = [layer for layer in layers if layer < 0 or layer >= model.aggregator.depth]
    if invalid_layers:
        raise ValueError(f"Layer indices out of range 0..{model.aggregator.depth - 1}: {invalid_layers}")

    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
        patch_size=model.aggregator.patch_size,
    ).to(device, non_blocking=device.type == "cuda")
    _, _, height, width = images.shape
    patch_grid_size = (height // model.aggregator.patch_size, width // model.aggregator.patch_size)

    matrices: dict[str, np.ndarray] = {}
    layouts: dict[str, dict[str, int]] = {}
    if args.input_feature == "rgb":
        matrix, layout = pooled_cosine_frame_similarity(
            images.unsqueeze(0),
            pool_size=args.pool_size,
            eps=args.eps,
            position_chunk=args.position_chunk,
        )
        matrices["input_rgb"] = matrix
        layouts["input_rgb"] = layout

    with FrameSimilarityCollector(
        model,
        num_frames=len(image_paths),
        layers=layers,
        patch_grid_size=patch_grid_size,
        pool_size=args.pool_size,
        eps=args.eps,
        position_chunk=args.position_chunk,
        capture_patch_input=args.input_feature == "patch_embed",
    ) as collector:
        with torch.inference_mode():
            run_aggregator_for_collection(model, images, use_amp=not args.no_amp)
        matrices.update(collector.matrices)
        layouts.update(collector.layouts)
        layer_attention_types = collector.layer_attention_types

    missing_keys = [f"layer_{layer:02d}" for layer in layers if f"layer_{layer:02d}" not in matrices]
    if missing_keys:
        raise RuntimeError(f"Missing requested stage matrices: {missing_keys}")

    ordered_keys = (
        ["input_rgb"] if args.input_feature == "rgb" else ["input_patch_embed"]
    ) + [f"layer_{layer:02d}" for layer in layers]
    matrices = {key: matrices[key] for key in ordered_keys}
    layouts = {key: layouts[key] for key in ordered_keys}

    frame_labels = [f"F{index:03d}" for index in range(len(image_paths))]
    metadata = {
        "checkpoint": str(args.checkpoint),
        "image_paths": [str(path) for path in image_paths],
        "input_feature": args.input_feature,
        "layers": layers,
        "layer_indexing": "0-based aggregator block indices",
        "layer_attention_types": {str(layer): layer_attention_types.get(layer) for layer in layers},
        "pool_size": args.pool_size,
        "pooling": "torch.nn.functional.avg_pool2d(kernel_size=pool_size, stride=pool_size)",
        "similarity": "mean cosine similarity over corresponding pooled spatial positions",
        "image_resolution": args.image_resolution,
        "resize_mode": args.resize_mode,
        "preprocessing": "vggt_omega.utils.load_fn.load_and_preprocess_images",
        "model_variant": "VGGTOmega(global_merging=False, merging=None, merge_ratio=0.0)",
        "patch_grid_size": list(patch_grid_size),
        "color_scale": args.color_scale,
    }
    write_outputs(
        args.output_dir,
        matrices=matrices,
        layouts=layouts,
        metadata=metadata,
        frame_labels=frame_labels,
        color_scale=args.color_scale,
        annotate_max_frames=args.annotate_max_frames,
    )
    print(f"Saved frame similarity matrices to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
