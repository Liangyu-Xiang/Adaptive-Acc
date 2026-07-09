#!/usr/bin/env python3
"""Analyze layer-wise token evolution without modifying VGGT-Omega forward.

The analyzer attaches forward hooks to the frame and inter-frame blocks. It
reconstructs the complete token tensor at each layer boundary, including the
register-attention layers that update only camera/register tokens, and computes
adjacent-layer statistics online. Only the previous layer is retained on GPU;
per-token scalar metrics are immediately moved to CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images


DEFAULT_DATA_ROOT = Path("/data/mmc_lyxiang/dataset/TUM-Dynamics")
DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_SEQUENCES = (
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_walking_xyz",
)
TOKEN_TYPES = ("all", "camera", "register", "patch")
LOW_UPDATE_SENSITIVITY = (
    ("strict", 0.05, 1e-3),
    ("conservative", 0.10, 5e-3),
    ("moderate", 0.25, 2e-2),
    ("loose", 0.50, 5e-2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/token_evolution"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--frame-source", choices=("full", "rgb_90"), default="full")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--relative-l2-threshold", type=float, default=0.05)
    parser.add_argument("--cosine-distance-threshold", type=float, default=1e-3)
    parser.add_argument("--persistent-layers", type=int, default=3)
    parser.add_argument("--late-start-layer", type=int, default=12)
    parser.add_argument("--verify-output", action="store_true")
    return parser.parse_args()


def read_sequence_images(sequence_dir: Path, source: str) -> list[Path]:
    if source == "rgb_90":
        paths = list((sequence_dir / "rgb_90").glob("*.png"))
        paths.sort(key=lambda path: float(path.stem))
        return paths

    rgb_file = sequence_dir / "rgb.txt"
    if not rgb_file.is_file():
        raise FileNotFoundError(rgb_file)
    rows: list[tuple[float, Path]] = []
    for line in rgb_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, relative_path = line.split()[:2]
        rows.append((float(timestamp), sequence_dir / relative_path))
    rows.sort(key=lambda row: row[0])
    return [path for _, path in rows]


def evenly_spaced(paths: Sequence[Path], count: int) -> list[Path]:
    if count < 2:
        raise ValueError("--num-frames must be at least 2")
    if len(paths) < count:
        raise ValueError(f"Requested {count} frames from a pool of {len(paths)}")
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def load_model(checkpoint: Path, device: torch.device) -> VGGTOmega:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Explicitly disable global merging: this experiment must observe the
    # uncompressed model and must not alter model outputs.
    model = VGGTOmega(global_merging=False, merging=None, merge_ratio=0.0)
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


@dataclass
class TokenLayout:
    batch_size: int
    num_frames: int
    num_tokens_per_frame: int
    hidden_dim: int
    patch_token_start: int


class LayerTokenAnalyzer:
    """Forward-hook collector retaining at most two complete token layers."""

    def __init__(
        self,
        model: VGGTOmega,
        num_frames: int,
        eps: float,
        relative_l2_threshold: float,
        cosine_distance_threshold: float,
    ) -> None:
        self.aggregator = model.aggregator
        self.num_frames = num_frames
        self.eps = eps
        self.relative_l2_threshold = relative_l2_threshold
        self.cosine_distance_threshold = cosine_distance_threshold
        self.layout: TokenLayout | None = None
        self.previous: torch.Tensor | None = None
        self.pending_frame: torch.Tensor | None = None
        self.relative_l2: list[np.ndarray] = []
        self.cosine_distance: list[np.ndarray] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.captured_layers: list[int] = []

    def __enter__(self) -> "LayerTokenAnalyzer":
        for layer, block in enumerate(self.aggregator.frame_blocks):
            self.handles.append(block.register_forward_hook(self._frame_hook(layer)))
        for layer, block in enumerate(self.aggregator.inter_frame_blocks):
            self.handles.append(block.register_forward_hook(self._inter_frame_hook(layer)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending_frame = None
        self.previous = None

    def _frame_hook(self, layer: int):
        def hook(_module, _inputs, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise TypeError(f"Layer {layer} frame output has unexpected type/shape")
            batch_frames, num_tokens, hidden_dim = output.shape
            if batch_frames % self.num_frames:
                raise ValueError("Cannot infer batch size from frame-block output")
            batch_size = batch_frames // self.num_frames
            if self.layout is None:
                self.layout = TokenLayout(
                    batch_size=batch_size,
                    num_frames=self.num_frames,
                    num_tokens_per_frame=num_tokens,
                    hidden_dim=hidden_dim,
                    patch_token_start=self.aggregator.patch_token_start,
                )
            self.pending_frame = output.detach()

        return hook

    def _inter_frame_hook(self, layer: int):
        def hook(_module, _inputs, output: torch.Tensor) -> None:
            if self.layout is None or self.pending_frame is None:
                raise RuntimeError(f"Missing frame output before inter-frame layer {layer}")
            layout = self.layout
            frame = self.pending_frame.view(
                layout.batch_size,
                layout.num_frames,
                layout.num_tokens_per_frame,
                layout.hidden_dim,
            )
            attention_type = self.aggregator.inter_frame_attention_types[layer]
            if attention_type == "global":
                current = output.detach().view_as(frame)
            elif attention_type == "register":
                special = output.detach().view(
                    layout.batch_size,
                    layout.num_frames,
                    layout.patch_token_start,
                    layout.hidden_dim,
                )
                current = torch.cat((special, frame[:, :, layout.patch_token_start :]), dim=2)
            else:
                raise ValueError(f"Unsupported attention type {attention_type!r}")

            if self.previous is not None:
                previous_float = self.previous.float()
                current_float = current.float()
                relative = (current_float - previous_float).norm(dim=-1)
                relative /= previous_float.norm(dim=-1).clamp_min(self.eps)
                cosine = 1.0 - F.cosine_similarity(
                    current_float,
                    previous_float,
                    dim=-1,
                    eps=self.eps,
                )
                self.relative_l2.append(relative.detach().cpu().numpy())
                self.cosine_distance.append(cosine.detach().cpu().numpy())
                del previous_float, current_float, relative, cosine

            self.previous = current.detach()
            self.pending_frame = None
            self.captured_layers.append(layer)

        return hook

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.layout is None:
            raise RuntimeError("No token layout was captured")
        if self.captured_layers != list(range(self.aggregator.depth)):
            raise RuntimeError(f"Expected 24 layers, captured {self.captured_layers}")
        relative = np.stack(self.relative_l2)
        cosine = np.stack(self.cosine_distance)
        low = (relative <= self.relative_l2_threshold) & (
            cosine <= self.cosine_distance_threshold
        )
        return relative, cosine, low


def token_mask(layout: TokenLayout, token_type: str) -> np.ndarray:
    within_frame = np.arange(layout.num_tokens_per_frame)
    if token_type == "all":
        one_frame = np.ones_like(within_frame, dtype=bool)
    elif token_type == "camera":
        one_frame = within_frame == 0
    elif token_type == "register":
        one_frame = (within_frame >= 1) & (within_frame < layout.patch_token_start)
    elif token_type == "patch":
        one_frame = within_frame >= layout.patch_token_start
    else:
        raise ValueError(token_type)
    return np.broadcast_to(one_frame, (layout.batch_size, layout.num_frames, len(one_frame)))


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def maximum_consecutive(mask: np.ndarray) -> np.ndarray:
    """Return maximum true-run length over axis 0 for every token."""
    current = np.zeros(mask.shape[1:], dtype=np.int16)
    maximum = np.zeros_like(current)
    for layer_mask in mask:
        current = np.where(layer_mask, current + 1, 0)
        maximum = np.maximum(maximum, current)
    return maximum


def summarize_sequence(
    name: str,
    label: str,
    layout: TokenLayout,
    relative: np.ndarray,
    cosine: np.ndarray,
    low: np.ndarray,
    attention_types: Sequence[str],
    late_start_layer: int,
    persistent_layers: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    target_layers = np.arange(1, len(relative) + 1)
    late = target_layers >= late_start_layer
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"sequence": name, "label": label, "token_types": {}}

    for token_type in TOKEN_TYPES:
        mask = token_mask(layout, token_type)
        layer_relative = []
        layer_cosine = []
        layer_low = []
        for transition, target_layer in enumerate(target_layers):
            rel_stats = distribution(relative[transition][mask])
            cos_stats = distribution(cosine[transition][mask])
            low_fraction = float(np.mean(low[transition][mask]))
            layer_relative.append(rel_stats["mean"])
            layer_cosine.append(cos_stats["mean"])
            layer_low.append(low_fraction)
            rows.append(
                {
                    "sequence": name,
                    "label": label,
                    "source_layer": int(target_layer - 1),
                    "target_layer": int(target_layer),
                    "target_inter_frame_attention": attention_types[target_layer],
                    "token_type": token_type,
                    "relative_l2_mean": rel_stats["mean"],
                    "relative_l2_median": rel_stats["median"],
                    "relative_l2_p10": rel_stats["p10"],
                    "relative_l2_p90": rel_stats["p90"],
                    "cosine_distance_mean": cos_stats["mean"],
                    "cosine_distance_median": cos_stats["median"],
                    "cosine_distance_p10": cos_stats["p10"],
                    "cosine_distance_p90": cos_stats["p90"],
                    "joint_low_update_fraction": low_fraction,
                }
            )

        token_low = low[:, mask]
        max_streak = maximum_consecutive(token_low)
        late_low = token_low[late]
        late_max_streak = maximum_consecutive(late_low)
        type_summary = {
            "num_tokens": int(mask.sum()),
            "relative_l2_early_mean": float(np.mean(layer_relative[:8])),
            "relative_l2_middle_mean": float(np.mean(layer_relative[8:16])),
            "relative_l2_late_mean": float(np.mean(layer_relative[16:])),
            "relative_l2_linear_slope": float(np.polyfit(target_layers, layer_relative, 1)[0]),
            "cosine_distance_early_mean": float(np.mean(layer_cosine[:8])),
            "cosine_distance_middle_mean": float(np.mean(layer_cosine[8:16])),
            "cosine_distance_late_mean": float(np.mean(layer_cosine[16:])),
            "cosine_distance_linear_slope": float(np.polyfit(target_layers, layer_cosine, 1)[0]),
            "joint_low_update_fraction_all": float(np.mean(token_low)),
            "joint_low_update_fraction_late": float(np.mean(late_low)),
            "fraction_tokens_persistent_anywhere": float(np.mean(max_streak >= persistent_layers)),
            "fraction_tokens_persistent_late": float(np.mean(late_max_streak >= persistent_layers)),
            "fraction_tokens_low_at_least_half_late": float(np.mean(np.mean(late_low, axis=0) >= 0.5)),
            "maximum_observed_low_update_streak": int(np.max(max_streak)),
        }
        sensitivity = {}
        relative_for_type = relative[:, mask]
        cosine_for_type = cosine[:, mask]
        for threshold_name, relative_threshold, cosine_threshold in LOW_UPDATE_SENSITIVITY:
            threshold_low = (relative_for_type <= relative_threshold) & (
                cosine_for_type <= cosine_threshold
            )
            threshold_late = threshold_low[late]
            threshold_late_streak = maximum_consecutive(threshold_late)
            sensitivity[threshold_name] = {
                "relative_l2_threshold": relative_threshold,
                "cosine_distance_threshold": cosine_threshold,
                "low_update_fraction_late": float(np.mean(threshold_late)),
                "fraction_tokens_persistent_3_layers_late": float(
                    np.mean(threshold_late_streak >= 3)
                ),
                "maximum_late_streak": int(np.max(threshold_late_streak)),
            }
        type_summary["threshold_sensitivity"] = sensitivity
        summary["token_types"][token_type] = type_summary
    return summary, rows


def save_sequence_arrays(
    output_dir: Path,
    sequence: str,
    layout: TokenLayout,
    relative: np.ndarray,
    cosine: np.ndarray,
    low: np.ndarray,
) -> None:
    target = output_dir / sequence
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / "token_metrics.npz",
        relative_l2=relative,
        cosine_distance=cosine,
        joint_low_update=low,
        target_layers=np.arange(1, len(relative) + 1),
        token_index_within_frame=np.arange(layout.num_tokens_per_frame),
        patch_token_start=np.asarray(layout.patch_token_start),
    )


def plot_results(rows: list[dict[str, object]], output_dir: Path) -> None:
    metric_specs = (
        ("relative_l2_mean", "Mean relative L2 update", "layerwise_relative_l2.png"),
        ("cosine_distance_mean", "Mean cosine distance", "layerwise_cosine_distance.png"),
        ("joint_low_update_fraction", "Joint low-update fraction", "layerwise_low_update_fraction.png"),
    )
    labels = sorted({str(row["label"]) for row in rows})
    colors = {"all": "black", "camera": "tab:orange", "register": "tab:green", "patch": "tab:blue"}
    for metric, ylabel, filename in metric_specs:
        figure, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 4.5), squeeze=False)
        for axis, label in zip(axes[0], labels):
            for token_type in TOKEN_TYPES:
                selected = [
                    row for row in rows if row["label"] == label and row["token_type"] == token_type
                ]
                selected.sort(key=lambda row: int(row["target_layer"]))
                axis.plot(
                    [row["target_layer"] for row in selected],
                    [row[metric] for row in selected],
                    label=token_type,
                    color=colors[token_type],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.4,
                )
            axis.set_title(label)
            axis.set_xlabel("Target layer")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def output_difference(reference: dict[str, torch.Tensor], hooked: dict[str, torch.Tensor]) -> dict[str, float]:
    differences: dict[str, float] = {}
    for key in ("pose_enc", "depth", "depth_conf"):
        if key in reference and key in hooked:
            delta = (reference[key] - hooked[key].detach().float().cpu()).abs()
            differences[f"{key}_max_abs"] = float(delta.max())
            differences[f"{key}_mean_abs"] = float(delta.mean())
    return differences


def main() -> int:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.sequences):
        raise ValueError("--labels must have the same length as --sequences")
    if not 0 <= args.late_start_layer < 24:
        raise ValueError("--late-start-layer must be in [0, 23]")
    if args.persistent_layers < 1:
        raise ValueError("--persistent-layers must be positive")

    labels = args.labels or ["dynamic" if "walking" in name else "static" for name in args.sequences]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Token analysis requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)

    summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    frame_selections: dict[str, list[str]] = {}
    invariance: dict[str, dict[str, float]] = {}

    for sequence, label in zip(args.sequences, labels):
        paths = evenly_spaced(
            read_sequence_images(args.data_root / sequence, args.frame_source),
            args.num_frames,
        )
        frame_selections[sequence] = [str(path) for path in paths]
        images = load_and_preprocess_images(
            [str(path) for path in paths],
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(device)

        reference: dict[str, torch.Tensor] = {}
        if args.verify_output:
            with torch.inference_mode():
                predictions = model(images)
            reference = {
                key: value.detach().float().cpu()
                for key, value in predictions.items()
                if key in ("pose_enc", "depth", "depth_conf")
            }
            del predictions
            torch.cuda.empty_cache()

        with LayerTokenAnalyzer(
            model,
            num_frames=args.num_frames,
            eps=args.eps,
            relative_l2_threshold=args.relative_l2_threshold,
            cosine_distance_threshold=args.cosine_distance_threshold,
        ) as analyzer:
            with torch.inference_mode():
                predictions = model(images)
            if args.verify_output:
                invariance[sequence] = output_difference(reference, predictions)
            relative, cosine, low = analyzer.arrays()
            assert analyzer.layout is not None
            layout = analyzer.layout

        summary, rows = summarize_sequence(
            sequence,
            label,
            layout,
            relative,
            cosine,
            low,
            model.aggregator.inter_frame_attention_types,
            args.late_start_layer,
            args.persistent_layers,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        save_sequence_arrays(args.output_dir, sequence, layout, relative, cosine, low)
        del images, predictions, reference
        torch.cuda.empty_cache()

    fieldnames = list(all_rows[0])
    with (args.output_dir / "layer_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    result = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "num_frames": args.num_frames,
            "frame_source": args.frame_source,
            "image_resolution": args.image_resolution,
            "resize_mode": args.resize_mode,
            "global_token_merging": False,
            "relative_l2_threshold": args.relative_l2_threshold,
            "cosine_distance_threshold": args.cosine_distance_threshold,
            "persistent_layers": args.persistent_layers,
            "late_start_layer": args.late_start_layer,
        },
        "frame_selections": frame_selections,
        "output_invariance": invariance,
        "sequences": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_results(all_rows, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
