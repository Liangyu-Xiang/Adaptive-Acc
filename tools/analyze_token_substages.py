#!/usr/bin/env python3
"""Analyze VGGT-Omega block substages and patch/register attention non-invasively.

The model implementation is not changed. Forward hooks measure residual updates
at pre-block, post-attention, and post-MLP boundaries. For global inter-frame
blocks, attention probabilities are recomputed from the block input in small
query chunks to obtain patch-to-register attention mass without retaining the
full attention matrix.
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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_token_evolution import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images


DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "token_evolution_3frame"
TOKEN_TYPES = ("camera", "register", "patch")
SUBSTAGES = ("attention", "mlp", "whole_block")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--late-start-layer", type=int, default=12)
    parser.add_argument("--attention-query-chunk", type=int, default=128)
    return parser.parse_args()


def update_arrays(before: torch.Tensor, after: torch.Tensor, eps: float = 1e-8):
    before = before.detach().float()
    after = after.detach().float()
    relative = (after - before).norm(dim=-1) / before.norm(dim=-1).clamp_min(eps)
    cosine = 1.0 - F.cosine_similarity(after, before, dim=-1, eps=eps)
    return relative.cpu().numpy(), cosine.cpu().numpy()


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # Average ranks for ties. Attention values are normally unique, but this
    # keeps the Spearman implementation mathematically well-defined.
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


class SubstageCollector:
    def __init__(self, model, num_frames: int, attention_query_chunk: int) -> None:
        self.model = model
        self.aggregator = model.aggregator
        self.num_frames = num_frames
        self.chunk = attention_query_chunk
        self.rows: list[dict[str, object]] = []
        self.attention_mass: dict[int, np.ndarray] = {}
        self.token_metrics: dict[tuple[str, str, int, str], np.ndarray] = {}
        self.handles = []
        self.pending: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.dense_norm_call = 0

    def __enter__(self):
        for layer, block in enumerate(self.aggregator.frame_blocks):
            self._register_block(block, "frame", layer, "frame")
        for layer, block in enumerate(self.aggregator.inter_frame_blocks):
            kind = self.aggregator.inter_frame_attention_types[layer]
            self._register_block(block, "inter", layer, kind)
            if kind == "global":
                self.handles.append(block.register_forward_pre_hook(self._attention_hook(layer)))

        camera_head = self.model.camera_head
        if camera_head is not None:
            self._register_norm(camera_head.token_norm, "camera_token_norm", "special")
            for layer, block in enumerate(camera_head.trunk):
                self._register_block(block, "camera_trunk", layer, "camera_head")
            self._register_norm(camera_head.trunk_norm, "camera_trunk_norm", "camera")
        if self.model.dense_head is not None:
            self.handles.append(self.model.dense_head.norm.register_forward_hook(self._dense_norm_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()

    def _register_block(self, block, branch: str, layer: int, kind: str) -> None:
        key = f"{branch}:{layer}"

        def pre_hook(_module, inputs):
            self.pending[key] = (inputs[0].detach(), torch.empty(0))

        def post_attention_hook(_module, inputs):
            pre, _ = self.pending[key]
            self.pending[key] = (pre, inputs[0].detach())

        def post_hook(_module, _inputs, output):
            pre, post_attention = self.pending.pop(key)
            self._record_transition(branch, layer, kind, "attention", pre, post_attention)
            self._record_transition(branch, layer, kind, "mlp", post_attention, output)
            self._record_transition(branch, layer, kind, "whole_block", pre, output)

        self.handles.append(block.register_forward_pre_hook(pre_hook))
        self.handles.append(block.norm2.register_forward_pre_hook(post_attention_hook))
        self.handles.append(block.register_forward_hook(post_hook))

    def _reshape(self, tensor: torch.Tensor, branch: str, kind: str) -> torch.Tensor:
        if branch == "frame":
            batch_size = tensor.shape[0] // self.num_frames
            return tensor.view(batch_size, self.num_frames, tensor.shape[1], tensor.shape[2])
        if branch == "inter":
            per_frame = (
                self.aggregator.patch_token_start
                if kind == "register"
                else tensor.shape[1] // self.num_frames
            )
            return tensor.view(tensor.shape[0], self.num_frames, per_frame, tensor.shape[2])
        if branch == "camera_trunk":
            return tensor.view(tensor.shape[0], self.num_frames, self.aggregator.patch_token_start, tensor.shape[2])
        raise ValueError(branch)

    def _record_transition(self, branch, layer, kind, stage, before, after) -> None:
        before = self._reshape(before, branch, kind)
        after = self._reshape(after, branch, kind)
        relative, cosine = update_arrays(before, after)
        if branch in ("frame", "inter"):
            self.token_metrics[(branch, stage, layer, "relative_l2")] = relative
            self.token_metrics[(branch, stage, layer, "cosine_distance")] = cosine
        token_count = before.shape[2]
        ranges = {
            "camera": slice(0, 1),
            "register": slice(1, min(self.aggregator.patch_token_start, token_count)),
            "patch": slice(self.aggregator.patch_token_start, token_count),
        }
        for token_type, token_slice in ranges.items():
            selected_relative = relative[:, :, token_slice]
            if selected_relative.size == 0:
                # Register-only inter blocks do not process patch tokens.
                continue
            selected_cosine = cosine[:, :, token_slice]
            row = {
                "branch": branch,
                "layer": layer,
                "block_type": kind,
                "stage": stage,
                "token_type": token_type,
                "num_tokens": int(selected_relative.size),
            }
            row.update({f"relative_l2_{k}": v for k, v in distribution(selected_relative).items()})
            row.update({f"cosine_distance_{k}": v for k, v in distribution(selected_cosine).items()})
            self.rows.append(row)

    def token_metric_arrays(self) -> dict[str, np.ndarray]:
        """Return dense per-token arrays; absent inter/register patch updates are NaN."""
        sample = self.token_metrics[("frame", "attention", 0, "relative_l2")]
        batch_size, num_frames, num_tokens = sample.shape
        arrays: dict[str, np.ndarray] = {}
        for branch in ("frame", "inter"):
            for metric in ("relative_l2", "cosine_distance"):
                output = np.full(
                    (self.aggregator.depth, len(SUBSTAGES), batch_size, num_frames, num_tokens),
                    np.nan,
                    dtype=np.float32,
                )
                for layer in range(self.aggregator.depth):
                    for stage_index, stage in enumerate(SUBSTAGES):
                        values = self.token_metrics[(branch, stage, layer, metric)]
                        output[layer, stage_index, :, :, : values.shape[2]] = values
                arrays[f"{branch}_{metric}"] = output
        return arrays

    def _attention_hook(self, layer: int):
        def hook(block, inputs):
            x = inputs[0]
            batch_size, total_tokens, hidden_dim = x.shape
            tokens_per_frame = total_tokens // self.num_frames
            patch_start = self.aggregator.patch_token_start
            register_indices = torch.cat(
                [
                    torch.arange(frame * tokens_per_frame + 1, frame * tokens_per_frame + patch_start, device=x.device)
                    for frame in range(self.num_frames)
                ]
            )
            patch_indices = torch.cat(
                [
                    torch.arange(
                        frame * tokens_per_frame + patch_start,
                        (frame + 1) * tokens_per_frame,
                        device=x.device,
                    )
                    for frame in range(self.num_frames)
                ]
            )
            normalized = block.norm1(x)
            qkv = block.attn.qkv(normalized).reshape(
                batch_size, total_tokens, 3, block.attn.num_heads, hidden_dim // block.attn.num_heads
            )
            q, k, _ = torch.unbind(qkv, dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            if block.attn.use_qk_norm:
                q = block.attn.q_norm(q)
                k = block.attn.k_norm(k)
            k_t = k.float().transpose(-2, -1)
            masses = []
            for start in range(0, len(patch_indices), self.chunk):
                indices = patch_indices[start : start + self.chunk]
                logits = torch.matmul(q[:, :, indices].float(), k_t) * block.attn.scale
                probabilities = logits.softmax(dim=-1)
                masses.append(probabilities[..., register_indices].sum(dim=-1).mean(dim=1).cpu())
                del logits, probabilities
            mass = torch.cat(masses, dim=1).view(
                batch_size, self.num_frames, tokens_per_frame - patch_start
            )
            self.attention_mass[layer] = mass.numpy()
            del normalized, qkv, q, k, k_t, masses, mass

        return hook

    def _register_norm(self, module, name: str, token_type: str) -> None:
        def hook(_module, inputs, output):
            relative, cosine = update_arrays(inputs[0], output)
            row = {
                "branch": "head_norm",
                "layer": -1,
                "block_type": name,
                "stage": "normalization",
                "token_type": token_type,
                "num_tokens": int(relative.size),
            }
            row.update({f"relative_l2_{k}": v for k, v in distribution(relative).items()})
            row.update({f"cosine_distance_{k}": v for k, v in distribution(cosine).items()})
            self.rows.append(row)

        self.handles.append(module.register_forward_hook(hook))

    def _dense_norm_hook(self, _module, inputs, output):
        layers = self.model.dense_head.intermediate_layer_idx
        layer = layers[self.dense_norm_call % len(layers)]
        self.dense_norm_call += 1
        relative, cosine = update_arrays(inputs[0], output)
        row = {
            "branch": "head_norm",
            "layer": layer,
            "block_type": "dense_token_norm",
            "stage": "normalization",
            "token_type": "patch",
            "num_tokens": int(relative.size),
        }
        row.update({f"relative_l2_{k}": v for k, v in distribution(relative).items()})
        row.update({f"cosine_distance_{k}": v for k, v in distribution(cosine).items()})
        self.rows.append(row)


def plot_substages(rows: list[dict[str, object]], target: Path) -> None:
    selected = [row for row in rows if row["branch"] in ("frame", "inter")]
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for column, token_type in enumerate(TOKEN_TYPES):
        for row_index, metric in enumerate(("relative_l2_median", "cosine_distance_median")):
            axis = axes[row_index, column]
            for branch, stage, style in (
                ("frame", "attention", "-"),
                ("frame", "mlp", "--"),
                ("inter", "attention", "-"),
                ("inter", "mlp", "--"),
            ):
                points = [
                    row for row in selected
                    if row["token_type"] == token_type and row["branch"] == branch and row["stage"] == stage
                ]
                points.sort(key=lambda row: int(row["layer"]))
                axis.plot(
                    [int(row["layer"]) for row in points],
                    [float(row[metric]) for row in points],
                    linestyle=style,
                    marker="o",
                    markersize=2.5,
                    label=f"{branch}-{stage}",
                )
            axis.set_title(token_type)
            axis.set_ylabel(metric.replace("_", " "))
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("Block index")
            if column == 0 and row_index == 0:
                axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(target, dpi=180)
    plt.close(figure)


def _select_token_type(values: np.ndarray, token_type: str, patch_start: int) -> np.ndarray:
    if token_type == "camera":
        selected = values[..., 0:1]
    elif token_type == "register":
        selected = values[..., 1:patch_start]
    elif token_type == "patch":
        selected = values[..., patch_start:]
    else:
        raise ValueError(token_type)
    # [layer, stage, batch, frame, token] -> [layer, stage, flattened token]
    return selected.reshape(selected.shape[0], selected.shape[1], -1)


def save_and_plot_token_substages(
    sequence_dir: Path,
    arrays: dict[str, np.ndarray],
    patch_start: int,
    num_frames: int,
    attention_types: list[str],
    late_start_layer: int,
) -> None:
    np.savez_compressed(
        sequence_dir / "substage_token_metrics.npz",
        **arrays,
        block_indices=np.arange(arrays["frame_relative_l2"].shape[0]),
        stage_names=np.asarray(SUBSTAGES),
        inter_block_types=np.asarray(attention_types),
        patch_token_start=np.asarray(patch_start),
    )

    late = np.arange(arrays["frame_relative_l2"].shape[0]) >= late_start_layer
    orders: dict[str, list[int]] = {}
    for branch in ("frame", "inter"):
        for metric, metric_label in (
            ("relative_l2", "Relative L2 update"),
            ("cosine_distance", "Cosine distance"),
        ):
            source = arrays[f"{branch}_{metric}"]
            selected_by_type = {
                token_type: _select_token_type(source, token_type, patch_start)
                for token_type in TOKEN_TYPES
            }
            for ordering in ("original", "sorted_by_late_whole_block_mean"):
                figure, axes = plt.subplots(3, 3, figsize=(16, 13), squeeze=False)
                for row_index, token_type in enumerate(TOKEN_TYPES):
                    selected = selected_by_type[token_type]
                    if ordering == "original":
                        order = np.arange(selected.shape[-1])
                    else:
                        score = np.nanmean(selected[late, SUBSTAGES.index("whole_block")], axis=0)
                        order = np.argsort(score)
                    orders[f"{branch}_{metric}_{token_type}_{ordering}"] = order.tolist()

                    finite = selected[np.isfinite(selected)]
                    vmin, vmax = np.percentile(finite, (2, 98))
                    if vmax <= vmin:
                        vmax = vmin + 1e-6
                    cmap = plt.get_cmap("viridis").copy()
                    cmap.set_bad("lightgray")
                    for column, stage in enumerate(SUBSTAGES):
                        axis = axes[row_index, column]
                        rendered = axis.imshow(
                            selected[:, column, order].T,
                            aspect="auto",
                            interpolation="nearest",
                            origin="lower",
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                        )
                        axis.set_title(f"{token_type}: {stage}")
                        axis.set_xlabel("Block index")
                        axis.set_ylabel("Token index")
                        axis.set_xticks(np.arange(0, source.shape[0], 2))
                        if ordering == "original":
                            tokens_per_frame = selected.shape[-1] // num_frames
                            for boundary in range(1, num_frames):
                                axis.axhline(
                                    boundary * tokens_per_frame - 0.5,
                                    color="white",
                                    linewidth=0.8,
                                    alpha=0.9,
                                )
                    figure.colorbar(
                        rendered,
                        ax=axes[row_index].tolist(),
                        fraction=0.015,
                        pad=0.01,
                    )
                note = "gray patch columns = register-only inter blocks" if branch == "inter" else ""
                figure.suptitle(
                    f"{branch} block per-token {metric_label}: {ordering}\n{note}",
                    fontsize=14,
                )
                figure.savefig(
                    sequence_dir / f"substage_token_heatmap_{branch}_{metric}_{ordering}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                plt.close(figure)
    (sequence_dir / "substage_token_orders.json").write_text(json.dumps(orders) + "\n")


def analyze_attention(
    sequence_dir: Path,
    attention_mass: dict[int, np.ndarray],
    late_start: int,
    patch_grid: tuple[int, int],
):
    metrics = np.load(sequence_dir / "token_metrics.npz")
    target_layers = metrics["target_layers"]
    patch_start = int(metrics["patch_token_start"])
    late_update = metrics["relative_l2"][target_layers >= late_start, :, :, patch_start:].mean(axis=0)
    late_layers = sorted(layer for layer in attention_mass if layer >= late_start)
    late_attention = np.stack([attention_mass[layer] for layer in late_layers]).mean(axis=0)
    update_flat = late_update.reshape(-1)
    attention_flat = late_attention.reshape(-1)
    order = np.argsort(update_flat)
    count = min(20, len(order) // 2)
    result = {
        "late_start_layer": late_start,
        "global_attention_layers": late_layers,
        "pearson_late_update_vs_register_attention": correlation(update_flat, attention_flat),
        "spearman_late_update_vs_register_attention": correlation(
            rankdata(update_flat), rankdata(attention_flat)
        ),
        "lowest_20_update_attention_mean": float(attention_flat[order[:count]].mean()),
        "highest_20_update_attention_mean": float(attention_flat[order[-count:]].mean()),
        "lowest_quartile_attention_mean": float(attention_flat[order[: len(order) // 4]].mean()),
        "highest_quartile_attention_mean": float(attention_flat[order[-len(order) // 4 :]].mean()),
    }
    np.savez_compressed(
        sequence_dir / "patch_register_attention.npz",
        layers=np.asarray(sorted(attention_mass)),
        patch_to_register_mass=np.stack([attention_mass[layer] for layer in sorted(attention_mass)]),
        late_patch_update=late_update,
        late_patch_register_attention=late_attention,
    )
    figure, axis = plt.subplots(figsize=(6.5, 5))
    axis.hexbin(update_flat, attention_flat, gridsize=45, mincnt=1, cmap="viridis")
    axis.set_xlabel("Mean late relative L2 update")
    axis.set_ylabel("Mean late patch-to-register attention mass")
    axis.set_title(
        f"Pearson={result['pearson_late_update_vs_register_attention']:.3f}, "
        f"Spearman={result['spearman_late_update_vs_register_attention']:.3f}"
    )
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(sequence_dir / "patch_register_correlation.png", dpi=180)
    plt.close(figure)

    requested_layers = [4, 8, 12, 16, 23]
    shown_layers = [layer for layer in requested_layers if layer in attention_mass]
    spatial_values = [attention_mass[layer] for layer in shown_layers] + [late_attention]
    spatial_titles = [f"global block {layer}" for layer in shown_layers] + ["late global mean"]
    stacked = np.stack(spatial_values)
    vmin, vmax = np.percentile(stacked, (2, 98))
    spatial_figure, spatial_axes = plt.subplots(
        late_attention.shape[1], len(spatial_values),
        figsize=(3.0 * len(spatial_values), 2.6 * late_attention.shape[1]),
        squeeze=False,
    )
    for column, (values, title) in enumerate(zip(spatial_values, spatial_titles)):
        for frame, axis in enumerate(spatial_axes[:, column]):
            rendered = axis.imshow(
                values[0, frame].reshape(patch_grid),
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axis.set_title(f"{title}, F{frame}")
            axis.axis("off")
    spatial_figure.colorbar(
        rendered, ax=spatial_axes.ravel().tolist(), fraction=0.012, pad=0.01,
        label="Patch-query attention mass to register keys",
    )
    spatial_figure.savefig(
        sequence_dir / "patch_register_attention_spatial.png", dpi=180, bbox_inches="tight"
    )
    plt.close(spatial_figure)
    return result


def main() -> int:
    args = parse_args()
    summary_path = args.analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Substage analysis requires CUDA")
    model = load_model(args.checkpoint, device)
    all_results = {}

    for sequence, paths in summary["frame_selections"].items():
        images = load_and_preprocess_images(
            paths, mode=args.resize_mode, image_resolution=args.image_resolution
        ).to(device)
        with SubstageCollector(model, len(paths), args.attention_query_chunk) as collector:
            with torch.inference_mode():
                predictions = model(images)
        sequence_dir = args.analysis_dir / sequence
        fieldnames = list(collector.rows[0])
        with (sequence_dir / "substage_stats.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(collector.rows)
        plot_substages(collector.rows, sequence_dir / "substage_updates.png")
        token_arrays = collector.token_metric_arrays()
        save_and_plot_token_substages(
            sequence_dir,
            token_arrays,
            model.aggregator.patch_token_start,
            len(paths),
            model.aggregator.inter_frame_attention_types,
            args.late_start_layer,
        )
        attention_result = analyze_attention(
            sequence_dir,
            collector.attention_mass,
            args.late_start_layer,
            (images.shape[-2] // model.aggregator.patch_size, images.shape[-1] // model.aggregator.patch_size),
        )
        head_shapes = {
            "camera_token_norm": "2048 -> 2048; relative update defined",
            "camera_trunk_norm": "2048 -> 2048; relative update defined",
            "dense_token_norm": "2048 -> 2048; relative update defined for cached layers 4/11/17/23",
            "camera_prediction_branch": "2048 -> 1024 -> 9; token-relative update is not defined after dimensionality change",
            "dense_prediction_head": "token grid -> convolutional multiscale feature maps; token-relative update is not defined",
        }
        result = {
            "attention": attention_result,
            "head_stage_shapes": head_shapes,
            "note": "All hooks are read-only. Attention probabilities are recomputed and do not replace model attention.",
        }
        (sequence_dir / "substage_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        all_results[sequence] = result
        del images, predictions, token_arrays
        torch.cuda.empty_cache()

    (args.analysis_dir / "substage_summary.json").write_text(json.dumps(all_results, indent=2) + "\n")
    print(json.dumps(all_results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
