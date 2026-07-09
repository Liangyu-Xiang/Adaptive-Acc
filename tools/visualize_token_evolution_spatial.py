#!/usr/bin/env python3
"""Create spatial and distributional views from token-evolution NPZ outputs."""

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

from vggt_omega.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 23])
    parser.add_argument("--late-start-layer", type=int, default=12)
    parser.add_argument("--region-rows", type=int, default=4)
    parser.add_argument("--region-cols", type=int, default=4)
    parser.add_argument("--relative-l2-threshold", type=float, default=0.25)
    parser.add_argument("--cosine-distance-threshold", type=float, default=0.02)
    return parser.parse_args()


def robust_limits(values: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, low)), float(np.percentile(finite, high))


def load_sequence_data(
    root: Path,
    sequence: str,
    paths: list[str],
    config: dict[str, object],
) -> dict[str, np.ndarray]:
    metrics = np.load(root / sequence / "token_metrics.npz")
    images = load_and_preprocess_images(
        paths,
        mode=str(config["resize_mode"]),
        image_resolution=int(config["image_resolution"]),
    ).permute(0, 2, 3, 1).numpy()
    height, width = images.shape[1:3]
    patch_size = 16
    grid_h, grid_w = height // patch_size, width // patch_size
    patch_start = int(metrics["patch_token_start"])
    patch_count = metrics["relative_l2"].shape[-1] - patch_start
    if grid_h * grid_w != patch_count:
        raise ValueError(
            f"Patch grid {grid_h}x{grid_w} does not match {patch_count} tokens for {sequence}"
        )
    relative = metrics["relative_l2"][:, 0]
    cosine = metrics["cosine_distance"][:, 0]
    return {
        "images": images,
        "relative": relative,
        "cosine": cosine,
        "patch_relative": relative[:, :, patch_start:].reshape(-1, len(images), grid_h, grid_w),
        "patch_cosine": cosine[:, :, patch_start:].reshape(-1, len(images), grid_h, grid_w),
        "register_relative": relative[:, :, 1:patch_start],
        "register_cosine": cosine[:, :, 1:patch_start],
        "grid_shape": np.asarray((grid_h, grid_w)),
        "patch_start": np.asarray(patch_start),
    }


def upsample(values: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = F.interpolate(
        torch.from_numpy(values).float()[None, None],
        size=(height, width),
        mode="nearest",
    )
    return tensor[0, 0].numpy()


def plot_layer_overlays(
    sequence_dir: Path,
    data: dict[str, np.ndarray],
    layers: list[int],
    key: str,
    label: str,
) -> None:
    images = data["images"]
    values = data[key]
    figure, axes = plt.subplots(
        len(images),
        len(layers),
        figsize=(3.1 * len(layers), 2.7 * len(images)),
        squeeze=False,
    )
    for column, target_layer in enumerate(layers):
        layer_values = values[target_layer - 1]
        median = max(float(np.median(layer_values)), 1e-12)
        normalized = layer_values / median
        for frame, axis in enumerate(axes[:, column]):
            heatmap = upsample(normalized[frame], *images.shape[1:3])
            axis.imshow(images[frame])
            rendered = axis.imshow(heatmap, cmap="magma", vmin=0.25, vmax=2.0, alpha=0.58)
            axis.set_title(f"L{target_layer}, F{frame}\nmean={layer_values[frame].mean():.3g}")
            axis.axis("off")
    figure.suptitle(f"{label}: patch update / layer median", fontsize=13)
    figure.colorbar(rendered, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    figure.savefig(sequence_dir / f"spatial_{key}_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_late_summary(
    sequence_dir: Path,
    data: dict[str, np.ndarray],
    late_start_layer: int,
    relative_threshold: float,
    cosine_threshold: float,
) -> None:
    images = data["images"]
    relative = data["patch_relative"]
    cosine = data["patch_cosine"]
    target_layers = np.arange(1, len(relative) + 1)
    late = target_layers >= late_start_layer
    late_relative = relative[late].mean(axis=0)
    late_cosine = cosine[late].mean(axis=0)
    late_low = ((relative[late] <= relative_threshold) & (cosine[late] <= cosine_threshold)).mean(axis=0)
    summaries = (
        (late_relative, "Late mean relative L2", "viridis"),
        (late_cosine, "Late mean cosine distance", "plasma"),
        (late_low, "Moderate low-update frequency", "magma"),
    )
    figure, axes = plt.subplots(len(images), 3, figsize=(11, 3.1 * len(images)), squeeze=False)
    for column, (values, title, cmap) in enumerate(summaries):
        vmin, vmax = ((0.0, 1.0) if column == 2 else robust_limits(values))
        for frame, axis in enumerate(axes[:, column]):
            heatmap = upsample(values[frame], *images.shape[1:3])
            axis.imshow(images[frame])
            rendered = axis.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.60)
            axis.set_title(f"{title}, F{frame}")
            axis.axis("off")
        figure.colorbar(rendered, ax=axes[:, column].ravel().tolist(), fraction=0.025, pad=0.01)
    figure.savefig(sequence_dir / "spatial_late_patch_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_quantiles(sequence_dir: Path, data: dict[str, np.ndarray]) -> None:
    target_layers = np.arange(1, len(data["relative"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    specifications = (
        ("relative", "register_relative", "Relative L2 update"),
        ("cosine", "register_cosine", "Cosine distance"),
    )
    for axis, (patch_key, register_key, title) in zip(axes, specifications):
        patch = data[f"patch_{patch_key}"].reshape(len(target_layers), -1)
        register = data[register_key].reshape(len(target_layers), -1)
        for values, token_type, color in (
            (patch, "patch", "tab:blue"),
            (register, "register", "tab:green"),
        ):
            p10, median, p90 = np.percentile(values, (10, 50, 90), axis=1)
            axis.fill_between(target_layers, p10, p90, color=color, alpha=0.18)
            axis.plot(target_layers, median, color=color, marker="o", markersize=3, label=token_type)
        axis.set_xlabel("Target layer")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(sequence_dir / "token_update_quantile_bands.png", dpi=180)
    plt.close(figure)


def save_region_statistics(
    sequence_dir: Path,
    data: dict[str, np.ndarray],
    region_rows: int,
    region_cols: int,
    relative_threshold: float,
    cosine_threshold: float,
) -> None:
    relative = data["patch_relative"]
    cosine = data["patch_cosine"]
    row_groups = np.array_split(np.arange(relative.shape[-2]), region_rows)
    col_groups = np.array_split(np.arange(relative.shape[-1]), region_cols)
    rows = []
    for transition in range(len(relative)):
        for frame in range(relative.shape[1]):
            for region_row, ys in enumerate(row_groups):
                for region_col, xs in enumerate(col_groups):
                    selection = np.ix_(ys, xs)
                    rel = relative[transition, frame][selection]
                    cos = cosine[transition, frame][selection]
                    rows.append(
                        {
                            "target_layer": transition + 1,
                            "frame": frame,
                            "region_row": region_row,
                            "region_col": region_col,
                            "relative_l2_mean": float(rel.mean()),
                            "relative_l2_median": float(np.median(rel)),
                            "cosine_distance_mean": float(cos.mean()),
                            "cosine_distance_median": float(np.median(cos)),
                            "moderate_low_update_fraction": float(
                                np.mean((rel <= relative_threshold) & (cos <= cosine_threshold))
                            ),
                        }
                    )
    with (sequence_dir / "patch_region_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_extreme_patch_locations(
    sequence_dir: Path,
    data: dict[str, np.ndarray],
    late_start_layer: int,
) -> None:
    relative = data["patch_relative"]
    target_layers = np.arange(1, len(relative) + 1)
    scores = relative[target_layers >= late_start_layer].mean(axis=(0, 1)).reshape(-1)
    grid_h, grid_w = map(int, data["grid_shape"])
    count = min(16, len(scores) // 2)
    order = np.argsort(scores)

    def records(indices: np.ndarray) -> list[dict[str, float | int]]:
        return [
            {
                "patch_index": int(index),
                "row": int(index // grid_w),
                "col": int(index % grid_w),
                "late_relative_l2_mean": float(scores[index]),
            }
            for index in indices
        ]

    output = {
        "lowest_update_locations": records(order[:count]),
        "highest_update_locations": records(order[-count:][::-1]),
        "grid_height": grid_h,
        "grid_width": grid_w,
    }
    (sequence_dir / "extreme_patch_locations.json").write_text(json.dumps(output, indent=2) + "\n")


def flattened_token_values(data: dict[str, np.ndarray], metric: str, token_type: str) -> np.ndarray:
    values = data[metric]
    patch_start = int(data["patch_start"])
    if token_type == "camera":
        selected = values[:, :, 0:1]
    elif token_type == "register":
        selected = values[:, :, 1:patch_start]
    elif token_type == "patch":
        selected = values[:, :, patch_start:]
    else:
        raise ValueError(token_type)
    return selected.reshape(len(selected), -1)


def plot_token_layer_heatmaps(
    sequence_dir: Path,
    data: dict[str, np.ndarray],
    late_start_layer: int,
) -> None:
    target_layers = np.arange(1, len(data["relative"]) + 1)
    late = target_layers >= late_start_layer
    order_output = {}
    for metric, label in (("relative", "Relative L2 update"), ("cosine", "Cosine distance")):
        figure, axes = plt.subplots(3, 2, figsize=(15, 13), squeeze=False)
        for row, token_type in enumerate(("patch", "register", "camera")):
            values = flattened_token_values(data, metric, token_type)
            stability_order = np.argsort(values[late].mean(axis=0))
            order_output[f"{metric}_{token_type}"] = stability_order.tolist()
            vmin, vmax = robust_limits(values)
            for column, (title, order) in enumerate(
                (("original token order", np.arange(values.shape[1])), ("sorted by late mean", stability_order))
            ):
                rendered = axes[row, column].imshow(
                    values[:, order].T,
                    aspect="auto",
                    interpolation="nearest",
                    origin="lower",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[row, column].set_title(f"{token_type}: {title}")
                axes[row, column].set_xlabel("Target layer")
                axes[row, column].set_ylabel("Token index")
                axes[row, column].set_xticks(np.arange(0, len(target_layers), 2))
                axes[row, column].set_xticklabels(target_layers[::2])
                if column == 0:
                    tokens_per_frame = values.shape[1] // len(data["images"])
                    for boundary in range(1, len(data["images"])):
                        axes[row, column].axhline(
                            boundary * tokens_per_frame - 0.5,
                            color="white",
                            linewidth=0.8,
                            alpha=0.9,
                        )
                figure.colorbar(rendered, ax=axes[row, column], fraction=0.025, pad=0.01)
        figure.suptitle(label, fontsize=15)
        figure.tight_layout()
        figure.savefig(sequence_dir / f"token_layer_heatmap_{metric}.png", dpi=180)
        plt.close(figure)
    (sequence_dir / "token_stability_orders.json").write_text(json.dumps(order_output) + "\n")


def plot_token_trajectories(
    sequence_dir: Path,
    data: dict[str, np.ndarray],
    late_start_layer: int,
) -> None:
    target_layers = np.arange(1, len(data["relative"]) + 1)
    late = target_layers >= late_start_layer
    patch_relative = flattened_token_values(data, "relative", "patch")
    late_score = patch_relative[late].mean(axis=0)
    order = np.argsort(late_score)
    count = min(20, patch_relative.shape[1] // 3)
    generator = np.random.RandomState(42)
    groups = {
        "lowest-20 patch": ("patch", order[:count]),
        "highest-20 patch": ("patch", order[-count:]),
        "random-20 patch": (
            "patch",
            generator.choice(patch_relative.shape[1], count, replace=False),
        ),
        "all register": ("register", None),
        "all camera": ("camera", None),
    }
    selections = {}
    figure, axes = plt.subplots(len(groups), 2, figsize=(14, 3.1 * len(groups)), squeeze=False)
    for row, (group_name, (token_type, indices)) in enumerate(groups.items()):
        selections[group_name] = None if indices is None else np.asarray(indices).tolist()
        for column, (metric, ylabel) in enumerate(
            (("relative", "Relative L2 update"), ("cosine", "Cosine distance"))
        ):
            values = flattened_token_values(data, metric, token_type)
            if indices is not None:
                values = values[:, indices]
            axis = axes[row, column]
            axis.plot(target_layers, values, color="tab:blue", alpha=0.20, linewidth=0.7)
            axis.plot(
                target_layers,
                np.median(values, axis=1),
                color="black",
                linewidth=2.2,
                label="median",
            )
            axis.set_title(group_name)
            axis.set_xlabel("Target layer")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend()
    figure.tight_layout()
    figure.savefig(sequence_dir / "individual_token_trajectories.png", dpi=180)
    plt.close(figure)
    (sequence_dir / "trajectory_token_selections.json").write_text(json.dumps(selections, indent=2) + "\n")


def plot_patch_boxplots(sequence_dir: Path, data: dict[str, np.ndarray]) -> None:
    target_layers = np.arange(1, len(data["relative"]) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for axis, (metric, ylabel) in zip(
        axes,
        (("relative", "Relative L2 update"), ("cosine", "Cosine distance")),
    ):
        values = flattened_token_values(data, metric, "patch")
        axis.boxplot(
            [values[layer] for layer in range(len(values))],
            positions=target_layers,
            widths=0.6,
            showfliers=False,
            whis=(10, 90),
            patch_artist=True,
            boxprops={"facecolor": "tab:blue", "alpha": 0.45},
            medianprops={"color": "black", "linewidth": 1.4},
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Target layer")
    axes[-1].set_xticks(target_layers)
    figure.tight_layout()
    figure.savefig(sequence_dir / "patch_update_boxplots.png", dpi=180)
    plt.close(figure)


def plot_low_update_jaccard(
    sequence_dir: Path,
    data: dict[str, np.ndarray],
    relative_threshold: float,
    cosine_threshold: float,
) -> None:
    target_layers = np.arange(1, len(data["relative"]) + 1)
    rows = []
    figure, axis = plt.subplots(figsize=(12, 4.5))
    for token_type, color in (("patch", "tab:blue"), ("register", "tab:green"), ("camera", "tab:orange")):
        relative = flattened_token_values(data, "relative", token_type)
        cosine = flattened_token_values(data, "cosine", token_type)
        low = (relative <= relative_threshold) & (cosine <= cosine_threshold)
        similarities = []
        for layer in range(len(low) - 1):
            intersection = np.count_nonzero(low[layer] & low[layer + 1])
            union = np.count_nonzero(low[layer] | low[layer + 1])
            similarity = float(intersection / union) if union else float("nan")
            similarities.append(similarity)
            rows.append(
                {
                    "token_type": token_type,
                    "first_target_layer": int(target_layers[layer]),
                    "second_target_layer": int(target_layers[layer + 1]),
                    "intersection": int(intersection),
                    "union": int(union),
                    "jaccard": similarity,
                }
            )
        axis.plot(target_layers[1:], similarities, marker="o", markersize=3, label=token_type, color=color)
    axis.set_xlabel("Second target layer in adjacent pair")
    axis.set_ylabel("Low-update set Jaccard")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(sequence_dir / "low_update_jaccard.png", dpi=180)
    plt.close(figure)
    with (sequence_dir / "low_update_jaccard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    summary = json.loads((args.analysis_dir / "summary.json").read_text())
    config = summary["config"]
    for sequence in summary["frame_selections"]:
        sequence_dir = args.analysis_dir / sequence
        data = load_sequence_data(
            args.analysis_dir,
            sequence,
            summary["frame_selections"][sequence],
            config,
        )
        plot_layer_overlays(sequence_dir, data, args.layers, "patch_relative", "Relative L2")
        plot_layer_overlays(sequence_dir, data, args.layers, "patch_cosine", "Cosine distance")
        plot_late_summary(
            sequence_dir,
            data,
            args.late_start_layer,
            args.relative_l2_threshold,
            args.cosine_distance_threshold,
        )
        plot_quantiles(sequence_dir, data)
        save_region_statistics(
            sequence_dir,
            data,
            args.region_rows,
            args.region_cols,
            args.relative_l2_threshold,
            args.cosine_distance_threshold,
        )
        save_extreme_patch_locations(sequence_dir, data, args.late_start_layer)
        plot_token_layer_heatmaps(sequence_dir, data, args.late_start_layer)
        plot_token_trajectories(sequence_dir, data, args.late_start_layer)
        plot_patch_boxplots(sequence_dir, data)
        plot_low_update_jaccard(
            sequence_dir,
            data,
            args.relative_l2_threshold,
            args.cosine_distance_threshold,
        )
        print(f"Saved spatial analysis for {sequence} to {sequence_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
