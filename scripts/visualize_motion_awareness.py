#!/usr/bin/env python3
"""Visualize motion-aware VGGT-Omega patch-token clusters.

This follows Sec. 5 / Fig. 9 of the VGGT-Omega paper: normalize intermediate
image tokens jointly over space and time, reduce them with PCA, and cluster
them with k-means. Patch labels are mapped back to pixels and the selected
motion cluster is rendered in red over a darkened RGB frame.

The paper does not publish its k, PCA dimension, or cluster-selection rule.
This implementation therefore saves every cluster and uses a documented
compactness heuristic for the default red cluster; ``--motion-cluster`` can
override that label after inspecting ``all_clusters`` outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vggt_omega.utils.load_fn import load_and_preprocess_images

from eval_tum_dynamics_paper import DEFAULT_CHECKPOINT, DEFAULT_DATA_ROOT, load_model


PALETTE = np.asarray(
    [
        [230, 35, 35],
        [35, 170, 245],
        [245, 190, 35],
        [65, 190, 95],
        [170, 80, 220],
        [35, 210, 190],
        [245, 110, 35],
        [220, 90, 155],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce VGGT-Omega motion-aware token clustering.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--sequence",
        default="rgbd_dataset_freiburg3_walking_static",
        help="TUM-Dynamics sequence directory name.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/motion_awareness"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument(
        "--frame-source",
        choices=("rgb_90", "full"),
        default="rgb_90",
        help="Use the contiguous prepared subset or the complete RGB stream.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 13, 23])
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--pca-dim", type=int, default=3)
    parser.add_argument("--kmeans-iterations", type=int, default=50)
    parser.add_argument("--kmeans-restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument(
        "--normalization",
        choices=("standardize", "l2"),
        default="l2",
        help="Normalize tokens jointly across space/time before PCA.",
    )
    parser.add_argument(
        "--motion-cluster",
        type=int,
        default=None,
        help="Cluster ID to render red. Default: automatically choose the most spatially compact cluster.",
    )
    parser.add_argument("--min-cluster-fraction", type=float, default=0.02)
    parser.add_argument("--overlay-alpha", type=float, default=0.82)
    parser.add_argument("--background-brightness", type=float, default=0.72)
    parser.add_argument("--fps", type=float, default=5.0)
    return parser.parse_args()


def list_images(sequence_dir: Path, source: str) -> list[Path]:
    if source == "rgb_90":
        image_dir = sequence_dir / "rgb_90"
        paths = list(image_dir.glob("*.png"))
    else:
        rows = []
        with (sequence_dir / "rgb.txt").open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    fields = line.split()
                    rows.append(sequence_dir / fields[1])
        paths = rows
    paths = sorted(paths, key=lambda path: float(path.stem))
    if not paths:
        raise FileNotFoundError(f"No input frames found for {sequence_dir} ({source})")
    return paths


def evenly_sample(paths: Sequence[Path], count: int) -> list[Path]:
    if count < 2:
        raise ValueError("--num-frames must be at least 2")
    if count > len(paths):
        raise ValueError(f"Requested {count} frames but only {len(paths)} are available")
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def normalize_features(features: torch.Tensor, method: str) -> torch.Tensor:
    features = features.float()
    if method == "standardize":
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (features - mean) / std
    return F.normalize(features, dim=-1)


def pca_reduce(features: torch.Tensor, dimension: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if dimension < 1 or dimension >= min(features.shape):
        raise ValueError(f"Invalid PCA dimension {dimension} for features {tuple(features.shape)}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    centered = features - features.mean(dim=0, keepdim=True)
    # Low-rank PCA avoids materializing a 2048x2048 covariance matrix.
    _, _, components = torch.pca_lowrank(centered, q=dimension, center=False, niter=4)
    reduced = centered @ components
    reduced = (reduced - reduced.mean(dim=0, keepdim=True)) / reduced.std(dim=0, keepdim=True).clamp_min(1e-6)
    return reduced, components


def run_kmeans_once(
    features: torch.Tensor, num_clusters: int, iterations: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    initial_indices = torch.randperm(len(features), generator=generator, device=features.device)[:num_clusters]
    centers = features[initial_indices].clone()
    previous_labels = None
    for _ in range(iterations):
        distances = torch.cdist(features, centers)
        labels = distances.argmin(dim=1)
        if previous_labels is not None and torch.equal(labels, previous_labels):
            break
        previous_labels = labels
        new_centers = []
        for cluster_id in range(num_clusters):
            members = features[labels == cluster_id]
            if len(members) == 0:
                farthest = distances.min(dim=1).values.argmax()
                new_centers.append(features[farthest])
            else:
                new_centers.append(members.mean(dim=0))
        centers = torch.stack(new_centers)
    distances = torch.cdist(features, centers)
    labels = distances.argmin(dim=1)
    inertia = float(distances.gather(1, labels[:, None]).square().sum().item())
    return labels, centers, inertia


def run_kmeans(
    features: torch.Tensor,
    num_clusters: int,
    iterations: int,
    restarts: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if num_clusters < 2 or num_clusters > len(features):
        raise ValueError("--num-clusters must be between 2 and the number of patch tokens")
    best = None
    for restart in range(restarts):
        candidate = run_kmeans_once(features, num_clusters, iterations, seed + restart)
        if best is None or candidate[2] < best[2]:
            best = candidate
    assert best is not None
    return best


def cluster_statistics(labels: np.ndarray, num_clusters: int) -> list[dict[str, float | int]]:
    """Measure occupancy and within-frame localization for automatic selection."""
    frames, height, width = labels.shape
    statistics: list[dict[str, float | int]] = []
    for cluster_id in range(num_clusters):
        mask = labels == cluster_id
        bbox_fractions = []
        fill_ratios = []
        for frame_index in range(frames):
            ys, xs = np.where(mask[frame_index])
            if len(xs) == 0:
                continue
            bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
            bbox_fractions.append(bbox_area / (height * width))
            fill_ratios.append(len(xs) / bbox_area)
        statistics.append(
            {
                "cluster_id": cluster_id,
                "fraction": float(mask.mean()),
                "mean_bbox_fraction": float(np.mean(bbox_fractions)) if bbox_fractions else 1.0,
                "mean_bbox_fill": float(np.mean(fill_ratios)) if fill_ratios else 0.0,
                "frames_present": len(bbox_fractions),
            }
        )
    return statistics


def choose_motion_cluster(statistics: list[dict[str, float | int]], min_fraction: float) -> int:
    candidates = [row for row in statistics if float(row["fraction"]) >= min_fraction]
    if not candidates:
        candidates = statistics
    # Dynamic people/objects normally occupy fewer tokens than the aggregate
    # static background. This only resolves k-means' arbitrary label IDs; all
    # clusters are saved for verification and the choice can be overridden.
    selected = min(
        candidates,
        key=lambda row: (float(row["fraction"]), float(row["mean_bbox_fraction"])),
    )
    return int(selected["cluster_id"])


def tensor_images_to_uint8(images: torch.Tensor) -> np.ndarray:
    return (
        images.detach()
        .float()
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )


def upsample_labels(labels: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(labels.astype(np.float32))[:, None]
    return F.interpolate(tensor, size=(height, width), mode="nearest")[:, 0].numpy().astype(np.int64)


def render_motion_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    background_brightness: float,
) -> np.ndarray:
    base = image.astype(np.float32) * background_brightness
    red = np.asarray([235, 25, 25], dtype=np.float32)
    base[mask] = (1 - alpha) * base[mask] + alpha * red
    return np.clip(base, 0, 255).astype(np.uint8)


def render_all_clusters(image: np.ndarray, labels: np.ndarray, alpha: float = 0.62) -> np.ndarray:
    colors = PALETTE[labels % len(PALETTE)]
    rendered = (1 - alpha) * image.astype(np.float32) + alpha * colors
    return np.clip(rendered, 0, 255).astype(np.uint8)


def write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def save_layer_visualization(
    output_dir: Path,
    images: np.ndarray,
    labels: np.ndarray,
    timestamps: Sequence[str],
    selected_cluster: int,
    statistics: list[dict[str, float | int]],
    alpha: float,
    background_brightness: float,
    fps: float,
) -> None:
    height, width = images.shape[1:3]
    pixel_labels = upsample_labels(labels, height, width)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "patch_labels.npy", labels.astype(np.int16))
    np.save(output_dir / "motion_mask_pixels.npy", pixel_labels == selected_cluster)
    motion_dir = output_dir / "motion_red"
    clusters_dir = output_dir / "all_clusters"
    motion_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    motion_frames = []
    cluster_frames = []
    for index, (image, label_map, timestamp) in enumerate(zip(images, pixel_labels, timestamps)):
        motion = render_motion_overlay(
            image, label_map == selected_cluster, alpha, background_brightness
        )
        clusters = render_all_clusters(image, label_map)
        Image.fromarray(motion).save(motion_dir / f"{index:03d}_{timestamp}.png")
        Image.fromarray(clusters).save(clusters_dir / f"{index:03d}_{timestamp}.png")
        motion_frames.append(motion)
        cluster_frames.append(clusters)
    write_video(output_dir / "motion_red.mp4", motion_frames, fps)
    write_video(output_dir / "all_clusters.mp4", cluster_frames, fps)

    # Match Fig. 9's two-time-step, side-by-side presentation.
    first = max(len(motion_frames) // 3, 0)
    second = min(2 * len(motion_frames) // 3, len(motion_frames) - 1)
    Image.fromarray(np.concatenate([motion_frames[first], motion_frames[second]], axis=1)).save(
        output_dir / "paper_style_pair.png"
    )

    # Also render each k-means cluster independently in red. Cluster IDs are
    # arbitrary, so these views are essential for auditing or overriding the
    # automatically selected motion cluster.
    individual_root = output_dir / "individual_clusters"
    for cluster_id in range(len(statistics)):
        cluster_dir = individual_root / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        individual_frames = []
        for index, (image, label_map, timestamp) in enumerate(zip(images, pixel_labels, timestamps)):
            rendered = render_motion_overlay(
                image,
                label_map == cluster_id,
                alpha,
                background_brightness,
            )
            Image.fromarray(rendered).save(cluster_dir / f"{index:03d}_{timestamp}.png")
            individual_frames.append(rendered)
        write_video(individual_root / f"cluster_{cluster_id}.mp4", individual_frames, fps)
        Image.fromarray(
            np.concatenate([individual_frames[first], individual_frames[second]], axis=1)
        ).save(individual_root / f"cluster_{cluster_id}_pair.png")
    with (output_dir / "cluster_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"selected_motion_cluster": selected_cluster, "clusters": statistics},
            handle,
            indent=2,
        )
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if not (0 <= args.overlay_alpha <= 1 and 0 <= args.background_brightness <= 1):
        raise ValueError("Overlay alpha and background brightness must be in [0, 1]")
    sequence_dir = args.data_root / args.sequence
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence does not exist: {sequence_dir}")
    image_paths = evenly_sample(list_images(sequence_dir, args.frame_source), args.num_frames)
    timestamps = [path.stem for path in image_paths]

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega inference requires CUDA")
    print(f"Loading {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    invalid_layers = [layer for layer in args.layers if layer < 0 or layer >= model.aggregator.depth]
    if invalid_layers:
        raise ValueError(f"Layer indices out of range [0, {model.aggregator.depth - 1}]: {invalid_layers}")
    model.aggregator.cached_layer_indices.update(args.layers)

    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    ).to(device)
    print(f"Running {len(image_paths)} frames at {tuple(images.shape[-2:])}")
    with torch.inference_mode():
        layer_outputs, patch_token_start = model.aggregator(images.unsqueeze(0))
    rgb_images = tensor_images_to_uint8(images)
    patch_height = images.shape[-2] // model.aggregator.patch_size
    patch_width = images.shape[-1] // model.aggregator.patch_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summary: dict[str, object] = {
        "sequence": args.sequence,
        "frame_source": args.frame_source,
        "image_paths": [str(path) for path in image_paths],
        "image_shape_hw": list(images.shape[-2:]),
        "patch_grid_hw": [patch_height, patch_width],
        "patch_size": model.aggregator.patch_size,
        "normalization": args.normalization,
        "pca_dim": args.pca_dim,
        "num_clusters": args.num_clusters,
        "layers": {},
    }

    for layer_index in args.layers:
        output = layer_outputs[layer_index]
        if output is None:
            raise RuntimeError(f"Layer {layer_index} was not cached")
        patch_tokens = output[0, :, patch_token_start:].reshape(
            args.num_frames * patch_height * patch_width, -1
        )
        normalized = normalize_features(patch_tokens, args.normalization)
        reduced, _ = pca_reduce(normalized, args.pca_dim, args.seed + layer_index)
        labels_tensor, _, inertia = run_kmeans(
            reduced,
            args.num_clusters,
            args.kmeans_iterations,
            args.kmeans_restarts,
            args.seed + layer_index,
        )
        labels = labels_tensor.reshape(args.num_frames, patch_height, patch_width).cpu().numpy()
        statistics = cluster_statistics(labels, args.num_clusters)
        selected_cluster = (
            args.motion_cluster
            if args.motion_cluster is not None
            else choose_motion_cluster(statistics, args.min_cluster_fraction)
        )
        if selected_cluster < 0 or selected_cluster >= args.num_clusters:
            raise ValueError(f"Motion cluster {selected_cluster} is outside [0, {args.num_clusters - 1}]")
        layer_dir = args.output_dir / f"layer_{layer_index:02d}"
        save_layer_visualization(
            layer_dir,
            rgb_images,
            labels,
            timestamps,
            selected_cluster,
            statistics,
            args.overlay_alpha,
            args.background_brightness,
            args.fps,
        )
        run_summary["layers"][str(layer_index)] = {
            "selected_motion_cluster": selected_cluster,
            "kmeans_inertia": inertia,
            "clusters": statistics,
        }
        print(f"Layer {layer_index}: selected cluster {selected_cluster}; saved to {layer_dir}")

    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
        handle.write("\n")
    print(f"Motion-awareness visualizations saved to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
