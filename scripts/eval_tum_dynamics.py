#!/usr/bin/env python3
"""Evaluate VGGT-Omega camera poses on the TUM RGB-D dynamic sequences.

The default protocol follows the 90-frame TUM-Dynamics subset used by the
dataset preparation script in MonST3R: associate RGB/GT at 20 ms, take every
third associated frame, and retain the first 90 frames.  If ``rgb_90`` and
``groundtruth_90.txt`` exist, those prepared files are used directly.

Metrics are computed after a 7-DoF Sim(3) alignment, since monocular camera
reconstruction has an unknown global scale:
  * ATE RMSE in metres
  * frame-to-frame RPE translation RMSE in metres
  * frame-to-frame RPE rotation RMSE in degrees
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


DEFAULT_DATA_ROOT = Path("/mnt/nasdata/xly/dataset/TUM-Dynamics")
DEFAULT_CHECKPOINT = Path(
    "/mnt/nasdata/xly/3D/vggt-omega/pretrained_ckpts/vggt_omega_1b_512.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VGGT-Omega camera trajectories on TUM-Dynamics."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tum_dynamics"))
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Sequence directory names to evaluate (default: all eight sequences).",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:1.")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="Frames per inference window. 0 runs all 90 frames jointly (recommended).",
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=5,
        help="Overlap used to align adjacent windows when --window-size is non-zero.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=90,
        help="Maximum sampled frames per sequence; 0 keeps every associated frame.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="Fallback sampling stride if the prepared 90-frame subset is absent.",
    )
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset selection and print frame counts without loading the model.",
    )
    return parser.parse_args()


def read_file_list(path: Path) -> list[tuple[float, list[str]]]:
    rows: list[tuple[float, list[str]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.replace(",", " ").split()
            rows.append((float(fields[0]), fields[1:]))
    return rows


def associate_nearest(
    first: Sequence[tuple[float, list[str]]],
    second: Sequence[tuple[float, list[str]]],
    tolerance: float,
) -> list[tuple[int, int]]:
    """Greedily make unique timestamp pairs, matching the TUM association tool."""
    candidates: list[tuple[float, int, int]] = []
    second_times = np.asarray([row[0] for row in second], dtype=np.float64)
    for i, (timestamp, _) in enumerate(first):
        left = int(np.searchsorted(second_times, timestamp - tolerance, side="right"))
        right = int(np.searchsorted(second_times, timestamp + tolerance, side="left"))
        for j in range(left, right):
            candidates.append((abs(timestamp - second_times[j]), i, j))

    used_first: set[int] = set()
    used_second: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in sorted(candidates):
        if i not in used_first and j not in used_second:
            matches.append((i, j))
            used_first.add(i)
            used_second.add(j)
    return sorted(matches)


def load_sequence(
    sequence_dir: Path,
    max_frames: int,
    frame_stride: int,
    tolerance: float,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    prepared_images = sequence_dir / "rgb_90"
    prepared_gt = sequence_dir / "groundtruth_90.txt"

    if prepared_images.is_dir() and prepared_gt.is_file() and 0 < max_frames <= 90:
        image_paths = sorted(
            (path for path in prepared_images.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}),
            key=lambda path: float(path.stem),
        )
        gt_rows = read_file_list(prepared_gt)
        if len(image_paths) != len(gt_rows):
            raise ValueError(
                f"{sequence_dir.name}: rgb_90 has {len(image_paths)} images but "
                f"groundtruth_90.txt has {len(gt_rows)} poses"
            )
        if max_frames:
            image_paths = image_paths[:max_frames]
            gt_rows = gt_rows[:max_frames]
        rgb_timestamps = np.asarray([float(path.stem) for path in image_paths])
    else:
        rgb_rows = read_file_list(sequence_dir / "rgb.txt")
        gt_all_rows = read_file_list(sequence_dir / "groundtruth.txt")
        matches = associate_nearest(rgb_rows, gt_all_rows, tolerance)[::frame_stride]
        if max_frames:
            matches = matches[:max_frames]
        image_paths = [sequence_dir / rgb_rows[i][1][0] for i, _ in matches]
        gt_rows = [gt_all_rows[j] for _, j in matches]
        rgb_timestamps = np.asarray([rgb_rows[i][0] for i, _ in matches])

    if len(image_paths) < 2:
        raise ValueError(f"{sequence_dir.name}: need at least two associated frames")
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RGB image: {missing[0]}")

    gt_timestamps = np.asarray([row[0] for row in gt_rows], dtype=np.float64)
    gt_poses = np.stack([tum_fields_to_matrix(row[1]) for row in gt_rows])
    if np.max(np.abs(rgb_timestamps - gt_timestamps)) >= tolerance:
        raise ValueError(f"{sequence_dir.name}: RGB/GT timestamps exceed {tolerance}s")
    return image_paths, rgb_timestamps, gt_poses


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized scalar-last quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                          (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif i == 1:
            s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                          (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2
            q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                          0.25 * s, (m[1, 0] - m[0, 1]) / s])
    q /= np.linalg.norm(q)
    return q


def tum_fields_to_matrix(fields: Sequence[str]) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    if values.shape != (7,):
        raise ValueError(f"Expected tx ty tz qx qy qz qw, got {len(values)} values")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_xyzw_to_matrix(values[3:])
    pose[:3, 3] = values[:3]
    return pose


def load_model(checkpoint: Path, device: torch.device) -> VGGTOmega:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    # Camera-only construction avoids allocating and running the dense depth head.
    model = VGGTOmega(enable_depth=False)
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        state = torch.load(checkpoint, mmap=True, **load_kwargs)
    except TypeError:  # torch < 2.1 has no mmap argument
        state = torch.load(checkpoint, **load_kwargs)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(state).__name__}")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    expected = model.state_dict()
    camera_state = {key: value for key, value in state.items() if key in expected}
    missing, unexpected = model.load_state_dict(camera_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    del state, camera_state
    return model.to(device).eval()


def extrinsics_to_c2w(extrinsics: torch.Tensor) -> np.ndarray:
    w2c = extrinsics.detach().float().cpu().numpy()
    bottom = np.broadcast_to(np.array([0, 0, 0, 1], dtype=w2c.dtype), (*w2c.shape[:-2], 1, 4))
    return np.linalg.inv(np.concatenate((w2c, bottom), axis=-2)).astype(np.float64)


def infer_window(
    model: VGGTOmega,
    image_paths: Sequence[Path],
    device: torch.device,
    image_resolution: int,
    resize_mode: str,
) -> np.ndarray:
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], mode=resize_mode, image_resolution=image_resolution
    ).to(device, non_blocking=True)
    with torch.inference_mode():
        predictions = model(images)
        extrinsics, _ = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:], build_intrinsics=False
        )
    poses = extrinsics_to_c2w(extrinsics[0])
    del images, predictions, extrinsics
    return poses


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def stitch_window(global_overlap: np.ndarray, local_poses: np.ndarray) -> np.ndarray:
    overlap = len(global_overlap)
    local_overlap = local_poses[:overlap]
    rotation = project_to_rotation(
        sum(g[:3, :3] @ l[:3, :3].T for g, l in zip(global_overlap, local_overlap))
    )
    local_xyz = local_overlap[:, :3, 3] @ rotation.T
    global_xyz = global_overlap[:, :3, 3]
    local_centered = local_xyz - local_xyz.mean(axis=0)
    global_centered = global_xyz - global_xyz.mean(axis=0)
    denominator = float(np.sum(local_centered**2))
    scale = float(np.sum(local_centered * global_centered) / denominator) if denominator > 1e-12 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    translation = global_xyz.mean(axis=0) - scale * local_xyz.mean(axis=0)

    transformed = local_poses.copy()
    transformed[:, :3, :3] = rotation @ local_poses[:, :3, :3]
    transformed[:, :3, 3] = scale * (local_poses[:, :3, 3] @ rotation.T) + translation
    return transformed


def infer_trajectory(
    model: VGGTOmega,
    image_paths: Sequence[Path],
    device: torch.device,
    image_resolution: int,
    resize_mode: str,
    window_size: int,
    overlap: int,
) -> np.ndarray:
    count = len(image_paths)
    if window_size <= 0 or window_size >= count:
        return infer_window(model, image_paths, device, image_resolution, resize_mode)
    if window_size < 2 or overlap < 1 or overlap >= window_size:
        raise ValueError("Windowed inference requires window-size >= 2 and 1 <= overlap < window-size")

    step = window_size - overlap
    trajectory: np.ndarray | None = None
    start = 0
    while start < count:
        end = min(start + window_size, count)
        local = infer_window(model, image_paths[start:end], device, image_resolution, resize_mode)
        if trajectory is None:
            trajectory = local
        else:
            actual_overlap = len(trajectory) - start
            if actual_overlap <= 0:
                raise RuntimeError("Inference windows do not overlap")
            local = stitch_window(trajectory[start:start + actual_overlap], local)
            trajectory = np.concatenate((trajectory, local[actual_overlap:]), axis=0)
        if end == count:
            break
        start += step
    assert trajectory is not None and len(trajectory) == count
    return trajectory


def align_sim3(estimated: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    source = estimated[:, :3, 3]
    target = reference[:, :3, 3]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = np.sum(source_centered**2) / len(source)
    if variance <= 1e-15:
        raise ValueError("Estimated trajectory has no translational variance; Sim(3) alignment is undefined")
    scale = float(np.sum(singular_values * sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)

    aligned = estimated.copy()
    aligned[:, :3, :3] = rotation @ estimated[:, :3, :3]
    aligned[:, :3, 3] = scale * (estimated[:, :3, 3] @ rotation.T) + translation
    return aligned, scale, rotation, translation


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def evaluate_trajectory(estimated: np.ndarray, reference: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    aligned, scale, _, _ = align_sim3(estimated, reference)
    ate_errors = np.linalg.norm(aligned[:, :3, 3] - reference[:, :3, 3], axis=1)
    rpe_translation: list[float] = []
    rpe_rotation: list[float] = []
    for i in range(len(reference) - 1):
        gt_delta = np.linalg.inv(reference[i]) @ reference[i + 1]
        est_delta = np.linalg.inv(aligned[i]) @ aligned[i + 1]
        error = np.linalg.inv(gt_delta) @ est_delta
        rpe_translation.append(float(np.linalg.norm(error[:3, 3])))
        rpe_rotation.append(rotation_angle_degrees(error[:3, :3]))

    metrics = {
        "num_frames": len(reference),
        "alignment_scale": scale,
        "ate_rmse_m": float(np.sqrt(np.mean(ate_errors**2))),
        "ate_mean_m": float(np.mean(ate_errors)),
        "ate_median_m": float(np.median(ate_errors)),
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(np.square(rpe_translation)))),
        "rpe_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rpe_rotation)))),
    }
    return metrics, aligned


def save_tum_trajectory(path: Path, timestamps: np.ndarray, poses: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp, pose in zip(timestamps, poses):
            xyz = pose[:3, 3]
            quaternion = matrix_to_quaternion_xyzw(pose[:3, :3])
            values = " ".join(f"{value:.9f}" for value in np.concatenate((xyz, quaternion)))
            handle.write(f"{timestamp:.6f} {values}\n")


def select_sequences(data_root: Path, requested: Sequence[str] | None) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    sequences = sorted(path for path in data_root.iterdir() if path.is_dir() and (path / "rgb.txt").is_file())
    if requested:
        by_name = {path.name: path for path in sequences}
        unknown = sorted(set(requested) - set(by_name))
        if unknown:
            raise ValueError(f"Unknown sequence(s): {', '.join(unknown)}")
        sequences = [by_name[name] for name in requested]
    if not sequences:
        raise ValueError(f"No TUM sequences found under {data_root}")
    return sequences


def write_summary(output_dir: Path, results: list[dict[str, object]]) -> dict[str, object]:
    metric_names = ("ate_rmse_m", "rpe_translation_rmse_m", "rpe_rotation_rmse_deg")
    average = {name: float(np.mean([float(row[name]) for row in results])) for name in metric_names}
    summary: dict[str, object] = {"sequences": results, "average": average}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return summary


def main() -> int:
    args = parse_args()
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise ValueError("--image-resolution must be positive and divisible by 16")
    if args.max_frames < 0 or args.frame_stride < 1:
        raise ValueError("--max-frames must be >= 0 and --frame-stride must be >= 1")

    sequences = select_sequences(args.data_root, args.sequences)
    loaded = {
        sequence.name: load_sequence(
            sequence, args.max_frames, args.frame_stride, args.association_tolerance
        )
        for sequence in sequences
    }
    for name, (images, _, _) in loaded.items():
        print(f"{name}: {len(images)} associated frames")
    if args.dry_run:
        print("Dry run completed; model was not loaded.")
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega inference requires an available CUDA device")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    results: list[dict[str, object]] = []
    for sequence in sequences:
        image_paths, timestamps, gt_poses = loaded[sequence.name]
        print(f"[{sequence.name}] running inference ...", flush=True)
        started = time.perf_counter()
        predicted = infer_trajectory(
            model,
            image_paths,
            device,
            args.image_resolution,
            args.resize_mode,
            args.window_size,
            args.window_overlap,
        )
        metrics, aligned = evaluate_trajectory(predicted, gt_poses)
        elapsed = time.perf_counter() - started
        metrics["sequence"] = sequence.name
        metrics["inference_seconds"] = elapsed
        # Put sequence first in stable CSV/JSON output.
        metrics = {"sequence": metrics.pop("sequence"), **metrics}
        results.append(metrics)

        sequence_output = args.output_dir / sequence.name
        sequence_output.mkdir(parents=True, exist_ok=True)
        save_tum_trajectory(sequence_output / "pred_traj.txt", timestamps, predicted)
        save_tum_trajectory(sequence_output / "pred_traj_aligned.txt", timestamps, aligned)
        save_tum_trajectory(sequence_output / "gt_traj.txt", timestamps, gt_poses)
        with (sequence_output / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
            handle.write("\n")
        print(
            f"[{sequence.name}] ATE={metrics['ate_rmse_m']:.5f} m, "
            f"RPE-t={metrics['rpe_translation_rmse_m']:.5f} m, "
            f"RPE-r={metrics['rpe_rotation_rmse_deg']:.5f} deg ({elapsed:.1f}s)"
        )
        torch.cuda.empty_cache()

    summary = write_summary(args.output_dir, results)
    average = summary["average"]
    assert isinstance(average, dict)
    print(
        "Average: "
        f"ATE={average['ate_rmse_m']:.5f} m, "
        f"RPE-t={average['rpe_translation_rmse_m']:.5f} m, "
        f"RPE-r={average['rpe_rotation_rmse_deg']:.5f} deg"
    )
    print(f"Results saved to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
