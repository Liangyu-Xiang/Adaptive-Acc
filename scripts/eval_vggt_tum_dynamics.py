#!/usr/bin/env python3
"""Evaluate vanilla VGGT on long TUM-Dynamics sequences.

This reuses the dataset association, window stitching, Sim(3) alignment, and
trajectory output from eval_tum_dynamics.py so VGGT and VGGT-Omega are measured
with the same protocol. Model latency excludes checkpoint loading and metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval_tum_dynamics import (
    DEFAULT_DATA_ROOT,
    evaluate_trajectory,
    load_sequence,
    save_tum_trajectory,
    select_sequences,
    stitch_window,
)


DEFAULT_VGGT_ROOT = REPO_ROOT.parent / "vggt"
DEFAULT_CHECKPOINT = DEFAULT_VGGT_ROOT / "ckpt" / "model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VGGT on long TUM-Dynamics sequences")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--vggt-root", type=Path, default=DEFAULT_VGGT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-size", type=int, default=90)
    parser.add_argument("--window-overlap", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument("--resize-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def import_vggt(vggt_root: Path):
    if not (vggt_root / "vggt" / "models" / "vggt.py").is_file():
        raise FileNotFoundError(f"VGGT repository not found: {vggt_root}")
    sys.path.insert(0, str(vggt_root))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    return VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri


def load_model(model_class, checkpoint: Path, device: torch.device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    model = model_class(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        state = torch.load(checkpoint, mmap=True, **kwargs)
    except TypeError:
        state = torch.load(checkpoint, **kwargs)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    expected = model.state_dict()
    camera_state = {key: value for key, value in state.items() if key in expected}
    missing, unexpected = model.load_state_dict(camera_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    del state, camera_state
    return model.to(device).eval()


def infer_window(model, paths, device, resize_mode, preprocess, decode_pose) -> np.ndarray:
    images = preprocess([str(path) for path in paths], mode=resize_mode).to(device, non_blocking=True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        predictions = model(images)
    with torch.autocast(device_type="cuda", enabled=False):
        extrinsics, _ = decode_pose(predictions["pose_enc"].float(), images.shape[-2:])
    w2c = extrinsics[0].detach().float().cpu().numpy()
    bottom = np.broadcast_to(np.array([0, 0, 0, 1], dtype=w2c.dtype), (*w2c.shape[:-2], 1, 4))
    poses = np.linalg.inv(np.concatenate((w2c, bottom), axis=-2)).astype(np.float64)
    del images, predictions, extrinsics
    return poses


def infer_trajectory(model, paths, device, window_size, overlap, resize_mode, preprocess, decode_pose):
    if window_size < 2 or overlap < 1 or overlap >= window_size:
        raise ValueError("Require window-size >= 2 and 1 <= overlap < window-size")
    trajectory = None
    start = 0
    step = window_size - overlap
    while start < len(paths):
        end = min(start + window_size, len(paths))
        local = infer_window(model, paths[start:end], device, resize_mode, preprocess, decode_pose)
        if trajectory is None:
            trajectory = local
        else:
            actual_overlap = len(trajectory) - start
            local = stitch_window(trajectory[start:start + actual_overlap], local)
            trajectory = np.concatenate((trajectory, local[actual_overlap:]), axis=0)
        if end == len(paths):
            break
        start += step
    return trajectory


def main() -> int:
    args = parse_args()
    sequences = select_sequences(args.data_root, args.sequences)
    loaded = {
        sequence.name: load_sequence(sequence, args.max_frames, args.frame_stride, args.association_tolerance)
        for sequence in sequences
    }
    for name, (images, _, _) in loaded.items():
        print(f"{name}: {len(images)} associated frames", flush=True)
    if args.dry_run:
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT inference requires CUDA")
    VGGT, preprocess, decode_pose = import_vggt(args.vggt_root)
    model = load_model(VGGT, args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for sequence in sequences:
        paths, timestamps, gt_poses = loaded[sequence.name]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        predicted = infer_trajectory(
            model, paths, device, args.window_size, args.window_overlap,
            args.resize_mode, preprocess, decode_pose,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
        metrics, aligned = evaluate_trajectory(predicted, gt_poses)
        row = {
            "sequence": sequence.name,
            **metrics,
            "inference_seconds": elapsed,
            "fps": len(paths) / elapsed,
            "peak_memory_gib": peak_gib,
            "window_size": args.window_size,
            "window_overlap": args.window_overlap,
            "frame_stride": args.frame_stride,
        }
        results.append(row)
        seq_dir = args.output_dir / sequence.name
        seq_dir.mkdir(parents=True, exist_ok=True)
        save_tum_trajectory(seq_dir / "pred_traj.txt", timestamps, predicted)
        save_tum_trajectory(seq_dir / "pred_traj_aligned.txt", timestamps, aligned)
        save_tum_trajectory(seq_dir / "gt_traj.txt", timestamps, gt_poses)
        (seq_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
        print(
            f"[{sequence.name}] ATE={row['ate_rmse_m']:.5f}m "
            f"RPE-t={row['rpe_translation_rmse_m']:.5f}m "
            f"RPE-r={row['rpe_rotation_rmse_deg']:.5f}deg "
            f"FPS={row['fps']:.2f} peak={peak_gib:.2f}GiB",
            flush=True,
        )

    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump({"sequences": results}, handle, indent=2)
        handle.write("\n")
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
