#!/usr/bin/env python3
"""Evaluate VGGT-Omega on the MPI-Sintel training split.

The Sintel test split does not include public camera/depth ground truth, so this
script evaluates the training split. It runs both clean and final renderings
against the shared training/depth and training/camdata_left annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_7scenes_paper import official_auc, pairwise_pose_errors
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


DEFAULT_ROOT = Path("/data/mmc_lyxiang/dataset/Sintel")
DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
TAG_FLOAT = 202021.25


@dataclass(frozen=True)
class FrameRecord:
    index: int
    image_path: Path
    depth_path: Path
    cam_path: Path
    w2c: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "sintel_eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--passes", nargs="+", choices=("clean", "final"), default=["clean", "final"])
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--merge-ratio", type=float, default=0.9)
    parser.add_argument(
        "--depth-alignment",
        choices=("per-frame-median", "per-sequence-median"),
        default="per-frame-median",
    )
    parser.add_argument("--max-depth", type=float, default=1000.0)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def frame_index(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def read_dpt(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        tag = struct.unpack("f", handle.read(4))[0]
        if abs(tag - TAG_FLOAT) > 1e-4:
            raise ValueError(f"{path}: invalid Sintel depth tag {tag}")
        width = struct.unpack("i", handle.read(4))[0]
        height = struct.unpack("i", handle.read(4))[0]
        data = np.frombuffer(handle.read(), dtype=np.float32)
    if data.size != width * height:
        raise ValueError(f"{path}: expected {width * height} floats, got {data.size}")
    return data.reshape(height, width)


def read_cam(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        tag = struct.unpack("f", handle.read(4))[0]
        if abs(tag - TAG_FLOAT) > 1e-4:
            raise ValueError(f"{path}: invalid Sintel camera tag {tag}")
        data = np.frombuffer(handle.read(), dtype=np.float64)
    if data.size != 21:
        raise ValueError(f"{path}: expected 21 float64 values, got {data.size}")
    intrinsics = data[:9].reshape(3, 3)
    extrinsics = data[9:].reshape(3, 4)
    return intrinsics, extrinsics


def load_model(checkpoint: Path, device: torch.device, merge_ratio: float) -> VGGTOmega:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = VGGTOmega(merge_ratio=merge_ratio)
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        state = torch.load(checkpoint, mmap=True, **kwargs)
    except TypeError:
        state = torch.load(checkpoint, **kwargs)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(state).__name__}")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device).eval()


def select_sequences(root: Path, requested: Sequence[str] | None) -> list[str]:
    sequence_root = root / "training" / "clean"
    if not sequence_root.is_dir():
        raise FileNotFoundError(sequence_root)
    available = sorted(path.name for path in sequence_root.iterdir() if path.is_dir())
    if requested:
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"Unknown Sintel sequence(s): {', '.join(unknown)}")
        return list(requested)
    return available


def load_records(root: Path, render_pass: str, sequence: str) -> list[FrameRecord]:
    image_dir = root / "training" / render_pass / sequence
    depth_dir = root / "training" / "depth" / sequence
    cam_dir = root / "training" / "camdata_left" / sequence
    images = {frame_index(path): path for path in image_dir.glob("frame_*.png")}
    depths = {frame_index(path): path for path in depth_dir.glob("frame_*.dpt")}
    cams = {frame_index(path): path for path in cam_dir.glob("frame_*.cam")}
    indices = sorted(set(images) & set(depths) & set(cams))
    if not indices:
        raise ValueError(f"{render_pass}/{sequence}: no matched image/depth/camera frames")
    records = []
    for index in indices:
        _, w2c_3x4 = read_cam(cams[index])
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3] = w2c_3x4
        records.append(FrameRecord(index, images[index], depths[index], cams[index], w2c))
    return records


def crop_box_for_loader(width: int, height: int) -> tuple[int, int, int, int]:
    aspect = height / max(width, 1)
    if aspect < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        return left, 0, left + crop_width, height
    if aspect > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        top = max((height - crop_height) // 2, 0)
        return 0, top, width, top + crop_height
    return 0, 0, width, height


def resize_depth_like_loader(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    height, width = depth.shape
    left, top, right, bottom = crop_box_for_loader(width, height)
    cropped = depth[top:bottom, left:right].astype(np.float32, copy=True)
    tensor = torch.from_numpy(cropped)[None, None]
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="nearest",
    )
    return resized[0, 0].numpy()


def depth_metrics(
    predicted: np.ndarray,
    records: Sequence[FrameRecord],
    alignment: str,
    max_depth: float,
) -> tuple[float, float, int, list[float]]:
    _, height, width = predicted.shape
    gt = np.stack([resize_depth_like_loader(read_dpt(record.depth_path), height, width) for record in records])
    valid = np.isfinite(gt) & (gt > 0) & (gt < max_depth)
    valid &= np.isfinite(predicted) & (predicted > 0)
    aligned = predicted.astype(np.float64, copy=True)
    scales: list[float] = []
    if alignment == "per-frame-median":
        for index in range(len(predicted)):
            if not np.any(valid[index]):
                scales.append(float("nan"))
                continue
            scale = float(np.median(gt[index][valid[index]]) / np.median(predicted[index][valid[index]]))
            aligned[index] *= scale
            scales.append(scale)
    else:
        if not np.any(valid):
            raise ValueError("No valid Sintel depth pixels")
        scale = float(np.median(gt[valid]) / np.median(predicted[valid]))
        aligned *= scale
        scales = [scale] * len(predicted)
    gt_valid = gt[valid].astype(np.float64)
    pred_valid = aligned[valid]
    abs_rel = float(np.mean(np.abs(pred_valid - gt_valid) / gt_valid))
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta = float(np.mean(ratio < 1.25))
    return abs_rel, delta, int(len(gt_valid)), scales


def camera_metrics(
    predictions: dict[str, torch.Tensor],
    records: Sequence[FrameRecord],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    with torch.inference_mode():
        extrinsics, _ = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
            build_intrinsics=False,
        )
    pred_w2c = np.broadcast_to(np.eye(4), (len(records), 4, 4)).copy()
    pred_w2c[:, :3] = extrinsics[0].detach().float().cpu().numpy().astype(np.float64)
    gt_w2c = np.stack([record.w2c for record in records])
    rotation_errors, translation_errors = pairwise_pose_errors(pred_w2c, gt_w2c)
    metrics = {
        "auc_3_percent": 100.0 * official_auc(rotation_errors, translation_errors, 3),
        "auc_30_percent": 100.0 * official_auc(rotation_errors, translation_errors, 30),
        "rotation_median_deg": float(np.median(rotation_errors)),
        "translation_median_deg": float(np.median(translation_errors)),
        "pose_pairs": int(len(rotation_errors)),
    }
    return metrics, rotation_errors, translation_errors


def evaluate_sequence(
    model: VGGTOmega,
    records: Sequence[FrameRecord],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[float], np.ndarray, np.ndarray]:
    images = load_and_preprocess_images(
        [str(record.image_path) for record in records],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    ).to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    if not args.skip_warmup:
        with torch.inference_mode():
            warmup = model(images)
        torch.cuda.synchronize(device)
        del warmup
    timings = []
    predictions = None
    for _ in range(args.timing_repeats):
        if predictions is not None:
            del predictions
        start = time.perf_counter()
        with torch.inference_mode():
            predictions = model(images)
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    assert predictions is not None
    metrics, rotation_errors, translation_errors = camera_metrics(predictions, records)
    predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
    abs_rel, delta, valid_count, scales = depth_metrics(
        predicted_depth,
        records,
        args.depth_alignment,
        args.max_depth,
    )
    metrics.update(
        {
            "depth_absrel": abs_rel,
            "depth_delta_1_25_percent": 100.0 * delta,
            "valid_depth_pixels": valid_count,
            "input_height": int(images.shape[-2]),
            "input_width": int(images.shape[-1]),
            "latency_sec": float(np.median(timings)),
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        }
    )
    del images, predictions
    torch.cuda.empty_cache()
    return metrics, scales, rotation_errors, translation_errors


def aggregate(
    rows: list[dict[str, object]],
    errors: dict[str, dict[str, list[np.ndarray]]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for render_pass in sorted({str(row["pass"]) for row in rows}):
        group = [row for row in rows if row["pass"] == render_pass]
        total_pairs = sum(int(row["pose_pairs"]) for row in group)
        total_valid = sum(int(row["valid_depth_pixels"]) for row in group)
        rotation_errors = np.concatenate(errors[render_pass]["rotation"])
        translation_errors = np.concatenate(errors[render_pass]["translation"])
        latencies = np.asarray([float(row["latency_sec"]) for row in group], dtype=np.float64)
        result[render_pass] = {
            "num_sequences": len(group),
            "num_frames": sum(int(row["num_frames"]) for row in group),
            "pose_pairs": total_pairs,
            "auc_3_percent_global": 100.0 * official_auc(rotation_errors, translation_errors, 3),
            "auc_30_percent_global": 100.0 * official_auc(rotation_errors, translation_errors, 30),
            "auc_3_percent_mean": float(np.mean([float(row["auc_3_percent"]) for row in group])),
            "auc_30_percent_mean": float(np.mean([float(row["auc_30_percent"]) for row in group])),
            "depth_absrel_pixel_weighted": float(
                sum(float(row["depth_absrel"]) * int(row["valid_depth_pixels"]) for row in group) / total_valid
            ),
            "depth_delta_1_25_percent_pixel_weighted": float(
                sum(float(row["depth_delta_1_25_percent"]) * int(row["valid_depth_pixels"]) for row in group)
                / total_valid
            ),
            "latency_sec_mean": float(np.mean(latencies)),
            "latency_sec_median": float(np.median(latencies)),
            "latency_sec_total": float(np.sum(latencies)),
            "peak_allocated_gb_max": float(np.max([float(row["peak_allocated_gb"]) for row in group])),
        }
    return result


def main() -> int:
    args = parse_args()
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be positive")
    if not 0.0 <= args.merge_ratio <= 1.0:
        raise ValueError("--merge-ratio must be in [0, 1]")
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise ValueError("--image-resolution must be positive and divisible by 16")

    sequences = select_sequences(args.data_root, args.sequences)
    plan = {
        render_pass: {
            sequence: len(load_records(args.data_root, render_pass, sequence))
            for sequence in sequences
        }
        for render_pass in args.passes
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device, args.merge_ratio)

    rows: list[dict[str, object]] = []
    scale_summary: dict[str, dict[str, list[float]]] = {}
    pose_errors: dict[str, dict[str, list[np.ndarray]]] = {
        render_pass: {"rotation": [], "translation": []}
        for render_pass in args.passes
    }
    for render_pass in args.passes:
        scale_summary[render_pass] = {}
        for sequence in sequences:
            records = load_records(args.data_root, render_pass, sequence)
            print(f"[{render_pass}/{sequence}] frames={len(records)}", flush=True)
            metrics, scales, rotation_errors, translation_errors = evaluate_sequence(model, records, device, args)
            pose_errors[render_pass]["rotation"].append(rotation_errors)
            pose_errors[render_pass]["translation"].append(translation_errors)
            row: dict[str, object] = {
                "pass": render_pass,
                "sequence": sequence,
                "num_frames": len(records),
                "merge_ratio": args.merge_ratio,
                **metrics,
            }
            rows.append(row)
            scale_summary[render_pass][sequence] = scales
            print(
                f"  AUC@3={metrics['auc_3_percent']:.2f}, "
                f"AUC@30={metrics['auc_30_percent']:.2f}, "
                f"AbsRel={metrics['depth_absrel']:.4f}, "
                f"delta1.25={metrics['depth_delta_1_25_percent']:.2f}",
                flush=True,
            )

    fields = list(rows[0])
    with (args.output_dir / "per_sequence_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": {
            "dataset": "MPI-Sintel training split",
            "passes": args.passes,
            "sequences": sequences,
            "checkpoint": str(args.checkpoint),
            "merge_ratio": args.merge_ratio,
            "image_resolution": args.image_resolution,
            "resize_mode": args.resize_mode,
            "depth_alignment": args.depth_alignment,
            "max_depth": args.max_depth,
            "timing_repeats": args.timing_repeats,
            "warmup": not args.skip_warmup,
            "paper_sintel_ours_1b": {
                "auc_3_percent": 35.3,
                "auc_30_percent": 73.0,
                "depth_delta_1_25_percent": 89.5,
                "depth_absrel": 0.097,
            },
            "paper_sintel_ours_10b": {
                "auc_3_percent": 40.0,
                "auc_30_percent": 79.1,
                "depth_delta_1_25_percent": 93.5,
                "depth_absrel": 0.081,
            },
        },
        "overall": aggregate(rows, pose_errors),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (args.output_dir / "depth_scales.json").open("w", encoding="utf-8") as handle:
        json.dump(scale_summary, handle)
        handle.write("\n")
    np.savez_compressed(
        args.output_dir / "pose_errors.npz",
        **{
            f"{render_pass}_rotation_error_deg": np.concatenate(pose_errors[render_pass]["rotation"])
            for render_pass in args.passes
        },
        **{
            f"{render_pass}_translation_error_deg": np.concatenate(pose_errors[render_pass]["translation"])
            for render_pass in args.passes
        },
    )
    print(f"Saved Sintel evaluation to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
