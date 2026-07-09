#!/usr/bin/env python3
"""Evaluate VGGT-Omega on 7 Scenes with the paper's 10-view protocol.

The VGGT-Omega paper samples 10 frames per scene/sequence and reports pairwise
relative-pose AUC plus scale-aligned depth metrics.  The paper does not publish
the sampled frame IDs, so this implementation uses the same deterministic
RandomState(42) convention as the public VGGT evaluation code and saves the
selection alongside the metrics.

The dataset must contain RGB-registered ``*.depth.proj.png`` maps.  These can be
generated once by FastVGGT's ``scripts/prepare_7scenes.py``; this evaluator only
reads those files and is therefore compatible with both projects.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.gpu_guard import assert_exclusive_gpu
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


DEFAULT_DATA_ROOT = Path("/data/mmc_lyxiang/dataset/7scenes")
DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
DEFAULT_REGISTER_ONLY_GLOBAL_LAYER_SPEC = "9-23"
PAPER_TARGETS = {
    "auc_3_percent": 29.6,
    "auc_30_percent": 83.1,
    "delta_1_25_percent": 94.6,
    "abs_rel": 0.058,
}


@dataclass(frozen=True)
class FrameRecord:
    index: int
    rgb_path: Path
    depth_path: Path
    pose_path: Path
    c2w: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce VGGT-Omega Table 1/2 metrics on 7 Scenes."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/7scenes_paper"))
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument(
        "--attention-mode",
        choices=("default", "register-only-zero-shot"),
        default="default",
        help=(
            "Attention schedule to evaluate. The register-only option changes the released "
            "checkpoint at inference time and is not a separately trained architecture."
        ),
    )
    parser.add_argument(
        "--register-only-global-layers",
        default="",
        help=(
            "Comma-separated layer indices or inclusive ranges to keep as global attention "
            "inside register-only-zero-shot mode, e.g. '9-23' or 'none'. "
            f"Default in register-only-zero-shot mode: {DEFAULT_REGISTER_ONLY_GLOBAL_LAYER_SPEC}. "
            "Indices are 0-based inter-frame block IDs."
        ),
    )
    parser.add_argument(
        "--inter-frame-only-global-layers",
        default="",
        help=(
            "Comma-separated 0-based global inter-frame layer indices or inclusive ranges "
            "where same-frame token attention is disabled, e.g. '15'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument(
        "--sampling-unit",
        choices=("scene", "sequence"),
        default="sequence",
        help="Sampling unit (default matches common 7 Scenes evaluation loaders).",
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument(
        "--merge-ratio",
        type=float,
        default=0.0,
        help="Token merge ratio. The paper baseline is 0 (disabled).",
    )
    parser.add_argument(
        "--register-patch-inter-frame-mode",
        choices=("none", "random", "least-register"),
        default="none",
        help=(
            "For register-only attention, additionally let selected patch tokens participate "
            "in inter-frame attention. 'least-register' selects patches least attended by "
            "same-frame registers in the preceding frame-attention block."
        ),
    )
    parser.add_argument(
        "--register-patch-inter-frame-percent",
        type=float,
        default=0.0,
        help="Per-frame percentage of patch tokens selected for register-only inter-frame attention.",
    )
    parser.add_argument(
        "--depth-alignment",
        choices=("per-frame-median", "per-sequence-median"),
        default="per-frame-median",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.2,
        help="Minimum valid sensor depth in metres (filters wrapped invalid-depth sentinels).",
    )
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional scene/seq-NN names, e.g. chess/seq-03.",
    )
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Run one forward per sequence for metrics and omit CUDA-event latency measurements.",
    )
    parser.add_argument(
        "--require-exclusive-gpu",
        action="store_true",
        help="Abort if the target physical GPU is shared with other compute workloads.",
    )
    parser.add_argument(
        "--exclusive-gpu-index",
        type=int,
        default=None,
        help="Physical GPU index to guard when --require-exclusive-gpu is enabled.",
    )
    parser.add_argument(
        "--exclusive-gpu-max-other-memory-mib",
        type=int,
        default=512,
        help="Allowed non-target residual memory on the guarded physical GPU.",
    )
    parser.add_argument("--sparse-attention", action="store_true", help="Use block-sparse inter-frame global attention.")
    parser.add_argument(
        "--sparse-ratio",
        type=float,
        default=None,
        help="Sparse-VGGT ratio of key blocks to prune, e.g. 0.1.",
    )
    parser.add_argument(
        "--sparse-cdf-threshold",
        type=float,
        default=None,
        help="Sparse-VGGT cumulative attention threshold, e.g. 0.97.",
    )
    parser.add_argument("--sparse-pool-mode", choices=("avg", "max"), default="avg")
    parser.add_argument(
        "--use-adaptive-kv-anchor",
        "--use-register-mediated-anchor",
        "--use_register_mediated_anchor",
        action="store_true",
        help="Enable adaptive K/V anchor token selection in global inter-frame attention.",
    )
    parser.add_argument(
        "--adaptive-anchor-layers",
        "--anchor-layers",
        "--anchor_layers",
        dest="adaptive_anchor_layers",
        default="6-18",
        help="Layer spec passed to VGGTOmega for adaptive K/V anchors, e.g. '6-18'.",
    )
    parser.add_argument(
        "--adaptive-anchor-ratio",
        "--anchor-ratio",
        "--anchor_ratio",
        dest="adaptive_anchor_ratio",
        type=float,
        default=0.2,
        help="Per-frame adaptive anchor ratio passed to VGGTOmega.",
    )
    parser.add_argument(
        "--adaptive-anchor-total",
        "--anchor-total",
        "--anchor_total",
        dest="adaptive_anchor_total",
        type=int,
        default=None,
        help="Optional total adaptive anchor budget passed to VGGTOmega.",
    )
    parser.add_argument(
        "--adaptive-anchor-min-per-frame",
        "--anchor-min-per-frame",
        "--anchor_min_per_frame",
        dest="adaptive_anchor_min_per_frame",
        type=int,
        default=4,
        help="Minimum adaptive anchors retained per frame.",
    )
    parser.add_argument(
        "--adaptive-anchor-tau",
        "--anchor-tau",
        "--anchor_tau",
        dest="adaptive_anchor_tau",
        type=float,
        default=1.0,
        help="Temperature for adaptive anchor score weighting.",
    )
    parser.add_argument(
        "--adaptive-anchor-uniform-mix",
        "--anchor-uniform-mix",
        "--anchor_uniform_mix",
        dest="adaptive_anchor_uniform_mix",
        type=float,
        default=0.2,
        help="Uniform mixing coefficient for adaptive anchor allocation.",
    )
    parser.add_argument(
        "--adaptive-anchor-mode",
        "--anchor-mode",
        "--anchor_mode",
        dest="adaptive_anchor_strategy",
        choices=("lifting", "frame_pair_gated", "hybrid", "register_intra", "fixed_grid", "intra_only", "proxy", "proxy_intra", "oracle", "random"),
        default="lifting",
        help="Adaptive K/V anchor mode. register-mediated modes are lifting, frame_pair_gated, and hybrid.",
    )
    parser.add_argument(
        "--adaptive-anchor-score-alpha-cross",
        "--anchor-score-alpha-cross",
        "--anchor_score_alpha_cross",
        dest="adaptive_anchor_score_alpha_cross",
        type=float,
        default=1.0,
        help="Weight for register-mediated cross-view patch score.",
    )
    parser.add_argument(
        "--adaptive-anchor-score-beta-intra",
        "--anchor-score-beta-intra",
        "--anchor_score_beta_intra",
        dest="adaptive_anchor_score_beta_intra",
        type=float,
        default=0.2,
        help="Weight for intra-frame structural patch score.",
    )
    parser.add_argument(
        "--adaptive-anchor-topm-frames",
        "--anchor-topm-frames",
        "--anchor_topm_frames",
        dest="adaptive_anchor_topm_frames",
        type=int,
        default=4,
        help="Top-M key frames per query frame for frame-pair gated modes. Use 0 to disable gating.",
    )
    parser.add_argument(
        "--adaptive-anchor-debug",
        "--anchor-debug",
        "--anchor_debug",
        dest="adaptive_anchor_debug",
        action="store_true",
        help="Enable adaptive anchor debug output in the model.",
    )
    parser.add_argument(
        "--adaptive-anchor-debug-dir",
        "--anchor-debug-dir",
        "--anchor_debug_dir",
        dest="adaptive_anchor_debug_dir",
        type=Path,
        default=Path("outputs/debug_register_mediated_anchor"),
        help="Directory for per-layer adaptive anchor debug .pt files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_layer_indices(spec: str, depth: int) -> list[int]:
    normalized = spec.strip().lower()
    if not normalized or normalized == "none":
        return []
    if normalized == "all":
        return list(range(depth))
    layers: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid layer range {part!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= depth)
    if invalid:
        raise ValueError(f"Layer indices out of range 0..{depth - 1}: {invalid}")
    return sorted(layers)


def parse_split_sequence(line: str) -> str:
    digits = "".join(character for character in line if character.isdigit())
    if not digits:
        raise ValueError(f"Invalid 7 Scenes split entry: {line!r}")
    return f"seq-{int(digits):02d}"


def select_sequence_dirs(data_root: Path, requested: Sequence[str] | None) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    sequence_dirs: list[Path] = []
    for scene in SCENES:
        split_path = data_root / scene / "TestSplit.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing official test split: {split_path}")
        for line in split_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                sequence_dirs.append(data_root / scene / parse_split_sequence(line))

    if requested:
        mapping = {
            f"{path.parent.name}/{path.name}": path
            for path in sequence_dirs
        }
        unknown = sorted(set(requested) - set(mapping))
        if unknown:
            raise ValueError(f"Unknown test sequence(s): {', '.join(unknown)}")
        sequence_dirs = [mapping[name] for name in requested]
    return sequence_dirs


def frame_index(path: Path) -> int:
    return int(path.name.split("-", 1)[1].split(".", 1)[0])


def load_frame_records(sequence_dir: Path) -> list[FrameRecord]:
    rgb_by_index = {frame_index(path): path for path in sequence_dir.glob("frame-*.color.png")}
    pose_by_index = {frame_index(path): path for path in sequence_dir.glob("frame-*.pose.txt")}
    indices = sorted(set(rgb_by_index) & set(pose_by_index))
    if not indices:
        raise FileNotFoundError(f"No extracted RGB/pose frames in {sequence_dir}")

    records: list[FrameRecord] = []
    for index in indices:
        pose = np.loadtxt(pose_by_index[index], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            continue
        depth_path = sequence_dir / f"frame-{index:06d}.depth.proj.png"
        records.append(
            FrameRecord(
                index=index,
                rgb_path=rgb_by_index[index],
                depth_path=depth_path,
                pose_path=pose_by_index[index],
                c2w=pose,
            )
        )
    if not records:
        raise ValueError(f"{sequence_dir}: no frames have finite 4x4 poses")
    return records


def sample_records(
    pools: dict[str, list[FrameRecord]], num_frames: int, seed: int
) -> tuple[dict[str, list[FrameRecord]], dict[str, list[int]]]:
    # This matches np.random.seed(seed) followed by sequential np.random.choice
    # calls, as used by the public VGGT/Pi3 evaluation implementations.
    rng = np.random.RandomState(seed)
    sampled: dict[str, list[FrameRecord]] = {}
    sampled_indices: dict[str, list[int]] = {}
    for sequence_name, records in pools.items():
        if len(records) < num_frames:
            raise ValueError(f"{sequence_name}: only {len(records)} frames, need {num_frames}")
        pool_indices = rng.choice(len(records), num_frames, replace=False).tolist()
        sampled_indices[sequence_name] = pool_indices
        sampled[sequence_name] = [records[index] for index in pool_indices]
    return sampled, sampled_indices


def load_model(
    checkpoint: Path,
    device: torch.device,
    merge_ratio: float,
    sparse_attention: bool,
    sparse_ratio: float | None,
    sparse_cdf_threshold: float | None,
    sparse_pool_mode: str,
    use_adaptive_kv_anchor: bool,
    adaptive_anchor_layers: str,
    adaptive_anchor_ratio: float,
    adaptive_anchor_total: int | None,
    adaptive_anchor_min_per_frame: int,
    adaptive_anchor_tau: float,
    adaptive_anchor_uniform_mix: float,
    adaptive_anchor_strategy: str,
    adaptive_anchor_score_alpha_cross: float,
    adaptive_anchor_score_beta_intra: float,
    adaptive_anchor_topm_frames: int | None,
    adaptive_anchor_debug: bool,
    adaptive_anchor_debug_dir: Path,
) -> VGGTOmega:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    # Pass the ratio explicitly because local FastVGGT experiments may change
    # the model constructor's default.  Zero is the unmodified paper model.
    model_kwargs = {
        "merge_ratio": merge_ratio,
        "sparse_attention": sparse_attention,
        "sparse_ratio": sparse_ratio,
        "sparse_cdf_threshold": sparse_cdf_threshold,
        "sparse_pool_mode": sparse_pool_mode,
    }
    adaptive_kwargs = {
        "use_adaptive_kv_anchor": use_adaptive_kv_anchor,
        "adaptive_anchor_layers": adaptive_anchor_layers,
        "adaptive_anchor_ratio": adaptive_anchor_ratio,
        "adaptive_anchor_total": adaptive_anchor_total,
        "adaptive_anchor_min_per_frame": adaptive_anchor_min_per_frame,
        "adaptive_anchor_tau": adaptive_anchor_tau,
        "adaptive_anchor_uniform_mix": adaptive_anchor_uniform_mix,
        "adaptive_anchor_strategy": adaptive_anchor_strategy,
        "adaptive_anchor_score_alpha_cross": adaptive_anchor_score_alpha_cross,
        "adaptive_anchor_score_beta_intra": adaptive_anchor_score_beta_intra,
        "adaptive_anchor_topm_frames": adaptive_anchor_topm_frames,
        "adaptive_anchor_debug": adaptive_anchor_debug,
        "adaptive_anchor_debug_dir": adaptive_anchor_debug_dir,
    }
    signature = inspect.signature(VGGTOmega)
    accepts_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    unsupported_adaptive_kwargs = [
        key for key in adaptive_kwargs if key not in signature.parameters
    ]
    if accepts_extra_kwargs or not unsupported_adaptive_kwargs:
        model_kwargs.update(adaptive_kwargs)
    elif use_adaptive_kv_anchor:
        missing = ", ".join(unsupported_adaptive_kwargs)
        raise RuntimeError(
            "This VGGTOmega build does not support adaptive K/V anchor parameters "
            f"({missing}). Update vggt_omega.models.VGGTOmega before running with "
            "--use-adaptive-kv-anchor."
        )
    model = VGGTOmega(**model_kwargs)
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


def to_homogeneous_w2c(extrinsics: torch.Tensor) -> np.ndarray:
    w2c = extrinsics.detach().float().cpu().numpy().astype(np.float64)
    result = np.broadcast_to(np.eye(4), (len(w2c), 4, 4)).copy()
    result[:, :3] = w2c
    return result


def pairwise_pose_errors(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match facebookresearch/vggt's official pairwise angular metric."""
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for first in range(len(pred_w2c)):
        for second in range(first + 1, len(pred_w2c)):
            gt_relative = gt_w2c[first] @ np.linalg.inv(gt_w2c[second])
            pred_relative = pred_w2c[first] @ np.linalg.inv(pred_w2c[second])

            rotation_delta = gt_relative[:3, :3].T @ pred_relative[:3, :3]
            cosine = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
            rotation_errors.append(math.degrees(math.acos(float(cosine))))

            gt_translation = gt_relative[:3, 3]
            pred_translation = pred_relative[:3, 3]
            denominator = np.linalg.norm(gt_translation) * np.linalg.norm(pred_translation)
            if denominator <= 1e-15:
                translation_errors.append(1e6)
            else:
                cosine_t = np.clip(
                    abs(float(np.dot(gt_translation, pred_translation))) / denominator,
                    0.0,
                    1.0,
                )
                translation_errors.append(math.degrees(math.acos(cosine_t)))
    return np.asarray(rotation_errors), np.asarray(translation_errors)


def official_auc(rotation_errors: np.ndarray, translation_errors: np.ndarray, threshold: int) -> float:
    max_errors = np.maximum(rotation_errors, translation_errors)
    histogram, _ = np.histogram(max_errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(np.float64) / len(max_errors))))


def read_resized_depth(path: Path, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        raw = np.asarray(image, dtype=np.uint16)
    resized = Image.fromarray(raw).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.float32) / 1000.0


def depth_sums(
    predicted: np.ndarray,
    records: Sequence[FrameRecord],
    alignment: str,
    min_depth: float,
    max_depth: float,
) -> tuple[float, int, int, list[float]]:
    height, width = predicted.shape[1:]
    ground_truth = np.stack(
        [read_resized_depth(record.depth_path, height, width) for record in records]
    )
    valid = np.isfinite(ground_truth) & (ground_truth > min_depth) & (ground_truth < max_depth)
    valid &= np.isfinite(predicted) & (predicted > 0)
    aligned = predicted.astype(np.float64, copy=True)
    scales: list[float] = []
    if alignment == "per-frame-median":
        for index in range(len(predicted)):
            if not np.any(valid[index]):
                scales.append(float("nan"))
                continue
            scale = float(
                np.median(ground_truth[index][valid[index]])
                / np.median(predicted[index][valid[index]])
            )
            aligned[index] *= scale
            scales.append(scale)
    else:
        if not np.any(valid):
            raise ValueError("Sequence has no valid depth pixels")
        scale = float(np.median(ground_truth[valid]) / np.median(predicted[valid]))
        aligned *= scale
        scales = [scale] * len(predicted)

    # Standard monocular depth evaluation clips predictions to the dataset's
    # valid range after resolving scale. Without this, a tiny number of very
    # large edge predictions dominate AbsRel while barely affecting delta1.25.
    aligned = np.clip(aligned, min_depth, max_depth)

    gt_valid = ground_truth[valid].astype(np.float64)
    pred_valid = aligned[valid]
    if len(gt_valid) == 0:
        raise ValueError("Sequence has no valid depth pixels")
    abs_rel_sum = float(np.sum(np.abs(pred_valid - gt_valid) / gt_valid))
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta_count = int(np.count_nonzero(ratio < 1.25))
    return abs_rel_sum, delta_count, len(gt_valid), scales


def main() -> int:
    args = parse_args()
    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2")
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be at least 1")
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise ValueError("--image-resolution must be positive and divisible by 16")
    if not 0 < args.min_depth < args.max_depth:
        raise ValueError("Depth range must satisfy 0 < --min-depth < --max-depth")
    if not 0.0 <= args.merge_ratio <= 1.0:
        raise ValueError("--merge-ratio must be in [0, 1]")
    if args.sparse_attention and args.merge_ratio > 0.0:
        raise ValueError("--sparse-attention requires --merge-ratio 0; sparse attention replaces token merging")
    if args.sparse_attention and args.sparse_ratio is None and args.sparse_cdf_threshold is None:
        raise ValueError("--sparse-attention requires --sparse-ratio and/or --sparse-cdf-threshold")
    if not 0.0 <= args.adaptive_anchor_ratio <= 1.0:
        raise ValueError("--adaptive-anchor-ratio must be in [0, 1]")
    if args.adaptive_anchor_total is not None and args.adaptive_anchor_total <= 0:
        raise ValueError("--adaptive-anchor-total must be positive when set")
    if args.adaptive_anchor_min_per_frame < 0:
        raise ValueError("--adaptive-anchor-min-per-frame must be non-negative")
    if args.adaptive_anchor_tau <= 0.0:
        raise ValueError("--adaptive-anchor-tau must be positive")
    if not 0.0 <= args.adaptive_anchor_uniform_mix <= 1.0:
        raise ValueError("--adaptive-anchor-uniform-mix must be in [0, 1]")
    if args.adaptive_anchor_topm_frames is not None and args.adaptive_anchor_topm_frames <= 0:
        args.adaptive_anchor_topm_frames = None
    if args.use_adaptive_kv_anchor and args.merge_ratio != 0.0:
        raise ValueError("--use-adaptive-kv-anchor requires --merge-ratio 0")
    if args.use_adaptive_kv_anchor and args.sparse_attention:
        raise ValueError("--use-adaptive-kv-anchor is not compatible with --sparse-attention")

    sequence_dirs = select_sequence_dirs(args.data_root, args.sequences)
    pools: dict[str, list[FrameRecord]] = {}
    for sequence_dir in sequence_dirs:
        sequence_name = f"{sequence_dir.parent.name}/{sequence_dir.name}"
        records = load_frame_records(sequence_dir)
        print(f"{sequence_name}: found {len(records)} frames")
        pool_name = sequence_dir.parent.name if args.sampling_unit == "scene" else sequence_name
        pools.setdefault(pool_name, []).extend(records)
    for pool_name, records in pools.items():
        print(f"{pool_name}: sampling pool has {len(records)} frames")

    sampled, sampled_pool_indices = sample_records(pools, args.num_frames, args.seed)
    selection = {
        name: {
            "pool_indices": sampled_pool_indices[name],
            "frame_indices": [record.index for record in records],
            "rgb_paths": [str(record.rgb_path) for record in records],
            "depth_paths": [str(record.depth_path) for record in records],
        }
        for name, records in sampled.items()
    }
    missing_depth = [
        str(record.depth_path)
        for records in sampled.values()
        for record in records
        if not record.depth_path.is_file()
    ]
    if args.dry_run:
        print(json.dumps({"sampled_frames": selection, "missing_depth": missing_depth}, indent=2))
        return 0 if not missing_depth else 3
    if missing_depth:
        raise FileNotFoundError(
            f"Missing {len(missing_depth)} sampled RGB-registered depth maps; first: {missing_depth[0]}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega inference requires CUDA")
    exclusive_gpu_index = args.exclusive_gpu_index
    if args.require_exclusive_gpu and exclusive_gpu_index is None:
        if device.index is None:
            raise ValueError(
                "--exclusive-gpu-index is required with --require-exclusive-gpu when "
                "--device does not encode a physical CUDA index."
            )
        exclusive_gpu_index = int(device.index)
    if args.require_exclusive_gpu:
        assert_exclusive_gpu(
            exclusive_gpu_index,
            allowed_pids={os.getpid()},
            max_other_memory_mib=args.exclusive_gpu_max_other_memory_mib,
        )
        print(
            "Exclusive GPU guard: "
            f"physical_gpu={exclusive_gpu_index}, "
            f"max_other_memory_mib={args.exclusive_gpu_max_other_memory_mib}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sampled_frames.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
        handle.write("\n")

    print(f"Loading {args.checkpoint}")
    model = load_model(
        args.checkpoint,
        device,
        args.merge_ratio,
        args.sparse_attention,
        args.sparse_ratio,
        args.sparse_cdf_threshold,
        args.sparse_pool_mode,
        args.use_adaptive_kv_anchor,
        args.adaptive_anchor_layers,
        args.adaptive_anchor_ratio,
        args.adaptive_anchor_total,
        args.adaptive_anchor_min_per_frame,
        args.adaptive_anchor_tau,
        args.adaptive_anchor_uniform_mix,
        args.adaptive_anchor_strategy,
        args.adaptive_anchor_score_alpha_cross,
        args.adaptive_anchor_score_beta_intra,
        args.adaptive_anchor_topm_frames,
        args.adaptive_anchor_debug,
        args.adaptive_anchor_debug_dir,
    )
    if args.attention_mode == "register-only-zero-shot":
        model.aggregator.inter_frame_attention_types = ["register"] * model.aggregator.depth
        register_only_global_layers = parse_layer_indices(
            args.register_only_global_layers or DEFAULT_REGISTER_ONLY_GLOBAL_LAYER_SPEC,
            model.aggregator.depth,
        )
        for layer in register_only_global_layers:
            model.aggregator.inter_frame_attention_types[layer] = "global"
        model.aggregator.set_register_patch_inter_frame(
            mode=args.register_patch_inter_frame_mode,
            percent=args.register_patch_inter_frame_percent,
            seed=args.seed,
        )
    elif (
        args.register_only_global_layers
        or
        args.register_patch_inter_frame_mode != "none"
        or args.register_patch_inter_frame_percent != 0.0
    ):
        raise ValueError(
            "register-only global layers and patch selection are only supported with register-only-zero-shot"
        )
    else:
        register_only_global_layers = []
    inter_frame_only_global_layers = parse_layer_indices(
        args.inter_frame_only_global_layers,
        model.aggregator.depth,
    )
    model.aggregator.set_inter_frame_only_layers(inter_frame_only_global_layers)
    num_register_blocks = model.aggregator.inter_frame_attention_types.count("register")
    global_blocks = [
        index
        for index, attention_type in enumerate(model.aggregator.inter_frame_attention_types)
        if attention_type == "global"
    ]
    print(
        f"Attention schedule: {args.attention_mode} "
        f"({num_register_blocks}/{model.aggregator.depth} inter-frame blocks use register attention; "
        f"global blocks={global_blocks})"
    )
    if inter_frame_only_global_layers:
        print(f"Inter-frame-only global layers: {inter_frame_only_global_layers}")
    if args.sparse_attention:
        print(
            "Sparse global attention: "
            f"sparse_ratio={args.sparse_ratio}, "
            f"cdf_threshold={args.sparse_cdf_threshold}, "
            f"pool={args.sparse_pool_mode}"
        )
    if args.use_adaptive_kv_anchor:
        print(
            "Adaptive K/V anchor: "
            f"layers={args.adaptive_anchor_layers}, "
            f"ratio={args.adaptive_anchor_ratio}, "
            f"total={args.adaptive_anchor_total}, "
            f"min_per_frame={args.adaptive_anchor_min_per_frame}, "
            f"tau={args.adaptive_anchor_tau}, "
            f"uniform_mix={args.adaptive_anchor_uniform_mix}, "
            f"mode={args.adaptive_anchor_strategy}, "
            f"alpha_cross={args.adaptive_anchor_score_alpha_cross}, "
            f"beta_intra={args.adaptive_anchor_score_beta_intra}, "
            f"topm_frames={args.adaptive_anchor_topm_frames}, "
            f"debug={args.adaptive_anchor_debug}, "
            f"debug_dir={args.adaptive_anchor_debug_dir}"
        )
    all_rotation_errors: list[np.ndarray] = []
    all_translation_errors: list[np.ndarray] = []
    total_abs_rel = 0.0
    total_delta = 0
    total_valid = 0
    per_sequence: list[dict[str, object]] = []

    for sequence_name, records in sampled.items():
        images = load_and_preprocess_images(
            [str(record.rgb_path) for record in records],
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(device, non_blocking=True)
        if args.require_exclusive_gpu:
            assert_exclusive_gpu(
                exclusive_gpu_index,
                allowed_pids={os.getpid()},
                max_other_memory_mib=args.exclusive_gpu_max_other_memory_mib,
            )
        if not args.skip_timing:
            with torch.inference_mode():
                warmup = model(images)
            del warmup
            torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings_ms: list[float] = []
        predictions = None
        repeats = 1 if args.skip_timing else args.timing_repeats
        for _ in range(repeats):
            if not args.skip_timing:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            with torch.inference_mode():
                current_predictions = model(images)
            if not args.skip_timing:
                end_event.record()
                torch.cuda.synchronize(device)
                timings_ms.append(float(start_event.elapsed_time(end_event)))
            else:
                torch.cuda.synchronize(device)
            if predictions is not None:
                del predictions
            predictions = current_predictions
        assert predictions is not None

        with torch.inference_mode():
            extrinsics, _ = encoding_to_camera(
                predictions["pose_enc"], predictions["images"].shape[-2:], build_intrinsics=False
            )
        pred_w2c = to_homogeneous_w2c(extrinsics[0])
        gt_w2c = np.linalg.inv(np.stack([record.c2w for record in records]))
        rotation_errors, translation_errors = pairwise_pose_errors(pred_w2c, gt_w2c)
        predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
        abs_rel_sum, delta_count, valid_count, scales = depth_sums(
            predicted_depth, records, args.depth_alignment, args.min_depth, args.max_depth
        )
        all_rotation_errors.append(rotation_errors)
        all_translation_errors.append(translation_errors)
        total_abs_rel += abs_rel_sum
        total_delta += delta_count
        total_valid += valid_count

        row: dict[str, object] = {
            "sequence": sequence_name,
            "auc_3_percent": 100 * official_auc(rotation_errors, translation_errors, 3),
            "auc_30_percent": 100 * official_auc(rotation_errors, translation_errors, 30),
            "delta_1_25_percent": 100 * delta_count / valid_count,
            "abs_rel": abs_rel_sum / valid_count,
            "valid_depth_pixels": valid_count,
            "depth_scales": scales,
            "model_latency_ms": None if args.skip_timing else float(np.median(timings_ms)),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        per_sequence.append(row)
        latency_text = "skipped" if args.skip_timing else f"{row['model_latency_ms']:.1f}ms"
        print(
            f"[{sequence_name}] AUC@3={row['auc_3_percent']:.2f}, "
            f"AUC@30={row['auc_30_percent']:.2f}, delta1.25={row['delta_1_25_percent']:.2f}, "
            f"AbsRel={row['abs_rel']:.4f}, latency={latency_text}"
        )
        del images, predictions, extrinsics
        torch.cuda.empty_cache()

    rotation_errors = np.concatenate(all_rotation_errors)
    translation_errors = np.concatenate(all_translation_errors)
    overall = {
        "auc_3_percent": 100 * official_auc(rotation_errors, translation_errors, 3),
        "auc_30_percent": 100 * official_auc(rotation_errors, translation_errors, 30),
        "delta_1_25_percent": 100 * total_delta / total_valid,
        "abs_rel": total_abs_rel / total_valid,
        "valid_depth_pixels": total_valid,
        "model_latency_ms_mean": None
        if args.skip_timing
        else float(np.mean([float(row["model_latency_ms"]) for row in per_sequence])),
        "peak_allocated_gib_max": float(
            np.max([float(row["peak_allocated_gib"]) for row in per_sequence])
        ),
    }
    result = {
        "protocol": {
            "dataset_split": "official TestSplit.txt files",
            "sampling_unit": args.sampling_unit,
            "seed": args.seed,
            "num_frames_per_sequence": args.num_frames,
            "num_sequences": len(sampled),
            "num_pose_pairs": len(rotation_errors),
            "image_resolution": args.image_resolution,
            "resize_mode": args.resize_mode,
            "depth_alignment": args.depth_alignment,
            "min_depth_m": args.min_depth,
            "max_depth_m": args.max_depth,
            "prediction_clip_m": [args.min_depth, args.max_depth],
            "merge_ratio": args.merge_ratio,
            "sparse_attention": args.sparse_attention,
            "sparse_ratio": args.sparse_ratio,
            "sparse_cdf_threshold": args.sparse_cdf_threshold,
            "sparse_pool_mode": args.sparse_pool_mode,
            "use_adaptive_kv_anchor": args.use_adaptive_kv_anchor,
            "adaptive_anchor_layers": args.adaptive_anchor_layers,
            "adaptive_anchor_active_layers": sorted(model.aggregator.adaptive_anchor_layers),
            "adaptive_anchor_ratio": args.adaptive_anchor_ratio,
            "adaptive_anchor_total": args.adaptive_anchor_total,
            "adaptive_anchor_min_per_frame": args.adaptive_anchor_min_per_frame,
            "adaptive_anchor_tau": args.adaptive_anchor_tau,
            "adaptive_anchor_uniform_mix": args.adaptive_anchor_uniform_mix,
            "adaptive_anchor_strategy": args.adaptive_anchor_strategy,
            "adaptive_anchor_score_alpha_cross": args.adaptive_anchor_score_alpha_cross,
            "adaptive_anchor_score_beta_intra": args.adaptive_anchor_score_beta_intra,
            "adaptive_anchor_topm_frames": args.adaptive_anchor_topm_frames,
            "adaptive_anchor_debug": args.adaptive_anchor_debug,
            "adaptive_anchor_debug_dir": str(args.adaptive_anchor_debug_dir),
            "attention_mode": args.attention_mode,
            "register_only_global_layers": register_only_global_layers,
            "inter_frame_only_global_layers": inter_frame_only_global_layers,
            "attention_schedule": model.aggregator.inter_frame_attention_types,
            "register_patch_inter_frame_mode": args.register_patch_inter_frame_mode,
            "register_patch_inter_frame_percent": args.register_patch_inter_frame_percent,
            "register_attention_blocks": num_register_blocks,
            "total_inter_frame_blocks": model.aggregator.depth,
            "timing_repeats": args.timing_repeats,
            "skip_timing": args.skip_timing,
            "require_exclusive_gpu": args.require_exclusive_gpu,
            "exclusive_gpu_index": exclusive_gpu_index,
            "exclusive_gpu_max_other_memory_mib": args.exclusive_gpu_max_other_memory_mib,
        },
        "paper_targets_1b": PAPER_TARGETS,
        "overall": overall,
        "difference_from_paper": {
            key: float(overall[key]) - target for key, target in PAPER_TARGETS.items()
        },
        "per_sequence": per_sequence,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    np.savez_compressed(
        args.output_dir / "pose_errors.npz",
        rotation_error_deg=rotation_errors,
        translation_error_deg=translation_errors,
    )

    print("\nPaper reproduction result (Ours-1B target in parentheses):")
    print(f"  AUC@3:     {overall['auc_3_percent']:.2f} ({PAPER_TARGETS['auc_3_percent']:.1f})")
    print(f"  AUC@30:    {overall['auc_30_percent']:.2f} ({PAPER_TARGETS['auc_30_percent']:.1f})")
    print(
        f"  delta1.25: {overall['delta_1_25_percent']:.2f} "
        f"({PAPER_TARGETS['delta_1_25_percent']:.1f})"
    )
    print(f"  AbsRel:    {overall['abs_rel']:.4f} ({PAPER_TARGETS['abs_rel']:.3f})")
    print(f"Saved reproducible results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        started = time.perf_counter()
        exit_code = main()
        print(f"Total elapsed: {time.perf_counter() - started:.1f}s")
        sys.exit(exit_code)
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
