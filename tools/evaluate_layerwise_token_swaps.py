#!/usr/bin/env python3
"""Evaluate layerwise frame-token swaps and continue the remaining network."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval_7scenes_paper as seven_eval  # noqa: E402
import scripts.eval_tum_dynamics_paper as tum_eval  # noqa: E402
from tools.analyze_token_evolution import load_model  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


DEFAULT_MATRIX_ROOTS = (
    REPO_ROOT / "outputs" / "frame_similarity_matrices__tum_halfsphere_300f__layers_2_6_10_16_23",
    REPO_ROOT / "outputs" / "frame_similarity_matrices__tum_rpy_300f__layers_2_6_10_16_23",
    REPO_ROOT / "outputs" / "frame_similarity_matrices__7scenes_chess_seq03_300f__layers_2_6_10_16_23",
    REPO_ROOT / "outputs" / "frame_similarity_matrices__7scenes_chess_seq05_300f__layers_2_6_10_16_23",
)
SWAP_KINDS = ("patch", "special", "whole")


@dataclass(frozen=True)
class SequenceSpec:
    dataset: str
    sequence: str
    root: Path
    image_paths: list[Path]
    records: list[object]
    pairs: list[tuple[int, int, float]]
    candidate_edges: int
    pair_stage: str
    image_resolution: int
    resize_mode: str
    checkpoint: Path

    @property
    def slug(self) -> str:
        return slugify(f"{self.dataset}__{self.sequence}")


class DepthEvaluator:
    def __init__(
        self,
        dataset: str,
        records: Sequence[object],
        image_hw: tuple[int, int],
        *,
        min_depth: float,
        max_depth: float,
    ) -> None:
        self.dataset = dataset
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        height, width = image_hw
        if dataset == "7Scenes":
            self.ground_truth = np.stack(
                [seven_eval.read_resized_depth(record.depth_path, height, width) for record in records]
            )
            self.base_valid = (
                np.isfinite(self.ground_truth)
                & (self.ground_truth > self.min_depth)
                & (self.ground_truth < self.max_depth)
            )
        elif dataset == "TUM-Dynamics":
            self.ground_truth = np.stack(
                [tum_eval.read_resized_depth(record.depth_path, height, width) for record in records]
            )
            self.base_valid = (
                np.isfinite(self.ground_truth)
                & (self.ground_truth > 0)
                & (self.ground_truth < self.max_depth)
            )
        else:
            raise ValueError(f"unsupported dataset: {dataset}")

    def summarize(
        self,
        predicted: np.ndarray,
        *,
        alignment: str,
        frame_indices: Sequence[int] | None = None,
    ) -> dict[str, float]:
        if frame_indices is not None:
            indices = np.asarray(list(frame_indices), dtype=np.int64)
            ground_truth = self.ground_truth[indices]
            base_valid = self.base_valid[indices]
            predicted = predicted[indices]
        else:
            ground_truth = self.ground_truth
            base_valid = self.base_valid

        valid = base_valid & np.isfinite(predicted) & (predicted > 0)
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
        elif alignment == "per-sequence-median":
            if not np.any(valid):
                raise ValueError("sequence has no valid depth pixels")
            scale = float(np.median(ground_truth[valid]) / np.median(predicted[valid]))
            aligned *= scale
            scales = [scale] * len(predicted)
        else:
            raise ValueError(f"unknown depth alignment: {alignment}")

        if self.dataset == "7Scenes":
            aligned = np.clip(aligned, self.min_depth, self.max_depth)

        gt_valid = ground_truth[valid].astype(np.float64)
        pred_valid = aligned[valid]
        if len(gt_valid) == 0:
            raise ValueError("sequence has no valid depth pixels")
        abs_rel_sum = float(np.sum(np.abs(pred_valid - gt_valid) / gt_valid))
        ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
        delta_count = int(np.count_nonzero(ratio < 1.25))
        return {
            "valid_depth_pixels": int(len(gt_valid)),
            "abs_rel": float(abs_rel_sum / len(gt_valid)),
            "delta_1_25_percent": float(100.0 * delta_count / len(gt_valid)),
            "median_scale": float(np.nanmedian(scales)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-roots", nargs="+", type=Path, default=list(DEFAULT_MATRIX_ROOTS))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/layerwise_token_swap_300f_2datasets_2seq"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0-23")
    parser.add_argument("--swap-kinds", nargs="+", choices=SWAP_KINDS, default=list(SWAP_KINDS))
    parser.add_argument("--pair-stage", default="layer_23")
    parser.add_argument("--threshold", type=float, default=0.76)
    parser.add_argument("--top-percent", type=float, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--expected-num-frames", type=int, default=300)
    parser.add_argument("--image-resolution", type=int, default=None)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default=None)
    parser.add_argument("--association-tolerance", type=float, default=0.02)
    parser.add_argument("--depth-alignment", choices=("per-frame-median", "per-sequence-median"), default="per-frame-median")
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    return parser.parse_args()


def parse_layers(spec: str, depth: int = 24) -> list[int]:
    normalized = spec.strip().lower()
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
                raise ValueError(f"invalid layer range: {part!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= depth)
    if invalid:
        raise ValueError(f"layer indices out of range 0..{depth - 1}: {invalid}")
    return sorted(layers)


def slugify(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def infer_dataset_sequence(paths: Sequence[Path]) -> tuple[str, str]:
    first = paths[0]
    parts = first.parts
    if "TUM-Dynamics" in parts:
        index = parts.index("TUM-Dynamics")
        return "TUM-Dynamics", parts[index + 1]
    if "7scenes" in parts:
        index = parts.index("7scenes")
        return "7Scenes", f"{parts[index + 1]}/{parts[index + 2]}"
    raise ValueError(f"cannot infer dataset from path: {first}")


def load_records(dataset: str, image_paths: Sequence[Path], tolerance: float) -> list[object]:
    if dataset == "TUM-Dynamics":
        sequence_dir = image_paths[0].parents[1]
        all_records = tum_eval.load_frame_records(sequence_dir, tolerance)
    elif dataset == "7Scenes":
        sequence_dir = image_paths[0].parent
        all_records = seven_eval.load_frame_records(sequence_dir)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    by_rgb_path = {record.rgb_path.resolve(): record for record in all_records}
    records: list[object] = []
    for path in image_paths:
        try:
            records.append(by_rgb_path[path.resolve()])
        except KeyError as exc:
            raise ValueError(f"metadata frame is not present in evaluated records: {path}") from exc
    return records


def upper_edges(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return rows, cols, matrix[rows, cols]


def select_pairs(
    matrix: np.ndarray,
    *,
    threshold: float,
    top_percent: float | None,
    max_pairs: int | None,
) -> tuple[list[tuple[int, int, float]], int]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected square similarity matrix, got {matrix.shape}")
    rows, cols, values = upper_edges(matrix)
    if top_percent is not None:
        if top_percent <= 0.0 or top_percent > 100.0:
            raise ValueError("--top-percent must be in (0, 100]")
        candidate_count = max(1, int(np.ceil(len(values) * top_percent / 100.0)))
        candidate_indices = np.argpartition(-values, candidate_count - 1)[:candidate_count]
    else:
        candidate_indices = np.flatnonzero(values > threshold)
        candidate_count = int(len(candidate_indices))
    if candidate_count == 0:
        raise ValueError("no candidate frame pairs selected")

    edges = [
        (float(values[index]), int(rows[index]), int(cols[index]))
        for index in candidate_indices
    ]
    edges.sort(reverse=True)
    used: set[int] = set()
    selected: list[tuple[int, int, float]] = []
    for similarity, first, second in edges:
        if first in used or second in used:
            continue
        used.add(first)
        used.add(second)
        selected.append((first, second, similarity))
        if max_pairs is not None and len(selected) >= max_pairs:
            break
    if not selected:
        raise ValueError("candidate pairs collapsed to an empty non-overlapping matching")
    return selected, candidate_count


def load_sequence_spec(root: Path, args: argparse.Namespace) -> SequenceSpec:
    metadata_path = root / "metadata.json"
    matrix_path = root / "frame_similarity_matrices.npz"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_paths = [Path(path) for path in metadata["image_paths"]]
    if args.expected_num_frames and len(image_paths) != args.expected_num_frames:
        raise ValueError(
            f"{root}: expected {args.expected_num_frames} frames, found {len(image_paths)}"
        )
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing image from metadata: {missing[0]}")

    dataset, sequence = infer_dataset_sequence(image_paths)
    loaded = np.load(matrix_path)
    if args.pair_stage not in loaded:
        raise ValueError(f"{matrix_path} lacks stage {args.pair_stage!r}; stages={loaded.files}")
    pairs, candidate_edges = select_pairs(
        loaded[args.pair_stage],
        threshold=args.threshold,
        top_percent=args.top_percent,
        max_pairs=args.max_pairs,
    )
    checkpoint = args.checkpoint or Path(metadata.get("checkpoint", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    records = load_records(dataset, image_paths, args.association_tolerance)
    return SequenceSpec(
        dataset=dataset,
        sequence=sequence,
        root=root,
        image_paths=image_paths,
        records=records,
        pairs=pairs,
        candidate_edges=candidate_edges,
        pair_stage=args.pair_stage,
        image_resolution=int(args.image_resolution or metadata.get("image_resolution", 512)),
        resize_mode=str(args.resize_mode or metadata.get("resize_mode", "balanced")),
        checkpoint=checkpoint,
    )


def write_sequence_manifest(output_dir: Path, spec: SequenceSpec) -> None:
    sequence_dir = output_dir / spec.slug
    write_json(
        sequence_dir / "sequence_manifest.json",
        {
            "dataset": spec.dataset,
            "sequence": spec.sequence,
            "matrix_root": str(spec.root),
            "pair_stage": spec.pair_stage,
            "candidate_edges": spec.candidate_edges,
            "selected_pairs": len(spec.pairs),
            "selected_frames": len({index for pair in spec.pairs for index in pair[:2]}),
            "checkpoint": str(spec.checkpoint),
            "image_resolution": spec.image_resolution,
            "resize_mode": spec.resize_mode,
            "num_input_frames": len(spec.image_paths),
            "image_paths": [str(path) for path in spec.image_paths],
        },
    )
    pairs_path = sequence_dir / "selected_pairs.csv"
    temporary = pairs_path.with_name(f"{pairs_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_index", "frame_i", "frame_j", "similarity"])
        writer.writeheader()
        for index, (first, second, similarity) in enumerate(spec.pairs):
            writer.writerow(
                {
                    "pair_index": index,
                    "frame_i": first,
                    "frame_j": second,
                    "similarity": f"{similarity:.8f}",
                }
            )
    temporary.replace(pairs_path)


def pose_pair_errors(
    pred_w2c: np.ndarray,
    gt_w2c: np.ndarray,
    pairs: Sequence[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for first, second, _ in pairs:
        gt_relative = gt_w2c[first] @ np.linalg.inv(gt_w2c[second])
        pred_relative = pred_w2c[first] @ np.linalg.inv(pred_w2c[second])

        rotation_delta = gt_relative[:3, :3].T @ pred_relative[:3, :3]
        cosine = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))

        gt_translation = gt_relative[:3, 3]
        pred_translation = pred_relative[:3, 3]
        denominator = np.linalg.norm(gt_translation) * np.linalg.norm(pred_translation)
        if denominator <= 1e-15:
            translation_errors.append(1e6)
        else:
            cosine_t = np.clip(abs(float(np.dot(gt_translation, pred_translation))) / denominator, 0.0, 1.0)
            translation_errors.append(float(np.degrees(np.arccos(cosine_t))))
    return np.asarray(rotation_errors), np.asarray(translation_errors)


def pose_summary(rotation_errors: np.ndarray, translation_errors: np.ndarray) -> dict[str, float]:
    max_errors = np.maximum(rotation_errors, translation_errors)
    return {
        "num_pose_pairs": int(len(max_errors)),
        "auc_3_percent": 100.0 * tum_eval.official_auc(rotation_errors, translation_errors, 3),
        "auc_30_percent": 100.0 * tum_eval.official_auc(rotation_errors, translation_errors, 30),
        "rotation_error_mean_deg": float(rotation_errors.mean()),
        "translation_error_mean_deg": float(translation_errors.mean()),
        "max_error_mean_deg": float(max_errors.mean()),
        "max_error_median_deg": float(np.median(max_errors)),
    }


def metric_delta(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        f"delta_{key}": float(current[key] - baseline[key])
        for key in current
        if key in baseline and isinstance(current[key], (int, float))
    }


def summarize_predictions(
    predictions: dict[str, torch.Tensor],
    *,
    dataset: str,
    records: Sequence[object],
    pairs: Sequence[tuple[int, int, float]],
    depth_evaluator: DepthEvaluator,
    depth_alignment: str,
) -> dict[str, dict[str, float]]:
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        extrinsics, _ = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
            build_intrinsics=False,
        )
    to_w2c = seven_eval.to_homogeneous_w2c if dataset == "7Scenes" else tum_eval.to_homogeneous_w2c
    pairwise = seven_eval.pairwise_pose_errors if dataset == "7Scenes" else tum_eval.pairwise_pose_errors
    pred_w2c = to_w2c(extrinsics[0])
    gt_w2c = np.linalg.inv(np.stack([record.c2w for record in records]))
    rotation_errors, translation_errors = pairwise(pred_w2c, gt_w2c)
    pair_rotation_errors, pair_translation_errors = pose_pair_errors(pred_w2c, gt_w2c, pairs)

    predicted_depth = predictions["depth"][0, ..., 0].detach().float().cpu().numpy()
    selected_frames = sorted({index for pair in pairs for index in pair[:2]})
    return {
        "pose_all_pairs": pose_summary(rotation_errors, translation_errors),
        "pose_swap_pairs_only": pose_summary(pair_rotation_errors, pair_translation_errors),
        "depth_all_frames": depth_evaluator.summarize(
            predicted_depth,
            alignment=depth_alignment,
        ),
        "depth_swap_frames": depth_evaluator.summarize(
            predicted_depth,
            alignment=depth_alignment,
            frame_indices=selected_frames,
        ),
    }


def run_forward(
    model,
    images: torch.Tensor,
    *,
    device: torch.device,
    layer: int | None,
    swap_kind: str,
    pairs: Sequence[tuple[int, int, float]],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if layer is None:
        model.aggregator.set_layer_token_swap(None)
    else:
        model.aggregator.set_layer_token_swap(
            layer,
            kind=swap_kind,
            pairs=[(first, second) for first, second, _ in pairs],
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        predictions = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated_gib = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    peak_reserved_gib = (
        torch.cuda.max_memory_reserved(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    model.aggregator.set_layer_token_swap(None)
    return predictions, {
        "wall_seconds": float(elapsed),
        "peak_allocated_gib": float(peak_allocated_gib),
        "peak_reserved_gib": float(peak_reserved_gib),
    }


def result_path(output_dir: Path, spec: SequenceSpec, layer: int, kind: str) -> Path:
    return output_dir / spec.slug / f"layer_{layer:02d}_{kind}.json"


def baseline_path(output_dir: Path, spec: SequenceSpec) -> Path:
    return output_dir / spec.slug / "baseline.json"


def load_images(spec: SequenceSpec, device: torch.device) -> torch.Tensor:
    return load_and_preprocess_images(
        [str(path) for path in spec.image_paths],
        mode=spec.resize_mode,
        image_resolution=spec.image_resolution,
    ).to(device, non_blocking=True)


def run_tasks(args: argparse.Namespace, specs: list[SequenceSpec], layers: list[int]) -> int:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    all_tasks: list[tuple[int, SequenceSpec, int, str]] = []
    task_index = 0
    for spec in specs:
        for layer in layers:
            for kind in args.swap_kinds:
                all_tasks.append((task_index, spec, layer, kind))
                task_index += 1
    tasks = [
        task
        for task in all_tasks
        if task[0] % args.num_shards == args.shard_index
    ]

    if args.dry_run:
        print(json.dumps(
            [
                {
                    "task_index": index,
                    "dataset": spec.dataset,
                    "sequence": spec.sequence,
                    "layer": layer,
                    "swap_kind": kind,
                    "result_path": str(result_path(args.output_dir, spec, layer, kind)),
                }
                for index, spec, layer, kind in tasks
            ],
            indent=2,
        ))
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("layerwise token-swap evaluation requires CUDA")
    checkpoint = specs[0].checkpoint
    model = load_model(checkpoint, device)

    baseline_cache: dict[str, dict[str, object]] = {}
    current_slug: str | None = None
    images: torch.Tensor | None = None
    depth_evaluator: DepthEvaluator | None = None

    for _, spec, layer, kind in tasks:
        output_path = result_path(args.output_dir, spec, layer, kind)
        if args.skip_existing and output_path.is_file():
            print(f"skip existing {output_path}", flush=True)
            continue

        if current_slug != spec.slug:
            if images is not None:
                del images
            if device.type == "cuda":
                torch.cuda.empty_cache()
            current_slug = spec.slug
            write_sequence_manifest(args.output_dir, spec)
            images = load_images(spec, device)
            image_hw = tuple(int(value) for value in images.shape[-2:])
            depth_evaluator = DepthEvaluator(
                spec.dataset,
                spec.records,
                image_hw,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )

        assert images is not None
        assert depth_evaluator is not None
        baseline = baseline_cache.get(spec.slug)
        if baseline is None:
            baseline_file = baseline_path(args.output_dir, spec)
            if args.skip_existing and baseline_file.is_file():
                baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
            else:
                predictions, timing = run_forward(
                    model,
                    images,
                    device=device,
                    layer=None,
                    swap_kind="none",
                    pairs=spec.pairs,
                )
                metrics = summarize_predictions(
                    predictions,
                    dataset=spec.dataset,
                    records=spec.records,
                    pairs=spec.pairs,
                    depth_evaluator=depth_evaluator,
                    depth_alignment=args.depth_alignment,
                )
                baseline = {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset": spec.dataset,
                    "sequence": spec.sequence,
                    "num_input_frames": len(spec.image_paths),
                    "pair_stage": spec.pair_stage,
                    "candidate_edges": spec.candidate_edges,
                    "selected_pairs": len(spec.pairs),
                    "selected_frames": len({index for pair in spec.pairs for index in pair[:2]}),
                    "checkpoint": str(spec.checkpoint),
                    "image_resolution": spec.image_resolution,
                    "resize_mode": spec.resize_mode,
                    "image_shape_hw": list(images.shape[-2:]),
                    "patch_token_start": int(model.aggregator.patch_token_start),
                    "timing": timing,
                    "metrics": metrics,
                }
                write_json(baseline_file, baseline)
                del predictions
            baseline_cache[spec.slug] = baseline

        print(f"run {spec.dataset} {spec.sequence} layer={layer:02d} kind={kind}", flush=True)
        predictions, timing = run_forward(
            model,
            images,
            device=device,
            layer=layer,
            swap_kind=kind,
            pairs=spec.pairs,
        )
        metrics = summarize_predictions(
            predictions,
            dataset=spec.dataset,
            records=spec.records,
            pairs=spec.pairs,
            depth_evaluator=depth_evaluator,
            depth_alignment=args.depth_alignment,
        )
        del predictions

        baseline_metrics = baseline["metrics"]
        deltas = {
            key: metric_delta(metrics[key], baseline_metrics[key])
            for key in metrics
        }
        write_json(
            output_path,
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": spec.dataset,
                "sequence": spec.sequence,
                "matrix_root": str(spec.root),
                "num_input_frames": len(spec.image_paths),
                "layer": layer,
                "swap_kind": kind,
                "swap_semantics": (
                    "swap after the selected aggregator layer's inter-frame block, "
                    "then continue the remaining aggregator layers and heads"
                ),
                "pair_stage": spec.pair_stage,
                "threshold": None if args.top_percent is not None else args.threshold,
                "top_percent": args.top_percent,
                "candidate_edges": spec.candidate_edges,
                "selected_pairs": len(spec.pairs),
                "selected_frames": len({index for pair in spec.pairs for index in pair[:2]}),
                "checkpoint": str(spec.checkpoint),
                "image_resolution": spec.image_resolution,
                "resize_mode": spec.resize_mode,
                "image_shape_hw": list(images.shape[-2:]),
                "patch_token_start": int(model.aggregator.patch_token_start),
                "timing": timing,
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "delta_vs_baseline": deltas,
            },
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


def flatten_for_row(result: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": result["dataset"],
        "sequence": result["sequence"],
        "layer": result["layer"],
        "swap_kind": result["swap_kind"],
        "selected_pairs": result["selected_pairs"],
        "selected_frames": result["selected_frames"],
        "wall_seconds": result["timing"]["wall_seconds"],
        "peak_allocated_gib": result["timing"]["peak_allocated_gib"],
    }
    metrics = result["metrics"]
    deltas = result["delta_vs_baseline"]
    for group in ("pose_all_pairs", "pose_swap_pairs_only", "depth_all_frames", "depth_swap_frames"):
        for key, value in metrics[group].items():
            if isinstance(value, (int, float)):
                row[f"{group}.{key}"] = value
        for key, value in deltas[group].items():
            if isinstance(value, (int, float)):
                row[f"{group}.{key}"] = value
    return row


def collect_results(output_dir: Path) -> int:
    result_files = sorted(output_dir.glob("*/layer_*.json"))
    if not result_files:
        raise FileNotFoundError(f"no layer result JSON files under {output_dir}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
    rows = [flatten_for_row(result) for result in results]
    fieldnames = sorted({key for row in rows for key in row})
    summary_dir = output_dir
    with (summary_dir / "layer_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate: list[dict[str, object]] = []
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["layer"]), str(row["swap_kind"]))
        grouped.setdefault(key, []).append(row)
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key not in {"layer"}
    ]
    for (dataset, layer, kind), group in sorted(grouped.items()):
        item: dict[str, object] = {
            "dataset": dataset,
            "layer": layer,
            "swap_kind": kind,
            "num_sequences": len(group),
        }
        for key in numeric_keys:
            item[f"{key}.mean"] = float(np.mean([float(row[key]) for row in group if key in row]))
        aggregate.append(item)
    aggregate_fields = sorted({key for row in aggregate for key in row})
    with (summary_dir / "dataset_layer_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate)
    write_json(
        summary_dir / "summary.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_files": len(result_files),
            "sequences": sorted({f"{result['dataset']}::{result['sequence']}" for result in results}),
            "layers": sorted({int(result["layer"]) for result in results}),
            "swap_kinds": sorted({str(result["swap_kind"]) for result in results}),
            "layer_results_csv": str(summary_dir / "layer_results.csv"),
            "dataset_layer_summary_csv": str(summary_dir / "dataset_layer_summary.csv"),
        },
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.collect_only:
        return collect_results(args.output_dir)
    layers = parse_layers(args.layers)
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    specs = [load_sequence_spec(root, args) for root in args.matrix_roots]
    checkpoints = {spec.checkpoint.resolve() for spec in specs}
    if len(checkpoints) != 1:
        raise ValueError(f"all specs must use the same checkpoint, got {sorted(str(path) for path in checkpoints)}")
    return run_tasks(args, specs, layers)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
