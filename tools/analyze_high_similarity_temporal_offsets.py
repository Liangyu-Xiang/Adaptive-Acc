#!/usr/bin/env python3
"""Analyze whether high-similarity frame pairs are temporally adjacent."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_ROOTS = (
    REPO_ROOT / "outputs" / "frame_similarity_matrices__tum_halfsphere_300f__layers_2_6_10_16_23",
    REPO_ROOT / "outputs" / "frame_similarity_matrices__tum_rpy_300f__layers_2_6_10_16_23",
    REPO_ROOT / "outputs" / "frame_similarity_matrices__7scenes_chess_seq03_300f__layers_2_6_10_16_23",
    REPO_ROOT / "outputs" / "frame_similarity_matrices__7scenes_chess_seq05_300f__layers_2_6_10_16_23",
)
DEFAULT_STAGES = ("input_rgb", "layer_02", "layer_06", "layer_10", "layer_16", "layer_23")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-roots", nargs="+", type=Path, default=list(DEFAULT_MATRIX_ROOTS))
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/high_similarity_temporal_offsets_300f"))
    parser.add_argument("--examples", type=int, default=8)
    return parser.parse_args()


def infer_dataset_sequence(paths: list[str]) -> tuple[str, str]:
    first = Path(paths[0])
    parts = first.parts
    if "TUM-Dynamics" in parts:
        index = parts.index("TUM-Dynamics")
        return "TUM-Dynamics", parts[index + 1]
    if "7scenes" in parts:
        index = parts.index("7scenes")
        return "7Scenes", f"{parts[index + 1]}/{parts[index + 2]}"
    return "unknown", first.parent.name


def temporal_value(path: str) -> float:
    name = Path(path).name
    if name.startswith("frame-"):
        match = re.match(r"frame-(\d+)", name)
        if match:
            return float(int(match.group(1)))
    return float(Path(path).stem)


def rank_temporal_values(paths: list[str]) -> np.ndarray:
    values = np.asarray([temporal_value(path) for path in paths], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks


def summarize_values(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_max": float(np.max(values)),
    }


def top_pairs(matrix: np.ndarray, top_percent: float, top_k: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got {matrix.shape}")
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    values = matrix[rows, cols]
    if top_k is None:
        count = max(1, int(math.ceil(len(values) * top_percent / 100.0)))
    else:
        count = max(1, min(int(top_k), len(values)))
    indices = np.argpartition(-values, count - 1)[:count]
    indices = indices[np.argsort(-values[indices])]
    return rows[indices], cols[indices], values[indices]


def all_pair_gap_stats(count: int, temporal_ranks: np.ndarray) -> dict[str, float]:
    rows, cols = np.triu_indices(count, k=1)
    input_gaps = np.abs(rows - cols)
    temporal_gaps = np.abs(temporal_ranks[rows] - temporal_ranks[cols])
    result = {}
    result.update(summarize_values(input_gaps, "all_input_gap"))
    result.update(summarize_values(temporal_gaps, "all_temporal_rank_gap"))
    return result


def analyze_root(
    matrix_root: Path,
    stages: list[str],
    top_percent: float,
    top_k: int | None,
    examples: int,
) -> list[dict[str, object]]:
    metadata_path = matrix_root / "metadata.json"
    npz_path = matrix_root / "frame_similarity_matrices.npz"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_paths = list(metadata["image_paths"])
    dataset, sequence = infer_dataset_sequence(image_paths)
    temporal_ranks = rank_temporal_values(image_paths)
    base_stats = all_pair_gap_stats(len(image_paths), temporal_ranks)
    matrices = np.load(npz_path)

    rows: list[dict[str, object]] = []
    for stage in stages:
        if stage not in matrices:
            raise KeyError(f"{stage} is not present in {npz_path}")
        matrix = matrices[stage]
        pair_i, pair_j, similarity = top_pairs(matrix, top_percent, top_k)
        input_gaps = np.abs(pair_i - pair_j)
        temporal_rank_gaps = np.abs(temporal_ranks[pair_i] - temporal_ranks[pair_j])

        row: dict[str, object] = {
            "dataset": dataset,
            "sequence": sequence,
            "matrix_root": str(matrix_root),
            "stage": stage,
            "num_frames": int(matrix.shape[0]),
            "selected_pairs": int(len(similarity)),
            "selection": f"top_{top_percent:g}_percent" if top_k is None else f"top_{top_k}",
            "similarity_min": float(np.min(similarity)),
            "similarity_mean": float(np.mean(similarity)),
            "similarity_max": float(np.max(similarity)),
            "input_non_adjacent_fraction": float(np.mean(input_gaps > 1)),
            "input_gap_gt_10_fraction": float(np.mean(input_gaps > 10)),
            "input_gap_gt_50_fraction": float(np.mean(input_gaps > 50)),
            "temporal_non_adjacent_fraction": float(np.mean(temporal_rank_gaps > 1)),
            "temporal_gap_gt_10_fraction": float(np.mean(temporal_rank_gaps > 10)),
            "temporal_gap_gt_50_fraction": float(np.mean(temporal_rank_gaps > 50)),
            "examples": [
                {
                    "frame_i": int(i),
                    "frame_j": int(j),
                    "similarity": float(value),
                    "input_gap": int(abs(i - j)),
                    "temporal_rank_gap": int(abs(temporal_ranks[i] - temporal_ranks[j])),
                    "image_i": image_paths[int(i)],
                    "image_j": image_paths[int(j)],
                }
                for i, j, value in zip(pair_i[:examples], pair_j[:examples], similarity[:examples])
            ],
        }
        row.update(summarize_values(input_gaps, "input_gap"))
        row.update(summarize_values(temporal_rank_gaps, "temporal_rank_gap"))
        row.update(base_stats)
        rows.append(row)
    return rows


def mean_numeric(rows: list[dict[str, object]], keys: list[str]) -> dict[str, float]:
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        result[key] = float(np.mean(values))
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    excluded = {"examples"}
    fieldnames = [key for key in rows[0] if key not in excluded]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key not in excluded})


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_keys = [
        "selected_pairs",
        "similarity_min",
        "similarity_mean",
        "similarity_max",
        "input_non_adjacent_fraction",
        "input_gap_gt_10_fraction",
        "input_gap_gt_50_fraction",
        "temporal_non_adjacent_fraction",
        "temporal_gap_gt_10_fraction",
        "temporal_gap_gt_50_fraction",
        "input_gap_mean",
        "input_gap_median",
        "input_gap_p90",
        "input_gap_max",
        "temporal_rank_gap_mean",
        "temporal_rank_gap_median",
        "temporal_rank_gap_p90",
        "temporal_rank_gap_max",
    ]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["stage"]))].append(row)
    output = []
    for (dataset, stage), group in sorted(grouped.items()):
        aggregate = {
            "dataset": dataset,
            "stage": stage,
            "num_sequences": len(group),
        }
        aggregate.update(mean_numeric(group, numeric_keys))
        output.append(aggregate)
    return output


def plot_temporal_nonlocal(output_dir: Path, aggregate: list[dict[str, object]]) -> None:
    stage_order = list(DEFAULT_STAGES)
    datasets = sorted({row["dataset"] for row in aggregate})
    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets):
        rows = [row for row in aggregate if row["dataset"] == dataset]
        rows.sort(key=lambda row: stage_order.index(row["stage"]) if row["stage"] in stage_order else 999)
        labels = [row["stage"] for row in rows]
        values = [100.0 * float(row["temporal_non_adjacent_fraction"]) for row in rows]
        axis.bar(labels, values)
        axis.set_ylim(0, 100)
        axis.set_title(dataset)
        axis.set_ylabel("Top-pair temporal non-adjacent (%)")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "temporal_non_adjacent_fraction.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.top_percent <= 0.0:
        raise ValueError("--top-percent must be positive")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive when set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for root in args.matrix_roots:
        rows.extend(
            analyze_root(
                root,
                args.stages,
                args.top_percent,
                args.top_k,
                args.examples,
            )
        )
    aggregate = aggregate_rows(rows)
    write_csv(args.output_dir / "sequence_stage_summary.csv", rows)
    write_csv(args.output_dir / "dataset_stage_summary.csv", aggregate)
    payload = {
        "config": {
            "matrix_roots": [str(root) for root in args.matrix_roots],
            "stages": args.stages,
            "top_percent": args.top_percent,
            "top_k": args.top_k,
            "examples": args.examples,
            "temporal_gap": "rank distance after sorting the selected 300 frames by original frame timestamp/index",
            "input_gap": "absolute distance on the model input/matrix axis",
        },
        "sequence_stage": rows,
        "dataset_stage": aggregate,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_temporal_nonlocal(args.output_dir, aggregate)
    print(json.dumps({"output_dir": str(args.output_dir), "dataset_stage": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
