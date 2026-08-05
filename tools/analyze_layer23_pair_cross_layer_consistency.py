#!/usr/bin/env python3
"""Check whether layer-23 high-similarity frame pairs stay similar earlier."""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_QUERY_PERCENTS = (1.0, 0.1)
DEFAULT_OTHER_STAGES = ("input_rgb", "layer_02", "layer_06", "layer_10", "layer_16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-roots", nargs="+", type=Path, default=list(DEFAULT_MATRIX_ROOTS))
    parser.add_argument("--query-stage", default="layer_23")
    parser.add_argument("--other-stages", nargs="+", default=list(DEFAULT_OTHER_STAGES))
    parser.add_argument("--query-top-percent", nargs="+", type=float, default=list(DEFAULT_QUERY_PERCENTS))
    parser.add_argument("--query-threshold", type=float, default=0.76)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/layer23_pair_cross_layer_consistency"))
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


def upper_values(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return rows, cols, matrix[rows, cols]


def top_percent_indices(values: np.ndarray, percent: float) -> np.ndarray:
    count = max(1, int(np.ceil(len(values) * percent / 100.0)))
    indices = np.argpartition(-values, count - 1)[:count]
    return indices[np.argsort(-values[indices])]


def percentile_ranks(all_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(all_values)
    return np.searchsorted(sorted_values, query_values, side="right") / len(sorted_values)


def spearman_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    order_a = np.argsort(a, kind="mergesort")
    order_b = np.argsort(b, kind="mergesort")
    ranks_a = np.empty(len(a), dtype=np.float64)
    ranks_b = np.empty(len(b), dtype=np.float64)
    ranks_a[order_a] = np.arange(len(a), dtype=np.float64)
    ranks_b[order_b] = np.arange(len(b), dtype=np.float64)
    ranks_a -= ranks_a.mean()
    ranks_b -= ranks_b.mean()
    denominator = np.linalg.norm(ranks_a) * np.linalg.norm(ranks_b)
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(ranks_a, ranks_b) / denominator)


def analyze_query(
    *,
    dataset: str,
    sequence: str,
    matrices: dict[str, np.ndarray],
    query_stage: str,
    other_stages: list[str],
    query_name: str,
    query_indices: np.ndarray,
) -> list[dict[str, object]]:
    _, _, query_values = upper_values(matrices[query_stage])
    query_set = set(int(index) for index in query_indices)
    rows = []
    for stage in other_stages:
        _, _, stage_values = upper_values(matrices[stage])
        ranks = percentile_ranks(stage_values, stage_values[query_indices])
        top1 = set(int(index) for index in top_percent_indices(stage_values, 1.0))
        top5 = set(int(index) for index in top_percent_indices(stage_values, 5.0))
        rows.append(
            {
                "dataset": dataset,
                "sequence": sequence,
                "query": query_name,
                "query_stage": query_stage,
                "stage": stage,
                "query_pairs": int(len(query_indices)),
                "query_stage_similarity_min": float(query_values[query_indices].min()),
                "query_stage_similarity_mean": float(query_values[query_indices].mean()),
                "query_stage_similarity_max": float(query_values[query_indices].max()),
                "stage_query_similarity_mean": float(stage_values[query_indices].mean()),
                "stage_all_similarity_mean": float(stage_values.mean()),
                "stage_query_minus_all_mean": float(stage_values[query_indices].mean() - stage_values.mean()),
                "stage_query_percentile_mean": float(ranks.mean()),
                "stage_query_percentile_median": float(np.median(ranks)),
                "stage_query_percentile_p10": float(np.percentile(ranks, 10)),
                "query_overlap_stage_top1_fraction": float(len(query_set & top1) / len(query_set)),
                "query_overlap_stage_top5_fraction": float(len(query_set & top5) / len(query_set)),
                "query_raw_gt_0p76_fraction": float(np.mean(stage_values[query_indices] > 0.76)),
                "query_raw_gt_0p90_fraction": float(np.mean(stage_values[query_indices] > 0.90)),
                "spearman_stage_vs_query_stage_all_pairs": spearman_rank_correlation(query_values, stage_values),
            }
        )
    return rows


def analyze_root(
    root: Path,
    query_stage: str,
    other_stages: list[str],
    query_percents: list[float],
    query_threshold: float,
) -> list[dict[str, object]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    dataset, sequence = infer_dataset_sequence(metadata["image_paths"])
    loaded = np.load(root / "frame_similarity_matrices.npz")
    matrices = {stage: loaded[stage] for stage in [query_stage, *other_stages]}
    _, _, query_values = upper_values(matrices[query_stage])

    rows: list[dict[str, object]] = []
    for percent in query_percents:
        rows.extend(
            analyze_query(
                dataset=dataset,
                sequence=sequence,
                matrices=matrices,
                query_stage=query_stage,
                other_stages=other_stages,
                query_name=f"{query_stage}_top_{percent:g}pct",
                query_indices=top_percent_indices(query_values, percent),
            )
        )
    threshold_indices = np.flatnonzero(query_values > query_threshold)
    if len(threshold_indices):
        rows.extend(
            analyze_query(
                dataset=dataset,
                sequence=sequence,
                matrices=matrices,
                query_stage=query_stage,
                other_stages=other_stages,
                query_name=f"{query_stage}_gt_{str(query_threshold).replace('.', 'p')}",
                query_indices=threshold_indices,
            )
        )
    return rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["query"]), str(row["stage"]))].append(row)
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key != "query_pairs"
    ]
    output: list[dict[str, object]] = []
    for (dataset, query, stage), group in sorted(grouped.items()):
        aggregate: dict[str, object] = {
            "dataset": dataset,
            "query": query,
            "stage": stage,
            "num_sequences": len(group),
            "query_pairs_mean": float(np.mean([float(row["query_pairs"]) for row in group])),
        }
        for key in numeric_keys:
            aggregate[key] = float(np.mean([float(row[key]) for row in group]))
        output.append(aggregate)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_top1_overlap(output_dir: Path, aggregate: list[dict[str, object]]) -> None:
    rows = [row for row in aggregate if row["query"].endswith("top_1pct")]
    datasets = sorted({str(row["dataset"]) for row in rows})
    stages = list(DEFAULT_OTHER_STAGES)
    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        values = []
        for stage in stages:
            match = [row for row in subset if row["stage"] == stage]
            values.append(100.0 * float(match[0]["query_overlap_stage_top1_fraction"]) if match else 0.0)
        axis.bar(stages, values)
        axis.set_title(dataset)
        axis.set_ylim(0, 100)
        axis.set_ylabel("Layer-23 top-1% pairs also in stage top-1% (%)")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "layer23_top1_overlap_stage_top1.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for root in args.matrix_roots:
        rows.extend(
            analyze_root(
                root=root,
                query_stage=args.query_stage,
                other_stages=args.other_stages,
                query_percents=args.query_top_percent,
                query_threshold=args.query_threshold,
            )
        )
    aggregate = aggregate_rows(rows)
    write_csv(args.output_dir / "sequence_stage_summary.csv", rows)
    write_csv(args.output_dir / "dataset_stage_summary.csv", aggregate)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "matrix_roots": [str(root) for root in args.matrix_roots],
                    "query_stage": args.query_stage,
                    "other_stages": args.other_stages,
                    "query_top_percent": args.query_top_percent,
                    "query_threshold": args.query_threshold,
                    "percentile_semantics": "fraction of all non-diagonal pairs in this stage with similarity <= query-pair similarity",
                },
                "sequence_stage": rows,
                "dataset_stage": aggregate,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_top1_overlap(args.output_dir, aggregate)
    print(json.dumps({"output_dir": str(args.output_dir), "dataset_stage": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
