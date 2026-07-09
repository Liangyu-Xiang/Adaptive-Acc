#!/usr/bin/env python3
"""Summarize full-dataset FastVGGT merge-ratio evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("outputs/fastvggt_merge_rates_50")
DATASETS = {"7scenes": "7 Scenes", "tum_dynamics": "TUM-Dynamics"}
TAGS = ("00", "10", "30", "50", "70", "90")
METRICS = (
    ("auc_3_percent", "AUC@3 (%)", True),
    ("auc_30_percent", "AUC@30 (%)", True),
    ("delta_1_25_percent", "delta<1.25 (%)", True),
    ("abs_rel", "AbsRel", False),
    ("model_latency_ms", "Latency (ms)", False),
)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def plot_per_sequence_curves(loaded):
    output_dir = ROOT / "per_sequence_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    index = ["# Per-sequence merge-ratio curves", ""]
    ratios = [float(tag) for tag in TAGS]
    plot_specs = (
        ("auc_3_percent", "AUC@3 (%)"),
        ("auc_30_percent", "AUC@30 (%)"),
        ("delta_1_25_percent", "delta<1.25 (%)"),
        ("abs_rel", "AbsRel"),
        ("model_latency_ms", "Latency (ms)"),
        ("speedup", "Speedup vs 0%"),
    )
    for dataset, display in DATASETS.items():
        index.extend([f"## {display}", ""])
        sequence_names = [row["sequence"] for row in loaded[dataset]["00"]["per_sequence"]]
        by_tag = {
            tag: {row["sequence"]: row for row in loaded[dataset][tag]["per_sequence"]}
            for tag in TAGS
        }
        for sequence in sequence_names:
            rows = [by_tag[tag][sequence] for tag in TAGS]
            baseline_latency = float(rows[0]["model_latency_ms"])
            figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
            for axis, (metric, label) in zip(axes.ravel(), plot_specs):
                values = (
                    [baseline_latency / float(row["model_latency_ms"]) for row in rows]
                    if metric == "speedup"
                    else [float(row[metric]) for row in rows]
                )
                axis.plot(ratios, values, marker="o", linewidth=1.8)
                axis.set_xlabel("Merge ratio (%)")
                axis.set_ylabel(label)
                axis.set_xticks(ratios)
                axis.grid(alpha=0.25)
                if metric == "speedup":
                    axis.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            figure.suptitle(f"{display}: {sequence}", fontsize=14)
            figure.tight_layout()
            filename = f"{dataset}__{sequence.replace('/', '__')}.png"
            figure.savefig(output_dir / filename, dpi=180)
            plt.close(figure)
            index.extend([f"### {sequence}", "", f"![{sequence}]({filename})", ""])
    (output_dir / "INDEX.md").write_text("\n".join(index) + "\n")


def plot_dataset_sequence_curves(loaded):
    """Plot all per-sequence merge-ratio curves in one figure per dataset."""
    output_dir = ROOT / "dataset_sequence_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = [float(tag) for tag in TAGS]
    plot_specs = (
        ("auc_3_percent", "AUC@3 (%)"),
        ("auc_30_percent", "AUC@30 (%)"),
        ("delta_1_25_percent", "delta<1.25 (%)"),
        ("abs_rel", "AbsRel"),
        ("model_latency_ms", "Latency (ms)"),
        ("speedup", "Speedup vs 0%"),
    )
    index = ["# Dataset-level per-sequence merge-ratio curves", ""]

    for dataset, display in DATASETS.items():
        sequence_names = [row["sequence"] for row in loaded[dataset]["00"]["per_sequence"]]
        by_tag = {
            tag: {row["sequence"]: row for row in loaded[dataset][tag]["per_sequence"]}
            for tag in TAGS
        }

        figure, axes = plt.subplots(2, 3, figsize=(20, 10.5))
        for axis, (metric, label) in zip(axes.ravel(), plot_specs):
            for sequence in sequence_names:
                rows = [by_tag[tag][sequence] for tag in TAGS]
                baseline_latency = float(rows[0]["model_latency_ms"])
                values = (
                    [baseline_latency / float(row["model_latency_ms"]) for row in rows]
                    if metric == "speedup"
                    else [float(row[metric]) for row in rows]
                )
                axis.plot(ratios, values, marker="o", linewidth=1.2, markersize=3, label=sequence)
            axis.set_xlabel("Merge ratio (%)")
            axis.set_ylabel(label)
            axis.set_xticks(ratios)
            axis.grid(alpha=0.25)
            if metric == "speedup":
                axis.axhline(1.0, color="gray", linestyle="--", linewidth=1)

        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.005, 0.5),
            fontsize=7 if len(sequence_names) > 10 else 9,
            frameon=False,
        )
        figure.suptitle(f"{display}: all sequences", fontsize=15)
        figure.tight_layout(rect=(0, 0, 0.86, 0.96))
        filename = f"{dataset}_all_sequences.png"
        figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)
        index.extend([f"## {display}", "", f"![{display}]({filename})", ""])

    (output_dir / "INDEX.md").write_text("\n".join(index) + "\n")


def plot_dataset_average_curves(aggregate_rows):
    """Plot sequence-averaged merge-ratio curves as one figure per dataset."""
    output_dir = ROOT / "dataset_average_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = (
        ("auc_3_percent", "AUC@3 (%)"),
        ("auc_30_percent", "AUC@30 (%)"),
        ("delta_1_25_percent", "delta<1.25 (%)"),
        ("abs_rel", "AbsRel"),
        ("model_latency_ms_mean", "Latency (ms)"),
        ("speedup_vs_no_merge", "Speedup vs 0%"),
    )
    index = ["# Dataset-level sequence-averaged merge-ratio curves", ""]

    for dataset, display in DATASETS.items():
        rows = sorted(
            [row for row in aggregate_rows if row["dataset"] == dataset],
            key=lambda row: row["merge_ratio"],
        )
        figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for axis, (metric, label) in zip(axes.ravel(), plot_specs):
            axis.plot(
                [100 * row["merge_ratio"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                linewidth=1.8,
            )
            axis.set_xlabel("Merge ratio (%)")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            if metric == "speedup_vs_no_merge":
                axis.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        figure.suptitle(f"{display}: sequence-averaged results", fontsize=14)
        figure.tight_layout()
        filename = f"{dataset}_average.png"
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)
        index.extend([f"## {display}", "", f"![{display}]({filename})", ""])

    (output_dir / "INDEX.md").write_text("\n".join(index) + "\n")


def main() -> int:
    aggregate_rows = []
    sequence_rows = []
    loaded = {}
    for dataset in DATASETS:
        loaded[dataset] = {}
        for tag in TAGS:
            path = ROOT / dataset / f"ratio_{tag}" / "metrics.json"
            result = json.loads(path.read_text())
            loaded[dataset][tag] = result
            overall = result["overall"]
            latency = overall["model_latency_ms_mean"]
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "merge_ratio": float(tag) / 100,
                    "auc_3_percent": overall["auc_3_percent"],
                    "auc_30_percent": overall["auc_30_percent"],
                    "delta_1_25_percent": overall["delta_1_25_percent"],
                    "abs_rel": overall["abs_rel"],
                    "model_latency_ms_mean": latency,
                    "peak_allocated_gib_max": overall["peak_allocated_gib_max"],
                }
            )
            for row in result["per_sequence"]:
                sequence_rows.append(
                    {
                        "dataset": dataset,
                        "merge_ratio": float(tag) / 100,
                        "sequence": row["sequence"],
                        "auc_3_percent": row["auc_3_percent"],
                        "auc_30_percent": row["auc_30_percent"],
                        "delta_1_25_percent": row["delta_1_25_percent"],
                        "abs_rel": row["abs_rel"],
                        "model_latency_ms": row["model_latency_ms"],
                        "peak_allocated_gib": row["peak_allocated_gib"],
                    }
                )

    for dataset in DATASETS:
        baseline = next(
            row["model_latency_ms_mean"]
            for row in aggregate_rows
            if row["dataset"] == dataset and row["merge_ratio"] == 0
        )
        for row in aggregate_rows:
            if row["dataset"] == dataset:
                row["speedup_vs_no_merge"] = baseline / row["model_latency_ms_mean"]

    for filename, rows in (("aggregate_summary.csv", aggregate_rows), ("per_sequence_summary.csv", sequence_rows)):
        with (ROOT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    (ROOT / "summary.json").write_text(
        json.dumps({"aggregate": aggregate_rows, "per_sequence": sequence_rows}, indent=2) + "\n"
    )

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    plot_specs = (
        ("auc_3_percent", "AUC@3 (%)"),
        ("auc_30_percent", "AUC@30 (%)"),
        ("delta_1_25_percent", "delta<1.25 (%)"),
        ("abs_rel", "AbsRel"),
        ("model_latency_ms_mean", "Latency (ms)"),
        ("speedup_vs_no_merge", "Speedup"),
    )
    for axis, (metric, label) in zip(axes.ravel(), plot_specs):
        for dataset, display in DATASETS.items():
            rows = [row for row in aggregate_rows if row["dataset"] == dataset]
            axis.plot(
                [100 * row["merge_ratio"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                label=display,
            )
        axis.set_xlabel("Merge ratio (%)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(ROOT / "merge_ratio_comparison.png", dpi=180)
    plt.close(figure)
    plot_per_sequence_curves(loaded)
    plot_dataset_sequence_curves(loaded)
    plot_dataset_average_curves(aggregate_rows)

    report = [
        "# FastVGGT token merging on 50-frame full datasets",
        "",
        "Ordinary FastVGGT bipartite token merging; mixed global/register architecture; seed 42; "
        "512-pixel preprocessing; one warmup and three CUDA-event timings per sequence. "
        "7 Scenes was timed on GPU5 and TUM-Dynamics on GPU4. Each dataset uses identical sampled "
        "frames across all ratios.",
        "",
        "## Overall",
        "",
    ]
    overall_table = []
    for row in aggregate_rows:
        overall_table.append(
            [
                DATASETS[row["dataset"]],
                f'{100 * row["merge_ratio"]:.0f}%',
                f'{row["auc_3_percent"]:.2f}',
                f'{row["auc_30_percent"]:.2f}',
                f'{row["delta_1_25_percent"]:.2f}',
                f'{row["abs_rel"]:.4f}',
                f'{row["model_latency_ms_mean"]:.1f}',
                f'{row["speedup_vs_no_merge"]:.3f}x',
                f'{row["peak_allocated_gib_max"]:.2f}',
            ]
        )
    report.append(
        markdown_table(
            ["Dataset", "Ratio", "AUC@3", "AUC@30", "delta<1.25", "AbsRel", "ms", "Speedup", "Peak GiB"],
            overall_table,
        )
    )

    for dataset, display in DATASETS.items():
        report.extend(["", f"## {display}: all sequences", ""])
        sequence_names = [row["sequence"] for row in loaded[dataset]["00"]["per_sequence"]]
        by_tag = {
            tag: {row["sequence"]: row for row in loaded[dataset][tag]["per_sequence"]}
            for tag in TAGS
        }
        for metric, label, _ in METRICS:
            rows = []
            for sequence in sequence_names:
                values = []
                for tag in TAGS:
                    value = by_tag[tag][sequence][metric]
                    values.append(f"{value:.4f}" if metric == "abs_rel" else f"{value:.2f}")
                rows.append([sequence, *values])
            report.extend(
                [
                    "",
                    f"### {label}",
                    "",
                    markdown_table(["Sequence", *[f"{tag}%" for tag in TAGS]], rows),
                ]
            )
    (ROOT / "REPORT.md").write_text("\n".join(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
