#!/usr/bin/env python3
"""Summarize and plot available multi-frame FastVGGT merge-rate results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("outputs/fastvggt_merge_rates_multiframe")
DATASET_NAMES = {
    "7scenes": "7Scenes",
    "tum_dynamics": "TUM-Dynamics",
}
DEFAULT_RATIOS = (0, 10, 30, 50, 70, 90)
METRICS = (
    ("auc_3_percent", "AUC@3 (%)"),
    ("auc_30_percent", "AUC@30 (%)"),
    ("delta_1_25_percent", "delta<1.25 (%)"),
    ("abs_rel", "AbsRel"),
    ("model_latency_ms_mean", "Latency (ms)"),
    ("speedup_vs_no_merge", "Speedup vs 0%"),
)


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame_dir in sorted(ROOT.glob("*frames"), key=lambda p: int(p.name.removesuffix("frames"))):
        frame_count = int(frame_dir.name.removesuffix("frames"))
        for dataset_dir in sorted(frame_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            for ratio_dir in sorted(dataset_dir.glob("ratio_*")):
                if not ratio_dir.is_dir():
                    continue
                try:
                    ratio = int(ratio_dir.name.removeprefix("ratio_"))
                except ValueError:
                    continue
                metrics_path = ratio_dir / "metrics.json"
                if not metrics_path.is_file():
                    continue
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
                overall = result["overall"]
                rows.append(
                    {
                        "dataset": dataset,
                        "frame_count": frame_count,
                        "merge_ratio": ratio,
                        "auc_3_percent": float(overall["auc_3_percent"]),
                        "auc_30_percent": float(overall["auc_30_percent"]),
                        "delta_1_25_percent": float(overall["delta_1_25_percent"]),
                        "abs_rel": float(overall["abs_rel"]),
                        "model_latency_ms_mean": float(overall["model_latency_ms_mean"]),
                        "peak_allocated_gib_max": float(overall["peak_allocated_gib_max"]),
                    }
                )
    baselines = {
        (row["dataset"], row["frame_count"]): row["model_latency_ms_mean"]
        for row in rows
        if row["merge_ratio"] == 0
    }
    for row in rows:
        baseline = baselines.get((row["dataset"], row["frame_count"]))
        row["speedup_vs_no_merge"] = (
            float(baseline) / float(row["model_latency_ms_mean"]) if baseline else float("nan")
        )
    return rows


def load_sequence_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame_dir in sorted(ROOT.glob("*frames"), key=lambda p: int(p.name.removesuffix("frames"))):
        frame_count = int(frame_dir.name.removesuffix("frames"))
        for dataset_dir in sorted(frame_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            for ratio_dir in sorted(dataset_dir.glob("ratio_*")):
                if not ratio_dir.is_dir():
                    continue
                try:
                    ratio = int(ratio_dir.name.removeprefix("ratio_"))
                except ValueError:
                    continue
                metrics_path = ratio_dir / "metrics.json"
                if not metrics_path.is_file():
                    continue
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
                for sequence in result["per_sequence"]:
                    rows.append(
                        {
                            "dataset": dataset,
                            "frame_count": frame_count,
                            "merge_ratio": ratio,
                            "sequence": str(sequence["sequence"]),
                            "auc_3_percent": float(sequence["auc_3_percent"]),
                            "auc_30_percent": float(sequence["auc_30_percent"]),
                            "delta_1_25_percent": float(sequence["delta_1_25_percent"]),
                            "abs_rel": float(sequence["abs_rel"]),
                            "model_latency_ms": float(sequence["model_latency_ms"]),
                            "peak_allocated_gib": float(sequence["peak_allocated_gib"]),
                        }
                    )
    baselines = {
        (row["dataset"], row["frame_count"], row["sequence"]): row["model_latency_ms"]
        for row in rows
        if row["merge_ratio"] == 0
    }
    for row in rows:
        baseline = baselines.get((row["dataset"], row["frame_count"], row["sequence"]))
        row["speedup_vs_no_merge"] = (
            float(baseline) / float(row["model_latency_ms"]) if baseline else float("nan")
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    output_path = ROOT / "multiframe_aggregate_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_sequence_csv(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    output_path = ROOT / "multiframe_per_sequence_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def complete_frame_groups(rows: list[dict[str, object]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        frame_counts = sorted({int(row["frame_count"]) for row in rows if row["dataset"] == dataset})
        complete = []
        for frame_count in frame_counts:
            available = {
                int(row["merge_ratio"])
                for row in rows
                if row["dataset"] == dataset and int(row["frame_count"]) == frame_count
            }
            if set(DEFAULT_RATIOS).issubset(available):
                complete.append(frame_count)
        if complete:
            groups[dataset] = complete
    return groups


def plot_dataset_curves(rows: list[dict[str, object]]) -> list[Path]:
    output_dir = ROOT / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    complete = complete_frame_groups(rows)

    for dataset, frame_counts in complete.items():
        figure, axes = plt.subplots(2, 3, figsize=(16, 9))
        for axis, (metric, label) in zip(axes.ravel(), METRICS):
            for frame_count in frame_counts:
                curve = sorted(
                    [
                        row
                        for row in rows
                        if row["dataset"] == dataset and int(row["frame_count"]) == frame_count
                    ],
                    key=lambda row: int(row["merge_ratio"]),
                )
                axis.plot(
                    [int(row["merge_ratio"]) for row in curve],
                    [float(row[metric]) for row in curve],
                    marker="o",
                    linewidth=1.8,
                    label=f"{frame_count} frames",
                )
            axis.set_xlabel("Merge ratio (%)")
            axis.set_ylabel(label)
            axis.set_xticks(sorted({int(row["merge_ratio"]) for row in rows if row["dataset"] == dataset}))
            axis.grid(alpha=0.25)
            if metric == "speedup_vs_no_merge":
                axis.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            axis.legend(frameon=False)
        display = DATASET_NAMES.get(dataset, dataset)
        figure.suptitle(f"{display}: merge-ratio curves by frame count", fontsize=15)
        figure.tight_layout()
        output_path = output_dir / f"{dataset}_multiframe_merge_curves.png"
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        generated.append(output_path)

    return generated


def plot_latency_scaling(rows: list[dict[str, object]]) -> list[Path]:
    output_dir = ROOT / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    complete = complete_frame_groups(rows)

    for dataset, frame_counts in complete.items():
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ratio in sorted({int(row["merge_ratio"]) for row in rows if row["dataset"] == dataset}):
            curve = sorted(
                [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and int(row["frame_count"]) in frame_counts
                    and int(row["merge_ratio"]) == ratio
                ],
                key=lambda row: int(row["frame_count"]),
            )
            if len(curve) < 2:
                continue
            axes[0].plot(
                [int(row["frame_count"]) for row in curve],
                [float(row["model_latency_ms_mean"]) for row in curve],
                marker="o",
                label=f"{ratio}%",
            )
            axes[1].plot(
                [int(row["frame_count"]) for row in curve],
                [float(row["speedup_vs_no_merge"]) for row in curve],
                marker="o",
                label=f"{ratio}%",
            )
        axes[0].set_title("Latency scaling")
        axes[0].set_xlabel("Frames")
        axes[0].set_ylabel("Latency (ms)")
        axes[1].set_title("Speedup scaling")
        axes[1].set_xlabel("Frames")
        axes[1].set_ylabel("Speedup vs 0%")
        axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, ncol=2)
        display = DATASET_NAMES.get(dataset, dataset)
        figure.suptitle(f"{display}: scaling over completed frame counts", fontsize=14)
        figure.tight_layout()
        output_path = output_dir / f"{dataset}_latency_speedup_scaling.png"
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        generated.append(output_path)

    return generated


def complete_sequence_groups(rows: list[dict[str, object]]) -> dict[tuple[str, int], list[str]]:
    groups: dict[tuple[str, int], list[str]] = {}
    expected = set(DEFAULT_RATIOS)
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        frame_counts = sorted({int(row["frame_count"]) for row in rows if row["dataset"] == dataset})
        for frame_count in frame_counts:
            sequences = sorted(
                {
                    str(row["sequence"])
                    for row in rows
                    if row["dataset"] == dataset and int(row["frame_count"]) == frame_count
                }
            )
            complete_sequences = []
            for sequence in sequences:
                available = {
                    int(row["merge_ratio"])
                    for row in rows
                    if row["dataset"] == dataset
                    and int(row["frame_count"]) == frame_count
                    and row["sequence"] == sequence
                }
                if expected.issubset(available):
                    complete_sequences.append(sequence)
            if complete_sequences:
                groups[(dataset, frame_count)] = complete_sequences
    return groups


def plot_per_sequence_overlay(rows: list[dict[str, object]]) -> list[Path]:
    output_dir = ROOT / "plots" / "per_sequence"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    groups = complete_sequence_groups(rows)
    plot_specs = (
        ("auc_30_percent", "AUC@30 (%)"),
        ("abs_rel", "AbsRel"),
        ("model_latency_ms", "Latency (ms)"),
        ("speedup_vs_no_merge", "Speedup vs 0%"),
    )

    for (dataset, frame_count), sequences in groups.items():
        figure, axes = plt.subplots(2, 2, figsize=(15, 9))
        for axis, (metric, label) in zip(axes.ravel(), plot_specs):
            for sequence in sequences:
                curve = sorted(
                    [
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and int(row["frame_count"]) == frame_count
                        and row["sequence"] == sequence
                    ],
                    key=lambda row: int(row["merge_ratio"]),
                )
                axis.plot(
                    [int(row["merge_ratio"]) for row in curve],
                    [float(row[metric]) for row in curve],
                    marker="o",
                    linewidth=1.0,
                    markersize=2.8,
                    label=sequence,
                )
            axis.set_xlabel("Merge ratio (%)")
            axis.set_ylabel(label)
            axis.set_xticks(
                sorted(
                    {
                        int(row["merge_ratio"])
                        for row in rows
                        if row["dataset"] == dataset and int(row["frame_count"]) == frame_count
                    }
                )
            )
            axis.grid(alpha=0.25)
            if metric == "speedup_vs_no_merge":
                axis.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.005, 0.5),
            fontsize=7 if len(sequences) > 10 else 9,
            frameon=False,
        )
        display = DATASET_NAMES.get(dataset, dataset)
        figure.suptitle(f"{display} {frame_count} frames: per-sequence curves", fontsize=15)
        figure.tight_layout(rect=(0, 0, 0.84, 0.96))
        output_path = output_dir / f"{dataset}_{frame_count}frames_per_sequence_overlay.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        generated.append(output_path)
    return generated


def write_sequence_variation_report(rows: list[dict[str, object]]) -> Path | None:
    groups = complete_sequence_groups(rows)
    if not groups:
        return None
    report_path = ROOT / "per_sequence_variation.md"
    lines = ["# Per-sequence merge-rate variation", ""]
    for (dataset, frame_count), sequences in groups.items():
        display = DATASET_NAMES.get(dataset, dataset)
        lines.extend(
            [
                f"## {display}, {frame_count} frames",
                "",
                "| Sequence | AUC@30 range | AUC@30 drop 0→90 | AbsRel rel range | AbsRel change 0→90 | Speedup 90% |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        items = []
        for sequence in sequences:
            curve = sorted(
                [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and int(row["frame_count"]) == frame_count
                    and row["sequence"] == sequence
                ],
                key=lambda row: int(row["merge_ratio"]),
            )
            by_ratio = {int(row["merge_ratio"]): row for row in curve}
            auc_values = [float(row["auc_30_percent"]) for row in curve]
            abs_values = [float(row["abs_rel"]) for row in curve]
            baseline_abs = float(by_ratio[0]["abs_rel"])
            auc_range = max(auc_values) - min(auc_values)
            auc_drop = float(by_ratio[0]["auc_30_percent"]) - float(by_ratio[90]["auc_30_percent"])
            abs_rel_range = (max(abs_values) - min(abs_values)) / (baseline_abs + 1e-12) * 100
            abs_change = (float(by_ratio[90]["abs_rel"]) - baseline_abs) / (baseline_abs + 1e-12) * 100
            speedup_90 = float(by_ratio[90]["speedup_vs_no_merge"])
            items.append((auc_range, sequence, auc_drop, abs_rel_range, abs_change, speedup_90))
        for auc_range, sequence, auc_drop, abs_rel_range, abs_change, speedup_90 in sorted(
            items, reverse=True
        ):
            lines.append(
                f"| `{sequence}` | {auc_range:.2f} | {auc_drop:.2f} | "
                f"{abs_rel_range:.2f}% | {abs_change:+.2f}% | {speedup_90:.2f}x |"
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    rows = load_rows()
    sequence_rows = load_sequence_rows()
    write_csv(rows)
    write_sequence_csv(sequence_rows)
    generated = (
        plot_dataset_curves(rows)
        + plot_latency_scaling(rows)
        + plot_per_sequence_overlay(sequence_rows)
    )
    variation_report = write_sequence_variation_report(sequence_rows)
    index = ["# Multi-frame FastVGGT merge-rate plots", ""]
    if not generated:
        index.append("No complete frame-count groups are available yet.")
    for path in generated:
        index.extend([f"## {path.stem}", "", f"![{path.stem}](plots/{path.name})", ""])
    (ROOT / "PLOTS.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    if variation_report:
        print(variation_report)
    print("\n".join(str(path) for path in generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
