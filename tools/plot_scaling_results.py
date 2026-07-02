#!/usr/bin/env python3
"""Plot timing, memory, and optional metrics from scaling_summary.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_series(rows: list[dict[str, str]], fields: list[tuple[str, str]], ylabel: str,
                output: Path) -> bool:
    sequences = sorted({row["sequence"] for row in rows})
    plotted = False
    plt.figure(figsize=(8, 5))
    for sequence in sequences:
        selected = sorted((row for row in rows if row["sequence"] == sequence),
                          key=lambda row: int(row["frame_count"]))
        for field, label in fields:
            points = [(int(row["frame_count"]), float(row[field])) for row in selected
                      if field in row and row[field] and math.isfinite(float(row[field]))]
            if points:
                x, y = zip(*points)
                suffix = f" ({label})" if len(fields) > 1 else ""
                plt.plot(x, y, marker="o", label=sequence + suffix)
                plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel("Number of input frames")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return True


def main() -> int:
    args = parse_args()
    rows = read_rows(args.summary_csv)
    if not rows:
        raise ValueError(f"No rows in {args.summary_csv}")
    output_dir = args.output_dir or args.summary_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_series(rows, [("mean_time_sec", "inference")], "Inference time (s)",
                output_dir / "time_vs_frames.png")
    plot_series(rows, [
        ("mean_peak_memory_allocated_gb", "allocated"),
        ("mean_peak_memory_reserved_gb", "reserved"),
    ], "Peak CUDA memory (GB)", output_dir / "memory_vs_frames.png")
    plot_series(rows, [
        ("mean_depth_absrel", "Depth AbsRel"),
        ("mean_camera_ate", "Camera ATE (m)"),
    ], "Metric value (lower is better)", output_dir / "metric_vs_frames.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
