#!/usr/bin/env python3
"""Plot static and dynamic camera/depth quality versus token merge ratio."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["results"]


def write_report(output_dir: Path, groups: dict[str, list[dict[str, object]]]) -> None:
    lines = [
        "# VGGT-Omega Merge-Ratio Evaluation",
        "",
        "## Protocol",
        "",
        "- Dataset: Bonn RGB-D.",
        "- Static: `rgbd_bonn_static`.",
        "- Dynamic: `rgbd_bonn_crowd` and `rgbd_bonn_person_tracking`.",
        "- Frames: 4, 8, 64, 128, and 256 per sequence; seed 42.",
        "- Merge ratios: 0%, 10%, 25%, 50%, and 75%; global attention only.",
        "- Camera: pairwise pose AUC@30 (higher is better).",
        "- Depth: median-aligned AbsRel (lower is better).",
        "",
    ]
    for scene_type, rows in groups.items():
        lines.extend([
            f"## {scene_type}",
            "",
            "| Frames | Baseline AUC@30 | Best AUC@30 (ratio) | Baseline AbsRel | Best AbsRel (ratio) |",
            "|---:|---:|---:|---:|---:|",
        ])
        frame_counts = sorted({int(row["frame_count"]) for row in rows})
        for frame_count in frame_counts:
            curve = [
                row for row in rows
                if int(row["frame_count"]) == frame_count and row["status"] == "success"
            ]
            baseline = next(row for row in curve if float(row["merge_percent"]) == 0.0)
            best_camera = max(curve, key=lambda row: float(row["auc_30_percent"]))
            best_depth = min(curve, key=lambda row: float(row["abs_rel"]))
            lines.append(
                f"| {frame_count} | {baseline['auc_30_percent']:.3f} | "
                f"{best_camera['auc_30_percent']:.3f} ({best_camera['merge_percent']:.0f}%) | "
                f"{baseline['abs_rel']:.5f} | "
                f"{best_depth['abs_rel']:.5f} ({best_depth['merge_percent']:.0f}%) |"
            )
        lines.extend([
            "",
            f"![{scene_type} curves]({scene_type.lower()}_camera_depth_vs_merge_ratio.png)",
            "",
        ])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "Static": load_results(args.static),
        "Dynamic": load_results(args.dynamic),
    }
    fields = (
        "auc_3_percent",
        "auc_30_percent",
        "abs_rel",
        "delta_1_25_percent",
        "latency_ms_mean",
        "peak_allocated_gib_max",
    )
    with (args.output_dir / "merge_ratio_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scene_type", "frame_count", "merge_percent", "status", *fields),
        )
        writer.writeheader()
        for scene_type, rows in groups.items():
            for row in rows:
                writer.writerow({
                    "scene_type": scene_type.lower(),
                    "frame_count": row["frame_count"],
                    "merge_percent": row["merge_percent"],
                    "status": row["status"],
                    **{field: row.get(field, "") for field in fields},
                })

    panels = (
        ("auc_3_percent", "Camera AUC@3 (%)", True),
        ("auc_30_percent", "Camera AUC@30 (%)", True),
        ("abs_rel", "Depth AbsRel", False),
        ("delta_1_25_percent", "Depth delta < 1.25 (%)", True),
    )
    for scene_type, rows in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        frame_counts = sorted({int(row["frame_count"]) for row in rows})
        colors = plt.cm.tab10.colors
        for axis, (field, title, _) in zip(axes.flat, panels):
            for index, frame_count in enumerate(frame_counts):
                curve = [
                    row for row in rows
                    if int(row["frame_count"]) == frame_count and row["status"] == "success"
                ]
                curve.sort(key=lambda row: float(row["merge_percent"]))
                if not curve:
                    continue
                axis.plot(
                    [row["merge_percent"] for row in curve],
                    [row[field] for row in curve],
                    marker="o",
                    linewidth=2,
                    label=f"{frame_count} frames",
                    color=colors[index % len(colors)],
                )
            axis.set_title(title)
            axis.set_xlabel("Merged tokens (%)")
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)
        stem = f"{scene_type.lower()}_camera_depth_vs_merge_ratio"
        fig.savefig(args.output_dir / f"{stem}.png", dpi=180)
        fig.savefig(args.output_dir / f"{stem}.pdf")
        plt.close(fig)
    write_report(args.output_dir, groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
