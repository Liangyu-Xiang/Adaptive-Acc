#!/usr/bin/env python3
"""Aggregate metrics and pose errors from sharded dataset evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_7scenes_paper import official_auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-metrics", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker_metrics = [json.loads(path.read_text(encoding="utf-8")) for path in args.worker_metrics]
    rows = [row for result in worker_metrics for row in result["per_sequence"]]
    rows.sort(key=lambda row: row["sequence"])
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for path in args.worker_metrics:
        pose_path = path.parent / "pose_errors.npz"
        with np.load(pose_path) as pose:
            rotations.append(pose["rotation_error_deg"])
            translations.append(pose["translation_error_deg"])
    rotation_errors = np.concatenate(rotations)
    translation_errors = np.concatenate(translations)
    total_valid = sum(int(row["valid_depth_pixels"]) for row in rows)
    total_abs_rel = sum(float(row["abs_rel"]) * int(row["valid_depth_pixels"]) for row in rows)
    total_delta = sum(
        float(row["delta_1_25_percent"]) / 100.0 * int(row["valid_depth_pixels"])
        for row in rows
    )
    latency_values = [row["model_latency_ms"] for row in rows if row.get("model_latency_ms") is not None]
    overall = {
        "auc_3_percent": 100.0 * official_auc(rotation_errors, translation_errors, 3),
        "auc_30_percent": 100.0 * official_auc(rotation_errors, translation_errors, 30),
        "delta_1_25_percent": 100.0 * total_delta / max(total_valid, 1),
        "abs_rel": total_abs_rel / max(total_valid, 1),
        "valid_depth_pixels": total_valid,
        "model_latency_ms_mean": float(np.mean(latency_values)) if latency_values else None,
        "peak_allocated_gib_max": float(max(row["peak_allocated_gib"] for row in rows)),
    }
    result = {
        "protocol": worker_metrics[0]["protocol"] | {"num_sequences": len(rows)},
        "paper_targets_1b": worker_metrics[0].get("paper_targets_1b", {}),
        "overall": overall,
        "per_sequence": rows,
        "worker_metrics": [str(path) for path in args.worker_metrics],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "pose_errors.npz",
        rotation_error_deg=rotation_errors,
        translation_error_deg=translation_errors,
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
