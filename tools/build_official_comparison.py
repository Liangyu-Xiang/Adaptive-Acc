#!/usr/bin/env python3
"""Build canonical three-method comparison artifacts from evaluator outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


METHODS = (
    ("vggt_omega", "VGGT-Omega", "baseline_dir"),
    (
        "vggt_omega_fastvggt",
        "VGGT-Omega + FastVGGT",
        "fastvggt_dir",
    ),
    ("proposed", "Proposed", "proposed_dir"),
)
METRIC_KEYS = (
    "auc_3_percent",
    "auc_30_percent",
    "abs_rel",
    "delta_1_25_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--fastvggt-dir", type=Path, required=True)
    parser.add_argument("--proposed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--proposed-config", type=Path, required=True)
    parser.add_argument("--gpu-model", default=None)
    parser.add_argument(
        "--method-gpus",
        action="append",
        default=[],
        metavar="METHOD_ID=GPU_IDS",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ("git", *args),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_method_gpus(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        method_id, separator, gpu_ids = value.partition("=")
        if not separator or not method_id or not gpu_ids:
            raise ValueError(
                "--method-gpus values must use METHOD_ID=GPU_IDS"
            )
        parsed[method_id] = gpu_ids
    return parsed


def method_config(
    method_id: str,
    protocol: dict[str, Any],
    proposed_config: Path,
) -> dict[str, Any]:
    if method_id == "vggt_omega":
        return {
            "attention_mode": protocol.get("attention_mode"),
            "merge_ratio": protocol.get("merge_ratio"),
        }
    if method_id == "vggt_omega_fastvggt":
        return {
            "merge_ratio": protocol.get("merge_ratio"),
            "merge_similarity_mode": protocol.get(
                "merge_similarity_mode"
            ),
            "merge_preserve_first_frame": protocol.get(
                "merge_preserve_first_frame"
            ),
            "merge_protected_ratio": protocol.get(
                "merge_protected_ratio"
            ),
            "merge_protection_mode": protocol.get(
                "merge_protection_mode"
            ),
        }
    return {
        "algorithm": "adaptive_pair_scope",
        "config_path": str(proposed_config),
        "progressive_attention": protocol.get(
            "progressive_attention_config"
        ),
    }


def render_markdown(dataset: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {dataset}: official 300-frame all-sequence comparison",
        "",
        "## Accuracy",
        "",
        "| Method | Camera AUC@3 ↑ | Camera AUC@30 ↑ | Depth AbsRel ↓ | Depth δ1.25 ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['display_name']} | "
            f"{metrics['auc_3_percent']:.2f} | "
            f"{metrics['auc_30_percent']:.2f} | "
            f"{metrics['abs_rel']:.4f} | "
            f"{metrics['delta_1_25_percent']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "| Method | Frames | Sequence Scope | Model Latency (ms) ↓ | Peak Allocated (GiB) ↓ | Speedup ↑ |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        efficiency = row["efficiency"]
        lines.append(
            f"| {row['display_name']} | 300 | All | "
            f"{efficiency['model_latency_ms_mean']:.2f} | "
            f"{efficiency['peak_allocated_gib_max']:.2f} | "
            f"{efficiency['speedup']:.2f}× |"
        )
    return "\n".join(lines) + "\n"


def render_latex(dataset: str, rows: list[dict[str, Any]]) -> str:
    body = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Camera and depth estimation on {dataset}.}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & AUC@3 $\uparrow$ & AUC@30 $\uparrow$ & AbsRel $\downarrow$ & $\delta_{1.25}$ $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        metrics = row["metrics"]
        body.append(
            f"{row['display_name']} & "
            f"{metrics['auc_3_percent']:.2f} & "
            f"{metrics['auc_30_percent']:.2f} & "
            f"{metrics['abs_rel']:.4f} & "
            f"{metrics['delta_1_25_percent']:.4f} \\\\"
        )
    body.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    return "\n".join(body) + "\n"


def main() -> int:
    args = parse_args()
    method_gpus = parse_method_gpus(args.method_gpus)
    loaded: list[tuple[str, str, Path, dict[str, Any]]] = []
    reference_sequences: list[str] | None = None
    reference_samples: dict[str, Any] | None = None
    for method_id, display_name, attr in METHODS:
        directory = getattr(args, attr)
        metrics = load_json(directory / "metrics.json")
        samples = load_json(directory / "sampled_frames.json")
        sequences = sorted(
            str(row["sequence"]) for row in metrics["per_sequence"]
        )
        if len(sequences) != len(set(sequences)):
            raise ValueError(f"{method_id} contains duplicate sequences")
        if reference_sequences is None:
            reference_sequences = sequences
            reference_samples = samples
        elif sequences != reference_sequences or samples != reference_samples:
            raise ValueError(
                f"{method_id} does not use the baseline sequence/sample set"
            )
        loaded.append((method_id, display_name, directory, metrics))

    baseline_latency = float(loaded[0][3]["overall"][
        "model_latency_ms_mean"
    ])
    rows: list[dict[str, Any]] = []
    for method_id, display_name, directory, metrics in loaded:
        overall = metrics["overall"]
        latency = float(overall["model_latency_ms_mean"])
        rows.append(
            {
                "method_id": method_id,
                "display_name": display_name,
                "status": "completed",
                "implementation_status": "success",
                "config": method_config(
                    method_id,
                    metrics["protocol"],
                    args.proposed_config,
                ),
                "metrics": {
                    key: float(overall[key]) for key in METRIC_KEYS
                },
                "efficiency": {
                    "model_latency_ms_mean": latency,
                    "peak_allocated_gib_max": float(
                        overall["peak_allocated_gib_max"]
                    ),
                    "speedup": baseline_latency / latency,
                },
                "physical_gpu_ids": method_gpus.get(method_id),
                "raw_metrics_path": str(directory / "metrics.json"),
            }
        )

    assert reference_sequences is not None
    baseline_protocol = loaded[0][3]["protocol"]
    result = {
        "schema_version": "1.0",
        "experiment_id": args.experiment_id,
        "evaluation_mode": "speed",
        "dataset": args.dataset,
        "split": (
            "official_test"
            if args.dataset == "7scenes"
            else "protocol_eight_sequences"
        ),
        "sequence_scope": "all",
        "sequences": reference_sequences,
        "frame_count": int(
            baseline_protocol["num_frames_per_sequence"]
        ),
        "sampling_strategy": (
            "shared_sequential_randomstate_choice"
        ),
        "sampled_frames_path": str(
            args.baseline_dir / "sampled_frames.json"
        ),
        "checkpoint": args.checkpoint,
        "seed": int(baseline_protocol["seed"]),
        "git_commit": git_value("rev-parse", "HEAD"),
        "working_tree_clean": not bool(git_value("status", "--porcelain")),
        "image_resolution": int(baseline_protocol["image_resolution"]),
        "resize_mode": baseline_protocol["resize_mode"],
        "batch_size": 1,
        "gpu": {
            "model": args.gpu_model,
            "physical_device_id": None,
            "effective_autocast_dtype": None,
        },
        "results": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "comparison.json", result)
    (args.output_dir / "comparison.md").write_text(
        render_markdown(args.dataset, rows),
        encoding="utf-8",
    )
    (args.output_dir / "comparison.tex").write_text(
        render_latex(args.dataset, rows),
        encoding="utf-8",
    )
    with (args.output_dir / "comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method_id",
                "display_name",
                *METRIC_KEYS,
                "model_latency_ms_mean",
                "peak_allocated_gib_max",
                "speedup",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method_id": row["method_id"],
                    "display_name": row["display_name"],
                    **row["metrics"],
                    **row["efficiency"],
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
