#!/usr/bin/env python3
"""Profile high-level VGGT-Omega/FastVGGT operation time shares on 300-frame inputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_7scenes_paper, eval_tum_dynamics_paper
from vggt_omega.utils.load_fn import load_and_preprocess_images


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_7SCENES_ROOT = Path("/data/mmc_lyxiang/dataset/7scenes")
DEFAULT_TUM_ROOT = Path("/data/mmc_lyxiang/dataset/TUM-Dynamics")


@dataclass
class TimedCall:
    name: str
    category: str
    layer: int | None
    elapsed_ms: float


class ModuleTimer:
    def __init__(self) -> None:
        self.records: list[TimedCall] = []
        self.handles = []

    def add(self, module: torch.nn.Module, name: str, category: str, layer: int | None = None) -> None:
        start_events: dict[int, torch.cuda.Event] = {}

        def pre_hook(_module, _inputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            start_events[id(_module)] = event

        def post_hook(_module, _inputs, _output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            start = start_events.pop(id(_module))
            self.records.append(TimedCall(name, category, layer, float(start.elapsed_time(end))))

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seven-scenes-root", type=Path, default=DEFAULT_7SCENES_ROOT)
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_TUM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/operation_breakdown_300frames"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.99])
    parser.add_argument("--seven-scenes-sequence", default="chess/seq-03")
    parser.add_argument("--tum-sequence", default="rgbd_dataset_freiburg3_sitting_static")
    parser.add_argument("--datasets", nargs="+", choices=("7scenes", "tum_dynamics"), default=["7scenes", "tum_dynamics"])
    return parser.parse_args()


def records_for_dataset(args: argparse.Namespace, dataset: str):
    if dataset == "7scenes":
        sequence_dir = args.seven_scenes_root / args.seven_scenes_sequence
        records = eval_7scenes_paper.load_frame_records(sequence_dir)
        sampled, sampled_indices = eval_7scenes_paper.sample_records(
            {args.seven_scenes_sequence: records}, args.num_frames, args.seed
        )
        sequence = args.seven_scenes_sequence
        return sequence, sampled[sequence], sampled_indices[sequence]
    sequence_dir = args.tum_root / args.tum_sequence
    records = eval_tum_dynamics_paper.load_frame_records(sequence_dir, tolerance=0.02)
    records = [record for record in records if record.rgb_path.is_file() and record.depth_path.is_file()]
    sampled, sampled_indices = eval_tum_dynamics_paper.sample_records(
        {args.tum_sequence: records}, args.num_frames, args.seed
    )
    sequence = args.tum_sequence
    return sequence, sampled[sequence], sampled_indices[sequence]


def load_model_for_ratio(args: argparse.Namespace, dataset: str, ratio: float, device: torch.device):
    loader = eval_7scenes_paper.load_model if dataset == "7scenes" else eval_tum_dynamics_paper.load_model
    return loader(
        args.checkpoint,
        device,
        merge_ratio=ratio,
        sparse_attention=False,
        sparse_ratio=None,
        sparse_cdf_threshold=None,
        sparse_pool_mode="avg",
    )


def attach_timers(model) -> ModuleTimer:
    timer = ModuleTimer()
    agg = model.aggregator
    timer.add(agg.patch_embed, "patch_embed", "patch_embed")
    for layer, block in enumerate(agg.frame_blocks):
        timer.add(block, f"frame_block_L{layer:02d}", "frame_attention", layer)
    for layer, block in enumerate(agg.inter_frame_blocks):
        kind = agg.inter_frame_attention_types[layer]
        category = "inter_global_attention" if kind == "global" else "inter_register_attention"
        timer.add(block, f"inter_{kind}_block_L{layer:02d}", category, layer)
    if model.camera_head is not None:
        timer.add(model.camera_head, "camera_head", "camera_head")
    if model.dense_head is not None:
        timer.add(model.dense_head, "dense_head", "dense_head")
    return timer


def profile_once(model, images: torch.Tensor) -> tuple[float, list[TimedCall], float]:
    timer = attach_timers(model)
    torch.cuda.reset_peak_memory_stats(images.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        _ = model(images)
    end.record()
    torch.cuda.synchronize()
    total_ms = float(start.elapsed_time(end))
    peak_gib = float(torch.cuda.max_memory_allocated(images.device) / (1024**3))
    records = list(timer.records)
    timer.close()
    return total_ms, records, peak_gib


def summarize_records(total_ms: float, records: list[TimedCall]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_category: dict[str, float] = defaultdict(float)
    by_name: dict[tuple[str, str, int | None], float] = defaultdict(float)
    for record in records:
        by_category[record.category] += record.elapsed_ms
        by_name[(record.name, record.category, record.layer)] += record.elapsed_ms
    measured_sum = sum(by_category.values())
    by_category["unmeasured_or_overlap"] += max(0.0, total_ms - measured_sum)
    category_rows = [
        {
            "category": category,
            "time_ms": value,
            "share_of_total_percent": 100.0 * value / total_ms if total_ms > 0 else np.nan,
        }
        for category, value in sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)
    ]
    detail_rows = [
        {
            "name": name,
            "category": category,
            "layer": "" if layer is None else layer,
            "time_ms": value,
            "share_of_total_percent": 100.0 * value / total_ms if total_ms > 0 else np.nan,
        }
        for (name, category, layer), value in sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return category_rows, detail_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_category_breakdown(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    rows = [row for row in rows if float(row["time_ms"]) > 0]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=170)
    labels = [str(row["category"]) for row in rows]
    values = [float(row["share_of_total_percent"]) for row in rows]
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel("Share of end-to-end model forward time (%)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    for dataset in args.datasets:
        sequence, records, sampled_indices = records_for_dataset(args, dataset)
        image_paths = [str(record.rgb_path) for record in records]
        images = load_and_preprocess_images(
            image_paths,
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(device, non_blocking=True)
        dataset_dir = args.output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for ratio in args.ratios:
            ratio_tag = f"ratio_{int(round(ratio * 100)):02d}"
            ratio_dir = dataset_dir / ratio_tag
            ratio_dir.mkdir(parents=True, exist_ok=True)
            model = load_model_for_ratio(args, dataset, ratio, device)
            # Warmup without timers.
            with torch.inference_mode():
                _ = model(images)
            torch.cuda.synchronize()
            total_ms, records_timed, peak_gib = profile_once(model, images)
            category_rows, detail_rows = summarize_records(total_ms, records_timed)
            write_csv(ratio_dir / "category_breakdown.csv", category_rows)
            write_csv(ratio_dir / "module_breakdown.csv", detail_rows)
            plot_category_breakdown(
                ratio_dir / "category_breakdown.png",
                f"{dataset} {sequence}, {args.num_frames} frames, merge ratio={ratio}",
                category_rows,
            )
            metadata = {
                "dataset": dataset,
                "sequence": sequence,
                "sampled_indices": sampled_indices,
                "num_frames": args.num_frames,
                "merge_ratio": ratio,
                "total_forward_ms": total_ms,
                "peak_allocated_gib": peak_gib,
                "image_resolution": args.image_resolution,
                "resize_mode": args.resize_mode,
                "note": "CUDA-event module hooks. Category times are measured at high-level modules and may include hook synchronization overhead, but are useful for relative shares.",
            }
            (ratio_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            for row in category_rows:
                all_summary.append(
                    {
                        "dataset": dataset,
                        "sequence": sequence,
                        "merge_ratio": ratio,
                        "num_frames": args.num_frames,
                        "total_forward_ms": total_ms,
                        "peak_allocated_gib": peak_gib,
                        **row,
                    }
                )
            del model
            torch.cuda.empty_cache()
        del images
        torch.cuda.empty_cache()

    write_csv(args.output_dir / "operation_breakdown_summary.csv", all_summary)
    print(f"Saved profiling results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
