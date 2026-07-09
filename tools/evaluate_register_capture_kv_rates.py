#!/usr/bin/env python3
"""Evaluate source-side K/V merge rates selected by frame-register capture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_tum_dynamics_paper import load_frame_records, sample_records
from tools.analyze_attention_information_flow import qkv_from_block
from tools.analyze_token_evolution import load_model
from tools.evaluate_coverage_guided_merge import (
    configure_baseline,
    configure_guided,
    task_metrics,
    timed_forward,
)
from vggt_omega.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/mmc_lyxiang/dataset/TUM-Dynamics"),
    )
    parser.add_argument(
        "--sequence",
        default="rgbd_dataset_freiburg3_sitting_static",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", nargs="+", type=int, default=[12, 13, 15])
    parser.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "register_capture_kv_50_static" / "results.json",
    )
    return parser.parse_args()


class RegisterCaptureCollector:
    def __init__(self, model, num_frames: int, layers: list[int]) -> None:
        self.model = model
        self.num_frames = num_frames
        self.layers = layers
        self.scores: dict[int, np.ndarray] = {}
        self.handles = []

    def __enter__(self):
        for layer in self.layers:
            self.handles.append(
                self.model.aggregator.frame_blocks[layer].register_forward_pre_hook(
                    self._hook(layer)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer: int):
        def hook(block, inputs):
            x = inputs[0]
            rope = inputs[1] if len(inputs) > 1 else None
            patch_start = self.model.aggregator.patch_token_start
            q, k, v = qkv_from_block(block, x, rope)
            logits = torch.matmul(
                q[:, :, 1:patch_start].float(), k.float().transpose(-2, -1)
            ) * block.attn.scale
            patch_probability = logits.softmax(dim=-1)[..., patch_start:]
            patch_value_norm = v[..., patch_start:, :].float().norm(dim=-1)
            score = (patch_probability * patch_value_norm[:, :, None, :]).mean(dim=(1, 2))
            batch = x.shape[0] // self.num_frames
            self.scores[layer] = score.view(batch, self.num_frames, -1).cpu().numpy()
            del q, k, v, logits, patch_probability, patch_value_norm
        return hook


def masks_for_rate(
    scores: dict[int, np.ndarray],
    rate: float,
    tokens_per_frame: int,
    patch_start: int,
) -> dict[int, torch.Tensor]:
    masks = {}
    for layer, values in scores.items():
        batch, frames, patches = values.shape
        full = np.zeros((batch, frames, tokens_per_frame), dtype=bool)
        count = int(round(rate * patches))
        if count:
            for batch_index in range(batch):
                for frame in range(1, frames):
                    selected = np.argpartition(values[batch_index, frame], -count)[-count:]
                    full[batch_index, frame, patch_start + selected] = True
        masks[layer] = torch.from_numpy(full.reshape(batch, frames * tokens_per_frame))
    return masks


def main() -> int:
    args = parse_args()
    if any(rate < 0 or rate > 1 for rate in args.rates):
        raise ValueError("Rates must be in [0, 1]")
    device = torch.device(args.device)
    records_all = load_frame_records(args.data_root / args.sequence, 0.02)
    sampled, sampled_indices = sample_records(
        {args.sequence: records_all}, args.num_frames, args.seed
    )
    records = sampled[args.sequence]
    images = load_and_preprocess_images(
        [str(record.rgb_path) for record in records],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    ).to(device)
    model = load_model(args.checkpoint, device)
    configure_baseline(model)

    calibration_start = time.perf_counter()
    with RegisterCaptureCollector(model, args.num_frames, args.layers) as collector:
        with torch.inference_mode():
            calibration_prediction = model(images)
    calibration_seconds = time.perf_counter() - calibration_start
    scores = collector.scores
    del calibration_prediction

    tokens_per_frame = 1 + 16 + (images.shape[-2] // 16) * (images.shape[-1] // 16)
    rows = []
    for rate in args.rates:
        if rate == 0:
            configure_baseline(model)
        else:
            masks = masks_for_rate(
                scores, rate, tokens_per_frame, model.aggregator.patch_token_start
            )
            configure_guided(model, masks, device, set(args.layers), True)
        predictions, timings, peak = timed_forward(
            model, images, device, args.timing_repeats
        )
        metrics = task_metrics(predictions, records, predictions["images"].shape[-2:])
        actual_by_layer = {
            str(layer): int(model.aggregator.inter_frame_blocks[layer].attn.last_merged_tokens)
            for layer in args.layers
        }
        patch_kv_candidates = (args.num_frames - 1) * (tokens_per_frame - 17)
        rows.append(
            {
                "requested_eligibility_rate": rate,
                "actual_merged_tokens_by_layer": actual_by_layer,
                "actual_merged_fraction_mean": float(
                    np.mean(list(actual_by_layer.values())) / patch_kv_candidates
                ),
                "latency_ms": float(np.median(timings)),
                "timing_repeats_ms": timings,
                "peak_allocated_gib": peak,
                **metrics,
            }
        )
        del predictions
        torch.cuda.empty_cache()

    baseline_latency = rows[0]["latency_ms"]
    baseline = {key: rows[0][key] for key in ("auc_3_percent", "auc_30_percent", "delta_1_25_percent", "abs_rel")}
    for row in rows:
        row["speedup_vs_baseline"] = baseline_latency / row["latency_ms"]
        row["metric_delta_vs_baseline"] = {
            key: row[key] - baseline[key] for key in baseline
        }
    result = {
        "config": {
            "sequence": args.sequence,
            "num_frames": args.num_frames,
            "seed": args.seed,
            "layers": args.layers,
            "selector": "frame register-to-patch attention times value norm",
            "merge_direction": "source K/V only; all Q preserved",
            "selector_calibration_excluded_from_latency": True,
            "selector_calibration_seconds": calibration_seconds,
            "timing_repeats": args.timing_repeats,
            "sampled_pool_indices": sampled_indices[args.sequence],
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
