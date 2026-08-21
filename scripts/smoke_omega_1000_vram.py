#!/usr/bin/env python3
"""Fast forward-only 1000-frame VRAM/OOM check for VGGT-Omega adapters.

No metrics, image outputs, warm-up, or timing repeats are performed.  Each
method starts from an empty CUDA cache and records the end-to-end model + input
+ forward peak.  An OOM is recorded and does not prevent the next method from
being checked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_tum_dynamics_paper import load_frame_records, load_model, sample_records
from vggt_omega.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    # This TUM sequence has 1258 valid RGB/pose/depth triplets, enough for the
    # default 1000-frame validation.
    parser.add_argument("--sequence", default="rgbd_dataset_freiburg3_sitting_xyz")
    parser.add_argument("--num-frames", type=int, default=1000)
    parser.add_argument("--sampling-stride", type=int, default=2)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--methods", default="baseline,fastvggt,u-m")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def gib(value: int) -> float:
    return value / 2**30


def model_for(method: str, checkpoint: Path, device: torch.device):
    kwargs: dict[str, object] = {"merge_ratio": 0.0, "retain_only_cached_intermediates": False}
    if method == "fastvggt":
        kwargs["merge_ratio"] = 0.9
    elif method == "u-m":
        kwargs.update(
            frame_fusion_mode="u-m",
            frame_fusion_lambda_cost=0.04,
            frame_fusion_spatial_radius=2,
            frame_fusion_temporal_window=4,
            frame_fusion_recompute_layers="0,10,17",
            frame_fusion_attention_variant="representative",
        )
    return load_model(checkpoint, device, **kwargs)


def run_one(method: str, image_paths: list[str], args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = images = predictions = None
    try:
        model = model_for(method, args.checkpoint, device)
        images = load_and_preprocess_images(
            image_paths, mode="max_size", image_resolution=args.image_resolution
        ).to(device, non_blocking=True)
        with torch.inference_mode():
            predictions = model(images)
        torch.cuda.synchronize(device)
        return {
            "method": method,
            "success": True,
            "forward_seconds": time.perf_counter() - started,
            "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
            "output_keys": sorted(predictions),
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "method": method,
            "success": False,
            "error": "CUDA out of memory",
            "detail": str(exc),
            "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
        }
    finally:
        del predictions, images, model
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    invalid = sorted(set(methods) - {"baseline", "fastvggt", "u-m"})
    if invalid:
        raise ValueError(f"Unsupported --methods: {invalid}")

    records = load_frame_records(args.data_root / args.sequence, tolerance=0.02)
    sampled, indices, modes, skipped = sample_records(
        {args.sequence: records}, args.num_frames, seed=42, strategy="uniform",
        sampling_stride=args.sampling_stride,
    )
    if args.sequence not in sampled:
        raise RuntimeError(f"Sequence cannot provide {args.num_frames} frames: {skipped}")
    selected = sampled[args.sequence]
    paths = [str(record.rgb_path) for record in selected]
    device = torch.device("cuda")
    results = [run_one(method, paths, args, device) for method in methods]
    summary = {
        "sequence": args.sequence,
        "num_frames": len(selected),
        "sampling_mode": modes[args.sequence],
        "sampled_indices": indices[args.sequence],
        "image_resolution": args.image_resolution,
        "cache_policy": "all_24_layers",
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
