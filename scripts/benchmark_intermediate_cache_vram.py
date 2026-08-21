#!/usr/bin/env python3
"""Forward-only VRAM A/B for the VGGT* intermediate-cache policy.

The production evaluator also builds geometry metrics after inference, which is
not part of the encoder-memory optimisation described by FastVGGT.  This tool
therefore measures only one warmed-up model forward and CUDA peak memory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eval_tum_dynamics_paper import load_frame_records, load_model, sample_records
from vggt_omega.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--method", choices=("fastvggt", "u-m"), required=True)
    parser.add_argument("--legacy-cache-all", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2")

    records = load_frame_records(args.data_root / args.sequence, tolerance=0.02)
    sampled, _, _, skipped = sample_records(
        {args.sequence: records}, args.num_frames, seed=42, strategy="uniform", sampling_stride=3
    )
    if args.sequence not in sampled:
        raise RuntimeError(f"Sequence was skipped: {skipped}")
    records = sampled[args.sequence]
    device = torch.device("cuda")
    images = load_and_preprocess_images(
        [str(record.rgb_path) for record in records],
        mode="max_size",
        image_resolution=args.image_resolution,
    ).to(device, non_blocking=True)

    is_um = args.method == "u-m"
    model = load_model(
        args.checkpoint,
        device,
        merge_ratio=0.0 if is_um else 0.9,
        frame_fusion_mode="u-m" if is_um else "none",
        frame_fusion_lambda_cost=0.04,
        frame_fusion_spatial_radius=2,
        frame_fusion_temporal_window=4,
        frame_fusion_recompute_layers="0,10,17" if is_um else "",
        retain_only_cached_intermediates=not args.legacy_cache_all,
    )

    result: dict[str, object] = {
        "method": args.method,
        "num_frames": args.num_frames,
        "sequence": args.sequence,
        "cache_policy": "all_24_layers" if args.legacy_cache_all else "required_4_layers",
        "required_layers": sorted(model.aggregator.cached_layer_indices),
    }
    try:
        with torch.inference_mode():
            warmup_predictions = model(images)
        torch.cuda.synchronize()
        del warmup_predictions
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            predictions = model(images)
        torch.cuda.synchronize()
        result.update(
            success=True,
            forward_seconds=time.perf_counter() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            output_keys=sorted(predictions),
        )
        del predictions
    except torch.cuda.OutOfMemoryError as exc:
        result.update(success=False, error="CUDA out of memory", detail=str(exc))
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
