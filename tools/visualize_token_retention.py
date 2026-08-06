#!/usr/bin/env python3
"""Visualize patch tokens retained by frame-fusion and temporal schemes.

The script runs one sampled sequence with the same 512/max-size preprocessing
used by the paper evaluators.  It records the temporal representative mapping
and the actual FastVGGT merge trace from the last global attention block, then
renders the retained/merged patch mask over one input frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images


def load_model(checkpoint: Path, mode: str, device: torch.device) -> VGGTOmega:
    if mode == "least20":
        kwargs = {
            "merge_ratio": 0.0,
            "frame_fusion_mode": "pair-top-percent",
            "frame_fusion_start_layer": -1,
            "frame_fusion_pair_percent": 25.0,
            "frame_fusion_pool_size": 2,
            "frame_fusion_target_keep_policy": "least-similar",
            "frame_fusion_target_keep_percent": 20.0,
            "frame_fusion_target_keep_seed": 33,
        }
    elif mode == "temporal090":
        kwargs = {
            "merge_ratio": 0.0,
            "frame_fusion_mode": "temporal-representative",
            "frame_fusion_start_layer": -1,
            "frame_fusion_target_keep_threshold": 0.90,
        }
    elif mode == "adaptive-linear":
        kwargs = {
            "merge_ratio": 0.0,
            "frame_fusion_mode": "adaptive-temporal-representative",
            "frame_fusion_start_layer": -1,
        }
    elif mode == "fastvggt":
        kwargs = {"merge_ratio": 0.9}
    else:
        raise ValueError(f"unsupported mode: {mode}")

    model = VGGTOmega(first_frame_token_indices=(0,), **kwargs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device).eval()


def capture_temporal_plan(model: VGGTOmega, captured: dict[str, object]) -> None:
    aggregator = model.aggregator
    original = aggregator._build_temporal_representative_plans

    def wrapped(self, tokens: torch.Tensor, *, source_layer: int):
        plans = original(tokens, source_layer=source_layer)
        captured["plans"] = plans
        return plans

    aggregator._build_temporal_representative_plans = types.MethodType(wrapped, aggregator)


def capture_adaptive_temporal_plan(model: VGGTOmega, captured: dict[str, object]) -> None:
    aggregator = model.aggregator
    original = aggregator._build_adaptive_temporal_representative_plans

    def wrapped(self, tokens: torch.Tensor, *, source_layer: int):
        plans = original(tokens, source_layer=source_layer)
        captured["plans"] = plans
        return plans

    aggregator._build_adaptive_temporal_representative_plans = types.MethodType(
        wrapped, aggregator
    )


def capture_pair_plan(model: VGGTOmega, captured: dict[str, object]) -> None:
    aggregator = model.aggregator
    original = aggregator._build_frame_fusion_pair_plans

    def wrapped(
        self,
        tokens: torch.Tensor,
        *,
        patch_grid_size: tuple[int, int],
        source_layer: int,
    ):
        plans = original(
            tokens,
            patch_grid_size=patch_grid_size,
            source_layer=source_layer,
        )
        captured["plans"] = plans
        return plans

    aggregator._build_frame_fusion_pair_plans = types.MethodType(wrapped, aggregator)


def capture_fast_merge_traces(model: VGGTOmega) -> None:
    for block in model.aggregator.inter_frame_blocks:
        block.attn.record_merge_trace = True


def image_from_tensor(image: torch.Tensor) -> Image.Image:
    array = image.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(np.round(array * 255.0)), mode="RGB")


def temporal_mask(model: VGGTOmega, captured: dict[str, object], frame_index: int, patch_count: int):
    plans = captured.get("plans")
    if not plans:
        raise RuntimeError("temporal representative plan was not captured")
    plan = plans[0]
    mapping = plan.position_to_representative.detach().cpu().long().numpy()
    source_indices = plan.representative_source_indices.detach().cpu().long().numpy()
    representative_source = source_indices[mapping[frame_index]]
    own_source = (representative_source // patch_count) == frame_index
    return own_source.astype(bool), {
        "retained_patch_tokens": int(own_source.sum()),
        "full_patch_tokens": int(patch_count),
        "representative_patch_tokens_total": int(source_indices.size),
    }


def least20_mask(captured: dict[str, object], frame_index: int, patch_count: int):
    plans = captured.get("plans")
    if not plans:
        raise RuntimeError("least20 pair plan was not captured")
    plan = plans[0]
    mask = np.ones(patch_count, dtype=bool)
    target_frames = plan.target_frames.detach().cpu().long().numpy()
    keep_indices = plan.target_keep_patch_indices.detach().cpu().long().numpy()
    matches = np.flatnonzero(target_frames == frame_index)
    if matches.size:
        mask[:] = False
        mask[keep_indices[int(matches[0])]] = True
    return mask, {
        "retained_patch_tokens": int(mask.sum()),
        "full_patch_tokens": int(patch_count),
        "selected_pairs": int(len(plan.pairs)),
        "target_frames": [int(value) for value in target_frames.tolist()],
        "target_keep_percent": 20.0,
        "target_frame": bool(matches.size),
    }


def least20_sequence_mask(captured: dict[str, object], num_frames: int, patch_count: int):
    plans = captured.get("plans")
    if not plans:
        raise RuntimeError("least20 pair plan was not captured")
    plan = plans[0]
    mask = np.ones((num_frames, patch_count), dtype=bool)
    target_frames = plan.target_frames.detach().cpu().long().numpy()
    keep_indices = plan.target_keep_patch_indices.detach().cpu().long().numpy()
    for pair_index, frame in enumerate(target_frames):
        mask[int(frame), :] = False
        mask[int(frame), keep_indices[pair_index]] = True
    return mask


def temporal_sequence_mask(captured: dict[str, object], num_frames: int, patch_count: int):
    plans = captured.get("plans")
    if not plans:
        raise RuntimeError("temporal representative plan was not captured")
    plan = plans[0]
    mapping = plan.position_to_representative.detach().cpu().long().numpy()
    source_indices = plan.representative_source_indices.detach().cpu().long().numpy()
    representative_source = source_indices[mapping]
    frame_indices = representative_source // patch_count
    return frame_indices == np.arange(num_frames)[:, None]


def fast_mask(model: VGGTOmega, frame_index: int, patch_count: int, num_frames: int, num_special: int):
    traces = []
    for layer, block in enumerate(model.aggregator.inter_frame_blocks):
        selected = getattr(block.attn, "last_merge_source_indices", None)
        if selected is None:
            continue
        selected = selected.detach().cpu().long().reshape(-1).numpy()
        traces.append((layer, selected))
    if not traces:
        raise RuntimeError("FastVGGT merge trace was not captured")

    layer, selected = traces[-1]
    start = frame_index * (patch_count + num_special) + num_special
    positions = np.arange(start, start + patch_count)
    retained = ~np.isin(positions, selected)
    total_tokens = num_frames * (patch_count + num_special)
    all_patch_positions = np.concatenate(
        [
            np.arange(frame * (patch_count + num_special) + num_special,
                      frame * (patch_count + num_special) + num_special + patch_count)
            for frame in range(num_frames)
        ]
    )
    retained_patch_tokens_total = int((~np.isin(all_patch_positions, selected)).sum())
    return retained, {
        "retained_patch_tokens": int(retained.sum()),
        "full_patch_tokens": int(patch_count),
        "retained_patch_tokens_total": retained_patch_tokens_total,
        "full_patch_tokens_total": int(num_frames * patch_count),
        "patch_retention_vs_full_total": float(
            retained_patch_tokens_total / max(num_frames * patch_count, 1)
        ),
        "retained_all_tokens": int(total_tokens - selected.size),
        "full_all_tokens": int(total_tokens),
        "last_global_layer": int(layer),
        "merged_source_tokens": int(selected.size),
        "global_layers": [
            {
                "layer": int(layer_index),
                "retained_all_tokens": int(total_tokens - values.size),
                "retention_vs_full": float((total_tokens - values.size) / total_tokens),
            }
            for layer_index, values in traces
        ],
    }


def render_overlay(image: Image.Image, mask: np.ndarray, title: str, output: Path) -> None:
    width, height = image.size
    patch_h = int(round(height / 16))
    patch_w = int(round(width / 16))
    if patch_h * patch_w != mask.size:
        raise ValueError(f"image grid {patch_h}x{patch_w} does not match {mask.size} tokens")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, retained in enumerate(mask.reshape(patch_h, patch_w).flat):
        row, col = divmod(index, patch_w)
        left, top = col * 16, row * 16
        right, bottom = (col + 1) * 16, (row + 1) * 16
        color = (30, 190, 80, 145) if retained else (220, 55, 55, 120)
        draw.rectangle((left, top, right - 1, bottom - 1), fill=color, outline=(255, 255, 255, 100))

    composite = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    canvas = Image.new("RGB", (width, height + 38), "white")
    canvas.paste(composite, (0, 38))
    text = ImageDraw.Draw(canvas)
    text.text((8, 10), title, fill="black")
    canvas.save(output)


def render_sequence_heatmap(mask: np.ndarray, title: str, output: Path) -> None:
    """Render one comparable local-retention mask for every frame."""

    colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colors[mask] = (45, 190, 90)
    colors[~mask] = (220, 65, 65)
    image = Image.fromarray(colors, mode="RGB")
    canvas = Image.new("RGB", (image.width, image.height + 38), "white")
    canvas.paste(image, (0, 38))
    ImageDraw.Draw(canvas).text((8, 10), title, fill="black")
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sampled-frames", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-index", type=int, default=150)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("least20", "temporal090", "adaptive-linear"),
        default=("least20", "adaptive-linear"),
        help="Visualization modes to compare.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sampled = json.loads(args.sampled_frames.read_text())
    if args.sequence not in sampled:
        raise KeyError(f"sequence not found: {args.sequence}")
    paths = sampled[args.sequence]["rgb_paths"]
    if not 0 <= args.frame_index < len(paths):
        raise IndexError(f"frame index {args.frame_index} outside [0, {len(paths)})")

    device = torch.device(args.device)
    images = load_and_preprocess_images(paths, mode="max_size", image_resolution=512)
    num_frames, _, height, width = images.shape
    patch_h, patch_w = height // 16, width // 16
    patch_count = patch_h * patch_w
    num_special = 17  # one camera token plus sixteen register tokens
    source_image = image_from_tensor(images[args.frame_index])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "sequence": args.sequence,
        "input_frame_index": args.frame_index,
        "num_frames": num_frames,
        "image_shape": [height, width],
        "patch_grid": [patch_h, patch_w],
        "patch_tokens_per_frame": patch_count,
    }

    for mode in args.modes:
        captured: dict[str, object] = {}
        model = load_model(args.checkpoint, mode, device)
        if mode == "temporal090":
            capture_temporal_plan(model, captured)
        elif mode == "adaptive-linear":
            capture_adaptive_temporal_plan(model, captured)
        elif mode == "least20":
            capture_pair_plan(model, captured)
        with torch.inference_mode():
            model(images.to(device, non_blocking=True))
        if mode == "least20":
            mask, stats = least20_mask(captured, args.frame_index, patch_count)
            sequence_mask = least20_sequence_mask(captured, num_frames, patch_count)
        else:
            mask, stats = temporal_mask(model, captured, args.frame_index, patch_count)
            sequence_mask = temporal_sequence_mask(captured, num_frames, patch_count)
        stats["patch_retention_vs_full"] = float(mask.mean())
        stats["sequence_local_patch_retention_vs_full"] = float(sequence_mask.mean())
        metadata[mode] = stats
        render_overlay(
            source_image,
            mask,
            f"{mode}: green=local, red=shared/omitted | frame={args.frame_index} | {mask.sum()}/{mask.size}",
            args.output_dir / f"{mode}_frame{args.frame_index:03d}_overlay.png",
        )
        render_sequence_heatmap(
            sequence_mask,
            f"{mode}: green=local, red=shared/omitted | sequence local retention={sequence_mask.mean():.3f}",
            args.output_dir / f"{mode}_sequence_retention_heatmap.png",
        )
        del model
        torch.cuda.empty_cache()

    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
