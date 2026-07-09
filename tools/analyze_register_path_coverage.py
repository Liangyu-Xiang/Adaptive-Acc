#!/usr/bin/env python3
"""Compare direct cross-frame patch attention with a two-hop register path.

For source-frame patch p and destination-frame patch q at global layer l:

  direct(q,p) = A_global(q, p)
  register_path(q,p) = sum_r A_global(q, register_r) * A_frame(register_r, p)

Head-mean attention matrices are used because heads are not aligned between the
frame and global blocks. This is an attention-rollout proxy, not causal proof
that a register retains or loses a patch's semantic content.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_attention_information_flow import patch_center, qkv_from_block
from tools.analyze_token_evolution import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images


COVERAGE_K = (5, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "token_evolution_3frame",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--uncovered-links", type=int, default=15)
    return parser.parse_args()


class RegisterPathCoverageCollector:
    def __init__(self, model, num_frames: int, query_chunk: int) -> None:
        self.aggregator = model.aggregator
        self.num_frames = num_frames
        self.query_chunk = query_chunk
        self.global_layers = [
            layer
            for layer, kind in enumerate(self.aggregator.inter_frame_attention_types)
            if kind == "global"
        ]
        self.frame_register_attention: dict[int, torch.Tensor] = {}
        self.results: dict[int, dict[str, np.ndarray]] = {}
        self.handles = []

    def __enter__(self):
        for layer in self.global_layers:
            self.handles.append(
                self.aggregator.frame_blocks[layer].register_forward_pre_hook(
                    self._frame_hook(layer)
                )
            )
            self.handles.append(
                self.aggregator.inter_frame_blocks[layer].register_forward_pre_hook(
                    self._global_hook(layer)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.frame_register_attention.clear()

    def _frame_hook(self, layer: int):
        def hook(block, inputs):
            x = inputs[0]
            rope = inputs[1] if len(inputs) > 1 else None
            patch_start = self.aggregator.patch_token_start
            batch_frames, num_tokens, _ = x.shape
            q, k, _ = qkv_from_block(block, x, rope)
            # The denominator must include camera/register keys as in the real attention.
            all_logits = torch.matmul(
                q[:, :, 1:patch_start].float(), k.float().transpose(-2, -1)
            ) * block.attn.scale
            probabilities = all_logits.softmax(dim=-1)[..., patch_start:].mean(dim=1)
            batch_size = batch_frames // self.num_frames
            self.frame_register_attention[layer] = probabilities.view(
                batch_size,
                self.num_frames,
                patch_start - 1,
                num_tokens - patch_start,
            ).detach()
            del q, k, all_logits, probabilities

        return hook

    def _global_hook(self, layer: int):
        def hook(block, inputs):
            x = inputs[0]
            batch_size, total_tokens, _ = x.shape
            tokens_per_frame = total_tokens // self.num_frames
            patch_start = self.aggregator.patch_token_start
            patch_count = tokens_per_frame - patch_start
            frame_attention = self.frame_register_attention.pop(layer)
            q, k, _ = qkv_from_block(block, x, None)
            k_t = k.float().transpose(-2, -1)

            shape = (batch_size, self.num_frames, self.num_frames, patch_count)
            direct_mass = np.full(shape, np.nan, dtype=np.float32)
            register_mass = np.full(shape, np.nan, dtype=np.float32)
            mass_ratio = np.full(shape, np.nan, dtype=np.float32)
            total_variation = np.full(shape, np.nan, dtype=np.float32)
            direct_top1_index = np.full(shape, -1, dtype=np.int16)
            direct_top1_weight = np.full(shape, np.nan, dtype=np.float16)
            direct_top1_register_weight = np.full(shape, np.nan, dtype=np.float16)
            covered_top1 = np.zeros(shape + (len(COVERAGE_K),), dtype=bool)
            direct_top5_recall = np.full(shape + (len(COVERAGE_K),), np.nan, dtype=np.float16)

            for destination in range(self.num_frames):
                query_start = destination * tokens_per_frame + patch_start
                query_indices = torch.arange(
                    query_start, query_start + patch_count, device=x.device
                )
                for offset in range(0, patch_count, self.query_chunk):
                    chunk = query_indices[offset : offset + self.query_chunk]
                    logits = torch.matmul(q[:, :, chunk].float(), k_t) * block.attn.scale
                    probabilities = logits.softmax(dim=-1).mean(dim=1)
                    end = offset + len(chunk)
                    for source in range(self.num_frames):
                        if source == destination:
                            continue
                        source_start = source * tokens_per_frame
                        register_slice = slice(source_start + 1, source_start + patch_start)
                        patch_slice = slice(source_start + patch_start, source_start + tokens_per_frame)
                        direct = probabilities[..., patch_slice]
                        patch_to_register = probabilities[..., register_slice]
                        indirect = torch.matmul(
                            patch_to_register,
                            frame_attention[:, source].float(),
                        )

                        d_mass = direct.sum(dim=-1)
                        r_mass = indirect.sum(dim=-1)
                        d_norm = direct / d_mass[..., None].clamp_min(1e-12)
                        r_norm = indirect / r_mass[..., None].clamp_min(1e-12)
                        tv = 0.5 * (d_norm - r_norm).abs().sum(dim=-1)
                        direct_values, direct_indices = d_norm.topk(5, dim=-1)
                        register_values, register_indices = r_norm.topk(max(COVERAGE_K), dim=-1)
                        direct_on_register = r_norm.gather(-1, direct_indices[..., :1]).squeeze(-1)

                        target = (slice(None), destination, source, slice(offset, end))
                        direct_mass[target] = d_mass.cpu().numpy()
                        register_mass[target] = r_mass.cpu().numpy()
                        mass_ratio[target] = (r_mass / d_mass.clamp_min(1e-12)).cpu().numpy()
                        total_variation[target] = tv.cpu().numpy()
                        direct_top1_index[target] = direct_indices[..., 0].cpu().numpy().astype(np.int16)
                        direct_top1_weight[target] = direct_values[..., 0].cpu().numpy().astype(np.float16)
                        direct_top1_register_weight[target] = direct_on_register.cpu().numpy().astype(np.float16)
                        for coverage_index, coverage_k in enumerate(COVERAGE_K):
                            candidates = register_indices[..., :coverage_k]
                            matches = direct_indices[..., :, None] == candidates[..., None, :]
                            covered_top1[target + (coverage_index,)] = (
                                matches[..., 0, :].any(dim=-1).cpu().numpy()
                            )
                            direct_top5_recall[target + (coverage_index,)] = (
                                matches.any(dim=-1).float().mean(dim=-1).cpu().numpy().astype(np.float16)
                            )
                        del direct, patch_to_register, indirect, d_norm, r_norm
                    del logits, probabilities

            self.results[layer] = {
                "direct_patch_mass": direct_mass,
                "register_path_mass": register_mass,
                "register_to_direct_mass_ratio": mass_ratio,
                "direct_register_total_variation": total_variation,
                "direct_top1_patch_index": direct_top1_index,
                "direct_top1_normalized_weight": direct_top1_weight,
                "register_weight_at_direct_top1": direct_top1_register_weight,
                "direct_top1_covered": covered_top1,
                "direct_top5_recall": direct_top5_recall,
            }
            del q, k, k_t, frame_attention

        return hook


def plot_coverage_maps(
    target: Path,
    layer: int,
    result: dict[str, np.ndarray],
    grid: tuple[int, int],
) -> None:
    pairs = [
        (destination, source)
        for destination in range(result["direct_patch_mass"].shape[1])
        for source in range(result["direct_patch_mass"].shape[2])
        if destination != source
    ]
    specs = (
        ("direct_patch_mass", "Direct patch mass", "viridis", None),
        ("register_to_direct_mass_ratio", "Register/direct mass", "magma", None),
        ("direct_top1_covered", "Top-1 covered by register top-20", "gray_r", 1),
        ("direct_register_total_variation", "Direct/register TV distance", "plasma", None),
    )
    figure, axes = plt.subplots(len(pairs), len(specs), figsize=(12, 2.5 * len(pairs)), squeeze=False)
    for row, (destination, source) in enumerate(pairs):
        for column, (key, label, cmap, coverage_index) in enumerate(specs):
            values = result[key][0, destination, source]
            if coverage_index is not None:
                values = values[:, coverage_index].astype(np.float32)
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = np.percentile(values[np.isfinite(values)], (2, 98))
            rendered = axes[row, column].imshow(
                values.reshape(grid), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            axes[row, column].set_title(f"Q F{destination} ← K F{source}\n{label}")
            axes[row, column].axis("off")
            figure.colorbar(rendered, ax=axes[row, column], fraction=0.04, pad=0.02)
    figure.suptitle(f"Global block {layer}: direct patch path vs two-hop register path")
    figure.savefig(target / f"register_path_coverage_L{layer:02d}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_uncovered_links(
    target: Path,
    layer: int,
    result: dict[str, np.ndarray],
    images: np.ndarray,
    grid: tuple[int, int],
    count: int,
) -> None:
    pairs = [
        (destination, source)
        for destination in range(len(images))
        for source in range(len(images))
        if destination != source
    ]
    figure, axes = plt.subplots(len(pairs), 1, figsize=(12, 3.1 * len(pairs)), squeeze=False)
    height, width = images.shape[1:3]
    coverage_index = COVERAGE_K.index(20)
    for row, (destination, source) in enumerate(pairs):
        axis = axes[row, 0]
        canvas = np.concatenate((images[destination], images[source]), axis=1)
        axis.imshow(canvas)
        covered = result["direct_top1_covered"][0, destination, source, :, coverage_index]
        weights = result["direct_top1_normalized_weight"][0, destination, source].astype(np.float32)
        key_indices = result["direct_top1_patch_index"][0, destination, source]
        candidates = np.flatnonzero(~covered)
        selected = candidates[np.argsort(weights[candidates])[-min(count, len(candidates)):]]
        norm = plt.Normalize(weights[selected].min(), weights[selected].max() + 1e-12)
        for query_index in selected:
            key_index = int(key_indices[query_index])
            x0, y0 = patch_center(int(query_index), grid, (height, width))
            x1, y1 = patch_center(key_index, grid, (height, width))
            color = plt.cm.plasma(norm(weights[query_index]))
            axis.plot([x0, x1 + width], [y0, y1], color=color, linewidth=0.9, alpha=0.8)
            axis.scatter([x0, x1 + width], [y0, y1], color=[color], s=8)
        axis.axvline(width - 0.5, color="white", linewidth=1)
        axis.set_title(
            f"Query F{destination} ← source F{source}: strongest direct top-1 links "
            "absent from register top-20"
        )
        axis.axis("off")
    figure.suptitle(f"Global block {layer}: direct cross-frame links not covered by register rollout")
    figure.savefig(target / f"uncovered_direct_links_L{layer:02d}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_results(
    sequence_dir: Path,
    collector: RegisterPathCoverageCollector,
    images: np.ndarray,
    grid: tuple[int, int],
    uncovered_count: int,
) -> None:
    target = sequence_dir / "register_path_coverage"
    target.mkdir(parents=True, exist_ok=True)
    layers = collector.global_layers
    keys = tuple(next(iter(collector.results.values())))
    arrays = {
        key: np.stack([collector.results[layer][key] for layer in layers])
        for key in keys
    }
    np.savez_compressed(
        target / "register_path_coverage_metrics.npz",
        global_layers=np.asarray(layers),
        coverage_k=np.asarray(COVERAGE_K),
        **arrays,
    )

    rows = []
    for layer in layers:
        result = collector.results[layer]
        for destination in range(len(images)):
            for source in range(len(images)):
                if destination == source:
                    continue
                valid = (0, destination, source)
                row = {
                    "global_block": layer,
                    "query_frame": destination,
                    "source_key_frame": source,
                    "direct_patch_mass_mean": float(np.mean(result["direct_patch_mass"][valid])),
                    "register_path_mass_mean": float(np.mean(result["register_path_mass"][valid])),
                    "register_to_direct_mass_ratio_mean": float(
                        np.mean(result["register_to_direct_mass_ratio"][valid])
                    ),
                    "direct_register_tv_mean": float(
                        np.mean(result["direct_register_total_variation"][valid])
                    ),
                }
                for index, coverage_k in enumerate(COVERAGE_K):
                    row[f"direct_top1_covered_by_register_top{coverage_k}_fraction"] = float(
                        np.mean(result["direct_top1_covered"][valid + (slice(None), index)])
                    )
                    row[f"direct_top5_recall_by_register_top{coverage_k}"] = float(
                        np.mean(result["direct_top5_recall"][valid + (slice(None), index)])
                    )
                rows.append(row)
        plot_coverage_maps(target, layer, result, grid)
        plot_uncovered_links(target, layer, result, images, grid, uncovered_count)

    with (target / "register_path_coverage_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "formula": "C[q,p] = sum_r A_global[q, register_r] * A_frame[register_r, patch_p]",
        "global_layers": layers,
        "coverage_k": list(COVERAGE_K),
        "uncovered_definition": "direct top-1 source patch is absent from register-path top-20",
        "limitations": [
            "head-mean attention rollout is a routing proxy, not causal attribution",
            "the immediate two-hop path does not include register memory accumulated in earlier layers",
            "value/output projections, residuals, and MLP transformations are not represented",
        ],
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    summary = json.loads((args.analysis_dir / "summary.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Register-path coverage analysis requires CUDA")
    model = load_model(args.checkpoint, device)
    for sequence, paths in summary["frame_selections"].items():
        image_tensor = load_and_preprocess_images(
            paths, mode=args.resize_mode, image_resolution=args.image_resolution
        )
        display_images = image_tensor.permute(0, 2, 3, 1).numpy()
        images = image_tensor.to(device)
        grid = (
            images.shape[-2] // model.aggregator.patch_size,
            images.shape[-1] // model.aggregator.patch_size,
        )
        with RegisterPathCoverageCollector(model, len(paths), args.query_chunk) as collector:
            with torch.inference_mode():
                predictions = model(images)
        save_results(
            args.analysis_dir / sequence,
            collector,
            display_images,
            grid,
            args.uncovered_links,
        )
        print(f"Saved register-path coverage for {sequence}")
        del images, image_tensor, predictions
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
