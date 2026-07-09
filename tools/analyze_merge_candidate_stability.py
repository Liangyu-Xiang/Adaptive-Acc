#!/usr/bin/env python3
"""Measure how stable FastVGGT merge-source choices are across random seeds.

The model forward is run without global merging. At selected global inter-frame
blocks, this script recomputes the qkv merge metric and replays the FastVGGT
bipartite source/destination selection for many random seeds. It records how
often each token is eligible as a source and how often it is actually selected
for merging.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_token_evolution import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images


DEFAULT_METADATA = (
    REPO_ROOT
    / "outputs"
    / "global_attention_matrices_10f_3seq_special_first"
    / "7scenes"
    / "metadata.json"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"


@dataclass
class MergeReplayResult:
    candidate_counts: np.ndarray
    selected_counts: np.ndarray
    similarity_sums: np.ndarray
    selected_dest_counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "merge_candidate_stability")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 3, 4])
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--merge-ratio", type=float, default=0.9)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=25)
    return parser.parse_args()


def normalize_metric(metric: torch.Tensor) -> torch.Tensor:
    return metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def build_source_destination_indices(
    *,
    num_tokens_total: int,
    tokens_per_img: int,
    num_special_tokens: int,
    patch_h: int,
    patch_w: int,
    seed: int,
    device: torch.device,
    sx: int = 2,
    sy: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_imgs = num_tokens_total // tokens_per_img
    if tokens_per_img * num_imgs != num_tokens_total:
        raise ValueError("Token count does not match frame layout")

    idx_buffer_seq = torch.zeros(num_tokens_total, device=device, dtype=torch.int64)
    idx_buffer_seq[:tokens_per_img] = -1

    if num_imgs > 1:
        special_indices = torch.arange(1, num_imgs, device=device) * tokens_per_img
        special_indices = special_indices[:, None] + torch.arange(num_special_tokens, device=device)
        idx_buffer_seq[special_indices.flatten()] = -1

        hsy, wsx = patch_h // sy, patch_w // sx
        effective_h = min(hsy * sy, patch_h)
        effective_w = min(wsx * sx, patch_w)
        effective_grid_size = effective_h * effective_w

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        all_rand_idx = torch.randint(
            sy * sx,
            size=(num_imgs - 1, hsy, wsx),
            device=device,
            generator=generator,
        )
        scatter_src = -torch.ones(num_imgs - 1, hsy, wsx, device=device, dtype=torch.int64)
        idx_buffer_batch = torch.zeros(
            num_imgs - 1,
            hsy,
            wsx,
            sy * sx,
            device=device,
            dtype=torch.int64,
        )
        idx_buffer_batch.scatter_(dim=3, index=all_rand_idx.unsqueeze(-1), src=scatter_src.unsqueeze(-1))
        idx_buffer_batch = (
            idx_buffer_batch.view(num_imgs - 1, hsy, wsx, sy, sx)
            .transpose(2, 3)
            .reshape(num_imgs - 1, hsy * sy, wsx * sx)
        )

        for offset in range(num_imgs - 1):
            img_idx = offset + 1
            grid_start = img_idx * tokens_per_img + num_special_tokens
            flat_view = idx_buffer_batch[offset, :effective_h, :effective_w].flatten()
            idx_buffer_seq[grid_start : grid_start + effective_grid_size] = flat_view

    rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
    num_dst = int((idx_buffer_seq == -1).sum())
    a_idx = rand_idx[:, num_dst:, :]
    b_idx = rand_idx[:, :num_dst, :]
    src_indices = a_idx[0, :, 0]
    dst_indices = b_idx[0, :, 0]
    return src_indices, dst_indices, idx_buffer_seq


@torch.inference_mode()
def replay_one_seed(
    metric: torch.Tensor,
    *,
    tokens_per_img: int,
    num_special_tokens: int,
    patch_h: int,
    patch_w: int,
    merge_ratio: float,
    seed: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if metric.shape[0] != 1:
        raise ValueError("This analysis currently expects batch size 1")
    device = metric.device
    _, num_tokens_total, channels = metric.shape
    src_indices, dst_indices, _ = build_source_destination_indices(
        num_tokens_total=num_tokens_total,
        tokens_per_img=tokens_per_img,
        num_special_tokens=num_special_tokens,
        patch_h=patch_h,
        patch_w=patch_w,
        seed=seed,
        device=device,
    )

    num_protected = max(1, int(num_tokens_total * 0.1))
    step = max(1, num_tokens_total // num_protected)
    protected_indices = torch.arange(0, num_tokens_total, step, device=device)[:num_protected]

    src = metric[:, src_indices, :]
    dst = metric[:, dst_indices, :]
    dst_t = dst.transpose(-1, -2).to(torch.bfloat16)
    src_bf16 = src.to(torch.bfloat16)
    node_max = torch.empty(src.shape[1], device=device, dtype=metric.dtype)
    node_idx = torch.empty(src.shape[1], device=device, dtype=torch.long)

    for start in range(0, src.shape[1], chunk_size):
        end = min(start + chunk_size, src.shape[1])
        scores = torch.bmm(src_bf16[:, start:end, :], dst_t)
        values, indices = torch.max(scores, dim=2)
        node_max[start:end] = values[0].to(metric.dtype)
        node_idx[start:end] = indices[0]

    edge_idx = node_max.argsort(descending=True)
    protected_mask_src = torch.isin(src_indices, protected_indices)
    valid_edges = edge_idx[~protected_mask_src[edge_idx]]
    r = min(int(num_tokens_total * merge_ratio), valid_edges.shape[0])
    selected_src_local = valid_edges[:r]
    selected_src_indices = src_indices[selected_src_local]
    selected_dst_indices = dst_indices[node_idx[selected_src_local]]
    selected_similarity = node_max[selected_src_local]
    return src_indices, selected_src_indices, selected_dst_indices, selected_similarity


def token_label(index: int, tokens_per_img: int, patch_start: int, patch_w: int) -> dict[str, int | str]:
    frame = index // tokens_per_img
    within = index % tokens_per_img
    if within == 0:
        return {"frame": frame, "kind": "camera", "within": within, "patch_row": -1, "patch_col": -1}
    if within < patch_start:
        return {"frame": frame, "kind": "register", "within": within, "patch_row": -1, "patch_col": -1}
    patch = within - patch_start
    return {
        "frame": frame,
        "kind": "patch",
        "within": within,
        "patch_row": patch // patch_w,
        "patch_col": patch % patch_w,
    }


def replay_layer(
    metric: torch.Tensor,
    *,
    tokens_per_img: int,
    patch_start: int,
    patch_h: int,
    patch_w: int,
    merge_ratio: float,
    seeds: list[int],
    chunk_size: int,
) -> MergeReplayResult:
    metric = normalize_metric(metric.float())
    num_tokens_total = metric.shape[1]
    candidate_counts = np.zeros(num_tokens_total, dtype=np.int32)
    selected_counts = np.zeros(num_tokens_total, dtype=np.int32)
    selected_dest_counts = np.zeros(num_tokens_total, dtype=np.int32)
    similarity_sums = np.zeros(num_tokens_total, dtype=np.float64)

    for seed in seeds:
        src_indices, selected_src, selected_dst, selected_similarity = replay_one_seed(
            metric,
            tokens_per_img=tokens_per_img,
            num_special_tokens=patch_start,
            patch_h=patch_h,
            patch_w=patch_w,
            merge_ratio=merge_ratio,
            seed=seed,
            chunk_size=chunk_size,
        )
        src_np = src_indices.detach().cpu().numpy()
        selected_np = selected_src.detach().cpu().numpy()
        dst_np = selected_dst.detach().cpu().numpy()
        sim_np = selected_similarity.detach().cpu().numpy().astype(np.float64)
        candidate_counts[src_np] += 1
        selected_counts[selected_np] += 1
        selected_dest_counts[dst_np] += 1
        similarity_sums[selected_np] += sim_np

    return MergeReplayResult(candidate_counts, selected_counts, similarity_sums, selected_dest_counts)


class MergeMetricCollector:
    def __init__(self, model, layers: set[int]) -> None:
        self.aggregator = model.aggregator
        self.layers = layers
        self.metrics: dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        for layer in self.layers:
            self.handles.append(
                self.aggregator.inter_frame_blocks[layer].register_forward_pre_hook(self._hook(layer))
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer: int):
        def hook(block, inputs):
            x = inputs[0]
            normalized = block.norm1(x)
            self.metrics[layer] = block.attn.qkv(normalized).detach()

        return hook


def write_top_csv(
    path: Path,
    result: MergeReplayResult,
    *,
    sequence: str,
    layer: int,
    tokens_per_img: int,
    patch_start: int,
    patch_w: int,
    top_k: int,
) -> list[dict[str, object]]:
    candidate = result.candidate_counts
    selected = result.selected_counts
    with np.errstate(divide="ignore", invalid="ignore"):
        conditional = np.divide(selected, candidate, out=np.zeros_like(selected, dtype=np.float64), where=candidate > 0)
    trial_count = max(1, int(candidate.max()))
    all_rate = selected / trial_count
    # np.lexsort uses the last key as primary. Negative values give descending
    # order for conditional rate, then selected trials, then candidate trials.
    order = np.lexsort((-candidate, -selected, -conditional))
    rows = []
    for rank, index in enumerate(order[:top_k], start=1):
        label = token_label(int(index), tokens_per_img, patch_start, patch_w)
        mean_similarity = (
            result.similarity_sums[index] / selected[index] if selected[index] > 0 else 0.0
        )
        rows.append(
            {
                "sequence": sequence,
                "layer": layer,
                "rank": rank,
                "token_index": int(index),
                **label,
                "candidate_trials": int(candidate[index]),
                "selected_trials": int(selected[index]),
                "selected_rate_given_candidate": float(conditional[index]),
                "selected_rate_vs_trials": float(all_rate[index]),
                "mean_selected_similarity": float(mean_similarity),
                "selected_as_destination": int(result.selected_dest_counts[index]),
            }
        )
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_patch_heatmap(
    path: Path,
    values: np.ndarray,
    *,
    title: str,
    num_frames: int,
    tokens_per_img: int,
    patch_start: int,
    patch_h: int,
    patch_w: int,
) -> None:
    fig, axes = plt.subplots(1, num_frames, figsize=(max(12, 2.4 * num_frames), 2.8), squeeze=False)
    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    for frame in range(num_frames):
        start = frame * tokens_per_img + patch_start
        grid = values[start : start + patch_h * patch_w].reshape(patch_h, patch_w)
        axis = axes[0, frame]
        im = axis.imshow(grid, cmap="magma", vmin=0.0, vmax=max(vmax, 1e-12))
        axis.set_title(f"F{frame}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize_distribution(result: MergeReplayResult) -> dict[str, float]:
    candidate = result.candidate_counts
    selected = result.selected_counts
    mask = candidate > 0
    conditional = np.divide(selected, candidate, out=np.zeros_like(selected, dtype=np.float64), where=mask)
    active = conditional[mask]
    return {
        "candidate_tokens": int(mask.sum()),
        "selected_tokens": int((selected > 0).sum()),
        "mean_conditional_rate": float(active.mean()) if active.size else 0.0,
        "std_conditional_rate": float(active.std()) if active.size else 0.0,
        "p95_conditional_rate": float(np.percentile(active, 95)) if active.size else 0.0,
        "p99_conditional_rate": float(np.percentile(active, 99)) if active.size else 0.0,
        "max_conditional_rate": float(active.max()) if active.size else 0.0,
        "top1_selected_trials": int(selected.max()) if selected.size else 0,
        "top10_selected_share": float(np.sort(selected)[-10:].sum() / max(1, selected.sum())),
        "top50_selected_share": float(np.sort(selected)[-50:].sum() / max(1, selected.sum())),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.merge_ratio <= 1.0:
        raise ValueError("--merge-ratio must be in [0, 1]")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this analysis")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    image_paths = metadata["image_paths"]
    num_frames = int(metadata["num_frames"])
    patch_start = int(metadata["patch_token_start"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_csv = args.output_dir / "top_merge_source_tokens.csv"
    if top_csv.exists():
        top_csv.unlink()

    model = load_model(args.checkpoint, device)
    layers = set(args.layers)
    seeds = list(range(args.seed_start, args.seed_start + args.trials))
    summary_rows = []

    for sequence, paths in image_paths.items():
        print(f"Collecting merge metrics for {sequence}", flush=True)
        images = load_and_preprocess_images(
            paths,
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        ).to(device)

        with MergeMetricCollector(model, layers) as collector:
            with torch.inference_mode():
                _ = model(images)

        for layer in args.layers:
            if layer not in collector.metrics:
                print(f"Skipping layer {layer}: no metric captured", flush=True)
                continue
            metric = collector.metrics[layer]
            tokens_per_img = metric.shape[1] // num_frames
            patch_count = tokens_per_img - patch_start
            patch_h = images.shape[-2] // model.aggregator.patch_size
            patch_w = images.shape[-1] // model.aggregator.patch_size
            if patch_h * patch_w != patch_count:
                raise ValueError(
                    f"Patch layout mismatch: {patch_h}x{patch_w} != {patch_count}"
                )

            print(f"  Replaying layer {layer:02d} for {args.trials} seeds", flush=True)
            result = replay_layer(
                metric,
                tokens_per_img=tokens_per_img,
                patch_start=patch_start,
                patch_h=patch_h,
                patch_w=patch_w,
                merge_ratio=args.merge_ratio,
                seeds=seeds,
                chunk_size=args.chunk_size,
            )

            safe_sequence = sequence.replace("/", "_")
            np.savez_compressed(
                args.output_dir / f"{safe_sequence}_L{layer:02d}_merge_stability.npz",
                candidate_counts=result.candidate_counts,
                selected_counts=result.selected_counts,
                selected_dest_counts=result.selected_dest_counts,
                similarity_sums=result.similarity_sums,
                tokens_per_img=np.asarray(tokens_per_img),
                patch_token_start=np.asarray(patch_start),
                patch_h=np.asarray(patch_h),
                patch_w=np.asarray(patch_w),
                seeds=np.asarray(seeds),
                merge_ratio=np.asarray(args.merge_ratio),
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                conditional = np.divide(
                    result.selected_counts,
                    result.candidate_counts,
                    out=np.zeros_like(result.selected_counts, dtype=np.float64),
                    where=result.candidate_counts > 0,
                )
            plot_patch_heatmap(
                args.output_dir / f"{safe_sequence}_L{layer:02d}_selected_rate_given_candidate.png",
                conditional,
                title=f"{sequence} L{layer:02d}: selected / candidate trials",
                num_frames=num_frames,
                tokens_per_img=tokens_per_img,
                patch_start=patch_start,
                patch_h=patch_h,
                patch_w=patch_w,
            )
            top_rows = write_top_csv(
                top_csv,
                result,
                sequence=sequence,
                layer=layer,
                tokens_per_img=tokens_per_img,
                patch_start=patch_start,
                patch_w=patch_w,
                top_k=args.top_k,
            )
            summary = summarize_distribution(result)
            summary_rows.append({"sequence": sequence, "layer": layer, **summary})
            print(
                "    "
                f"max conditional={summary['max_conditional_rate']:.3f}, "
                f"p99={summary['p99_conditional_rate']:.3f}, "
                f"top token={top_rows[0]['token_index']} "
                f"rate={top_rows[0]['selected_rate_given_candidate']:.3f}",
                flush=True,
            )

        del images
        torch.cuda.empty_cache()

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    run_metadata = {
        "metadata": str(args.metadata),
        "checkpoint": str(args.checkpoint),
        "layers": args.layers,
        "trials": args.trials,
        "seeds": seeds,
        "merge_ratio": args.merge_ratio,
        "image_resolution": args.image_resolution,
        "resize_mode": args.resize_mode,
        "num_frames": num_frames,
        "patch_token_start": patch_start,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved merge stability analysis to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
