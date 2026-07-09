#!/usr/bin/env python3
"""Analyze register-mediated proxy paths against full global attention.

The collector is non-invasive: it attaches forward pre-hooks to selected
global inter-frame blocks, recomputes Q/K attention probabilities in chunks,
and leaves the model outputs unchanged. Full VGGT-Omega token merging is
disabled because the experiment needs uncompressed full global attention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
ANCHOR_STRATEGIES = ("fixed_grid", "intra_only", "proxy", "proxy_intra", "oracle", "random")
EVAL_STRATEGIES = (
    "full_global_attention",
    "fixed_grid_kv",
    "intra_only_kv",
    "proxy_kv",
    "proxy_intra_kv",
    "oracle_kv",
    "random_kv",
)


@dataclass
class Sample:
    sample_id: str
    image_paths: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--layers", default="0-23")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--save_attention_stats", action="store_true")
    parser.add_argument("--save_visualization", action="store_true")
    parser.add_argument("--anchor_ratio", type=float, default=0.2)
    parser.add_argument("--anchor_total", type=int, default=None)
    parser.add_argument("--topk_list", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--eval_anchor_strategies", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--num_frames", type=int, default=0, help="0 keeps all image files; videos default to 4.")
    parser.add_argument("--video_sample_fps", type=float, default=1.0)
    parser.add_argument("--query_chunk", type=int, default=64)
    parser.add_argument(
        "--query_sample_total",
        type=int,
        default=0,
        help="If >0, estimate direct scores from this many evenly spaced patch queries.",
    )
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--normalization", choices=("sum", "minmax"), default="sum")
    parser.add_argument("--anchor_min_per_frame", type=int, default=1)
    parser.add_argument("--budget_tau", type=float, default=1.0)
    parser.add_argument("--budget_uniform_mix", type=float, default=0.1)
    parser.add_argument("--timing_repeats", type=int, default=3)
    return parser.parse_args()


def parse_layer_spec(spec: str, depth: int) -> list[int]:
    selected: set[int] = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Invalid layer range {item!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    invalid = sorted(layer for layer in selected if layer < 0 or layer >= depth)
    if invalid:
        raise ValueError(f"Layer indices out of range 0..{depth - 1}: {invalid}")
    return sorted(selected)


def parse_topk_list(spec: str) -> list[float]:
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise ValueError("--topk_list entries must be in (0, 1]")
    return values


def image_files_in_dir(path: Path) -> list[Path]:
    files = [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=lambda item: item.name)


def evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if count <= 0 or count >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[int(index)] for index in indices]


def extract_video_frames(video_path: Path, output_dir: Path, num_frames: int, fps: float) -> list[Path]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video input requires opencv-python/cv2") from exc

    target = output_dir / "_video_frames" / video_path.stem
    target.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if num_frames <= 0:
        num_frames = 4
    if total > 0:
        frame_indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=np.int64).tolist()
    else:
        interval = max(int(round((native_fps if native_fps > 0 else 1.0) / max(fps, 0.1))), 1)
        frame_indices = list(range(0, num_frames * interval, interval))

    paths: list[Path] = []
    for output_index, frame_index in enumerate(frame_indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        out_path = target / f"frame_{output_index:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        paths.append(out_path)
    capture.release()
    if len(paths) < 2:
        raise RuntimeError(f"Extracted only {len(paths)} frames from {video_path}")
    return paths


def discover_samples(args: argparse.Namespace) -> list[Sample]:
    path = args.input_path
    if not path.exists():
        raise FileNotFoundError(path)
    samples: list[Sample] = []
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        samples.append(Sample(path.stem, extract_video_frames(path, args.output_dir, args.num_frames, args.video_sample_fps)))
    elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        samples.append(Sample(path.stem, [path]))
    elif path.is_file():
        rows = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        samples.append(Sample(path.stem, rows))
    else:
        direct_images = image_files_in_dir(path)
        if direct_images:
            samples.append(Sample(path.name, evenly_spaced(direct_images, args.num_frames)))
        else:
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                if not child.is_dir():
                    continue
                images = image_files_in_dir(child)
                if images:
                    samples.append(Sample(child.name, evenly_spaced(images, args.num_frames)))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    for sample in samples:
        if len(sample.image_paths) < 2:
            raise ValueError(f"{sample.sample_id}: need at least 2 images, got {len(sample.image_paths)}")
    if not samples:
        raise ValueError(f"No image/video samples found in {path}")
    return samples


def load_model(checkpoint: Path, device: torch.device) -> VGGTOmega:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = VGGTOmega(global_merging=False, merging=None, merge_ratio=0.0)
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        state = torch.load(checkpoint, mmap=True, **kwargs)
    except TypeError:
        state = torch.load(checkpoint, **kwargs)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device).eval()


def qkv_from_block(block, x: torch.Tensor):
    normalized = block.norm1(x)
    batch, tokens, hidden = normalized.shape
    heads = block.attn.num_heads
    qkv = block.attn.qkv(normalized).reshape(batch, tokens, 3, heads, hidden // heads)
    q, k, v = torch.unbind(qkv, dim=2)
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    if block.attn.use_qk_norm:
        q = block.attn.q_norm(q)
        k = block.attn.k_norm(k)
    return q, k, v


def patch_indices(num_frames: int, tokens_per_frame: int, patch_start: int, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(
                frame * tokens_per_frame + patch_start,
                (frame + 1) * tokens_per_frame,
                device=device,
            )
            for frame in range(num_frames)
        ]
    )


def normalize_scores(scores: torch.Tensor, mode: str) -> torch.Tensor:
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if mode == "sum":
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    minimum = scores.amin(dim=-1, keepdim=True)
    maximum = scores.amax(dim=-1, keepdim=True)
    return (scores - minimum) / (maximum - minimum).clamp_min(1.0e-12)


def compute_proxy_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    num_frames: int,
    tokens_per_frame: int,
    patch_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, _, head_dim = q.shape
    patch_count = tokens_per_frame - patch_start
    q_frames = q.reshape(batch, heads, num_frames, tokens_per_frame, head_dim)
    k_frames = k.reshape(batch, heads, num_frames, tokens_per_frame, head_dim)
    register_start = 1 if patch_start > 1 else 0
    reg_q = q_frames[:, :, :, register_start:patch_start]
    reg_k = k_frames[:, :, :, register_start:patch_start]
    if reg_q.shape[-2] == 0:
        reg_q = q_frames[:, :, :, :1]
        reg_k = k_frames[:, :, :, :1]
    reg_count = reg_q.shape[-2]
    patch_k = k_frames[:, :, :, patch_start:]

    reg_q_flat = reg_q.reshape(batch, heads, num_frames * reg_count, head_dim).float()
    reg_k_flat = reg_k.reshape(batch, heads, num_frames * reg_count, head_dim).float()
    reg_prob = (torch.matmul(reg_q_flat, reg_k_flat.transpose(-2, -1)) * scale).softmax(dim=-1)
    frame_index = torch.arange(num_frames, device=q.device).repeat_interleave(reg_count)
    cross_mask = frame_index[:, None] != frame_index[None, :]
    reg_prob_cross = reg_prob * cross_mask.to(dtype=reg_prob.dtype)
    reg_recv = reg_prob_cross.sum(dim=-2).reshape(batch, heads, num_frames, reg_count).mean(dim=1)

    graph = torch.empty(batch, num_frames, num_frames, device=q.device, dtype=torch.float32)
    for source in range(num_frames):
        source_slice = slice(source * reg_count, (source + 1) * reg_count)
        for target in range(num_frames):
            target_slice = slice(target * reg_count, (target + 1) * reg_count)
            graph[:, source, target] = reg_prob[:, :, source_slice, target_slice].mean(dim=(1, 2, 3))

    reg_to_patch = torch.empty(batch, num_frames, reg_count, patch_count, device=q.device, dtype=torch.float32)
    for frame in range(num_frames):
        logits = torch.matmul(reg_q[:, :, frame].float(), patch_k[:, :, frame].float().transpose(-2, -1)) * scale
        reg_to_patch[:, frame] = logits.softmax(dim=-1).mean(dim=1)
    proxy = (reg_recv[:, :, :, None] * reg_to_patch).sum(dim=2)
    return torch.nan_to_num(proxy, nan=0.0, posinf=0.0, neginf=0.0), graph


def compute_intra_scores_and_metrics(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    num_frames: int,
    tokens_per_frame: int,
    patch_start: int,
    grid: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    batch, heads, _, head_dim = q.shape
    patch_count = tokens_per_frame - patch_start
    q_frames = q.reshape(batch, heads, num_frames, tokens_per_frame, head_dim)
    k_frames = k.reshape(batch, heads, num_frames, tokens_per_frame, head_dim)
    patch_q = q_frames[:, :, :, patch_start:]
    patch_k = k_frames[:, :, :, patch_start:]
    col_mass = torch.empty(batch, num_frames, patch_count, device=q.device, dtype=torch.float32)
    entropy = torch.empty(batch, num_frames, device=q.device, dtype=torch.float32)
    diag_ratio = torch.empty(batch, num_frames, device=q.device, dtype=torch.float32)
    nonlocal_ratio = torch.empty(batch, num_frames, device=q.device, dtype=torch.float32)

    rows, cols = grid
    yy, xx = torch.meshgrid(torch.arange(rows, device=q.device), torch.arange(cols, device=q.device), indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1).float()
    distances = torch.cdist(coords, coords, p=2)
    nonlocal_mask = distances > max(rows, cols) * 0.25

    for frame in range(num_frames):
        logits = torch.matmul(patch_q[:, :, frame].float(), patch_k[:, :, frame].float().transpose(-2, -1)) * scale
        prob = logits.softmax(dim=-1)
        col_mass[:, frame] = prob.sum(dim=-2).mean(dim=1)
        entropy[:, frame] = (-(prob * prob.clamp_min(1.0e-12).log()).sum(dim=-1)).mean(dim=(1, 2))
        diag_ratio[:, frame] = prob.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean(dim=1) / patch_count
        nonlocal_ratio[:, frame] = (prob * nonlocal_mask.to(dtype=prob.dtype)).sum(dim=-1).mean(dim=(1, 2))
    metrics = {
        "entropy": entropy.detach().cpu().numpy(),
        "diag_ratio": diag_ratio.detach().cpu().numpy(),
        "nonlocal_ratio": nonlocal_ratio.detach().cpu().numpy(),
    }
    return torch.nan_to_num(col_mass, nan=0.0, posinf=0.0, neginf=0.0), metrics


def compute_direct_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    num_frames: int,
    tokens_per_frame: int,
    patch_start: int,
    query_chunk: int,
    query_sample_total: int,
) -> tuple[torch.Tensor, np.ndarray]:
    batch, heads, _, _ = q.shape
    patch_count = tokens_per_frame - patch_start
    indices = patch_indices(num_frames, tokens_per_frame, patch_start, q.device)
    query_frames_all = torch.arange(num_frames, device=q.device).repeat_interleave(patch_count)
    key_frames = query_frames_all
    if query_sample_total > 0 and query_sample_total < indices.numel():
        sample_positions = torch.linspace(0, indices.numel() - 1, query_sample_total, device=q.device).round().long()
        query_indices_all = indices[sample_positions]
        query_frames = query_frames_all[sample_positions]
    else:
        query_indices_all = indices
        query_frames = query_frames_all
    direct = torch.zeros(batch, heads, num_frames * patch_count, device=q.device, dtype=torch.float32)
    cross_mass = torch.zeros(batch, heads, device=q.device, dtype=torch.float32)
    key_t = k.float().transpose(-2, -1)
    query_chunk = max(1, min(query_chunk, query_indices_all.numel()))
    for start in range(0, query_indices_all.numel(), query_chunk):
        end = min(start + query_chunk, query_indices_all.numel())
        q_indices = query_indices_all[start:end]
        logits = torch.matmul(q[:, :, q_indices].float(), key_t) * scale
        prob = logits.softmax(dim=-1)
        patch_prob = prob[..., indices]
        mask = query_frames[start:end, None] != key_frames[None, :]
        cross_patch_prob = patch_prob * mask.to(dtype=patch_prob.dtype)
        direct += cross_patch_prob.sum(dim=-2)
        cross_mass += cross_patch_prob.sum(dim=(-2, -1))
    direct = direct.mean(dim=1).reshape(batch, num_frames, patch_count)
    cross_mass_np = (cross_mass / max(query_indices_all.numel(), 1)).mean(dim=1).detach().cpu().numpy()
    return torch.nan_to_num(direct, nan=0.0, posinf=0.0, neginf=0.0), cross_mass_np


class RegisterProxyCollector:
    def __init__(
        self,
        model: VGGTOmega,
        num_frames: int,
        layers: Iterable[int],
        query_chunk: int,
        query_sample_total: int,
        alpha: float,
        beta: float,
        normalization: str,
        grid: tuple[int, int],
    ) -> None:
        self.model = model
        self.aggregator = model.aggregator
        self.num_frames = num_frames
        self.layers = set(layers)
        self.query_chunk = query_chunk
        self.query_sample_total = query_sample_total
        self.alpha = alpha
        self.beta = beta
        self.normalization = normalization
        self.grid = grid
        self.results: dict[int, dict[str, object]] = {}
        self.handles = []

    def __enter__(self):
        for layer in self.layers:
            self.handles.append(self.aggregator.inter_frame_blocks[layer].register_forward_pre_hook(self._hook(layer)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer: int):
        def hook(block, inputs):
            x = inputs[0]
            batch, total_tokens, _ = x.shape
            tokens_per_frame = total_tokens // self.num_frames
            patch_start = self.aggregator.patch_token_start
            q, k, _ = qkv_from_block(block, x)
            direct, cross_mass = compute_direct_scores(
                q,
                k,
                block.attn.scale,
                self.num_frames,
                tokens_per_frame,
                patch_start,
                self.query_chunk,
                self.query_sample_total,
            )
            proxy, graph = compute_proxy_scores(q, k, block.attn.scale, self.num_frames, tokens_per_frame, patch_start)
            intra, intra_metrics = compute_intra_scores_and_metrics(
                q, k, block.attn.scale, self.num_frames, tokens_per_frame, patch_start, self.grid
            )
            proxy_intra = self.alpha * normalize_scores(proxy, self.normalization) + self.beta * normalize_scores(
                intra, self.normalization
            )
            self.results[layer] = {
                "direct": direct.detach().cpu(),
                "proxy": proxy.detach().cpu(),
                "intra": intra.detach().cpu(),
                "proxy_intra": proxy_intra.detach().cpu(),
                "frame_graph": graph.detach().cpu(),
                "cross_mass": cross_mass,
                "intra_metrics": intra_metrics,
                "tokens_per_frame": tokens_per_frame,
                "patch_start": patch_start,
            }
            del q, k, direct, proxy, intra, proxy_intra, graph

        return hook


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    x = x[finite].astype(np.float64)
    y = y[finite].astype(np.float64)
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    return pearson(rankdata(x[finite]), rankdata(y[finite]))


def topk_overlap(reference: np.ndarray, candidate: np.ndarray, ratio: float) -> float:
    count = max(1, int(math.ceil(reference.size * ratio)))
    ref_idx = np.argpartition(reference, -count)[-count:]
    cand_idx = np.argpartition(candidate, -count)[-count:]
    return float(len(set(ref_idx.tolist()) & set(cand_idx.tolist())) / count)


def ndcg_at_k(reference: np.ndarray, candidate: np.ndarray, ratio: float) -> float:
    count = max(1, int(math.ceil(reference.size * ratio)))
    order = np.argsort(candidate)[::-1][:count]
    gains = reference[order]
    discounts = 1.0 / np.log2(np.arange(2, count + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(reference)[::-1][:count]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0.0 else float("nan")


def distribution_concentration(values: np.ndarray) -> dict[str, float]:
    values = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return {"entropy": float("nan"), "gini": float("nan"), "top10_mass": float("nan"), "top20_mass": float("nan")}
    prob = values.reshape(-1) / total
    entropy = float(-(prob * np.log(prob + 1.0e-12)).sum() / max(math.log(len(prob)), 1.0e-12))
    sorted_values = np.sort(values.reshape(-1))
    n = len(sorted_values)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(sorted_values) / (n * sorted_values.sum() + 1.0e-12))
    top10 = max(1, int(math.ceil(0.10 * n)))
    top20 = max(1, int(math.ceil(0.20 * n)))
    return {
        "entropy": entropy,
        "gini": gini,
        "top10_mass": float(np.sort(prob)[-top10:].sum()),
        "top20_mass": float(np.sort(prob)[-top20:].sum()),
    }


def layer_group(layer: int, selected_layers: list[int]) -> str:
    position = selected_layers.index(layer)
    third = max(1, math.ceil(len(selected_layers) / 3))
    if position < third:
        return "early"
    if position < 2 * third:
        return "middle"
    return "late"


def build_stage1_rows(
    sample_id: str,
    results: dict[int, dict[str, object]],
    selected_layers: list[int],
    topk_list: list[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer in selected_layers:
        result = results[layer]
        direct = result["direct"].numpy()
        score_map = {
            "proxy": result["proxy"].numpy(),
            "intra": result["intra"].numpy(),
            "proxy_intra": result["proxy_intra"].numpy(),
        }
        for batch_idx in range(direct.shape[0]):
            row: dict[str, object] = {
                "sample_id": sample_id if direct.shape[0] == 1 else f"{sample_id}_b{batch_idx}",
                "layer_id": layer,
                "layer_group": layer_group(layer, selected_layers),
                "cross_frame_mass": float(result["cross_mass"][batch_idx]),
            }
            direct_flat = direct[batch_idx].reshape(-1)
            for name, scores in score_map.items():
                flat = scores[batch_idx].reshape(-1)
                row[f"spearman_{name}"] = spearman(direct_flat, flat)
                row[f"pearson_{name}"] = pearson(direct_flat, flat)
                for ratio in topk_list:
                    suffix = f"{ratio:g}"
                    row[f"topk_overlap_{name}_{suffix}"] = topk_overlap(direct_flat, flat, ratio)
                    row[f"ndcg_{name}_{suffix}"] = ndcg_at_k(direct_flat, flat, ratio)
            direct_conc = distribution_concentration(direct_flat)
            intra_conc = distribution_concentration(score_map["intra"][batch_idx].reshape(-1))
            row.update({f"direct_{key}": value for key, value in direct_conc.items()})
            row.update({f"intra_col_{key}": value for key, value in intra_conc.items()})
            intra_metrics = result["intra_metrics"]
            row["intra_attention_entropy"] = float(np.nanmean(intra_metrics["entropy"][batch_idx]))
            row["intra_diagonal_ratio"] = float(np.nanmean(intra_metrics["diag_ratio"][batch_idx]))
            row["intra_nonlocal_ratio"] = float(np.nanmean(intra_metrics["nonlocal_ratio"][batch_idx]))
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, object]], selected_layers: list[int]) -> dict[str, object]:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"sample_id", "layer_id", "layer_group"} and isinstance(value, (int, float, np.floating))
    ]
    summary: dict[str, object] = {"num_rows": len(rows), "layers": selected_layers, "by_layer": {}, "by_group": {}}
    for layer in selected_layers:
        layer_rows = [row for row in rows if int(row["layer_id"]) == layer]
        summary["by_layer"][str(layer)] = {
            key: {
                "mean": float(np.nanmean([row[key] for row in layer_rows])),
                "std": float(np.nanstd([row[key] for row in layer_rows])),
            }
            for key in numeric_keys
        }
    for group in ("early", "middle", "late"):
        group_rows = [row for row in rows if row["layer_group"] == group]
        if not group_rows:
            continue
        summary["by_group"][group] = {
            key: {
                "mean": float(np.nanmean([row[key] for row in group_rows])),
                "std": float(np.nanstd([row[key] for row in group_rows])),
            }
            for key in numeric_keys
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def anchor_budget(total_patches: int, anchor_ratio: float, anchor_total: int | None) -> int:
    if anchor_total is not None:
        return max(0, min(total_patches, int(anchor_total)))
    return max(1, min(total_patches, int(math.ceil(anchor_ratio * total_patches))))


def allocate_counts(
    frame_scores: np.ndarray,
    budget: int,
    patch_count: int,
    min_per_frame: int,
    adaptive: bool,
    tau: float,
    uniform_mix: float,
) -> np.ndarray:
    frames = len(frame_scores)
    min_per_frame = min(max(min_per_frame, 0), patch_count)
    budget = max(min(budget, frames * patch_count), frames * min_per_frame)
    counts = np.full(frames, min_per_frame, dtype=np.int64)
    remaining = budget - int(counts.sum())
    if remaining <= 0:
        return counts
    capacity = np.full(frames, patch_count - min_per_frame, dtype=np.int64)
    if adaptive:
        scores = np.nan_to_num(frame_scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.clip(scores, 0.0, None)
        if scores.sum() <= 0.0:
            probs = np.full(frames, 1.0 / frames)
        else:
            logits = np.log(scores + 1.0e-12) / tau
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
        probs = (1.0 - uniform_mix) * probs + uniform_mix / frames
        raw = probs * remaining
    else:
        raw = np.full(frames, remaining / frames)
        probs = np.full(frames, 1.0 / frames)
    extra = np.minimum(np.floor(raw).astype(np.int64), capacity)
    leftover = remaining - int(extra.sum())
    fractional = raw - np.floor(raw)
    while leftover > 0 and np.any(extra < capacity):
        available = extra < capacity
        priority = np.where(available, fractional + probs * 1.0e-6, -np.inf)
        chosen = np.argsort(priority)[-min(leftover, int(available.sum())) :]
        extra[chosen] += 1
        leftover -= len(chosen)
        fractional = probs
    return counts + extra


def fixed_grid_indices(count: int, grid: tuple[int, int]) -> np.ndarray:
    rows, cols = grid
    patch_count = rows * cols
    if count >= patch_count:
        return np.arange(patch_count, dtype=np.int64)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    grid_rows = max(1, min(rows, int(round(math.sqrt(count * rows / max(cols, 1))))))
    grid_cols = max(1, min(cols, int(math.ceil(count / grid_rows))))
    while grid_rows * grid_cols < count and grid_rows < rows:
        grid_rows += 1
    while grid_rows * grid_cols < count and grid_cols < cols:
        grid_cols += 1
    rr = np.unique(np.rint(np.linspace(0, rows - 1, grid_rows)).astype(np.int64))
    cc = np.unique(np.rint(np.linspace(0, cols - 1, grid_cols)).astype(np.int64))
    candidates = np.unique((rr[:, None] * cols + cc[None, :]).reshape(-1))
    if len(candidates) >= count:
        pick = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(np.int64)
        return candidates[pick]
    fallback = np.unique(np.rint(np.linspace(0, patch_count - 1, count)).astype(np.int64))
    combined = np.unique(np.concatenate([candidates, fallback]))
    if len(combined) >= count:
        return combined[:count]
    missing = np.setdiff1d(np.arange(patch_count, dtype=np.int64), combined, assume_unique=False)
    return np.concatenate([combined, missing[: count - len(combined)]])


def select_anchors(
    scores: dict[str, np.ndarray],
    strategy: str,
    budget: int,
    grid: tuple[int, int],
    min_per_frame: int,
    tau: float,
    uniform_mix: float,
    seed: int,
    adaptive_budget: bool | None = None,
    token_source: str | None = None,
) -> dict[str, object]:
    direct = scores["direct"]
    frames, patch_count = direct.shape
    source = token_source or strategy
    if source == "oracle":
        token_scores = scores["direct"]
    elif source == "intra_only":
        token_scores = scores["intra"]
    elif source == "proxy_intra":
        token_scores = scores["proxy_intra"]
    else:
        token_scores = scores["proxy"]
    if adaptive_budget is None:
        adaptive_budget = strategy in {"proxy", "proxy_intra", "oracle"}
    if strategy == "fixed_grid":
        frame_scores = np.ones(frames, dtype=np.float64)
        adaptive_budget = False
    elif strategy == "random":
        frame_scores = np.ones(frames, dtype=np.float64)
        adaptive_budget = False
    else:
        frame_scores = token_scores.sum(axis=-1)
    counts = allocate_counts(frame_scores, budget, patch_count, min_per_frame, adaptive_budget, tau, uniform_mix)

    rng = np.random.RandomState(seed)
    frame_indices: list[int] = []
    patch_indices_out: list[int] = []
    for frame, count in enumerate(counts):
        if count <= 0:
            continue
        if count >= patch_count:
            selected = np.arange(patch_count, dtype=np.int64)
        elif strategy == "fixed_grid" or source == "fixed_grid":
            selected = fixed_grid_indices(int(count), grid)
        elif strategy == "random" or source == "random":
            selected = rng.permutation(patch_count)[: int(count)]
        else:
            selected = np.argpartition(token_scores[frame], -int(count))[-int(count) :]
        frame_indices.extend([frame] * len(selected))
        patch_indices_out.extend(selected.astype(np.int64).tolist())
    frame_array = np.asarray(frame_indices, dtype=np.int64)
    patch_array = np.asarray(patch_indices_out, dtype=np.int64)
    flat_patch = frame_array * patch_count + patch_array
    return {
        "flat_patch_indices": flat_patch,
        "frame_indices": frame_array,
        "patch_indices": patch_array,
        "counts": counts,
        "budget": int(budget),
    }


def budget_stats(counts: np.ndarray) -> dict[str, float]:
    total = float(counts.sum())
    if total <= 0.0:
        return {"frame_budget_entropy": float("nan"), "frame_budget_gini": float("nan"), "top20_frames_budget_ratio": float("nan")}
    probs = counts / total
    entropy = float(-(probs * np.log(probs + 1.0e-12)).sum() / max(math.log(len(probs)), 1.0e-12))
    sorted_counts = np.sort(counts.astype(np.float64))
    n = len(sorted_counts)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(sorted_counts) / (n * sorted_counts.sum() + 1.0e-12))
    top_frames = max(1, int(math.ceil(0.2 * len(counts))))
    return {
        "frame_budget_entropy": entropy,
        "frame_budget_gini": gini,
        "top20_frames_budget_ratio": float(np.sort(counts)[-top_frames:].sum() / total),
    }


def save_stage2_and_ablation(
    output_dir: Path,
    sample_id: str,
    results: dict[int, dict[str, object]],
    selected_layers: list[int],
    args: argparse.Namespace,
    grid: tuple[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    anchor_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    anchor_dir = output_dir / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    for layer in selected_layers:
        result = results[layer]
        direct = result["direct"].numpy()
        batch, frames, patch_count = direct.shape
        budget = anchor_budget(frames * patch_count, args.anchor_ratio, args.anchor_total)
        for batch_idx in range(batch):
            score_dict = {
                "direct": direct[batch_idx],
                "proxy": result["proxy"].numpy()[batch_idx],
                "intra": result["intra"].numpy()[batch_idx],
                "proxy_intra": result["proxy_intra"].numpy()[batch_idx],
            }
            oracle = select_anchors(
                score_dict,
                "oracle",
                budget,
                grid,
                args.anchor_min_per_frame,
                args.budget_tau,
                args.budget_uniform_mix,
                args.seed,
            )
            oracle_set = set(oracle["flat_patch_indices"].tolist())
            for strategy in ANCHOR_STRATEGIES:
                selected = select_anchors(
                    score_dict,
                    strategy,
                    budget,
                    grid,
                    args.anchor_min_per_frame,
                    args.budget_tau,
                    args.budget_uniform_mix,
                    args.seed + layer * 1000 + batch_idx,
                )
                overlap = len(oracle_set & set(selected["flat_patch_indices"].tolist())) / max(budget, 1)
                tag = sample_id if batch == 1 else f"{sample_id}_b{batch_idx}"
                torch.save(
                    {
                        "sample_id": tag,
                        "layer_id": layer,
                        "strategy": strategy,
                        "flat_patch_indices": torch.from_numpy(selected["flat_patch_indices"]),
                        "frame_indices": torch.from_numpy(selected["frame_indices"]),
                        "patch_indices": torch.from_numpy(selected["patch_indices"]),
                        "counts": torch.from_numpy(selected["counts"]),
                    },
                    anchor_dir / f"{tag}_layer_{layer:02d}_{strategy}.pt",
                )
                budget_payload = {"sample_id": tag, "layer_id": layer, "strategy": strategy, "counts": selected["counts"].tolist()}
                (anchor_dir / f"{tag}_layer_{layer:02d}_{strategy}_budget.json").write_text(
                    json.dumps(budget_payload, indent=2) + "\n",
                    encoding="utf-8",
                )
                row = {
                    "sample_id": tag,
                    "layer_id": layer,
                    "strategy": strategy,
                    "K": budget,
                    "overlap_with_oracle": float(overlap),
                }
                row.update(budget_stats(selected["counts"]))
                anchor_rows.append(row)

            ablations = (
                ("s_proxy_only", True, "proxy"),
                ("s_intra_only", False, "intra_only"),
                ("s_proxy_plus_intra", True, "proxy_intra"),
                ("frame_budget_uniform_proxy_selection", False, "proxy"),
                ("frame_budget_adaptive_grid_selection", True, "fixed_grid"),
                ("frame_budget_adaptive_proxy_selection", True, "proxy"),
                ("oracle_direct_score", True, "oracle"),
            )
            total_tokens = frames * (result["patch_start"] + patch_count)
            full_pairs = total_tokens * total_tokens
            relative_cost = (frames * (result["patch_start"] + patch_count) * (frames * result["patch_start"] + budget)) / max(full_pairs, 1)
            for name, adaptive_budget_mode, source in ablations:
                selected = select_anchors(
                    score_dict,
                    "proxy" if source not in {"fixed_grid", "random"} else source,
                    budget,
                    grid,
                    args.anchor_min_per_frame,
                    args.budget_tau,
                    args.budget_uniform_mix,
                    args.seed + layer * 1000 + batch_idx,
                    adaptive_budget=adaptive_budget_mode,
                    token_source=source,
                )
                overlap = len(oracle_set & set(selected["flat_patch_indices"].tolist())) / max(budget, 1)
                ablation_rows.append(
                    {
                        "sample_id": sample_id if batch == 1 else f"{sample_id}_b{batch_idx}",
                        "layer_id": layer,
                        "ablation_name": name,
                        "anchor_ratio": args.anchor_ratio,
                        "topk_overlap_with_oracle": float(overlap),
                        "runtime_ms": float("nan"),
                        "relative_attention_cost": float(relative_cost),
                        "output_diff_to_full": float("nan"),
                        "task_metric": "TODO_standard_task_metric",
                    }
                )
    return anchor_rows, ablation_rows


def plot_curves(output_dir: Path, rows: list[dict[str, object]], topk_list: list[float]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted({int(row["layer_id"]) for row in rows})
    by_layer = {layer: [row for row in rows if int(row["layer_id"]) == layer] for layer in layers}

    def mean_series(key: str) -> list[float]:
        return [float(np.nanmean([row[key] for row in by_layer[layer]])) for layer in layers]

    plots = [
        ("proxy_vs_direct_spearman_by_layer.png", "spearman_proxy", "Spearman(proxy, direct)"),
        (f"topk_overlap_by_layer.png", f"topk_overlap_proxy_{topk_list[0]:g}", "Top-K overlap(proxy, direct)"),
        ("cross_frame_mass_by_layer.png", "cross_frame_mass", "Cross-frame patch mass"),
        ("intra_concentration_by_layer.png", "intra_col_top10_mass", "Intra column top-10% mass"),
        ("direct_score_concentration_by_layer.png", "direct_top10_mass", "Direct score top-10% mass"),
    ]
    for filename, key, ylabel in plots:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(layers, mean_series(key), marker="o")
        axis.set_xlabel("Layer")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=180)
        plt.close(figure)


def plot_layer_maps(
    output_dir: Path,
    sample_id: str,
    results: dict[int, dict[str, object]],
    selected_layers: list[int],
    grid: tuple[int, int],
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for layer in selected_layers:
        result = results[layer]
        for name in ("direct", "proxy"):
            values = result[name][0].numpy()
            finite = values[np.isfinite(values)]
            vmin, vmax = np.percentile(finite, (2, 98)) if finite.size else (0.0, 1.0)
            if vmax <= vmin:
                vmax = vmin + 1.0e-8
            figure, axes = plt.subplots(values.shape[0], 1, figsize=(4, 2.4 * values.shape[0]), squeeze=False)
            for frame, axis in enumerate(axes[:, 0]):
                image = axis.imshow(values[frame].reshape(grid), cmap="magma", vmin=vmin, vmax=vmax)
                axis.set_title(f"{sample_id} L{layer:02d} {name} F{frame}")
                axis.axis("off")
            figure.colorbar(image, ax=axes[:, 0].tolist(), fraction=0.04, pad=0.02)
            figure.savefig(plot_dir / f"{name}_score_heatmap_layer_{layer:02d}.png", dpi=180, bbox_inches="tight")
            plt.close(figure)

        graph = result["frame_graph"][0].numpy()
        figure, axis = plt.subplots(figsize=(4.5, 4))
        image = axis.imshow(graph, cmap="viridis")
        axis.set_xlabel("Key frame")
        axis.set_ylabel("Query frame")
        axis.set_title(f"{sample_id} L{layer:02d} register frame graph")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(plot_dir / f"frame_pair_graph_G_layer_{layer:02d}.png", dpi=180)
        plt.close(figure)

        scores = result["proxy_intra"][0].numpy()
        counts = allocate_counts(
            scores.sum(axis=-1),
            anchor_budget(scores.size, 0.2, None),
            scores.shape[-1],
            1,
            True,
            1.0,
            0.1,
        )
        figure, axis = plt.subplots(figsize=(7, 3))
        axis.bar(np.arange(len(counts)), counts)
        axis.set_xlabel("Frame")
        axis.set_ylabel("Anchor count")
        axis.set_title(f"{sample_id} L{layer:02d} proxy_intra budget")
        figure.tight_layout()
        figure.savefig(plot_dir / f"anchor_budget_by_frame_layer_{layer:02d}.png", dpi=180)
        plt.close(figure)


def save_attention_stats(output_dir: Path, sample_id: str, results: dict[int, dict[str, object]], selected_layers: list[int]) -> None:
    target = output_dir / "attention_stats"
    target.mkdir(parents=True, exist_ok=True)
    for layer in selected_layers:
        result = results[layer]
        torch.save(
            {
                "direct": result["direct"],
                "proxy": result["proxy"],
                "intra": result["intra"],
                "proxy_intra": result["proxy_intra"],
                "frame_graph": result["frame_graph"],
                "cross_mass": result["cross_mass"],
                "intra_metrics": result["intra_metrics"],
            },
            target / f"{sample_id}_layer_{layer:02d}_stats.pt",
        )


def timed_forward(model: VGGTOmega, images: torch.Tensor, device: torch.device, repeats: int):
    if device.type != "cuda":
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = model(images)
        return prediction, [(time.perf_counter() - start) * 1000.0], 0.0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    prediction = None
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.inference_mode():
            current = model(images)
        end_event.record()
        torch.cuda.synchronize(device)
        timings.append(float(start_event.elapsed_time(end_event)))
        prediction = current
    peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    assert prediction is not None
    return prediction, timings, float(peak)


def prediction_diff(prediction: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict[str, float]:
    output: dict[str, float] = {}
    if "depth" in prediction and "depth" in reference:
        output["depth_l1_to_full"] = float((prediction["depth"].float() - reference["depth"].float()).abs().mean().detach().cpu())
    else:
        output["depth_l1_to_full"] = float("nan")
    if "pose_enc" in prediction and "pose_enc" in reference:
        output["pose_l1_to_full"] = float((prediction["pose_enc"].float() - reference["pose_enc"].float()).abs().mean().detach().cpu())
    else:
        output["pose_l1_to_full"] = float("nan")
    if "camera_and_register_tokens" in prediction and "camera_and_register_tokens" in reference:
        pred = prediction["camera_and_register_tokens"].float().flatten(1)
        ref = reference["camera_and_register_tokens"].float().flatten(1)
        output["token_cosine_distance_to_full"] = float((1.0 - F.cosine_similarity(pred, ref, dim=1)).mean().detach().cpu())
    else:
        output["token_cosine_distance_to_full"] = float("nan")
    return output


def estimate_pair_count(
    model: VGGTOmega,
    selected_layers: list[int],
    images: torch.Tensor,
    anchor_ratio_value: float,
    anchor_total_value: int | None,
    compressed: bool,
) -> tuple[int, float]:
    patch_count = (images.shape[-2] // model.aggregator.patch_size) * (images.shape[-1] // model.aggregator.patch_size)
    special = model.aggregator.patch_token_start
    frames = images.shape[1] if images.ndim == 5 else images.shape[0]
    tokens_per_frame = special + patch_count
    total_tokens = frames * tokens_per_frame
    global_layers = [
        layer for layer in selected_layers if model.aggregator.inter_frame_attention_types[layer] == "global"
    ]
    full = len(global_layers) * total_tokens * total_tokens
    if not compressed:
        return full, 1.0
    budget = anchor_budget(frames * patch_count, anchor_ratio_value, anchor_total_value)
    compressed_pairs = len(global_layers) * total_tokens * (frames * special + budget)
    return compressed_pairs, compressed_pairs / max(full, 1)


def run_stage3_eval(
    model: VGGTOmega,
    images: torch.Tensor,
    sample_id: str,
    selected_layers: list[int],
    args: argparse.Namespace,
    full_reference: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    layer_spec = ",".join(str(layer) for layer in selected_layers)
    strategy_to_model = {
        "fixed_grid_kv": "fixed_grid",
        "intra_only_kv": "intra_only",
        "proxy_kv": "proxy",
        "proxy_intra_kv": "proxy_intra",
        "oracle_kv": "oracle",
        "random_kv": "random",
    }
    for strategy in EVAL_STRATEGIES:
        if strategy == "full_global_attention":
            model.set_adaptive_kv_anchor(enabled=False, layers="none")
            compressed = False
        else:
            model.set_adaptive_kv_anchor(
                enabled=True,
                layers=layer_spec,
                ratio=args.anchor_ratio,
                total=args.anchor_total,
                min_per_frame=args.anchor_min_per_frame,
                tau=args.budget_tau,
                uniform_mix=args.budget_uniform_mix,
                strategy=strategy_to_model[strategy],
                random_seed=args.seed,
                debug=False,
            )
            compressed = True
        prediction, timings, peak = timed_forward(model, images, torch.device(args.device), args.timing_repeats)
        diffs = prediction_diff(prediction, full_reference)
        pair_count, relative_cost = estimate_pair_count(
            model, selected_layers, images, args.anchor_ratio, args.anchor_total, compressed
        )
        rows.append(
            {
                "sample_id": sample_id,
                "strategy": strategy,
                "anchor_ratio": args.anchor_ratio,
                "anchor_total": args.anchor_total,
                "layers": layer_spec,
                "runtime_ms": float(np.median(timings)),
                "peak_memory_mb": peak,
                "attention_pair_count": int(pair_count),
                "relative_attention_cost": float(relative_cost),
                "task_metric_1": diffs["depth_l1_to_full"],
                "task_metric_2": diffs["pose_l1_to_full"],
                "output_diff_to_full": diffs["token_cosine_distance_to_full"],
                "oracle_gap": float("nan"),
            }
        )
        del prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    proxy = next((row for row in rows if row["strategy"] == "proxy_kv"), None)
    oracle = next((row for row in rows if row["strategy"] == "oracle_kv"), None)
    if proxy is not None and oracle is not None:
        gap = float(proxy["output_diff_to_full"] - oracle["output_diff_to_full"])
        for row in rows:
            if row["strategy"] == "proxy_kv":
                row["oracle_gap"] = gap
    model.set_adaptive_kv_anchor(enabled=False, layers="none")
    return rows


def main() -> int:
    args = parse_args()
    if args.anchor_ratio <= 0.0 or args.anchor_ratio > 1.0:
        raise ValueError("--anchor_ratio must be in (0, 1]")
    if args.anchor_total is not None and args.anchor_total <= 0:
        raise ValueError("--anchor_total must be positive when specified")
    if args.anchor_min_per_frame < 0:
        raise ValueError("--anchor_min_per_frame must be non-negative")
    if args.budget_tau <= 0.0:
        raise ValueError("--budget_tau must be positive")
    if not 0.0 <= args.budget_uniform_mix <= 1.0:
        raise ValueError("--budget_uniform_mix must be in [0, 1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topk_list = parse_topk_list(args.topk_list)
    samples = discover_samples(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_model(args.checkpoint, device)
    requested_layers = parse_layer_spec(args.layers, model.aggregator.depth)
    selected_layers = [
        layer for layer in requested_layers if model.aggregator.inter_frame_attention_types[layer] == "global"
    ]
    skipped_layers = sorted(set(requested_layers) - set(selected_layers))
    if not selected_layers:
        raise ValueError("No global inter-frame layers selected")
    if skipped_layers:
        print(f"Skipping non-global inter-frame layers: {skipped_layers}", flush=True)

    all_stage1_rows: list[dict[str, object]] = []
    all_anchor_rows: list[dict[str, object]] = []
    all_ablation_rows: list[dict[str, object]] = []
    all_stage3_rows: list[dict[str, object]] = []
    metadata = {
        "checkpoint": str(args.checkpoint),
        "input_path": str(args.input_path),
        "samples": {sample.sample_id: [str(path) for path in sample.image_paths] for sample in samples},
        "requested_layers": requested_layers,
        "global_layers_analyzed": selected_layers,
        "skipped_non_global_layers": skipped_layers,
        "anchor_ratio": args.anchor_ratio,
        "anchor_total": args.anchor_total,
        "topk_list": topk_list,
        "query_sample_total": args.query_sample_total,
        "note": "Standard task metrics are TODO; Stage 3 records output-level differences to full attention.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    for sample in samples:
        if args.debug:
            print(f"Loading sample {sample.sample_id}: {len(sample.image_paths)} frames", flush=True)
        image_cpu = load_and_preprocess_images(
            [str(path) for path in sample.image_paths],
            mode=args.resize_mode,
            image_resolution=args.image_resolution,
        )
        images = image_cpu.to(device, non_blocking=True)
        grid = (images.shape[-2] // model.aggregator.patch_size, images.shape[-1] // model.aggregator.patch_size)
        with RegisterProxyCollector(
            model,
            len(sample.image_paths),
            selected_layers,
            args.query_chunk,
            args.query_sample_total,
            args.alpha,
            args.beta,
            args.normalization,
            grid,
        ) as collector:
            with torch.inference_mode():
                full_prediction = model(images)

        stage1_rows = build_stage1_rows(sample.sample_id, collector.results, selected_layers, topk_list)
        all_stage1_rows.extend(stage1_rows)
        anchor_rows, ablation_rows = save_stage2_and_ablation(
            args.output_dir, sample.sample_id, collector.results, selected_layers, args, grid
        )
        all_anchor_rows.extend(anchor_rows)
        all_ablation_rows.extend(ablation_rows)
        if args.save_attention_stats:
            save_attention_stats(args.output_dir, sample.sample_id, collector.results, selected_layers)
        if args.save_visualization:
            plot_layer_maps(args.output_dir, sample.sample_id, collector.results, selected_layers, grid)

        if args.eval_anchor_strategies:
            all_stage3_rows.extend(run_stage3_eval(model, images, sample.sample_id, selected_layers, args, full_prediction))

        del images, image_cpu, full_prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "stage1_proxy_vs_direct.csv", all_stage1_rows)
    (args.output_dir / "stage1_summary.json").write_text(
        json.dumps(summarize_rows(all_stage1_rows, selected_layers), indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "stage2_anchor_overlap.csv", all_anchor_rows)
    write_csv(args.output_dir / "ablation_results.csv", all_ablation_rows)
    if all_stage3_rows:
        write_csv(args.output_dir / "stage3_eval_results.csv", all_stage3_rows)
    if args.save_visualization:
        plot_curves(args.output_dir, all_stage1_rows, topk_list)
    print(f"Saved register-mediated proxy analysis to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
