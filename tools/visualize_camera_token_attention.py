#!/usr/bin/env python3
"""Visualize special-token attention for high-similarity frame pairs.

The script runs the dense/original VGGT-Omega aggregator and attaches hooks to
the requested layers. It records selected special-token query attention before
SDPA:

* frame blocks: selected special token -> tokens inside the same frame
* inter-frame blocks: selected special token -> tokens across frames

For memory, it keeps only selected high-similarity query frames and aggregates
global attention by key frame and token type.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_token_evolution import load_model  # noqa: E402
from vggt_omega.models.aggregator import slice_expand_and_flatten  # noqa: E402
from vggt_omega.models.layers.attention import rope_apply  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.reference_frame import resolve_first_frame_token_indices  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained_ckpts" / "vggt_omega_1b_512.pt"
DEFAULT_SAMPLED_FRAMES = (
    REPO_ROOT
    / "outputs"
    / "frame-fusion-smoke__tum__300frames__K80_M5_pre0__20260730"
    / "sampled_frames.json"
)
DEFAULT_SIMILARITY_NPZ = (
    REPO_ROOT
    / "outputs"
    / "frame_similarity_matrices__tum_halfsphere_300f__layers_2_6_10_16_23"
    / "frame_similarity_matrices.npz"
)
DEFAULT_LAYERS = (2, 6, 10, 16, 23)
TOKEN_TYPE_NAMES = ("camera", "register", "patch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sampled-frames", type=Path, default=DEFAULT_SAMPLED_FRAMES)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--similarity-npz", type=Path, default=DEFAULT_SIMILARITY_NPZ)
    parser.add_argument("--similarity-stage", default="layer_23")
    parser.add_argument("--similarity-threshold", type=float, default=0.76)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument(
        "--pair-selection",
        choices=("greedy-matching", "topk"),
        default="greedy-matching",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=[str(layer) for layer in DEFAULT_LAYERS],
        help="0-based aggregator layer indices, e.g. 2 6 10 16 23 or 2,6,10,16,23.",
    )
    parser.add_argument(
        "--query-token-kind",
        choices=("camera", "register"),
        default="camera",
        help="Special-token query type to analyze. Register mode analyzes all 16 register tokens by default.",
    )
    parser.add_argument(
        "--register-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional 0-based register token indices to analyze when --query-token-kind=register.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/camera_token_attention"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="max_size")
    parser.add_argument("--key-frame-chunk", type=int, default=8)
    parser.add_argument("--query-frame-chunk", type=int, default=8)
    parser.add_argument(
        "--patch-embed-chunk-size",
        type=int,
        default=0,
        help="If positive, run the DINO patch embedder over this many frames at a time.",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def parse_layers(values: Iterable[str]) -> list[int]:
    layers: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                layers.append(int(part))
    if not layers:
        raise ValueError("At least one layer is required")
    return layers


def load_manifest_image_paths(
    sampled_frames: Path,
    sequence: str | None,
    num_frames: int | None,
) -> tuple[str, list[Path]]:
    manifest = json.loads(sampled_frames.read_text(encoding="utf-8"))
    if sequence is None:
        sequence = next(iter(manifest))
    if sequence not in manifest:
        raise ValueError(f"Sequence {sequence!r} is not in {sampled_frames}")
    paths = [Path(path) for path in manifest[sequence]["rgb_paths"]]
    if num_frames is not None:
        if num_frames < 2:
            raise ValueError("--num-frames must be at least 2")
        paths = paths[:num_frames]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return sequence, paths


def select_high_similarity_pairs(
    similarity: np.ndarray,
    *,
    threshold: float,
    max_pairs: int,
    selection: str,
) -> tuple[list[tuple[int, int, float]], int]:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"Expected a square similarity matrix, got {similarity.shape}")
    edges: list[tuple[float, int, int]] = []
    for i, j in zip(*np.triu_indices_from(similarity, k=1)):
        value = float(similarity[i, j])
        if value > threshold:
            edges.append((value, int(i), int(j)))
    edges.sort(reverse=True)
    if not edges:
        for i, j in zip(*np.triu_indices_from(similarity, k=1)):
            edges.append((float(similarity[i, j]), int(i), int(j)))
        edges.sort(reverse=True)

    selected: list[tuple[int, int, float]] = []
    if selection == "topk":
        selected = [(i, j, value) for value, i, j in edges[:max_pairs]]
    elif selection == "greedy-matching":
        used: set[int] = set()
        for value, i, j in edges:
            if i in used or j in used:
                continue
            selected.append((i, j, value))
            used.add(i)
            used.add(j)
            if len(selected) >= max_pairs:
                break
    else:
        raise ValueError(selection)
    return selected, len([edge for edge in edges if edge[0] > threshold])


def _reshape_q_or_k(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, num_tokens, hidden_dim = tensor.shape
    head_dim = hidden_dim // num_heads
    return tensor.reshape(batch_size, num_tokens, num_heads, head_dim).transpose(1, 2)


def _apply_k_rope_only(
    k: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    sin, cos = rope
    prefix = k.shape[-2] - sin.shape[-2]
    if prefix < 0:
        raise ValueError("RoPE length exceeds key sequence length")
    k_dtype = k.dtype
    k_prefix = k[:, :, :prefix]
    k_patch = rope_apply(k[:, :, prefix:].to(dtype=sin.dtype), sin, cos).to(dtype=k_dtype)
    return torch.cat((k_prefix, k_patch), dim=-2)


def _normalize_qk(block, q: torch.Tensor, k: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    if block.attn.use_qk_norm:
        q = block.attn.q_norm(q)
        if k is not None:
            k = block.attn.k_norm(k)
    return q, k


def _cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _rank_descending(row: np.ndarray, target_index: int) -> int:
    return int(np.sum(row > row[target_index]) + 1)


class CameraAttentionCollector:
    """Collect selected special-token query attention and features."""

    def __init__(
        self,
        model,
        *,
        layers: Sequence[int],
        num_frames: int,
        query_frames: Sequence[int],
        query_token_offsets: Sequence[int],
        query_token_labels: Sequence[str],
        query_token_kind: str,
        patch_grid_size: tuple[int, int],
        key_frame_chunk: int,
        query_frame_chunk: int,
    ) -> None:
        self.aggregator = model.aggregator
        self.layers = tuple(int(layer) for layer in layers)
        self.num_frames = int(num_frames)
        self.query_frames = tuple(int(frame) for frame in query_frames)
        self.query_token_offsets = tuple(int(offset) for offset in query_token_offsets)
        self.query_token_labels = tuple(str(label) for label in query_token_labels)
        self.query_token_kind = str(query_token_kind)
        self.patch_grid_size = patch_grid_size
        self.key_frame_chunk = int(key_frame_chunk)
        self.query_frame_chunk = int(query_frame_chunk)
        self.patch_token_start = int(self.aggregator.patch_token_start)
        if not self.query_token_offsets:
            raise ValueError("At least one query token offset is required")
        if len(self.query_token_offsets) != len(self.query_token_labels):
            raise ValueError("Query token offsets and labels must have the same length")
        invalid_offsets = [
            offset
            for offset in self.query_token_offsets
            if offset < 0 or offset >= self.patch_token_start
        ]
        if invalid_offsets:
            raise ValueError(
                f"Query token offsets must be within 0..{self.patch_token_start - 1}: {invalid_offsets}"
            )
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.intra_type_mass: dict[int, np.ndarray] = {}
        self.intra_patch_maps: dict[int, np.ndarray] = {}
        self.global_frame_type_mass: dict[int, np.ndarray] = {}
        self.frame_camera_features: dict[int, np.ndarray] = {}
        self.inter_camera_features: dict[int, np.ndarray] = {}
        self.inter_special_features: dict[int, np.ndarray] = {}
        self.layer_attention_types: dict[int, str] = {}
        self._frame_rope: tuple[torch.Tensor, torch.Tensor] | None = None

    def __enter__(self) -> "CameraAttentionCollector":
        for layer in self.layers:
            self.handles.append(
                self.aggregator.frame_blocks[layer].attn.qkv.register_forward_hook(
                    self._frame_qkv_hook(layer)
                )
            )
            self.handles.append(
                self.aggregator.frame_blocks[layer].register_forward_hook(
                    self._frame_output_hook(layer)
                )
            )
            self.handles.append(
                self.aggregator.inter_frame_blocks[layer].attn.qkv.register_forward_hook(
                    self._inter_qkv_hook(layer)
                )
            )
            self.handles.append(
                self.aggregator.inter_frame_blocks[layer].register_forward_hook(
                    self._inter_output_hook(layer)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def set_frame_rope(self, rope: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._frame_rope = rope

    def _query_index_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.query_frames, device=device, dtype=torch.long)

    def _query_token_offset_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.query_token_offsets, device=device, dtype=torch.long)

    def _frame_qkv_hook(self, layer: int):
        def hook(module, _inputs, output: torch.Tensor) -> None:
            if self._frame_rope is None:
                raise RuntimeError("Frame RoPE has not been initialized")
            qkv = output.detach()
            if qkv.ndim != 3:
                raise ValueError(f"Frame qkv output for layer {layer} has shape {tuple(qkv.shape)}")
            device_type = qkv.device.type
            query_indices = self._query_index_tensor(qkv.device)
            if int(query_indices.max().item()) >= qkv.shape[0]:
                raise ValueError("Selected query frame exceeds frame-block batch dimension")
            selected = qkv.index_select(0, query_indices)
            _, num_tokens, three_hidden = selected.shape
            hidden_dim = module.in_features
            num_heads = self.aggregator.frame_blocks[layer].attn.num_heads
            query_token_offsets = self._query_token_offset_tensor(qkv.device)
            q = _reshape_q_or_k(
                selected.index_select(1, query_token_offsets)[:, :, :hidden_dim],
                num_heads,
            )
            k = _reshape_q_or_k(selected[:, :, hidden_dim : 2 * hidden_dim], num_heads)
            block = self.aggregator.frame_blocks[layer]
            q, k = _normalize_qk(block, q, k)
            k = _apply_k_rope_only(k, self._frame_rope)
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * block.attn.scale
            probabilities = logits.softmax(dim=-1)
            camera_mass = probabilities[..., 0].mean(dim=1)
            register_mass = probabilities[..., 1 : self.patch_token_start].sum(dim=-1).mean(dim=1)
            patch_probs = probabilities[..., self.patch_token_start :]
            patch_mass = patch_probs.sum(dim=-1).mean(dim=1)
            self.intra_type_mass[layer] = (
                torch.stack((camera_mass, register_mass, patch_mass), dim=-1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            patch_h, patch_w = self.patch_grid_size
            expected_tokens = self.patch_token_start + patch_h * patch_w
            if num_tokens != expected_tokens:
                raise ValueError(f"Unexpected frame token count {num_tokens}; expected {expected_tokens}")
            self.intra_patch_maps[layer] = (
                patch_probs.mean(dim=1)
                .reshape(len(self.query_frames), len(self.query_token_offsets), patch_h, patch_w)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            del qkv, selected, q, k, logits, probabilities, patch_probs
            if device_type == "cuda":
                torch.cuda.empty_cache()

        return hook

    def _inter_qkv_hook(self, layer: int):
        def hook(module, _inputs, output: torch.Tensor) -> None:
            qkv = output.detach()
            if qkv.ndim != 3:
                raise ValueError(f"Inter-frame qkv output for layer {layer} has shape {tuple(qkv.shape)}")
            device_type = qkv.device.type
            attention_type = self.aggregator.inter_frame_attention_types[layer]
            self.layer_attention_types[layer] = attention_type
            if attention_type == "global":
                tokens_per_frame = qkv.shape[1] // self.num_frames
                key_patch_token_start = self.patch_token_start
            elif attention_type == "register":
                tokens_per_frame = self.patch_token_start
                key_patch_token_start = self.patch_token_start
            else:
                raise ValueError(f"Unsupported inter-frame attention type {attention_type!r}")
            if tokens_per_frame * self.num_frames != qkv.shape[1]:
                raise ValueError(
                    f"Cannot split inter-frame qkv length {qkv.shape[1]} over {self.num_frames} frames"
                )

            hidden_dim = module.in_features
            block = self.aggregator.inter_frame_blocks[layer]
            num_heads = block.attn.num_heads
            query_token_offsets = self._query_token_offset_tensor(qkv.device)
            q_indices = torch.tensor(
                [
                    frame * tokens_per_frame + offset
                    for frame in self.query_frames
                    for offset in self.query_token_offsets
                ],
                device=qkv.device,
                dtype=torch.long,
            )
            q = _reshape_q_or_k(qkv.index_select(1, q_indices)[:, :, :hidden_dim], num_heads)
            q, _ = _normalize_qk(block, q, None)

            batch_size, num_heads, query_count, _ = q.shape
            device = qkv.device
            max_logits = torch.full(
                (batch_size, num_heads, query_count),
                -float("inf"),
                device=device,
                dtype=torch.float32,
            )
            denominator = torch.zeros_like(max_logits)
            mass = torch.zeros(
                batch_size,
                num_heads,
                query_count,
                self.num_frames,
                len(TOKEN_TYPE_NAMES),
                device=device,
                dtype=torch.float32,
            )

            for key_start_frame in range(0, self.num_frames, self.key_frame_chunk):
                key_end_frame = min(self.num_frames, key_start_frame + self.key_frame_chunk)
                key_token_start = key_start_frame * tokens_per_frame
                key_token_end = key_end_frame * tokens_per_frame
                key_frame_count = key_end_frame - key_start_frame
                key_slice = qkv[:, key_token_start:key_token_end, hidden_dim : 2 * hidden_dim]
                k = _reshape_q_or_k(key_slice, num_heads)
                if block.attn.use_qk_norm:
                    k = block.attn.k_norm(k)

                for query_start in range(0, query_count, self.query_frame_chunk):
                    query_end = min(query_count, query_start + self.query_frame_chunk)
                    q_chunk = q[:, :, query_start:query_end]
                    logits = torch.matmul(q_chunk.float(), k.float().transpose(-2, -1)) * block.attn.scale
                    chunk_max = logits.max(dim=-1).values
                    old_max = max_logits[:, :, query_start:query_end]
                    new_max = torch.maximum(old_max, chunk_max)
                    old_scale = torch.exp(old_max - new_max)
                    exp_logits = torch.exp(logits - new_max.unsqueeze(-1))

                    mass[:, :, query_start:query_end] *= old_scale.unsqueeze(-1).unsqueeze(-1)
                    by_frame = exp_logits.reshape(
                        batch_size,
                        num_heads,
                        query_end - query_start,
                        key_frame_count,
                        tokens_per_frame,
                    )
                    mass[:, :, query_start:query_end, key_start_frame:key_end_frame, 0] += by_frame[..., 0]
                    if key_patch_token_start > 1:
                        mass[:, :, query_start:query_end, key_start_frame:key_end_frame, 1] += (
                            by_frame[..., 1:key_patch_token_start].sum(dim=-1)
                        )
                    if tokens_per_frame > key_patch_token_start:
                        mass[:, :, query_start:query_end, key_start_frame:key_end_frame, 2] += (
                            by_frame[..., key_patch_token_start:].sum(dim=-1)
                        )
                    denominator[:, :, query_start:query_end] = (
                        denominator[:, :, query_start:query_end] * old_scale + exp_logits.sum(dim=-1)
                    )
                    max_logits[:, :, query_start:query_end] = new_max

            normalized_mass = mass / denominator.clamp_min(1.0e-30).unsqueeze(-1).unsqueeze(-1)
            self.global_frame_type_mass[layer] = (
                normalized_mass.mean(dim=(0, 1))
                .reshape(len(self.query_frames), len(self.query_token_offsets), self.num_frames, len(TOKEN_TYPE_NAMES))
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            del qkv, q, mass, denominator, max_logits, normalized_mass
            if device_type == "cuda":
                torch.cuda.empty_cache()

        return hook

    def _frame_output_hook(self, layer: int):
        def hook(_module, _inputs, output: torch.Tensor) -> None:
            if output.ndim != 3:
                return
            query_indices = self._query_index_tensor(output.device)
            query_token_offsets = self._query_token_offset_tensor(output.device)
            self.frame_camera_features[layer] = (
                output.detach()
                .index_select(0, query_indices)
                .index_select(1, query_token_offsets)
                .float()
                .cpu()
                .numpy()
            )

        return hook

    def _inter_output_hook(self, layer: int):
        def hook(_module, _inputs, output: torch.Tensor) -> None:
            if output.ndim != 3:
                return
            attention_type = self.aggregator.inter_frame_attention_types[layer]
            if attention_type == "global":
                tokens_per_frame = output.shape[1] // self.num_frames
            elif attention_type == "register":
                tokens_per_frame = self.patch_token_start
            else:
                return
            indices = torch.tensor(
                [
                    frame * tokens_per_frame + offset
                    for frame in self.query_frames
                    for offset in self.query_token_offsets
                ],
                device=output.device,
                dtype=torch.long,
            )
            self.inter_camera_features[layer] = (
                output.detach()
                .index_select(1, indices)[0]
                .reshape(len(self.query_frames), len(self.query_token_offsets), -1)
                .float()
                .cpu()
                .numpy()
            )
            special_indices = torch.tensor(
                [
                    frame * tokens_per_frame + offset
                    for frame in self.query_frames
                    for offset in range(self.patch_token_start)
                ],
                device=output.device,
                dtype=torch.long,
            )
            self.inter_special_features[layer] = (
                output.detach()
                .index_select(1, special_indices)[0]
                .reshape(len(self.query_frames), self.patch_token_start, -1)
                .float()
                .cpu()
                .numpy()
            )

        return hook


def run_aggregator(
    model,
    images: torch.Tensor,
    collector: CameraAttentionCollector,
    use_amp: bool,
    patch_embed_chunk_size: int,
) -> None:
    patch_h = images.shape[-2] // model.aggregator.patch_size
    patch_w = images.shape[-1] // model.aggregator.patch_size
    with torch.inference_mode():
        rope_sin, rope_cos = model.aggregator.rope_embed(H=patch_h, W=patch_w)
    collector.set_frame_rope(
        (
            rope_sin.to(device=images.device, dtype=torch.float32),
            rope_cos.to(device=images.device, dtype=torch.float32),
        )
    )
    if use_amp and images.device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        context = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        context = nullcontext()
    with torch.inference_mode(), context:
        if patch_embed_chunk_size > 0:
            _ = run_aggregator_with_chunked_patch_embed(
                model,
                images.unsqueeze(0),
                frame_rope=collector._frame_rope,
                patch_embed_chunk_size=patch_embed_chunk_size,
            )
        else:
            _ = model.aggregator(images.unsqueeze(0))


def run_aggregator_with_chunked_patch_embed(
    model,
    images: torch.Tensor,
    *,
    frame_rope: tuple[torch.Tensor, torch.Tensor] | None,
    patch_embed_chunk_size: int,
) -> tuple[list[torch.Tensor | None], int]:
    """Run the dense aggregator while chunking only the DINO patch embedder."""

    aggregator = model.aggregator
    batch_size, num_frames, num_channels, height, width = images.shape
    if batch_size != 1:
        raise ValueError("Chunked patch embedding currently supports batch_size=1")
    if num_channels != 3:
        raise ValueError(f"Expected 3 input channels, got {num_channels}")
    if aggregator.frame_fusion_mode != "none":
        raise ValueError("Chunked patch embedding path expects frame_fusion_mode='none'")
    if getattr(aggregator, "layer_token_swap_kind", "none") != "none":
        raise ValueError("Chunked patch embedding path expects layer token swap to be disabled")
    if frame_rope is None:
        raise RuntimeError("Frame RoPE is required")

    normalized_images = (images - aggregator._resnet_mean) / aggregator._resnet_std
    flat_images = normalized_images.view(batch_size * num_frames, num_channels, height, width)

    patch_chunks: list[torch.Tensor] = []
    for start in range(0, flat_images.shape[0], patch_embed_chunk_size):
        end = min(flat_images.shape[0], start + patch_embed_chunk_size)
        patch_tokens = aggregator.patch_embed(flat_images[start:end])
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        patch_chunks.append(patch_tokens)
    patch_tokens = torch.cat(patch_chunks, dim=0)

    first_frame_token_indices = resolve_first_frame_token_indices(
        aggregator.first_frame_token_indices,
        num_frames,
    )
    camera_token = slice_expand_and_flatten(
        aggregator.camera_token,
        batch_size,
        num_frames,
        first_frame_token_indices=first_frame_token_indices,
    )
    register_token = slice_expand_and_flatten(
        aggregator.register_token,
        batch_size,
        num_frames,
        first_frame_token_indices=first_frame_token_indices,
    )
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
    _, num_tokens, embed_dim = tokens.shape
    patch_grid_size = (height // aggregator.patch_size, width // aggregator.patch_size)

    outputs: list[torch.Tensor | None] = []
    aggregator._register_patch_selection.clear()
    aggregator._adaptive_intra_scores.clear()
    aggregator._progressive_stage_states.clear()
    aggregator.last_progressive_attention_stats.clear()
    aggregator.last_progressive_sample_indices.clear()
    aggregator.last_adaptive_pair_scope_debug.clear()
    aggregator.last_frame_fusion_debug.clear()

    tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
    for block_idx in range(aggregator.depth):
        tokens, frame_tokens = aggregator._run_frame_block(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
            block_idx,
            frame_rope,
        )
        tokens = aggregator._run_inter_frame_attention_block(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
            block_idx,
            aggregator.inter_frame_attention_types[block_idx],
            patch_grid_size,
            frame_fusion_pair_plans=None,
        )
        if block_idx in aggregator.cached_layer_indices:
            outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
        else:
            outputs.append(None)
    return outputs, aggregator.patch_token_start


def write_selected_pairs_csv(
    output_path: Path,
    pairs: Sequence[tuple[int, int, float]],
    image_paths: Sequence[Path],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_index",
                "frame_i",
                "frame_j",
                "similarity",
                "abs_input_index_gap",
                "frame_i_path",
                "frame_j_path",
            ],
        )
        writer.writeheader()
        for pair_index, (i, j, value) in enumerate(pairs):
            writer.writerow(
                {
                    "pair_index": pair_index,
                    "frame_i": i,
                    "frame_j": j,
                    "similarity": f"{value:.8f}",
                    "abs_input_index_gap": abs(i - j),
                    "frame_i_path": str(image_paths[i]),
                    "frame_j_path": str(image_paths[j]),
                }
            )


def write_intra_type_csv(
    output_path: Path,
    layers: Sequence[int],
    query_frames: Sequence[int],
    collector: CameraAttentionCollector,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer",
                "frame",
                "query_token_index",
                "query_token_offset",
                "query_token_label",
                "camera_mass",
                "register_mass",
                "patch_mass",
            ],
        )
        writer.writeheader()
        for layer in layers:
            masses = collector.intra_type_mass[layer]
            for row, frame in enumerate(query_frames):
                for token_index, (offset, label) in enumerate(
                    zip(collector.query_token_offsets, collector.query_token_labels)
                ):
                    writer.writerow(
                        {
                            "layer": layer,
                            "frame": frame,
                            "query_token_index": token_index,
                            "query_token_offset": offset,
                            "query_token_label": label,
                            "camera_mass": f"{masses[row, token_index, 0]:.8f}",
                            "register_mass": f"{masses[row, token_index, 1]:.8f}",
                            "patch_mass": f"{masses[row, token_index, 2]:.8f}",
                        }
                    )


def build_pair_metrics(
    layers: Sequence[int],
    pairs: Sequence[tuple[int, int, float]],
    query_frames: Sequence[int],
    collector: CameraAttentionCollector,
) -> list[dict[str, object]]:
    frame_to_query = {frame: row for row, frame in enumerate(query_frames)}
    rows: list[dict[str, object]] = []
    for layer in layers:
        global_mass = collector.global_frame_type_mass[layer]
        total = global_mass.sum(axis=-1)
        frame_features = collector.frame_camera_features[layer]
        inter_features = collector.inter_camera_features[layer]
        for token_index, (offset, label) in enumerate(
            zip(collector.query_token_offsets, collector.query_token_labels)
        ):
            for pair_index, (i, j, similarity) in enumerate(pairs):
                qi = frame_to_query[i]
                qj = frame_to_query[j]
                i_to_j = global_mass[qi, token_index, j]
                j_to_i = global_mass[qj, token_index, i]
                i_to_i = global_mass[qi, token_index, i]
                j_to_j = global_mass[qj, token_index, j]
                rows.append(
                    {
                        "layer": layer,
                        "attention_type": collector.layer_attention_types.get(layer, "unknown"),
                        "query_token_kind": collector.query_token_kind,
                        "query_token_index": token_index,
                        "query_token_offset": offset,
                        "query_token_label": label,
                        "pair_index": pair_index,
                        "frame_i": i,
                        "frame_j": j,
                        "layer23_similarity": similarity,
                        "i_to_j_total": float(i_to_j.sum()),
                        "j_to_i_total": float(j_to_i.sum()),
                        "mean_cross_total": float((i_to_j.sum() + j_to_i.sum()) * 0.5),
                        "i_self_total": float(i_to_i.sum()),
                        "j_self_total": float(j_to_j.sum()),
                        "mean_self_total": float((i_to_i.sum() + j_to_j.sum()) * 0.5),
                        "i_to_j_camera": float(i_to_j[0]),
                        "i_to_j_register": float(i_to_j[1]),
                        "i_to_j_patch": float(i_to_j[2]),
                        "j_to_i_camera": float(j_to_i[0]),
                        "j_to_i_register": float(j_to_i[1]),
                        "j_to_i_patch": float(j_to_i[2]),
                        "i_to_j_rank": _rank_descending(total[qi, token_index], j),
                        "j_to_i_rank": _rank_descending(total[qj, token_index], i),
                        "global_attention_row_cosine": _cosine_np(total[qi, token_index], total[qj, token_index]),
                        "frame_block_camera_feature_cosine": _cosine_np(
                            frame_features[qi, token_index],
                            frame_features[qj, token_index],
                        ),
                        "inter_block_camera_feature_cosine": _cosine_np(
                            inter_features[qi, token_index],
                            inter_features[qj, token_index],
                        ),
                    }
                )
    return rows


def write_pair_metrics_csv(output_path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def special_token_label(offset: int) -> tuple[str, int, str]:
    if offset == 0:
        return "camera", 0, "camera"
    return "register", offset - 1, f"register_{offset - 1:02d}"


def build_final_special_feature_similarity(
    *,
    final_layer: int,
    pairs: Sequence[tuple[int, int, float]],
    query_frames: Sequence[int],
    collector: CameraAttentionCollector,
) -> list[dict[str, object]]:
    if final_layer not in collector.inter_special_features:
        raise RuntimeError(f"Final special features for layer {final_layer} were not collected")
    frame_to_query = {frame: row for row, frame in enumerate(query_frames)}
    features = collector.inter_special_features[final_layer]
    rows: list[dict[str, object]] = []
    for pair_index, (i, j, similarity) in enumerate(pairs):
        qi = frame_to_query[i]
        qj = frame_to_query[j]
        for offset in range(collector.patch_token_start):
            token_kind, token_index, label = special_token_label(offset)
            rows.append(
                {
                    "final_layer": final_layer,
                    "pair_index": pair_index,
                    "frame_i": i,
                    "frame_j": j,
                    "layer23_similarity": similarity,
                    "token_offset": offset,
                    "token_kind": token_kind,
                    "token_index": token_index,
                    "token_label": label,
                    "feature_cosine": _cosine_np(features[qi, offset], features[qj, offset]),
                }
            )
    return rows


def write_final_special_feature_similarity_csv(
    output_path: Path,
    rows: Sequence[dict[str, object]],
) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_final_special_feature_similarity(
    output_path: Path,
    pairs: Sequence[tuple[int, int, float]],
    rows: Sequence[dict[str, object]],
    patch_token_start: int,
) -> None:
    if not rows:
        return
    pair_count = len(pairs)
    matrix = np.full((patch_token_start, pair_count), np.nan, dtype=np.float32)
    for row in rows:
        matrix[int(row["token_offset"]), int(row["pair_index"])] = float(row["feature_cosine"])
    y_labels = [special_token_label(offset)[2] for offset in range(patch_token_start)]
    x_labels = [f"{i}-{j}\n{value:.3f}" for i, j, value in pairs]
    figure, axis = plt.subplots(figsize=(max(9.0, pair_count * 0.8), 7.0))
    finite = matrix[np.isfinite(matrix)]
    if finite.size and float(finite.min()) > 0.95:
        vmin = max(0.95, float(np.percentile(finite, 1)))
        vmax = 1.0
    else:
        vmin, vmax = -1.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0e-6
    image = axis.imshow(matrix, aspect="auto", origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    axis.set_title("Final-layer feature cosine for all special tokens in selected high-similarity pairs")
    axis.set_ylabel("special token")
    axis.set_xlabel("selected pair and layer-23 patch similarity")
    axis.set_yticks(np.arange(patch_token_start))
    axis.set_yticklabels(y_labels)
    axis.set_xticks(np.arange(pair_count))
    axis.set_xticklabels(x_labels, rotation=45, ha="right")
    axis.figure.colorbar(image, ax=axis, fraction=0.026, pad=0.02, label="cosine similarity")
    if pair_count <= 14 and patch_token_start <= 20:
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                axis.text(x, y, f"{matrix[y, x]:.5f}", ha="center", va="center", fontsize=6, color="white")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_global_attention_to_frames(
    output_path: Path,
    layers: Sequence[int],
    query_frames: Sequence[int],
    pairs: Sequence[tuple[int, int, float]],
    collector: CameraAttentionCollector,
) -> None:
    figure, axes = plt.subplots(
        len(layers),
        1,
        figsize=(14, max(3.0, 2.4 * len(layers))),
        squeeze=False,
    )
    frame_to_row = {frame: row for row, frame in enumerate(query_frames)}
    pair_lookup = [(i, j) for i, j, _ in pairs]
    for axis, layer in zip(axes[:, 0], layers):
        total = collector.global_frame_type_mass[layer].sum(axis=-1).mean(axis=1)
        vmax = float(np.percentile(total, 99.5))
        image = axis.imshow(total, aspect="auto", origin="upper", cmap="magma", vmin=0.0, vmax=vmax)
        for i, j in pair_lookup:
            if i in frame_to_row:
                axis.scatter(j, frame_to_row[i], s=18, facecolors="none", edgecolors="cyan", linewidths=0.9)
            if j in frame_to_row:
                axis.scatter(i, frame_to_row[j], s=18, facecolors="none", edgecolors="cyan", linewidths=0.9)
        axis.set_title(
            f"Layer {layer} inter-frame {collector.query_token_kind} attention "
            f"({collector.layer_attention_types.get(layer, 'unknown')}; mean over query tokens)"
        )
        axis.set_ylabel("query frame")
        y_ticks = np.arange(len(query_frames))
        axis.set_yticks(y_ticks)
        axis.set_yticklabels([str(frame) for frame in query_frames])
        axis.set_xlabel("key frame")
        axis.figure.colorbar(image, ax=axis, fraction=0.018, pad=0.01, label="attention mass")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_intra_type_mass(
    output_path: Path,
    layers: Sequence[int],
    collector: CameraAttentionCollector,
) -> None:
    means = np.stack([collector.intra_type_mass[layer].mean(axis=(0, 1)) for layer in layers], axis=0)
    x = np.arange(len(layers))
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(layers), dtype=np.float64)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for type_index, name in enumerate(TOKEN_TYPE_NAMES):
        axis.bar(x, means[:, type_index], bottom=bottom, label=name, color=colors[type_index])
        bottom += means[:, type_index]
    axis.set_xticks(x)
    axis.set_xticklabels([str(layer) for layer in layers])
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("layer")
    axis.set_ylabel("mean attention mass")
    axis.set_title(
        f"Frame-block {collector.query_token_kind} attention inside selected high-similarity frames"
    )
    axis.legend(frameon=False, ncols=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_selected_pair_cross_attention(
    output_path: Path,
    layers: Sequence[int],
    pairs: Sequence[tuple[int, int, float]],
    pair_metrics: Sequence[dict[str, object]],
    collector: CameraAttentionCollector,
) -> None:
    layer_to_row = {layer: idx for idx, layer in enumerate(layers)}
    token_count = len(collector.query_token_offsets)
    if token_count == 1:
        x_count = len(pairs)
        cross = np.zeros((len(layers), x_count), dtype=np.float32)
        row_cos = np.zeros_like(cross)
        feature_cos = np.zeros_like(cross)
        for row in pair_metrics:
            layer_row = layer_to_row[int(row["layer"])]
            pair_col = int(row["pair_index"])
            cross[layer_row, pair_col] = float(row["mean_cross_total"])
            row_cos[layer_row, pair_col] = float(row["global_attention_row_cosine"])
            feature_cos[layer_row, pair_col] = float(row["inter_block_camera_feature_cosine"])
        x_labels = [f"{i}-{j}\n{value:.3f}" for i, j, value in pairs]
        x_label = "selected pair and layer-23 similarity"
    else:
        x_count = token_count
        cross = np.zeros((len(layers), x_count), dtype=np.float32)
        row_cos = np.zeros_like(cross)
        feature_cos = np.zeros_like(cross)
        for layer in layers:
            for token_index in range(token_count):
                layer_rows = [
                    row
                    for row in pair_metrics
                    if int(row["layer"]) == layer and int(row["query_token_index"]) == token_index
                ]
                if not layer_rows:
                    continue
                layer_row = layer_to_row[layer]
                cross[layer_row, token_index] = float(
                    np.mean([float(row["mean_cross_total"]) for row in layer_rows])
                )
                row_cos[layer_row, token_index] = float(
                    np.mean([float(row["global_attention_row_cosine"]) for row in layer_rows])
                )
                feature_cos[layer_row, token_index] = float(
                    np.mean([float(row["inter_block_camera_feature_cosine"]) for row in layer_rows])
                )
        x_labels = list(collector.query_token_labels)
        x_label = f"{collector.query_token_kind} query token"

    figure, axes = plt.subplots(3, 1, figsize=(max(9, x_count * 0.75), 8.5), squeeze=False)
    payloads = [
        (cross, "direct pair cross-attention mass", "magma", 0.0, float(np.percentile(cross, 99))),
        (row_cos, "cosine between global attention rows", "viridis", -1.0, 1.0),
        (feature_cos, "cosine between inter-block query tokens", "viridis", -1.0, 1.0),
    ]
    for axis, (matrix, title, cmap, vmin, vmax) in zip(axes[:, 0], payloads):
        if vmax <= vmin:
            vmax = vmin + 1.0e-6
        image = axis.imshow(matrix, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_yticks(np.arange(len(layers)))
        axis.set_yticklabels([str(layer) for layer in layers])
        axis.set_xticks(np.arange(x_count))
        axis.set_xticklabels(x_labels, rotation=45, ha="right")
        axis.set_ylabel("layer")
        axis.figure.colorbar(image, ax=axis, fraction=0.018, pad=0.01)
        if x_count <= 18:
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    axis.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center", fontsize=7, color="white")
    axes[-1, 0].set_xlabel(x_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_top_pair_global_profiles(
    output_path: Path,
    layers: Sequence[int],
    pairs: Sequence[tuple[int, int, float]],
    query_frames: Sequence[int],
    collector: CameraAttentionCollector,
) -> None:
    if not pairs:
        return
    i, j, similarity = pairs[0]
    frame_to_query = {frame: row for row, frame in enumerate(query_frames)}
    qi = frame_to_query[i]
    qj = frame_to_query[j]
    figure, axes = plt.subplots(
        len(layers),
        1,
        figsize=(12, max(3.0, 2.3 * len(layers))),
        sharex=True,
        squeeze=False,
    )
    x = np.arange(collector.num_frames)
    for axis, layer in zip(axes[:, 0], layers):
        total = collector.global_frame_type_mass[layer].sum(axis=-1).mean(axis=1)
        axis.plot(x, total[qi], label=f"query {i}", linewidth=1.4)
        axis.plot(x, total[qj], label=f"query {j}", linewidth=1.4)
        axis.axvline(i, color="gray", linewidth=0.8, linestyle="--")
        axis.axvline(j, color="gray", linewidth=0.8, linestyle=":")
        axis.set_ylabel(f"L{layer}")
        axis.legend(frameon=False, loc="upper right")
    axes[0, 0].set_title(
        f"Top high-similarity pair global {collector.query_token_kind}-attention profiles "
        f"(mean over query tokens): {i}-{j}, sim={similarity:.4f}"
    )
    axes[-1, 0].set_xlabel("key frame")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_top_pair_intra_patch_maps(
    output_path: Path,
    layers: Sequence[int],
    pairs: Sequence[tuple[int, int, float]],
    query_frames: Sequence[int],
    collector: CameraAttentionCollector,
) -> None:
    if not pairs:
        return
    i, j, similarity = pairs[0]
    frame_to_query = {frame: row for row, frame in enumerate(query_frames)}
    qi = frame_to_query[i]
    qj = frame_to_query[j]
    figure, axes = plt.subplots(
        len(layers),
        2,
        figsize=(7.5, max(4.0, 2.5 * len(layers))),
        squeeze=False,
    )
    for row, layer in enumerate(layers):
        maps = collector.intra_patch_maps[layer].mean(axis=1)
        vmax = float(np.percentile(maps[[qi, qj]], 99.5))
        for col, (frame, query_row) in enumerate(((i, qi), (j, qj))):
            axis = axes[row, col]
            image = axis.imshow(maps[query_row], cmap="magma", origin="upper", vmin=0.0, vmax=vmax)
            axis.set_title(f"L{layer} frame {frame}")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    figure.suptitle(
        f"Frame-block {collector.query_token_kind} -> patch attention maps "
        f"(mean over query tokens) for pair {i}-{j}, sim={similarity:.4f}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    *,
    layers: Sequence[int],
    sequence: str,
    image_paths: Sequence[Path],
    similarity_stage: str,
    similarity_threshold: float,
    threshold_edge_count: int,
    pairs: Sequence[tuple[int, int, float]],
    query_frames: Sequence[int],
    collector: CameraAttentionCollector,
    pair_metrics: Sequence[dict[str, object]],
    final_special_similarity: Sequence[dict[str, object]],
    final_layer: int,
    patch_grid_size: tuple[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_selected_pairs_csv(output_dir / "selected_pairs.csv", pairs, image_paths)
    write_intra_type_csv(output_dir / "intra_camera_attention_type_mass.csv", layers, query_frames, collector)
    write_pair_metrics_csv(output_dir / "selected_pair_camera_attention_metrics.csv", pair_metrics)
    write_final_special_feature_similarity_csv(
        output_dir / "final_layer_special_token_feature_similarity.csv",
        final_special_similarity,
    )

    npz_payload: dict[str, np.ndarray] = {}
    for layer in layers:
        key = f"layer_{layer:02d}"
        npz_payload[f"intra_{key}_type_mass"] = collector.intra_type_mass[layer]
        npz_payload[f"intra_{key}_patch_maps"] = collector.intra_patch_maps[layer]
        npz_payload[f"global_{key}_frame_type_mass"] = collector.global_frame_type_mass[layer]
        npz_payload[f"frame_{key}_camera_features"] = collector.frame_camera_features[layer]
        npz_payload[f"inter_{key}_camera_features"] = collector.inter_camera_features[layer]
        npz_payload[f"inter_{key}_special_features"] = collector.inter_special_features[layer]
    np.savez_compressed(output_dir / "camera_token_attention.npz", **npz_payload)

    plot_global_attention_to_frames(
        output_dir / "global_camera_attention_to_frames.png",
        layers,
        query_frames,
        pairs,
        collector,
    )
    plot_intra_type_mass(output_dir / "intra_camera_attention_type_mass.png", layers, collector)
    plot_selected_pair_cross_attention(
        output_dir / "selected_pair_cross_attention.png",
        layers,
        pairs,
        pair_metrics,
        collector,
    )
    plot_top_pair_global_profiles(
        output_dir / "top_pair_global_attention_profiles.png",
        layers,
        pairs,
        query_frames,
        collector,
    )
    plot_top_pair_intra_patch_maps(
        output_dir / "top_pair_intra_patch_maps.png",
        layers,
        pairs,
        query_frames,
        collector,
    )
    plot_final_special_feature_similarity(
        output_dir / "final_layer_special_token_feature_similarity.png",
        pairs,
        final_special_similarity,
        collector.patch_token_start,
    )

    aggregates: dict[str, dict[str, float]] = {}
    for layer in layers:
        layer_rows = [row for row in pair_metrics if int(row["layer"]) == layer]
        aggregates[str(layer)] = {
            "attention_type": collector.layer_attention_types.get(layer, "unknown"),
            "mean_cross_attention_mass": float(np.mean([row["mean_cross_total"] for row in layer_rows])),
            "mean_self_attention_mass": float(np.mean([row["mean_self_total"] for row in layer_rows])),
            "mean_global_attention_row_cosine": float(
                np.mean([row["global_attention_row_cosine"] for row in layer_rows])
            ),
            "mean_inter_camera_feature_cosine": float(
                np.mean([row["inter_block_camera_feature_cosine"] for row in layer_rows])
            ),
            "mean_frame_camera_feature_cosine": float(
                np.mean([row["frame_block_camera_feature_cosine"] for row in layer_rows])
            ),
            "mean_i_to_j_rank": float(np.mean([row["i_to_j_rank"] for row in layer_rows])),
            "mean_j_to_i_rank": float(np.mean([row["j_to_i_rank"] for row in layer_rows])),
        }
    final_special_by_kind: dict[str, dict[str, float]] = {}
    for token_kind in sorted({str(row["token_kind"]) for row in final_special_similarity}):
        kind_rows = [row for row in final_special_similarity if row["token_kind"] == token_kind]
        values = np.asarray([float(row["feature_cosine"]) for row in kind_rows], dtype=np.float64)
        final_special_by_kind[token_kind] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    final_special_by_token = {}
    for offset in range(collector.patch_token_start):
        label = special_token_label(offset)[2]
        token_rows = [row for row in final_special_similarity if int(row["token_offset"]) == offset]
        values = np.asarray([float(row["feature_cosine"]) for row in token_rows], dtype=np.float64)
        final_special_by_token[label] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    metadata = {
        "sequence": sequence,
        "num_frames": len(image_paths),
        "layers": list(layers),
        "patch_grid_size": list(patch_grid_size),
        "patch_token_start": collector.patch_token_start,
        "query_token_kind": collector.query_token_kind,
        "query_token_offsets": list(collector.query_token_offsets),
        "query_token_labels": list(collector.query_token_labels),
        "token_type_names": TOKEN_TYPE_NAMES,
        "similarity_stage": similarity_stage,
        "similarity_threshold": similarity_threshold,
        "threshold_edge_count": threshold_edge_count,
        "selected_pair_count": len(pairs),
        "query_frames": list(query_frames),
        "attention_semantics": (
            "pre-SDPA softmax attention from the selected special-token query; frame-block attention is within-frame, "
            "inter-frame register layers attend over camera/register tokens only, and inter-frame global layers "
            "attend over all camera/register/patch tokens"
        ),
        "final_special_feature_similarity": {
            "layer": final_layer,
            "csv": "final_layer_special_token_feature_similarity.csv",
            "png": "final_layer_special_token_feature_similarity.png",
            "summary_by_kind": final_special_by_kind,
            "summary_by_token": final_special_by_token,
        },
        "plots": {
            "global_camera_attention_to_frames": "global_camera_attention_to_frames.png",
            "intra_camera_attention_type_mass": "intra_camera_attention_type_mass.png",
            "selected_pair_cross_attention": "selected_pair_cross_attention.png",
            "top_pair_global_attention_profiles": "top_pair_global_attention_profiles.png",
            "top_pair_intra_patch_maps": "top_pair_intra_patch_maps.png",
            "final_layer_special_token_feature_similarity": "final_layer_special_token_feature_similarity.png",
        },
        "aggregate_pair_metrics_by_layer": aggregates,
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def resolve_query_tokens(args: argparse.Namespace, patch_token_start: int) -> tuple[list[int], list[str]]:
    if args.query_token_kind == "camera":
        if args.register_indices is not None:
            raise ValueError("--register-indices is only valid with --query-token-kind=register")
        return [0], ["camera"]

    register_count = patch_token_start - 1
    if register_count <= 0:
        raise ValueError("The model has no register tokens")
    if args.register_indices is None:
        indices = list(range(register_count))
    else:
        indices = [int(index) for index in args.register_indices]
    invalid = [index for index in indices if index < 0 or index >= register_count]
    if invalid:
        raise ValueError(f"Register indices must be within 0..{register_count - 1}: {invalid}")
    offsets = [1 + index for index in indices]
    labels = [f"register_{index:02d}" for index in indices]
    return offsets, labels


def main() -> int:
    args = parse_args()
    layers = parse_layers(args.layers)
    if args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive")
    if args.key_frame_chunk <= 0 or args.query_frame_chunk <= 0:
        raise ValueError("Frame chunks must be positive")
    if args.patch_embed_chunk_size < 0:
        raise ValueError("--patch-embed-chunk-size must be non-negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    sequence, image_paths = load_manifest_image_paths(args.sampled_frames, args.sequence, args.num_frames)
    similarity_payload = np.load(args.similarity_npz)
    if args.similarity_stage not in similarity_payload:
        raise ValueError(
            f"Stage {args.similarity_stage!r} not in {args.similarity_npz}; "
            f"available={list(similarity_payload.keys())}"
        )
    similarity = np.asarray(similarity_payload[args.similarity_stage])
    if args.num_frames is not None:
        similarity = similarity[: args.num_frames, : args.num_frames]
    if similarity.shape != (len(image_paths), len(image_paths)):
        raise ValueError(
            f"Similarity matrix shape {similarity.shape} does not match {len(image_paths)} frames"
        )
    pairs, threshold_edge_count = select_high_similarity_pairs(
        similarity,
        threshold=args.similarity_threshold,
        max_pairs=args.max_pairs,
        selection=args.pair_selection,
    )
    query_frames = sorted({frame for pair in pairs for frame in pair[:2]})
    if not query_frames:
        raise RuntimeError("No query frames selected")

    model = load_model(args.checkpoint, device)
    final_layer = model.aggregator.depth - 1
    if final_layer not in layers:
        layers = sorted(set(layers) | {final_layer})
    invalid_layers = [layer for layer in layers if layer < 0 or layer >= model.aggregator.depth]
    if invalid_layers:
        raise ValueError(f"Layer indices out of range 0..{model.aggregator.depth - 1}: {invalid_layers}")
    query_token_offsets, query_token_labels = resolve_query_tokens(args, model.aggregator.patch_token_start)

    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
        patch_size=model.aggregator.patch_size,
    ).to(device, non_blocking=device.type == "cuda")
    patch_grid_size = (
        images.shape[-2] // model.aggregator.patch_size,
        images.shape[-1] // model.aggregator.patch_size,
    )

    collector = CameraAttentionCollector(
        model,
        layers=layers,
        num_frames=len(image_paths),
        query_frames=query_frames,
        query_token_offsets=query_token_offsets,
        query_token_labels=query_token_labels,
        query_token_kind=args.query_token_kind,
        patch_grid_size=patch_grid_size,
        key_frame_chunk=args.key_frame_chunk,
        query_frame_chunk=args.query_frame_chunk,
    )
    with collector:
        run_aggregator(
            model,
            images,
            collector,
            use_amp=not args.no_amp,
            patch_embed_chunk_size=args.patch_embed_chunk_size,
        )

    missing = [
        f"{kind}:layer_{layer}"
        for layer in layers
        for kind, store in (
            ("intra", collector.intra_type_mass),
            ("global", collector.global_frame_type_mass),
            ("frame_features", collector.frame_camera_features),
            ("inter_features", collector.inter_camera_features),
            ("special_features", collector.inter_special_features),
        )
        if layer not in store
    ]
    if missing:
        raise RuntimeError(f"Missing collected outputs: {missing}")

    pair_metrics = build_pair_metrics(layers, pairs, query_frames, collector)
    final_special_similarity = build_final_special_feature_similarity(
        final_layer=final_layer,
        pairs=pairs,
        query_frames=query_frames,
        collector=collector,
    )
    write_outputs(
        args.output_dir,
        layers=layers,
        sequence=sequence,
        image_paths=image_paths,
        similarity_stage=args.similarity_stage,
        similarity_threshold=args.similarity_threshold,
        threshold_edge_count=threshold_edge_count,
        pairs=pairs,
        query_frames=query_frames,
        collector=collector,
        pair_metrics=pair_metrics,
        final_special_similarity=final_special_similarity,
        final_layer=final_layer,
        patch_grid_size=patch_grid_size,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "query_token_kind": args.query_token_kind,
                "query_tokens": len(query_token_offsets),
                "pairs": len(pairs),
                "query_frames": len(query_frames),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
