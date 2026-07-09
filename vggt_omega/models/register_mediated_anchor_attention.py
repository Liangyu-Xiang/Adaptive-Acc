# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Register-mediated adaptive K/V patch-anchor attention.

The implementation keeps every query token and only compresses patch K/V tokens.
Per-frame camera/register/special prefix tokens are always retained in K/V.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TokenLayout:
    num_frames: int
    patch_count: int
    tokens_per_frame: int
    num_special_tokens: int
    reg_indices_by_frame: Tensor
    patch_indices_by_frame: Tensor
    all_reg_indices: Tensor
    all_patch_indices: Tensor


def get_token_layout(
    total_tokens: int,
    patch_grid_size: tuple[int, int],
    num_special_tokens: int,
    device: torch.device,
) -> TokenLayout:
    patch_h, patch_w = patch_grid_size
    patch_count = int(patch_h) * int(patch_w)
    if patch_count <= 0:
        raise ValueError("Register-mediated anchors require at least one patch token per frame")
    if num_special_tokens <= 0:
        raise ValueError(
            "Register-mediated anchors require explicit per-frame camera/register/special "
            "prefix-token metadata. Pass num_special_tokens/patch_token_start from the aggregator."
        )

    tokens_per_frame = patch_count + int(num_special_tokens)
    if total_tokens % tokens_per_frame != 0:
        raise ValueError(
            "Cannot infer token layout for register-mediated anchors: "
            f"total_tokens={total_tokens} is not divisible by "
            f"tokens_per_frame={tokens_per_frame} "
            f"(patch_grid_size={patch_grid_size}, num_special_tokens={num_special_tokens})."
        )

    num_frames = total_tokens // tokens_per_frame
    frame_offsets = torch.arange(num_frames, device=device, dtype=torch.long) * tokens_per_frame
    reg_offsets = torch.arange(num_special_tokens, device=device, dtype=torch.long)
    patch_offsets = num_special_tokens + torch.arange(patch_count, device=device, dtype=torch.long)
    reg_indices_by_frame = frame_offsets[:, None] + reg_offsets[None, :]
    patch_indices_by_frame = frame_offsets[:, None] + patch_offsets[None, :]
    return TokenLayout(
        num_frames=num_frames,
        patch_count=patch_count,
        tokens_per_frame=tokens_per_frame,
        num_special_tokens=num_special_tokens,
        reg_indices_by_frame=reg_indices_by_frame,
        patch_indices_by_frame=patch_indices_by_frame,
        all_reg_indices=reg_indices_by_frame.reshape(-1),
        all_patch_indices=patch_indices_by_frame.reshape(-1),
    )


def register_mediated_anchor_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    patch_grid_size: tuple[int, int],
    num_special_tokens: int,
    anchor_ratio: float,
    anchor_total: int | None,
    anchor_min_per_frame: int,
    anchor_tau: float,
    anchor_uniform_mix: float,
    anchor_mode: str,
    alpha_cross: float,
    beta_intra: float,
    topm_frames: int | None,
    scale: float,
    debug: bool = False,
) -> tuple[Tensor, dict[str, object]]:
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "Register-mediated anchors expect unmerged q/k/v with identical shapes, "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.ndim != 4:
        raise ValueError(f"Expected q/k/v in [B, H, N, Dh] format, got {tuple(q.shape)}")
    if not 0.0 <= anchor_ratio <= 1.0:
        raise ValueError(f"anchor_ratio must be in [0, 1], got {anchor_ratio}")
    if anchor_total is not None and anchor_total < 0:
        raise ValueError(f"anchor_total must be non-negative or None, got {anchor_total}")
    if anchor_min_per_frame < 0:
        raise ValueError(f"anchor_min_per_frame must be non-negative, got {anchor_min_per_frame}")
    if anchor_tau <= 0.0:
        raise ValueError(f"anchor_tau must be positive, got {anchor_tau}")
    if not 0.0 <= anchor_uniform_mix <= 1.0:
        raise ValueError(f"anchor_uniform_mix must be in [0, 1], got {anchor_uniform_mix}")

    mode = anchor_mode.replace("-", "_").lower()
    if mode not in {"lifting", "frame_pair_gated", "hybrid"}:
        raise ValueError("anchor_mode must be one of ['lifting', 'frame_pair_gated', 'hybrid']")

    batch_size, num_heads, total_tokens, head_dim = q.shape
    layout = get_token_layout(total_tokens, patch_grid_size, num_special_tokens, q.device)
    with torch.no_grad():
        scores = _compute_register_mediated_scores(
            q=q.detach(),
            k=k.detach(),
            layout=layout,
            scale=scale,
            alpha_cross=alpha_cross,
            beta_intra=beta_intra,
        )
        selection = _select_patch_anchors(
            patch_scores=scores["s_anchor"],
            frame_scores=scores["frame_scores"],
            layout=layout,
            anchor_ratio=anchor_ratio,
            anchor_total=anchor_total,
            anchor_min_per_frame=anchor_min_per_frame,
            anchor_tau=anchor_tau,
            anchor_uniform_mix=anchor_uniform_mix,
        )

    kv_indices = selection["kv_indices"]
    anchor_budget = int(selection["anchor_budget"])
    full_patch_kv = anchor_budget >= layout.num_frames * layout.patch_count
    use_gating = mode in {"frame_pair_gated", "hybrid"} and topm_frames is not None and topm_frames > 0
    if full_patch_kv or not use_gating:
        gather_index = kv_indices[:, None, :, None].expand(batch_size, num_heads, kv_indices.shape[1], head_dim)
        compressed_k = k.gather(dim=2, index=gather_index)
        compressed_v = v.gather(dim=2, index=gather_index)
        output = F.scaled_dot_product_attention(q, compressed_k, compressed_v)
        topm_key_frames = None
        gated_cost = None
    else:
        output, topm_key_frames, gated_cost = _frame_pair_gated_attention(
            q=q,
            k=k,
            v=v,
            layout=layout,
            selected_by_frame=selection["selected_by_frame"],
            anchor_counts=selection["anchor_counts"],
            frame_pair_graph=scores["frame_pair_graph"],
            topm_frames=topm_frames,
        )

    n_reg = int(layout.all_reg_indices.numel())
    debug_payload: dict[str, object] = {
        "mode": mode,
        "num_frames": int(layout.num_frames),
        "tokens_per_frame": int(layout.tokens_per_frame),
        "num_special_tokens": int(layout.num_special_tokens),
        "patch_count": int(layout.patch_count),
        "anchor_budget": int(anchor_budget),
        "kv_token_count": int(kv_indices.shape[1]),
        "anchor_counts": selection["anchor_counts"],
        "kv_indices": kv_indices,
        "selected_patch_anchor_indices_by_frame": selection["selected_by_frame"],
        "frame_scores": scores["frame_scores"],
        "theoretical_cost": {
            "original_cost": int(total_tokens * total_tokens),
            "compressed_cost_lifting": int(total_tokens * (n_reg + anchor_budget)),
            "compressed_cost_gated": gated_cost,
        },
    }
    if debug:
        debug_payload.update(
            {
                "frame_pair_graph": scores["frame_pair_graph"],
                "s_cross": scores["s_cross"],
                "col_mass": scores["col_mass"],
                "s_anchor": scores["s_anchor"],
                "topm_key_frames": topm_key_frames,
            }
        )
    return output, debug_payload


def _compute_register_mediated_scores(
    q: Tensor,
    k: Tensor,
    layout: TokenLayout,
    scale: float,
    alpha_cross: float,
    beta_intra: float,
) -> dict[str, Tensor]:
    batch_size, num_heads, _, head_dim = q.shape
    num_frames = layout.num_frames
    reg_count = layout.num_special_tokens
    patch_count = layout.patch_count
    tokens_per_frame = layout.tokens_per_frame

    q_frames = q.reshape(batch_size, num_heads, num_frames, tokens_per_frame, head_dim)
    k_frames = k.reshape(batch_size, num_heads, num_frames, tokens_per_frame, head_dim)
    q_reg = q_frames[:, :, :, :reg_count]
    k_reg = k_frames[:, :, :, :reg_count]
    q_patch = q_frames[:, :, :, reg_count:]
    k_patch = k_frames[:, :, :, reg_count:]

    q_reg_flat = q_reg.reshape(batch_size, num_heads, num_frames * reg_count, head_dim).float()
    k_reg_flat = k_reg.reshape(batch_size, num_heads, num_frames * reg_count, head_dim).float()
    reg_logits = torch.matmul(q_reg_flat, k_reg_flat.transpose(-2, -1)) * scale
    a_reg = reg_logits.softmax(dim=-1)
    a_reg_by_frame = a_reg.reshape(batch_size, num_heads, num_frames, reg_count, num_frames, reg_count)

    frame_pair_graph = a_reg_by_frame.mean(dim=(1, 3, 5))
    cross_frame_mask = ~torch.eye(num_frames, device=q.device, dtype=torch.bool)
    reg_recv = (
        a_reg_by_frame
        * cross_frame_mask.view(1, 1, num_frames, 1, num_frames, 1).to(dtype=a_reg_by_frame.dtype)
    ).sum(dim=(2, 3)).mean(dim=1)

    reg_to_patch = torch.empty(
        batch_size,
        num_frames,
        reg_count,
        patch_count,
        device=q.device,
        dtype=torch.float32,
    )
    denom = max(batch_size * num_heads * reg_count * patch_count, 1)
    frame_chunk = max(1, min(num_frames, 16_000_000 // denom))
    for start in range(0, num_frames, frame_chunk):
        end = min(start + frame_chunk, num_frames)
        logits = (
            torch.matmul(
                q_reg[:, :, start:end].float(),
                k_patch[:, :, start:end].float().transpose(-2, -1),
            )
            * scale
        )
        reg_to_patch[:, start:end] = logits.softmax(dim=-1).mean(dim=1)

    s_cross = (reg_recv[:, :, :, None] * reg_to_patch).sum(dim=2)
    s_cross = torch.nan_to_num(s_cross, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    col_mass = torch.empty(batch_size, num_frames, patch_count, device=q.device, dtype=torch.float32)
    denom = max(batch_size * num_heads * patch_count * patch_count, 1)
    frame_chunk = max(1, min(num_frames, 8_000_000 // denom))
    for start in range(0, num_frames, frame_chunk):
        end = min(start + frame_chunk, num_frames)
        logits = (
            torch.matmul(
                q_patch[:, :, start:end].float(),
                k_patch[:, :, start:end].float().transpose(-2, -1),
            )
            * scale
        )
        intra = logits.softmax(dim=-1)
        col_mass[:, start:end] = intra.sum(dim=-2).mean(dim=1)
    col_mass = torch.nan_to_num(col_mass, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    s_anchor = (
        float(alpha_cross) * _normalize_per_frame(s_cross)
        + float(beta_intra) * _normalize_per_frame(col_mass)
    )
    topk = max(1, int(math.ceil(patch_count * 0.1)))
    intra_het = col_mass.topk(topk, dim=-1).values.sum(dim=-1) / col_mass.sum(dim=-1).clamp_min(1.0e-12)
    frame_scores = s_cross.sum(dim=-1) + 0.5 * torch.nan_to_num(intra_het, nan=0.0, posinf=0.0, neginf=0.0)
    frame_scores = torch.nan_to_num(frame_scores, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "frame_pair_graph": frame_pair_graph,
        "s_cross": s_cross,
        "col_mass": col_mass,
        "s_anchor": s_anchor,
        "frame_scores": frame_scores,
    }


def _normalize_per_frame(scores: Tensor) -> Tensor:
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = scores.amin(dim=-1, keepdim=True)
    max_value = scores.amax(dim=-1, keepdim=True)
    denom = max_value - min_value
    return torch.where(denom > 1.0e-12, (scores - min_value) / denom.clamp_min(1.0e-12), torch.zeros_like(scores))


def _select_patch_anchors(
    patch_scores: Tensor,
    frame_scores: Tensor,
    layout: TokenLayout,
    anchor_ratio: float,
    anchor_total: int | None,
    anchor_min_per_frame: int,
    anchor_tau: float,
    anchor_uniform_mix: float,
) -> dict[str, Tensor | int]:
    batch_size = patch_scores.shape[0]
    total_patch_tokens = layout.num_frames * layout.patch_count
    if anchor_total is None:
        anchor_budget = int(math.ceil(total_patch_tokens * anchor_ratio))
    else:
        anchor_budget = int(anchor_total)
    anchor_budget = max(0, min(anchor_budget, total_patch_tokens))

    min_per_frame = min(int(anchor_min_per_frame), layout.patch_count)
    if layout.num_frames > 0 and anchor_budget < layout.num_frames * min_per_frame:
        min_per_frame = anchor_budget // layout.num_frames

    device = patch_scores.device
    anchor_indices = torch.empty(batch_size, anchor_budget, device=device, dtype=torch.long)
    anchor_counts = torch.empty(batch_size, layout.num_frames, device=device, dtype=torch.long)
    max_count = layout.patch_count if anchor_budget >= total_patch_tokens else min(layout.patch_count, anchor_budget)
    selected_by_frame = torch.full(
        (batch_size, layout.num_frames, max_count),
        -1,
        device=device,
        dtype=torch.long,
    )

    for batch_idx in range(batch_size):
        counts = _allocate_anchor_counts(
            frame_scores=frame_scores[batch_idx],
            num_frames=layout.num_frames,
            patch_count=layout.patch_count,
            anchor_budget=anchor_budget,
            min_per_frame=min_per_frame,
            tau=anchor_tau,
            uniform_mix=anchor_uniform_mix,
        )
        anchor_counts[batch_idx] = counts
        selected_frames = []
        for frame_idx, keep_count_tensor in enumerate(counts):
            keep_count = int(keep_count_tensor.item())
            if keep_count == 0:
                continue
            if keep_count >= layout.patch_count:
                local_patch = torch.arange(layout.patch_count, device=device, dtype=torch.long)
            else:
                local_patch = patch_scores[batch_idx, frame_idx].topk(
                    keep_count,
                    dim=-1,
                    largest=True,
                    sorted=False,
                ).indices
            global_patch = layout.patch_indices_by_frame[frame_idx, local_patch]
            selected_by_frame[batch_idx, frame_idx, :keep_count] = global_patch
            selected_frames.append(global_patch)
        if selected_frames:
            selected = torch.cat(selected_frames, dim=0)
        else:
            selected = torch.empty(0, device=device, dtype=torch.long)
        if selected.numel() != anchor_budget:
            raise RuntimeError(
                "Register-mediated anchor selection produced an unexpected count: "
                f"{selected.numel()} vs {anchor_budget}"
            )
        anchor_indices[batch_idx] = selected

    special_indices = layout.all_reg_indices.reshape(1, -1).expand(batch_size, -1)
    kv_indices = torch.cat([special_indices, anchor_indices], dim=1)
    return {
        "kv_indices": kv_indices,
        "anchor_counts": anchor_counts,
        "selected_by_frame": selected_by_frame,
        "anchor_budget": anchor_budget,
    }


def _allocate_anchor_counts(
    frame_scores: Tensor,
    num_frames: int,
    patch_count: int,
    anchor_budget: int,
    min_per_frame: int,
    tau: float,
    uniform_mix: float,
) -> Tensor:
    device = frame_scores.device
    counts = torch.full((num_frames,), min_per_frame, device=device, dtype=torch.long)
    remaining = anchor_budget - int(counts.sum().item())
    if remaining <= 0:
        return counts

    capacity = torch.full((num_frames,), patch_count - min_per_frame, device=device, dtype=torch.long)
    scores = torch.nan_to_num(frame_scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    probs = (scores / tau).softmax(dim=0)
    if uniform_mix > 0.0:
        probs = (1.0 - uniform_mix) * probs + uniform_mix / num_frames

    raw = probs * remaining
    extra = torch.minimum(torch.floor(raw).to(dtype=torch.long), capacity)
    leftover = remaining - int(extra.sum().item())
    fractional = raw - torch.floor(raw)
    while leftover > 0:
        available = extra < capacity
        if not bool(available.any().item()):
            break
        priority = (fractional + probs * 1.0e-6).masked_fill(~available, -float("inf"))
        take = min(leftover, int(available.sum().item()))
        chosen = priority.topk(take, dim=0, largest=True, sorted=False).indices
        extra[chosen] += 1
        leftover -= take
        fractional = probs
    return counts + extra


def _frame_pair_gated_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    layout: TokenLayout,
    selected_by_frame: Tensor,
    anchor_counts: Tensor,
    frame_pair_graph: Tensor,
    topm_frames: int,
) -> tuple[Tensor, Tensor, list[int]]:
    batch_size, num_heads, _, head_dim = q.shape
    output = torch.empty_like(q)
    topm = max(1, min(int(topm_frames), layout.num_frames))
    topm_key_frames = torch.empty(batch_size, layout.num_frames, topm, device=q.device, dtype=torch.long)
    gated_cost: list[int] = []

    for batch_idx in range(batch_size):
        all_anchor = selected_by_frame[batch_idx]
        all_anchor = all_anchor[all_anchor >= 0]
        lifting_kv = torch.cat([layout.all_reg_indices, all_anchor], dim=0)
        k_lift, v_lift = _gather_sample_kv(k, v, batch_idx, lifting_kv)
        q_special = q[batch_idx : batch_idx + 1, :, layout.all_reg_indices]
        special_out = F.scaled_dot_product_attention(q_special, k_lift, v_lift)
        output[batch_idx, :, layout.all_reg_indices] = special_out[0].to(dtype=output.dtype)

        sample_cost = int(layout.all_reg_indices.numel() * lifting_kv.numel())
        for query_frame in range(layout.num_frames):
            scores = frame_pair_graph[batch_idx, query_frame].float().clone()
            scores[query_frame] = float("inf")
            key_frames = scores.topk(topm, dim=0, largest=True, sorted=False).indices.sort().values
            topm_key_frames[batch_idx, query_frame] = key_frames

            frame_anchors = []
            for key_frame_tensor in key_frames:
                key_frame = int(key_frame_tensor.item())
                keep = int(anchor_counts[batch_idx, key_frame].item())
                if keep > 0:
                    frame_anchors.append(selected_by_frame[batch_idx, key_frame, :keep])
            if frame_anchors:
                anchor_indices = torch.cat(frame_anchors, dim=0)
                kv_indices = torch.cat([layout.all_reg_indices, anchor_indices], dim=0)
            else:
                kv_indices = layout.all_reg_indices
            k_gated, v_gated = _gather_sample_kv(k, v, batch_idx, kv_indices)
            patch_indices = layout.patch_indices_by_frame[query_frame]
            q_patch = q[batch_idx : batch_idx + 1, :, patch_indices]
            patch_out = F.scaled_dot_product_attention(q_patch, k_gated, v_gated)
            output[batch_idx, :, patch_indices] = patch_out[0].to(dtype=output.dtype)
            sample_cost += int(layout.patch_count * kv_indices.numel())
        gated_cost.append(sample_cost)

    return output, topm_key_frames, gated_cost


def _gather_sample_kv(k: Tensor, v: Tensor, batch_idx: int, indices: Tensor) -> tuple[Tensor, Tensor]:
    gather_index = indices.view(1, 1, -1, 1).expand(1, k.shape[1], indices.numel(), k.shape[-1])
    k_sample = k[batch_idx : batch_idx + 1].gather(dim=2, index=gather_index)
    v_sample = v[batch_idx : batch_idx + 1].gather(dim=2, index=gather_index)
    return k_sample, v_sample
