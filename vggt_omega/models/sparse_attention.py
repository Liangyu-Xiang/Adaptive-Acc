# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Block-sparse attention utilities adapted from sparse-vggt for VGGT-Omega."""

from __future__ import annotations

import math
from collections import namedtuple

import torch
import torch.nn.functional as F
from einops import rearrange


SortResult = namedtuple("SortResult", ["values", "indices"])


def validate_sparse_mode(sparse_ratio: float | None, cdf_threshold: float | None) -> None:
    use_ratio = sparse_ratio is not None
    use_cdf = cdf_threshold is not None
    if not (use_ratio or use_cdf):
        raise ValueError("Sparse attention requires --sparse-ratio, --sparse-cdf-threshold, or both")
    if sparse_ratio is not None and not 0.0 <= sparse_ratio <= 1.0:
        raise ValueError(f"sparse_ratio must be in [0, 1], got {sparse_ratio}")
    if cdf_threshold is not None and not 0.0 <= cdf_threshold <= 1.0:
        raise ValueError(f"cdf_threshold must be in [0, 1], got {cdf_threshold}")


def predict_pooled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    ks_q: int = 128,
    ks_k: int = 64,
    pool_mode: str = "avg",
) -> torch.Tensor:
    if pool_mode not in {"max", "avg"}:
        raise ValueError(f"pool_mode must be 'max' or 'avg', got {pool_mode!r}")

    pooling_fn = F.max_pool1d if pool_mode == "max" else F.avg_pool1d
    batch_size, num_heads, _, head_dim = query.shape

    query = rearrange(query, "B H T C -> (B H) C T")
    pooled_query = pooling_fn(query, kernel_size=ks_q, ceil_mode=True)
    pooled_query = rearrange(
        pooled_query,
        "(B H) C T -> B H T C",
        B=batch_size,
        H=num_heads,
    )

    key = rearrange(key, "B H T C -> (B H) C T")
    pooled_key = pooling_fn(key, kernel_size=ks_k, ceil_mode=True)
    pooled_key = rearrange(
        pooled_key,
        "(B H) C T -> B H T C",
        B=batch_size,
        H=num_heads,
    )

    pooled_score = pooled_query @ pooled_key.transpose(-1, -2)
    pooled_score = pooled_score * (head_dim**-0.5)
    return F.softmax(pooled_score, dim=-1)


def split_patch_tokens(x: torch.Tensor, num_frames: int, tokens_per_frame: int, special_tokens: int) -> torch.Tensor:
    batch_dims = x.shape[:-2]
    channels = x.shape[-1]
    x = x.view(batch_dims + (num_frames, tokens_per_frame, channels))
    return x[..., special_tokens:, :].reshape(batch_dims + (num_frames * (tokens_per_frame - special_tokens), channels))


def split_special_tokens(
    x: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    special_tokens: int,
) -> torch.Tensor | None:
    if special_tokens == 0:
        return None
    batch_dims = x.shape[:-2]
    channels = x.shape[-1]
    x = x.view(batch_dims + (num_frames, tokens_per_frame, channels))
    return x[..., :special_tokens, :].reshape(batch_dims + (num_frames * special_tokens, channels))


def _int32_idx(sort_result):
    return SortResult(sort_result.values, sort_result.indices.to(torch.int32))


def _mem_eff_sort(t: torch.Tensor, chunks: int = 4, dim: int = 1):
    sorted_chunks = [_int32_idx(torch.sort(tt, dim=-1, descending=True)) for tt in torch.chunk(t, chunks, dim=dim)]
    return SortResult(
        torch.cat([chunk.values for chunk in sorted_chunks], dim=dim),
        torch.cat([chunk.indices for chunk in sorted_chunks], dim=dim),
    )


def _validate_block_selection(
    topk: int | None,
    sparse_ratio: float | None,
    cdf_threshold: float | None,
) -> None:
    use_topk = topk is not None
    use_ratio = sparse_ratio is not None
    use_cdf = cdf_threshold is not None
    if not (
        (use_topk and not use_ratio and not use_cdf)
        or (use_ratio and not use_topk and not use_cdf)
        or (use_cdf and not use_topk and not use_ratio)
        or (use_ratio and use_cdf and not use_topk)
    ):
        raise ValueError(f"Invalid sparse mode: {topk=}, {sparse_ratio=}, {cdf_threshold=}")


def get_block_mask(
    pooled_score: torch.Tensor,
    sink_blocks: int,
    topk: int | None = None,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    try:
        from spas_sage_attn.utils import fill_block_map_triton, hyperparameter_check
    except ImportError as error:
        raise ImportError(
            "Sparse attention requires the SpargeAttn package used by sparse-vggt. "
            "Install sparse-vggt/external/SpargeAttn in the active environment."
        ) from error

    _validate_block_selection(topk, sparse_ratio, cdf_threshold)
    batch_size, num_heads, query_blocks, key_blocks = pooled_score.shape
    if sink_blocks < 0 or sink_blocks > key_blocks:
        raise ValueError(f"sink_blocks must be in [0, {key_blocks}], got {sink_blocks}")

    if sparse_ratio is not None:
        topk = int(key_blocks * (1 - sparse_ratio))
    if topk is not None and not 0 <= topk <= key_blocks:
        raise ValueError(f"topk must be in [0, {key_blocks}], got {topk}")

    sorted_score = torch.sort(pooled_score, dim=-1, descending=True) if pooled_score.numel() < 2e8 else _mem_eff_sort(pooled_score)
    num_to_select = None
    if cdf_threshold is not None:
        cdf = torch.cumsum(sorted_score.values, dim=-1)
        cdfthreshd = hyperparameter_check(cdf_threshold, num_heads, pooled_score.device)
        cdfthreshd = cdfthreshd.view(1, num_heads, 1, 1) + eps
        cdfthreshd = cdfthreshd.expand(batch_size, -1, query_blocks, 1).contiguous()
        num_to_select = torch.searchsorted(cdf, cdfthreshd, right=True).squeeze(-1)

    if topk is not None:
        topk_tensor = torch.full(
            (batch_size, num_heads, query_blocks),
            topk,
            device=pooled_score.device,
            dtype=torch.int64,
        )
        num_to_select = topk_tensor if num_to_select is None else torch.clamp(num_to_select, min=topk)

    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)

    if sink_blocks > 0:
        ones_shape = list(final_map.shape)
        ones_shape[-1] = sink_blocks
        trailing_ones = torch.ones(ones_shape, device=final_map.device, dtype=torch.bool)
        final_map = torch.cat([final_map, trailing_ones], dim=-1)

    return final_map


def block_sparse_attn_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pooled_score: torch.Tensor,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import spas_sage_attn._qattn as qattn
        from spas_sage_attn.quant_per_block import per_block_int8
        from spas_sage_attn.utils import block_map_lut_triton, hyperparameter_check
    except ImportError as error:
        raise ImportError(
            "Sparse attention requires the SpargeAttn package used by sparse-vggt. "
            "Install sparse-vggt/external/SpargeAttn in the active environment."
        ) from error

    out_dtype = query.dtype
    key_block_size = 64
    total_key_blocks = math.ceil(key.shape[-2] / key_block_size)
    sink_blocks = total_key_blocks - pooled_score.shape[-1]
    final_map = get_block_mask(
        pooled_score,
        sink_blocks=sink_blocks,
        sparse_ratio=sparse_ratio,
        cdf_threshold=cdf_threshold,
    )
    lut, valid_block_num = block_map_lut_triton(final_map)

    query = query.contiguous().to(dtype)
    key = key.contiguous().to(dtype)
    value = value.contiguous().to(dtype)

    key_mean = key.mean(dim=-2, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_block_int8(query, key - key_mean)
    q_scale = q_scale.squeeze(-1)
    k_scale = k_scale.squeeze(-1)

    head_dim = query.shape[-1]
    pvthreshd = hyperparameter_check(1e10, query.size(-3), query.device)
    output = torch.empty_like(query)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
        q_int8,
        k_int8,
        value,
        output,
        lut,
        valid_block_num,
        pvthreshd,
        q_scale,
        k_scale,
        1,
        0,
        1,
        head_dim**-0.5,
        0,
    )
    return output.to(out_dtype), 1 - final_map.float().mean()


def sparse_global_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    patch_grid_size: tuple[int, int],
    num_special_tokens: int,
    sparse_ratio: float | None,
    cdf_threshold: float | None,
    pool_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_sparse_mode(sparse_ratio, cdf_threshold)
    batch_size, num_heads, total_tokens, head_dim = q.shape
    patch_h, patch_w = patch_grid_size
    tokens_per_frame = patch_h * patch_w + num_special_tokens
    if total_tokens % tokens_per_frame != 0:
        raise ValueError(
            f"Cannot infer frame count for sparse attention: {total_tokens=} is not divisible by {tokens_per_frame=}"
        )
    num_frames = total_tokens // tokens_per_frame

    q_special = split_special_tokens(q, num_frames, tokens_per_frame, num_special_tokens)
    k_special = split_special_tokens(k, num_frames, tokens_per_frame, num_special_tokens)
    v_special = split_special_tokens(v, num_frames, tokens_per_frame, num_special_tokens)
    q_patch = split_patch_tokens(q, num_frames, tokens_per_frame, num_special_tokens)
    k_patch = split_patch_tokens(k, num_frames, tokens_per_frame, num_special_tokens)
    v_patch = split_patch_tokens(v, num_frames, tokens_per_frame, num_special_tokens)

    x_special = F.scaled_dot_product_attention(q_special, k, v) if q_special is not None else None
    if k_special is not None:
        key = torch.cat([k_patch, k_special], dim=-2)
        value = torch.cat([v_patch, v_special], dim=-2)
    else:
        key = k_patch
        value = v_patch

    pooled_attention = predict_pooled_attention(q_patch, k_patch, pool_mode=pool_mode)
    x_patch, sparsity = block_sparse_attn_cuda(
        query=q_patch,
        key=key,
        value=value,
        pooled_score=pooled_attention,
        sparse_ratio=sparse_ratio,
        cdf_threshold=cdf_threshold,
    )
    x_patch = rearrange(
        x_patch,
        "B H (F P) C -> B H F P C",
        F=num_frames,
        P=patch_h * patch_w,
    )

    if x_special is not None:
        x_special = x_special.view(batch_size, num_heads, num_frames, num_special_tokens, head_dim)
        x = torch.cat([x_special, x_patch], dim=-2)
    else:
        x = x_patch
    return x.view(batch_size, num_heads, total_tokens, head_dim), sparsity
