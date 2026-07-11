# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import os
from typing import List, Tuple

from torch import Tensor, nn
import torch
import torch.nn.functional as F
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except (ImportError, ModuleNotFoundError):
    create_block_mask = None
    flex_attention = None

from vggt_omega.models.merging import token_merge_bipartite2d
from vggt_omega.models.register_mediated_anchor_attention import register_mediated_anchor_attention
from vggt_omega.models.sparse_attention import sparse_global_attention

from .utils import cat_keep_shapes, uncat_with_shapes


ADAPTIVE_ANCHOR_STRATEGIES = {
    "all_frame_intra",
    "lifting",
    "frame_pair_gated",
    "hybrid",
    "random_frame_intra",
    "register_gated_intra",
    "register_gated_intra_query",
    "temporal_neighbor_intra",
    "oracle_frame_intra",
    "quota_intra_proxy",
    "register_intra",
    "fixed_grid",
    "intra_only",
    "proxy",
    "proxy_intra",
    "oracle",
    "random",
}


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


def _detach_debug_payload(payload):
    if isinstance(payload, Tensor):
        return payload.detach().cpu()
    if isinstance(payload, dict):
        return {key: _detach_debug_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_detach_debug_payload(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_detach_debug_payload(value) for value in payload)
    return payload


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        use_qk_norm: bool = False,
        merge_ratio: float = 0.9,
        use_adaptive_kv_anchor: bool = False,
        adaptive_anchor_ratio: float = 0.25,
        adaptive_anchor_total: int | None = None,
        adaptive_anchor_min_per_frame: int = 1,
        adaptive_anchor_tau: float = 1.0,
        adaptive_anchor_uniform_mix: float = 0.0,
        adaptive_anchor_strategy: str = "lifting",
        adaptive_anchor_score_alpha_cross: float = 1.0,
        adaptive_anchor_score_beta_intra: float = 0.2,
        adaptive_anchor_score_mode: str = "intra",
        adaptive_anchor_proxy_quota_ratio: float = 0.0,
        adaptive_anchor_intra_source: str = "cached_frame_qk",
        adaptive_anchor_frame_budget_mode: str = "hybrid",
        adaptive_anchor_frame_budget_top_frac: float = 0.1,
        adaptive_anchor_frame_budget_lambda_intra: float = 0.7,
        adaptive_anchor_frame_budget_lambda_reg: float = 0.3,
        adaptive_anchor_frame_budget_reg_topm: int = 4,
        adaptive_anchor_reg_patch_topk_ratio: float = 0.1,
        adaptive_anchor_reg_patch_topk_min: int = 8,
        adaptive_anchor_reg_patch_topk_max: int = 64,
        adaptive_anchor_reg_patch_conf_power: float = 1.0,
        adaptive_anchor_reg_patch_min_conf: float = 0.05,
        adaptive_anchor_query_conditioned_eta: float = 0.1,
        adaptive_anchor_gated_anchor_ratio_per_key_frame: float = 0.1,
        adaptive_anchor_gated_min_per_key_frame: int = 4,
        adaptive_anchor_gated_max_per_key_frame: int = 64,
        adaptive_anchor_always_include_self_frame: bool = True,
        adaptive_anchor_profile: bool = False,
        adaptive_anchor_topm_frames: int | None = 4,
        adaptive_anchor_random_seed: int = 33,
        adaptive_anchor_debug: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.merge_ratio = merge_ratio
        self._inter_frame_only_block_mask_cache = {}
        self.use_adaptive_kv_anchor = use_adaptive_kv_anchor
        self.adaptive_anchor_ratio = adaptive_anchor_ratio
        self.adaptive_anchor_total = adaptive_anchor_total
        self.adaptive_anchor_min_per_frame = adaptive_anchor_min_per_frame
        self.adaptive_anchor_tau = adaptive_anchor_tau
        self.adaptive_anchor_uniform_mix = adaptive_anchor_uniform_mix
        self.adaptive_anchor_strategy = adaptive_anchor_strategy
        self.adaptive_anchor_score_alpha_cross = float(adaptive_anchor_score_alpha_cross)
        self.adaptive_anchor_score_beta_intra = float(adaptive_anchor_score_beta_intra)
        self.adaptive_anchor_score_mode = adaptive_anchor_score_mode
        self.adaptive_anchor_proxy_quota_ratio = float(adaptive_anchor_proxy_quota_ratio)
        self.adaptive_anchor_intra_source = adaptive_anchor_intra_source
        self.adaptive_anchor_frame_budget_mode = adaptive_anchor_frame_budget_mode
        self.adaptive_anchor_frame_budget_top_frac = float(adaptive_anchor_frame_budget_top_frac)
        self.adaptive_anchor_frame_budget_lambda_intra = float(adaptive_anchor_frame_budget_lambda_intra)
        self.adaptive_anchor_frame_budget_lambda_reg = float(adaptive_anchor_frame_budget_lambda_reg)
        self.adaptive_anchor_frame_budget_reg_topm = int(adaptive_anchor_frame_budget_reg_topm)
        self.adaptive_anchor_reg_patch_topk_ratio = float(adaptive_anchor_reg_patch_topk_ratio)
        self.adaptive_anchor_reg_patch_topk_min = int(adaptive_anchor_reg_patch_topk_min)
        self.adaptive_anchor_reg_patch_topk_max = int(adaptive_anchor_reg_patch_topk_max)
        self.adaptive_anchor_reg_patch_conf_power = float(adaptive_anchor_reg_patch_conf_power)
        self.adaptive_anchor_reg_patch_min_conf = float(adaptive_anchor_reg_patch_min_conf)
        self.adaptive_anchor_query_conditioned_eta = float(adaptive_anchor_query_conditioned_eta)
        self.adaptive_anchor_gated_anchor_ratio_per_key_frame = float(
            adaptive_anchor_gated_anchor_ratio_per_key_frame
        )
        self.adaptive_anchor_gated_min_per_key_frame = int(adaptive_anchor_gated_min_per_key_frame)
        self.adaptive_anchor_gated_max_per_key_frame = int(adaptive_anchor_gated_max_per_key_frame)
        self.adaptive_anchor_always_include_self_frame = bool(adaptive_anchor_always_include_self_frame)
        self.adaptive_anchor_profile = bool(adaptive_anchor_profile)
        self.adaptive_anchor_topm_frames = adaptive_anchor_topm_frames
        self.adaptive_anchor_random_seed = int(adaptive_anchor_random_seed)
        self.adaptive_anchor_debug = adaptive_anchor_debug
        self.precomputed_intra_scores = None
        self.last_adaptive_anchor_debug = None
        self.last_adaptive_anchor_kv_tokens = None
        self.last_adaptive_anchor_patch_tokens = None
        # VGGT-Omega change: the aggregator checkpoint was trained with Q/K
        # normalization, while upstream DINOv3 attention does not expose it.
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(
        self,
        x: Tensor,
        attn_bias=None,
        rope: Tensor = None,
        global_merging=None,
        patch_grid_size: tuple[int, int] | None = None,
        num_special_tokens: int = 5,
        sparse_attention: bool = False,
        sparse_ratio: float | None = None,
        sparse_cdf_threshold: float | None = None,
        sparse_pool_mode: str = "avg",
        inter_frame_only_attention: bool = False,
        use_adaptive_kv_anchor: bool | None = None,
        adaptive_anchor_ratio: float | None = None,
        adaptive_anchor_total: int | None = None,
        adaptive_anchor_min_per_frame: int | None = None,
        adaptive_anchor_tau: float | None = None,
        adaptive_anchor_uniform_mix: float | None = None,
        adaptive_anchor_strategy: str | None = None,
        adaptive_anchor_score_alpha_cross: float | None = None,
        adaptive_anchor_score_beta_intra: float | None = None,
        adaptive_anchor_score_mode: str | None = None,
        adaptive_anchor_proxy_quota_ratio: float | None = None,
        adaptive_anchor_intra_source: str | None = None,
        adaptive_anchor_frame_budget_mode: str | None = None,
        adaptive_anchor_frame_budget_top_frac: float | None = None,
        adaptive_anchor_frame_budget_lambda_intra: float | None = None,
        adaptive_anchor_frame_budget_lambda_reg: float | None = None,
        adaptive_anchor_frame_budget_reg_topm: int | None = None,
        adaptive_anchor_reg_patch_topk_ratio: float | None = None,
        adaptive_anchor_reg_patch_topk_min: int | None = None,
        adaptive_anchor_reg_patch_topk_max: int | None = None,
        adaptive_anchor_reg_patch_conf_power: float | None = None,
        adaptive_anchor_reg_patch_min_conf: float | None = None,
        adaptive_anchor_query_conditioned_eta: float | None = None,
        adaptive_anchor_gated_anchor_ratio_per_key_frame: float | None = None,
        adaptive_anchor_gated_min_per_key_frame: int | None = None,
        adaptive_anchor_gated_max_per_key_frame: int | None = None,
        adaptive_anchor_always_include_self_frame: bool | None = None,
        adaptive_anchor_profile: bool | None = None,
        adaptive_anchor_topm_frames: int | None = None,
        adaptive_anchor_random_seed: int | None = None,
        adaptive_anchor_debug: bool | None = None,
    ) -> Tensor:
        merge_metric = x
        qkv = self.qkv(x)
        attn_v = self.compute_attention(
            qkv=qkv,
            attn_bias=attn_bias,
            rope=rope,
            global_merging=global_merging,
            patch_grid_size=patch_grid_size,
            num_special_tokens=num_special_tokens,
            merge_metric=merge_metric,
            sparse_attention=sparse_attention,
            sparse_ratio=sparse_ratio,
            sparse_cdf_threshold=sparse_cdf_threshold,
            sparse_pool_mode=sparse_pool_mode,
            inter_frame_only_attention=inter_frame_only_attention,
            use_adaptive_kv_anchor=use_adaptive_kv_anchor,
            adaptive_anchor_ratio=adaptive_anchor_ratio,
            adaptive_anchor_total=adaptive_anchor_total,
            adaptive_anchor_min_per_frame=adaptive_anchor_min_per_frame,
            adaptive_anchor_tau=adaptive_anchor_tau,
            adaptive_anchor_uniform_mix=adaptive_anchor_uniform_mix,
            adaptive_anchor_strategy=adaptive_anchor_strategy,
            adaptive_anchor_score_alpha_cross=adaptive_anchor_score_alpha_cross,
            adaptive_anchor_score_beta_intra=adaptive_anchor_score_beta_intra,
            adaptive_anchor_score_mode=adaptive_anchor_score_mode,
            adaptive_anchor_proxy_quota_ratio=adaptive_anchor_proxy_quota_ratio,
            adaptive_anchor_intra_source=adaptive_anchor_intra_source,
            adaptive_anchor_frame_budget_mode=adaptive_anchor_frame_budget_mode,
            adaptive_anchor_frame_budget_top_frac=adaptive_anchor_frame_budget_top_frac,
            adaptive_anchor_frame_budget_lambda_intra=adaptive_anchor_frame_budget_lambda_intra,
            adaptive_anchor_frame_budget_lambda_reg=adaptive_anchor_frame_budget_lambda_reg,
            adaptive_anchor_frame_budget_reg_topm=adaptive_anchor_frame_budget_reg_topm,
            adaptive_anchor_reg_patch_topk_ratio=adaptive_anchor_reg_patch_topk_ratio,
            adaptive_anchor_reg_patch_topk_min=adaptive_anchor_reg_patch_topk_min,
            adaptive_anchor_reg_patch_topk_max=adaptive_anchor_reg_patch_topk_max,
            adaptive_anchor_reg_patch_conf_power=adaptive_anchor_reg_patch_conf_power,
            adaptive_anchor_reg_patch_min_conf=adaptive_anchor_reg_patch_min_conf,
            adaptive_anchor_query_conditioned_eta=adaptive_anchor_query_conditioned_eta,
            adaptive_anchor_gated_anchor_ratio_per_key_frame=adaptive_anchor_gated_anchor_ratio_per_key_frame,
            adaptive_anchor_gated_min_per_key_frame=adaptive_anchor_gated_min_per_key_frame,
            adaptive_anchor_gated_max_per_key_frame=adaptive_anchor_gated_max_per_key_frame,
            adaptive_anchor_always_include_self_frame=adaptive_anchor_always_include_self_frame,
            adaptive_anchor_profile=adaptive_anchor_profile,
            adaptive_anchor_topm_frames=adaptive_anchor_topm_frames,
            adaptive_anchor_random_seed=adaptive_anchor_random_seed,
            adaptive_anchor_debug=adaptive_anchor_debug,
        )
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(
        self,
        qkv: Tensor,
        attn_bias=None,
        rope=None,
        global_merging=None,
        patch_grid_size: tuple[int, int] | None = None,
        num_special_tokens: int = 5,
        merge_metric: Tensor | None = None,
        sparse_attention: bool = False,
        sparse_ratio: float | None = None,
        sparse_cdf_threshold: float | None = None,
        sparse_pool_mode: str = "avg",
        inter_frame_only_attention: bool = False,
        use_adaptive_kv_anchor: bool | None = None,
        adaptive_anchor_ratio: float | None = None,
        adaptive_anchor_total: int | None = None,
        adaptive_anchor_min_per_frame: int | None = None,
        adaptive_anchor_tau: float | None = None,
        adaptive_anchor_uniform_mix: float | None = None,
        adaptive_anchor_strategy: str | None = None,
        adaptive_anchor_score_alpha_cross: float | None = None,
        adaptive_anchor_score_beta_intra: float | None = None,
        adaptive_anchor_score_mode: str | None = None,
        adaptive_anchor_proxy_quota_ratio: float | None = None,
        adaptive_anchor_intra_source: str | None = None,
        adaptive_anchor_frame_budget_mode: str | None = None,
        adaptive_anchor_frame_budget_top_frac: float | None = None,
        adaptive_anchor_frame_budget_lambda_intra: float | None = None,
        adaptive_anchor_frame_budget_lambda_reg: float | None = None,
        adaptive_anchor_frame_budget_reg_topm: int | None = None,
        adaptive_anchor_reg_patch_topk_ratio: float | None = None,
        adaptive_anchor_reg_patch_topk_min: int | None = None,
        adaptive_anchor_reg_patch_topk_max: int | None = None,
        adaptive_anchor_reg_patch_conf_power: float | None = None,
        adaptive_anchor_reg_patch_min_conf: float | None = None,
        adaptive_anchor_query_conditioned_eta: float | None = None,
        adaptive_anchor_gated_anchor_ratio_per_key_frame: float | None = None,
        adaptive_anchor_gated_min_per_key_frame: int | None = None,
        adaptive_anchor_gated_max_per_key_frame: int | None = None,
        adaptive_anchor_always_include_self_frame: bool | None = None,
        adaptive_anchor_profile: bool | None = None,
        adaptive_anchor_topm_frames: int | None = None,
        adaptive_anchor_random_seed: int | None = None,
        adaptive_anchor_debug: bool | None = None,
    ) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        self.last_merged_tokens = 0
        self.last_sparse_sparsity = None
        self.last_adaptive_anchor_debug = None
        self.last_adaptive_anchor_kv_tokens = None
        self.last_adaptive_anchor_patch_tokens = None

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        adaptive_enabled = self.use_adaptive_kv_anchor if use_adaptive_kv_anchor is None else use_adaptive_kv_anchor
        if adaptive_enabled:
            if global_merging is not None:
                raise ValueError("Adaptive K/V anchors and token merging are mutually exclusive")
            if sparse_attention:
                raise ValueError("Adaptive K/V anchors and sparse attention are mutually exclusive")
            if inter_frame_only_attention:
                raise ValueError("Adaptive K/V anchors and inter-frame-only attention are mutually exclusive")
            if patch_grid_size is None:
                raise ValueError("patch_grid_size is required when adaptive K/V anchors are enabled")
            return self._adaptive_kv_anchor_attention(
                q=q,
                k=k,
                v=v,
                patch_grid_size=patch_grid_size,
                num_special_tokens=num_special_tokens,
                original_num_tokens=N,
                embed_dim=C,
                adaptive_anchor_ratio=(
                    self.adaptive_anchor_ratio if adaptive_anchor_ratio is None else adaptive_anchor_ratio
                ),
                adaptive_anchor_total=(
                    self.adaptive_anchor_total if adaptive_anchor_total is None else adaptive_anchor_total
                ),
                adaptive_anchor_min_per_frame=(
                    self.adaptive_anchor_min_per_frame
                    if adaptive_anchor_min_per_frame is None
                    else adaptive_anchor_min_per_frame
                ),
                adaptive_anchor_tau=self.adaptive_anchor_tau if adaptive_anchor_tau is None else adaptive_anchor_tau,
                adaptive_anchor_uniform_mix=(
                    self.adaptive_anchor_uniform_mix
                    if adaptive_anchor_uniform_mix is None
                    else adaptive_anchor_uniform_mix
                ),
                adaptive_anchor_strategy=(
                    self.adaptive_anchor_strategy
                    if adaptive_anchor_strategy is None
                    else adaptive_anchor_strategy
                ),
                adaptive_anchor_score_alpha_cross=(
                    self.adaptive_anchor_score_alpha_cross
                    if adaptive_anchor_score_alpha_cross is None
                    else adaptive_anchor_score_alpha_cross
                ),
                adaptive_anchor_score_beta_intra=(
                    self.adaptive_anchor_score_beta_intra
                    if adaptive_anchor_score_beta_intra is None
                    else adaptive_anchor_score_beta_intra
                ),
                adaptive_anchor_score_mode=(
                    self.adaptive_anchor_score_mode
                    if adaptive_anchor_score_mode is None
                    else adaptive_anchor_score_mode
                ),
                adaptive_anchor_proxy_quota_ratio=(
                    self.adaptive_anchor_proxy_quota_ratio
                    if adaptive_anchor_proxy_quota_ratio is None
                    else adaptive_anchor_proxy_quota_ratio
                ),
                adaptive_anchor_intra_source=(
                    self.adaptive_anchor_intra_source
                    if adaptive_anchor_intra_source is None
                    else adaptive_anchor_intra_source
                ),
                adaptive_anchor_frame_budget_mode=(
                    self.adaptive_anchor_frame_budget_mode
                    if adaptive_anchor_frame_budget_mode is None
                    else adaptive_anchor_frame_budget_mode
                ),
                adaptive_anchor_frame_budget_top_frac=(
                    self.adaptive_anchor_frame_budget_top_frac
                    if adaptive_anchor_frame_budget_top_frac is None
                    else adaptive_anchor_frame_budget_top_frac
                ),
                adaptive_anchor_frame_budget_lambda_intra=(
                    self.adaptive_anchor_frame_budget_lambda_intra
                    if adaptive_anchor_frame_budget_lambda_intra is None
                    else adaptive_anchor_frame_budget_lambda_intra
                ),
                adaptive_anchor_frame_budget_lambda_reg=(
                    self.adaptive_anchor_frame_budget_lambda_reg
                    if adaptive_anchor_frame_budget_lambda_reg is None
                    else adaptive_anchor_frame_budget_lambda_reg
                ),
                adaptive_anchor_frame_budget_reg_topm=(
                    self.adaptive_anchor_frame_budget_reg_topm
                    if adaptive_anchor_frame_budget_reg_topm is None
                    else adaptive_anchor_frame_budget_reg_topm
                ),
                adaptive_anchor_reg_patch_topk_ratio=(
                    self.adaptive_anchor_reg_patch_topk_ratio
                    if adaptive_anchor_reg_patch_topk_ratio is None
                    else adaptive_anchor_reg_patch_topk_ratio
                ),
                adaptive_anchor_reg_patch_topk_min=(
                    self.adaptive_anchor_reg_patch_topk_min
                    if adaptive_anchor_reg_patch_topk_min is None
                    else adaptive_anchor_reg_patch_topk_min
                ),
                adaptive_anchor_reg_patch_topk_max=(
                    self.adaptive_anchor_reg_patch_topk_max
                    if adaptive_anchor_reg_patch_topk_max is None
                    else adaptive_anchor_reg_patch_topk_max
                ),
                adaptive_anchor_reg_patch_conf_power=(
                    self.adaptive_anchor_reg_patch_conf_power
                    if adaptive_anchor_reg_patch_conf_power is None
                    else adaptive_anchor_reg_patch_conf_power
                ),
                adaptive_anchor_reg_patch_min_conf=(
                    self.adaptive_anchor_reg_patch_min_conf
                    if adaptive_anchor_reg_patch_min_conf is None
                    else adaptive_anchor_reg_patch_min_conf
                ),
                adaptive_anchor_query_conditioned_eta=(
                    self.adaptive_anchor_query_conditioned_eta
                    if adaptive_anchor_query_conditioned_eta is None
                    else adaptive_anchor_query_conditioned_eta
                ),
                adaptive_anchor_gated_anchor_ratio_per_key_frame=(
                    self.adaptive_anchor_gated_anchor_ratio_per_key_frame
                    if adaptive_anchor_gated_anchor_ratio_per_key_frame is None
                    else adaptive_anchor_gated_anchor_ratio_per_key_frame
                ),
                adaptive_anchor_gated_min_per_key_frame=(
                    self.adaptive_anchor_gated_min_per_key_frame
                    if adaptive_anchor_gated_min_per_key_frame is None
                    else adaptive_anchor_gated_min_per_key_frame
                ),
                adaptive_anchor_gated_max_per_key_frame=(
                    self.adaptive_anchor_gated_max_per_key_frame
                    if adaptive_anchor_gated_max_per_key_frame is None
                    else adaptive_anchor_gated_max_per_key_frame
                ),
                adaptive_anchor_always_include_self_frame=(
                    self.adaptive_anchor_always_include_self_frame
                    if adaptive_anchor_always_include_self_frame is None
                    else adaptive_anchor_always_include_self_frame
                ),
                adaptive_anchor_profile=(
                    self.adaptive_anchor_profile
                    if adaptive_anchor_profile is None
                    else adaptive_anchor_profile
                ),
                adaptive_anchor_topm_frames=(
                    self.adaptive_anchor_topm_frames
                    if adaptive_anchor_topm_frames is None
                    else adaptive_anchor_topm_frames
                ),
                adaptive_anchor_random_seed=(
                    self.adaptive_anchor_random_seed
                    if adaptive_anchor_random_seed is None
                    else adaptive_anchor_random_seed
                ),
                adaptive_anchor_debug=(
                    self.adaptive_anchor_debug if adaptive_anchor_debug is None else adaptive_anchor_debug
                ),
            )

        if inter_frame_only_attention:
            if patch_grid_size is None:
                raise ValueError("patch_grid_size is required for inter-frame-only attention")
            patch_h, patch_w = patch_grid_size
            tokens_per_frame = patch_h * patch_w + num_special_tokens
            if tokens_per_frame <= 0 or N % tokens_per_frame != 0:
                raise ValueError(
                    "Cannot infer frame layout for inter-frame-only attention: "
                    f"{N=} is not divisible by {tokens_per_frame=}"
                )
            num_frames = N // tokens_per_frame
            x = self._inter_frame_only_attention(q, k, v, num_frames, tokens_per_frame)
            x = x.transpose(1, 2)
            return x.reshape([B, N, C])

        unmerge = None
        if sparse_attention:
            if patch_grid_size is None:
                raise ValueError("patch_grid_size is required when sparse_attention is enabled")
            if self.training:
                raise NotImplementedError("Sparse attention is currently inference-only")
            if global_merging is not None:
                raise ValueError("Sparse attention and token merging are mutually exclusive")
            x, sparsity = sparse_global_attention(
                q=q,
                k=k,
                v=v,
                patch_grid_size=patch_grid_size,
                num_special_tokens=num_special_tokens,
                sparse_ratio=sparse_ratio,
                cdf_threshold=sparse_cdf_threshold,
                pool_mode=sparse_pool_mode,
            )
            self.last_sparse_sparsity = float(sparsity.detach().float().cpu())
            x = x.transpose(1, 2)
            return x.reshape([B, N, C])

        if global_merging is not None and not getattr(self, "disable_global_merging", False):
            if patch_grid_size is None:
                raise ValueError("patch_grid_size is required when global_merging is enabled")
            patch_h, patch_w = patch_grid_size
            generator = torch.Generator(device=qkv.device)
            generator.manual_seed(int(getattr(self, "merge_random_seed", 33)))
            r = int(N * self.merge_ratio)

            merge, unmerge = token_merge_bipartite2d(
                qkv if merge_metric is None else merge_metric,
                w=patch_w,
                h=patch_h,
                sx=2,
                sy=2,
                r=r,
                no_rand=False,
                generator=generator,
                enable_protection=True,
                num_special_tokens=num_special_tokens,
                merge_eligible_mask=getattr(self, "merge_eligible_mask", None),
            )
            if getattr(self, "record_merge_trace", False):
                source_indices = getattr(merge, "selected_source_indices", None)
                destination_indices = getattr(merge, "selected_destination_indices", None)
                if source_indices is not None:
                    self.last_merge_source_count = int(source_indices.numel())
                    self.last_merge_source_checksum = int(source_indices.long().sum().item())
                    self.last_merge_source_indices = source_indices.detach().cpu()
                if destination_indices is not None:
                    self.last_merge_destination_checksum = int(destination_indices.long().sum().item())
                    self.last_merge_destination_indices = destination_indices.detach().cpu()

            k_merge_in = k.transpose(1, 2).reshape(B, N, C)
            v_merge_in = v.transpose(1, 2).reshape(B, N, C)
            if getattr(self, "merge_kv_only", False):
                k_out, v_out = merge(k_merge_in, mode="mean", extra_tensors=v_merge_in)
                merged_tokens = k_out.shape[1]
                self.last_merged_tokens = N - merged_tokens
                k = k_out.reshape(B, merged_tokens, self.num_heads, C // self.num_heads).transpose(1, 2)
                v = v_out.reshape(B, merged_tokens, self.num_heads, C // self.num_heads).transpose(1, 2)
                # Q stays full-length, so every patch keeps its output identity.
                unmerge = None
            else:
                q_merge_in = q.transpose(1, 2).reshape(B, N, C)
                q_out, k_out, v_out = merge(
                    q_merge_in,
                    mode="mean",
                    extra_tensors=k_merge_in,
                    extra_tensors_2=v_merge_in,
                )
                N = q_out.shape[1]
                self.last_merged_tokens = qkv.shape[1] - N
                q = q_out.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
                k = k_out.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
                v = v_out.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        x = x.reshape([B, N, C])
        if unmerge is not None:
            x = unmerge(x)
        return x

    def _adaptive_kv_anchor_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        patch_grid_size: tuple[int, int],
        num_special_tokens: int,
        original_num_tokens: int,
        embed_dim: int,
        adaptive_anchor_ratio: float,
        adaptive_anchor_total: int | None,
        adaptive_anchor_min_per_frame: int,
        adaptive_anchor_tau: float,
        adaptive_anchor_uniform_mix: float,
        adaptive_anchor_strategy: str,
        adaptive_anchor_score_alpha_cross: float,
        adaptive_anchor_score_beta_intra: float,
        adaptive_anchor_score_mode: str,
        adaptive_anchor_proxy_quota_ratio: float,
        adaptive_anchor_intra_source: str,
        adaptive_anchor_frame_budget_mode: str,
        adaptive_anchor_frame_budget_top_frac: float,
        adaptive_anchor_frame_budget_lambda_intra: float,
        adaptive_anchor_frame_budget_lambda_reg: float,
        adaptive_anchor_frame_budget_reg_topm: int,
        adaptive_anchor_reg_patch_topk_ratio: float,
        adaptive_anchor_reg_patch_topk_min: int,
        adaptive_anchor_reg_patch_topk_max: int,
        adaptive_anchor_reg_patch_conf_power: float,
        adaptive_anchor_reg_patch_min_conf: float,
        adaptive_anchor_query_conditioned_eta: float,
        adaptive_anchor_gated_anchor_ratio_per_key_frame: float,
        adaptive_anchor_gated_min_per_key_frame: int,
        adaptive_anchor_gated_max_per_key_frame: int,
        adaptive_anchor_always_include_self_frame: bool,
        adaptive_anchor_profile: bool,
        adaptive_anchor_topm_frames: int | None,
        adaptive_anchor_random_seed: int,
        adaptive_anchor_debug: bool,
    ) -> Tensor:
        strategy = adaptive_anchor_strategy.replace("-", "_").lower()
        patch_h, patch_w = patch_grid_size
        patch_count = patch_h * patch_w
        tokens_per_frame = patch_count + num_special_tokens
        if patch_count <= 0:
            raise ValueError("adaptive K/V anchors require at least one patch token per frame")
        if tokens_per_frame <= 0 or original_num_tokens % tokens_per_frame != 0:
            raise ValueError(
                "Cannot infer frame layout for adaptive K/V anchors: "
                f"{original_num_tokens=} is not divisible by {tokens_per_frame=}"
            )
        if not 0.0 <= adaptive_anchor_ratio <= 1.0:
            raise ValueError(f"adaptive_anchor_ratio must be in [0, 1], got {adaptive_anchor_ratio}")
        if adaptive_anchor_total is not None and adaptive_anchor_total < 0:
            raise ValueError(f"adaptive_anchor_total must be non-negative or None, got {adaptive_anchor_total}")
        if adaptive_anchor_min_per_frame < 0:
            raise ValueError(
                "adaptive_anchor_min_per_frame must be non-negative, "
                f"got {adaptive_anchor_min_per_frame}"
            )
        if adaptive_anchor_tau <= 0.0:
            raise ValueError(f"adaptive_anchor_tau must be positive, got {adaptive_anchor_tau}")
        if not 0.0 <= adaptive_anchor_uniform_mix <= 1.0:
            raise ValueError(
                "adaptive_anchor_uniform_mix must be in [0, 1], "
                f"got {adaptive_anchor_uniform_mix}"
            )
        if strategy not in ADAPTIVE_ANCHOR_STRATEGIES:
            raise ValueError(
                f"adaptive_anchor_strategy must be one of {sorted(ADAPTIVE_ANCHOR_STRATEGIES)}, "
                f"got {adaptive_anchor_strategy!r}"
            )

        batch_size = q.shape[0]
        num_frames = original_num_tokens // tokens_per_frame
        if strategy in {
            "all_frame_intra",
            "lifting",
            "frame_pair_gated",
            "hybrid",
            "random_frame_intra",
            "register_gated_intra",
            "register_gated_intra_query",
            "temporal_neighbor_intra",
            "oracle_frame_intra",
            "quota_intra_proxy",
        }:
            x, debug_payload = register_mediated_anchor_attention(
                q=q,
                k=k,
                v=v,
                patch_grid_size=patch_grid_size,
                num_special_tokens=num_special_tokens,
                anchor_ratio=adaptive_anchor_ratio,
                anchor_total=adaptive_anchor_total,
                anchor_min_per_frame=adaptive_anchor_min_per_frame,
                anchor_tau=adaptive_anchor_tau,
                anchor_uniform_mix=adaptive_anchor_uniform_mix,
                anchor_mode=strategy,
                score_mode=adaptive_anchor_score_mode,
                proxy_quota_ratio=adaptive_anchor_proxy_quota_ratio,
                intra_source=adaptive_anchor_intra_source,
                precomputed_intra_scores=self.precomputed_intra_scores,
                frame_budget_mode=adaptive_anchor_frame_budget_mode,
                frame_budget_top_frac=adaptive_anchor_frame_budget_top_frac,
                frame_budget_lambda_intra=adaptive_anchor_frame_budget_lambda_intra,
                frame_budget_lambda_reg=adaptive_anchor_frame_budget_lambda_reg,
                frame_budget_reg_topm=adaptive_anchor_frame_budget_reg_topm,
                reg_patch_topk_ratio=adaptive_anchor_reg_patch_topk_ratio,
                reg_patch_topk_min=adaptive_anchor_reg_patch_topk_min,
                reg_patch_topk_max=adaptive_anchor_reg_patch_topk_max,
                reg_patch_conf_power=adaptive_anchor_reg_patch_conf_power,
                reg_patch_min_conf=adaptive_anchor_reg_patch_min_conf,
                query_conditioned_eta=adaptive_anchor_query_conditioned_eta,
                gated_anchor_ratio_per_key_frame=adaptive_anchor_gated_anchor_ratio_per_key_frame,
                gated_min_per_key_frame=adaptive_anchor_gated_min_per_key_frame,
                gated_max_per_key_frame=adaptive_anchor_gated_max_per_key_frame,
                always_include_self_frame=adaptive_anchor_always_include_self_frame,
                alpha_cross=adaptive_anchor_score_alpha_cross,
                beta_intra=adaptive_anchor_score_beta_intra,
                topm_frames=adaptive_anchor_topm_frames,
                random_seed=adaptive_anchor_random_seed,
                scale=self.scale,
                profile=adaptive_anchor_profile,
                debug=adaptive_anchor_debug,
            )
            self.last_adaptive_anchor_kv_tokens = int(debug_payload["kv_token_count"])
            self.last_adaptive_anchor_patch_tokens = int(debug_payload["anchor_budget"])
            if adaptive_anchor_debug:
                self.last_adaptive_anchor_debug = _detach_debug_payload(debug_payload)
            x = x.transpose(1, 2)
            return x.reshape([batch_size, original_num_tokens, embed_dim])

        with torch.no_grad():
            patch_scores, frame_scores = self._adaptive_anchor_scores(
                q=q.detach(),
                k=k.detach(),
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                num_special_tokens=num_special_tokens,
                patch_count=patch_count,
                strategy=strategy,
            )
            kv_indices, anchor_counts, anchor_budget = self._select_adaptive_anchor_indices(
                patch_scores=patch_scores,
                frame_scores=frame_scores,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                num_special_tokens=num_special_tokens,
                patch_count=patch_count,
                patch_grid_size=patch_grid_size,
                adaptive_anchor_ratio=adaptive_anchor_ratio,
                adaptive_anchor_total=adaptive_anchor_total,
                adaptive_anchor_min_per_frame=adaptive_anchor_min_per_frame,
                adaptive_anchor_tau=adaptive_anchor_tau,
                adaptive_anchor_uniform_mix=adaptive_anchor_uniform_mix,
                strategy=strategy,
                random_seed=int(adaptive_anchor_random_seed),
            )

        gather_index = kv_indices[:, None, :, None].expand(batch_size, k.shape[1], kv_indices.shape[1], k.shape[-1])
        compressed_k = k.gather(dim=2, index=gather_index)
        compressed_v = v.gather(dim=2, index=gather_index)
        self.last_adaptive_anchor_kv_tokens = int(kv_indices.shape[1])
        self.last_adaptive_anchor_patch_tokens = int(anchor_budget)
        if adaptive_anchor_debug:
            self.last_adaptive_anchor_debug = {
                "num_frames": int(num_frames),
                "tokens_per_frame": int(tokens_per_frame),
                "num_special_tokens": int(num_special_tokens),
                "patch_count": int(patch_count),
                "strategy": strategy,
                "anchor_budget": int(anchor_budget),
                "kv_token_count": int(kv_indices.shape[1]),
                "anchor_counts": anchor_counts.detach().cpu(),
                "kv_indices": kv_indices.detach().cpu(),
                "frame_scores": frame_scores.detach().float().cpu(),
            }

        x = F.scaled_dot_product_attention(q, compressed_k, compressed_v)
        x = x.transpose(1, 2)
        return x.reshape([batch_size, original_num_tokens, embed_dim])

    def _adaptive_anchor_scores(
        self,
        q: Tensor,
        k: Tensor,
        num_frames: int,
        tokens_per_frame: int,
        num_special_tokens: int,
        patch_count: int,
        strategy: str,
    ) -> tuple[Tensor, Tensor]:
        batch_size, num_heads, _, head_dim = q.shape
        score_dtype = torch.float32
        if strategy in {"fixed_grid", "random"}:
            patch_scores = torch.zeros(batch_size, num_frames, patch_count, device=q.device, dtype=score_dtype)
            frame_scores = torch.ones(batch_size, num_frames, device=q.device, dtype=score_dtype)
            return patch_scores, frame_scores

        q_frames = q.reshape(batch_size, num_heads, num_frames, tokens_per_frame, head_dim)
        k_frames = k.reshape(batch_size, num_heads, num_frames, tokens_per_frame, head_dim)
        patch_q = q_frames[:, :, :, num_special_tokens:]
        patch_k = k_frames[:, :, :, num_special_tokens:]

        register_start = 1 if num_special_tokens > 1 else 0
        register_q = q_frames[:, :, :, register_start:num_special_tokens]
        if register_q.shape[-2] == 0:
            register_q = q_frames[:, :, :, :1]

        register_scores = None
        intra_scores = None
        target_logits = 8_000_000
        chunk_denom = max(batch_size * num_heads * patch_count * patch_count, 1)
        frame_chunk = max(1, min(num_frames, target_logits // chunk_denom))

        if strategy in {"register_intra"}:
            register_scores = torch.empty(batch_size, num_frames, patch_count, device=q.device, dtype=score_dtype)
        if strategy in {"register_intra", "intra_only", "proxy_intra"}:
            intra_scores = torch.empty(batch_size, num_frames, patch_count, device=q.device, dtype=score_dtype)

        if register_scores is not None or intra_scores is not None:
            for start in range(0, num_frames, frame_chunk):
                end = min(start + frame_chunk, num_frames)
                if register_scores is not None:
                    reg_logits = (
                        torch.matmul(
                            register_q[:, :, start:end].float(),
                            patch_k[:, :, start:end].float().transpose(-2, -1),
                        )
                        * self.scale
                    )
                    reg_prob = reg_logits.softmax(dim=-1)
                    register_scores[:, start:end] = reg_prob.mean(dim=(1, 3))

                if intra_scores is not None:
                    intra_logits = (
                        torch.matmul(
                            patch_q[:, :, start:end].float(),
                            patch_k[:, :, start:end].float().transpose(-2, -1),
                        )
                        * self.scale
                    )
                    intra_prob = intra_logits.softmax(dim=-1)
                    intra_scores[:, start:end] = intra_prob.mean(dim=(1, 3))

        if strategy == "register_intra":
            register_scores = torch.nan_to_num(register_scores, nan=0.0, posinf=0.0, neginf=0.0)
            intra_scores = torch.nan_to_num(intra_scores, nan=0.0, posinf=0.0, neginf=0.0)
            patch_scores = register_scores + intra_scores
        elif strategy == "intra_only":
            patch_scores = torch.nan_to_num(intra_scores, nan=0.0, posinf=0.0, neginf=0.0)
        elif strategy in {"proxy", "proxy_intra"}:
            proxy_scores = self._register_mediated_proxy_scores(
                q_frames=q_frames,
                k_frames=k_frames,
                register_start=register_start,
                num_special_tokens=num_special_tokens,
                patch_count=patch_count,
                num_frames=num_frames,
            )
            if strategy == "proxy":
                patch_scores = proxy_scores
            else:
                intra_scores = torch.nan_to_num(intra_scores, nan=0.0, posinf=0.0, neginf=0.0)
                patch_scores = self._normalize_scores_per_frame(proxy_scores) + 0.2 * self._normalize_scores_per_frame(
                    intra_scores
                )
        elif strategy == "oracle":
            patch_scores = self._oracle_cross_frame_direct_scores(
                q=q,
                k=k,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                num_special_tokens=num_special_tokens,
                patch_count=patch_count,
            )
        else:
            raise ValueError(f"Unhandled adaptive anchor strategy: {strategy}")

        patch_scores = torch.nan_to_num(patch_scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        frame_scores = patch_scores.amax(dim=-1) + patch_scores.mean(dim=-1)
        frame_scores = torch.nan_to_num(frame_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return patch_scores, frame_scores

    def _register_mediated_proxy_scores(
        self,
        q_frames: Tensor,
        k_frames: Tensor,
        register_start: int,
        num_special_tokens: int,
        patch_count: int,
        num_frames: int,
    ) -> Tensor:
        batch_size, num_heads, _, _, head_dim = q_frames.shape
        register_q = q_frames[:, :, :, register_start:num_special_tokens]
        register_k = k_frames[:, :, :, register_start:num_special_tokens]
        if register_q.shape[-2] == 0:
            register_q = q_frames[:, :, :, :1]
            register_k = k_frames[:, :, :, :1]
        register_count = register_q.shape[-2]
        patch_k = k_frames[:, :, :, num_special_tokens:]

        reg_q_flat = register_q.reshape(batch_size, num_heads, num_frames * register_count, head_dim).float()
        reg_k_flat = register_k.reshape(batch_size, num_heads, num_frames * register_count, head_dim).float()
        reg_logits = torch.matmul(reg_q_flat, reg_k_flat.transpose(-2, -1)) * self.scale
        reg_prob = reg_logits.softmax(dim=-1)
        key_frames = torch.arange(num_frames, device=q_frames.device).repeat_interleave(register_count)
        query_frames = key_frames
        cross_mask = query_frames[:, None] != key_frames[None, :]
        reg_prob = reg_prob * cross_mask.to(dtype=reg_prob.dtype)
        reg_recv = reg_prob.sum(dim=-2).reshape(batch_size, num_heads, num_frames, register_count).mean(dim=1)

        reg_to_patch = torch.empty(
            batch_size,
            num_frames,
            register_count,
            patch_count,
            device=q_frames.device,
            dtype=torch.float32,
        )
        for frame_idx in range(num_frames):
            logits = (
                torch.matmul(
                    register_q[:, :, frame_idx].float(),
                    patch_k[:, :, frame_idx].float().transpose(-2, -1),
                )
                * self.scale
            )
            reg_to_patch[:, frame_idx] = logits.softmax(dim=-1).mean(dim=1)
        scores = (reg_recv[:, :, :, None] * reg_to_patch).sum(dim=2)
        return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _oracle_cross_frame_direct_scores(
        self,
        q: Tensor,
        k: Tensor,
        num_frames: int,
        tokens_per_frame: int,
        num_special_tokens: int,
        patch_count: int,
    ) -> Tensor:
        batch_size, num_heads, _, _ = q.shape
        patch_indices = torch.cat(
            [
                torch.arange(
                    frame_idx * tokens_per_frame + num_special_tokens,
                    (frame_idx + 1) * tokens_per_frame,
                    device=q.device,
                )
                for frame_idx in range(num_frames)
            ]
        )
        patch_query_frames = torch.arange(num_frames, device=q.device).repeat_interleave(patch_count)
        patch_key_frames = patch_query_frames
        scores = torch.zeros(batch_size, num_heads, num_frames * patch_count, device=q.device, dtype=torch.float32)
        key_t = k.float().transpose(-2, -1)
        query_chunk = max(1, min(patch_indices.numel(), 8_000_000 // max(batch_size * num_heads * k.shape[-2], 1)))
        for start in range(0, patch_indices.numel(), query_chunk):
            end = min(start + query_chunk, patch_indices.numel())
            query_indices = patch_indices[start:end]
            logits = torch.matmul(q[:, :, query_indices].float(), key_t) * self.scale
            probabilities = logits.softmax(dim=-1)
            patch_prob = probabilities[..., patch_indices]
            mask = patch_query_frames[start:end, None] != patch_key_frames[None, :]
            patch_prob = patch_prob * mask.to(dtype=patch_prob.dtype)
            scores += patch_prob.sum(dim=-2)
        return torch.nan_to_num(scores.mean(dim=1).reshape(batch_size, num_frames, patch_count), nan=0.0)

    @staticmethod
    def _normalize_scores_per_frame(scores: Tensor) -> Tensor:
        scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        denominator = scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return scores / denominator

    def _select_adaptive_anchor_indices(
        self,
        patch_scores: Tensor,
        frame_scores: Tensor,
        num_frames: int,
        tokens_per_frame: int,
        num_special_tokens: int,
        patch_count: int,
        patch_grid_size: tuple[int, int],
        adaptive_anchor_ratio: float,
        adaptive_anchor_total: int | None,
        adaptive_anchor_min_per_frame: int,
        adaptive_anchor_tau: float,
        adaptive_anchor_uniform_mix: float,
        strategy: str,
        random_seed: int,
    ) -> tuple[Tensor, Tensor, int]:
        batch_size = patch_scores.shape[0]
        total_patch_tokens = num_frames * patch_count
        if adaptive_anchor_total is None:
            anchor_budget = int(math.ceil(total_patch_tokens * adaptive_anchor_ratio))
        else:
            anchor_budget = int(adaptive_anchor_total)
        min_per_frame = min(int(adaptive_anchor_min_per_frame), patch_count)
        anchor_budget = max(anchor_budget, num_frames * min_per_frame)
        anchor_budget = min(anchor_budget, total_patch_tokens)

        device = patch_scores.device
        anchor_indices = torch.empty(batch_size, anchor_budget, device=device, dtype=torch.long)
        anchor_counts = torch.empty(batch_size, num_frames, device=device, dtype=torch.long)
        for batch_idx in range(batch_size):
            if strategy in {"fixed_grid", "intra_only", "random"}:
                counts = self._allocate_uniform_anchor_counts(
                    num_frames=num_frames,
                    patch_count=patch_count,
                    anchor_budget=anchor_budget,
                    min_per_frame=min_per_frame,
                    device=device,
                )
            else:
                counts = self._allocate_adaptive_anchor_counts(
                    frame_scores=frame_scores[batch_idx],
                    num_frames=num_frames,
                    patch_count=patch_count,
                    anchor_budget=anchor_budget,
                    min_per_frame=min_per_frame,
                    tau=adaptive_anchor_tau,
                    uniform_mix=adaptive_anchor_uniform_mix,
                )
            anchor_counts[batch_idx] = counts
            selected_per_frame = []
            for frame_idx, keep_count_tensor in enumerate(counts):
                keep_count = int(keep_count_tensor.item())
                if keep_count == 0:
                    continue
                if keep_count >= patch_count:
                    patch_indices = torch.arange(patch_count, device=device, dtype=torch.long)
                elif strategy == "fixed_grid":
                    patch_indices = self._fixed_grid_patch_indices(
                        keep_count=keep_count,
                        patch_grid_size=patch_grid_size,
                        device=device,
                    )
                elif strategy == "random":
                    generator = torch.Generator(device=device)
                    generator.manual_seed(random_seed + batch_idx * 1009 + frame_idx)
                    patch_indices = torch.randperm(patch_count, device=device, generator=generator)[:keep_count]
                else:
                    patch_indices = patch_scores[batch_idx, frame_idx].topk(
                        keep_count,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    ).indices
                selected_per_frame.append(frame_idx * tokens_per_frame + num_special_tokens + patch_indices)
            if selected_per_frame:
                selected = torch.cat(selected_per_frame, dim=0)
            else:
                selected = torch.empty(0, device=device, dtype=torch.long)
            if selected.numel() != anchor_budget:
                raise RuntimeError(
                    "Adaptive K/V anchor selection produced an unexpected count: "
                    f"{selected.numel()} vs {anchor_budget}"
                )
            anchor_indices[batch_idx] = selected

        if num_special_tokens > 0:
            frame_offsets = torch.arange(num_frames, device=device, dtype=torch.long) * tokens_per_frame
            special_offsets = torch.arange(num_special_tokens, device=device, dtype=torch.long)
            special_indices = (frame_offsets[:, None] + special_offsets[None, :]).reshape(1, -1)
            special_indices = special_indices.expand(batch_size, -1)
            kv_indices = torch.cat([special_indices, anchor_indices], dim=1)
        else:
            kv_indices = anchor_indices
        return kv_indices, anchor_counts, anchor_budget

    @staticmethod
    def _allocate_uniform_anchor_counts(
        num_frames: int,
        patch_count: int,
        anchor_budget: int,
        min_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        counts = torch.full((num_frames,), min_per_frame, device=device, dtype=torch.long)
        remaining = anchor_budget - int(counts.sum().item())
        if remaining <= 0:
            return counts
        capacity = torch.full((num_frames,), patch_count - min_per_frame, device=device, dtype=torch.long)
        base = remaining // num_frames
        extra = torch.minimum(torch.full_like(counts, base), capacity)
        counts += extra
        leftover = anchor_budget - int(counts.sum().item())
        if leftover <= 0:
            return counts
        available = counts < patch_count
        while leftover > 0 and bool(available.any().item()):
            take = min(leftover, int(available.sum().item()))
            chosen = torch.nonzero(available, as_tuple=False).flatten()[:take]
            counts[chosen] += 1
            leftover -= take
            available = counts < patch_count
        return counts

    @staticmethod
    def _fixed_grid_patch_indices(
        keep_count: int,
        patch_grid_size: tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        patch_h, patch_w = patch_grid_size
        patch_count = patch_h * patch_w
        if keep_count >= patch_count:
            return torch.arange(patch_count, device=device, dtype=torch.long)
        if keep_count <= 0:
            return torch.empty(0, device=device, dtype=torch.long)

        rows = max(1, min(patch_h, int(round(math.sqrt(keep_count * patch_h / max(patch_w, 1))))))
        cols = max(1, min(patch_w, int(math.ceil(keep_count / rows))))
        while rows * cols < keep_count and rows < patch_h:
            rows += 1
        while rows * cols < keep_count and cols < patch_w:
            cols += 1

        row_idx = torch.linspace(0, patch_h - 1, rows, device=device).round().long().unique()
        col_idx = torch.linspace(0, patch_w - 1, cols, device=device).round().long().unique()
        candidates = (row_idx[:, None] * patch_w + col_idx[None, :]).flatten().unique(sorted=True)
        if candidates.numel() >= keep_count:
            select = torch.linspace(0, candidates.numel() - 1, keep_count, device=device).round().long()
            return candidates[select]

        fallback = torch.linspace(0, patch_count - 1, keep_count, device=device).round().long().unique(sorted=True)
        combined = torch.cat([candidates, fallback]).unique(sorted=True)
        if combined.numel() >= keep_count:
            return combined[:keep_count]
        padding = torch.arange(patch_count, device=device, dtype=torch.long)
        padding = padding[~torch.isin(padding, combined)]
        return torch.cat([combined, padding[: keep_count - combined.numel()]])

    def _allocate_adaptive_anchor_counts(
        self,
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
        scores = torch.nan_to_num(frame_scores.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if float(scores.sum().item()) <= 0.0:
            probs = torch.full((num_frames,), 1.0 / num_frames, device=device, dtype=torch.float32)
        else:
            logits = torch.log(scores + 1.0e-12) / tau
            probs = logits.softmax(dim=0)
        if uniform_mix > 0.0:
            probs = (1.0 - uniform_mix) * probs + uniform_mix / num_frames

        raw_extra = probs * remaining
        extra = torch.floor(raw_extra).to(dtype=torch.long)
        extra = torch.minimum(extra, capacity)
        leftover = remaining - int(extra.sum().item())
        fractional = raw_extra - torch.floor(raw_extra)
        while leftover > 0:
            available = extra < capacity
            if not bool(available.any().item()):
                break
            priority = fractional + probs * 1.0e-6
            priority = priority.masked_fill(~available, -float("inf"))
            take = min(leftover, int(available.sum().item()))
            chosen = priority.topk(take, dim=0, largest=True, sorted=False).indices
            extra[chosen] += 1
            leftover -= take
            fractional = probs

        counts += extra
        return counts

    def _inter_frame_only_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        num_frames: int,
        tokens_per_frame: int,
    ) -> Tensor:
        if num_frames < 2:
            raise ValueError("inter-frame-only attention requires at least two frames")

        total_tokens = num_frames * tokens_per_frame
        if q.shape[-2] != total_tokens or k.shape[-2] != total_tokens or v.shape[-2] != total_tokens:
            raise ValueError("inter-frame-only attention expects unmerged full-frame token sequences")

        if (
            os.environ.get("VGGT_OMEGA_USE_FLEX_INTER_FRAME_ONLY") == "1"
            and flex_attention is not None
            and create_block_mask is not None
            and q.is_cuda
        ):
            block_mask = self._get_inter_frame_only_block_mask(
                total_tokens=total_tokens,
                tokens_per_frame=tokens_per_frame,
                device=q.device,
            )
            return flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)

        outputs = []
        for frame_idx in range(num_frames):
            start = frame_idx * tokens_per_frame
            end = start + tokens_per_frame
            q_frame = q[:, :, start:end]
            k_other = torch.cat((k[:, :, :start], k[:, :, end:]), dim=-2)
            v_other = torch.cat((v[:, :, :start], v[:, :, end:]), dim=-2)
            outputs.append(F.scaled_dot_product_attention(q_frame, k_other, v_other))
        return torch.cat(outputs, dim=-2)

    def _get_inter_frame_only_block_mask(
        self,
        total_tokens: int,
        tokens_per_frame: int,
        device: torch.device,
    ):
        block_size = 128
        cache_key = (total_tokens, tokens_per_frame, device.type, device.index, block_size)
        block_mask = self._inter_frame_only_block_mask_cache.get(cache_key)
        if block_mask is not None:
            return block_mask

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx // tokens_per_frame != kv_idx // tokens_per_frame

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=total_tokens,
            KV_LEN=total_tokens,
            device=str(device),
            BLOCK_SIZE=block_size,
        )
        self._inter_frame_only_block_mask_cache[cache_key] = block_mask
        return block_mask


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
