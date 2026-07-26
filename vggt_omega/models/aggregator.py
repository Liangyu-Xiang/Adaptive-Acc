# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
import torch.nn as nn

from vggt_omega.models.adaptive_pair_scope_attention import (
    adaptive_pair_scope_attention_block,
)
from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer
from vggt_omega.models.progressive_attention import (
    ProgressiveAttentionConfig,
    ProgressiveMaskState,
    progressive_attention_block,
    progressive_config_from_dict,
    resolve_progressive_schedule,
)


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
        global_merging: bool = True,
        merging: int | None = 0,
        merge_ratio: float = 0.9,
        merge_random_seed: int = 33,
        register_patch_inter_frame_mode: str = "none",
        register_patch_inter_frame_percent: float = 0.0,
        register_patch_inter_frame_seed: int = 33,
        sparse_attention: bool = False,
        sparse_ratio: float | None = None,
        sparse_cdf_threshold: float | None = None,
        sparse_pool_mode: str = "avg",
        progressive_attention: ProgressiveAttentionConfig | dict | None = None,
        inter_frame_only_layers: tuple[int, ...] = (),
        use_adaptive_kv_anchor: bool = False,
        adaptive_anchor_layers: str | int | tuple[int, ...] | list[int] = "none",
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
        adaptive_anchor_debug_dir: str | Path = "outputs/debug_register_mediated_anchor",
    ) -> None:
        super().__init__()

        if not 0.0 <= merge_ratio <= 1.0:
            raise ValueError(f"merge_ratio must be between 0.0 and 1.0, got {merge_ratio}")
        if sparse_attention and self._merge_is_enabled(global_merging, merging, merge_ratio):
            raise ValueError("Sparse attention and token merging are mutually exclusive")
        if sparse_pool_mode not in {"avg", "max"}:
            raise ValueError(f"sparse_pool_mode must be 'avg' or 'max', got {sparse_pool_mode!r}")

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                    merge_ratio=merge_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                    merge_ratio=merge_ratio,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.global_merging = global_merging
        self.merging = merging
        self.merge_ratio = merge_ratio
        self.merge_random_seed = merge_random_seed
        self.register_patch_inter_frame_mode = "none"
        self.register_patch_inter_frame_percent = 0.0
        self.register_patch_inter_frame_seed = register_patch_inter_frame_seed
        self.sparse_attention = sparse_attention
        self.sparse_ratio = sparse_ratio
        self.sparse_cdf_threshold = sparse_cdf_threshold
        self.sparse_pool_mode = sparse_pool_mode
        self.progressive_attention_config = ProgressiveAttentionConfig()
        self.progressive_layer_schedule = {}
        self._progressive_stage_states: dict[int, ProgressiveMaskState] = {}
        self.last_progressive_attention_stats: dict[int, dict[str, object]] = {}
        self.last_progressive_sample_indices: dict[int, torch.Tensor] = {}
        self.last_adaptive_pair_scope_debug: dict[
            int,
            dict[str, object],
        ] = {}
        self.inter_frame_only_layers: set[int] = set()
        self.use_adaptive_kv_anchor = False
        self.adaptive_anchor_layers: set[int] = set()
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
        self.adaptive_anchor_debug_dir = Path(adaptive_anchor_debug_dir)
        self._adaptive_anchor_debug_step = 0
        self._adaptive_intra_scores: dict[int, torch.Tensor] = {}
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens
        self._register_patch_selection: dict[int, torch.Tensor] = {}
        self.set_register_patch_inter_frame(
            mode=register_patch_inter_frame_mode,
            percent=register_patch_inter_frame_percent,
            seed=register_patch_inter_frame_seed,
        )

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"
        self.set_inter_frame_only_layers(inter_frame_only_layers)
        self.set_adaptive_kv_anchor(
            enabled=use_adaptive_kv_anchor,
            layers=adaptive_anchor_layers,
            ratio=adaptive_anchor_ratio,
            total=adaptive_anchor_total,
            min_per_frame=adaptive_anchor_min_per_frame,
            tau=adaptive_anchor_tau,
            uniform_mix=adaptive_anchor_uniform_mix,
            strategy=adaptive_anchor_strategy,
            score_alpha_cross=adaptive_anchor_score_alpha_cross,
            score_beta_intra=adaptive_anchor_score_beta_intra,
            score_mode=adaptive_anchor_score_mode,
            proxy_quota_ratio=adaptive_anchor_proxy_quota_ratio,
            intra_source=adaptive_anchor_intra_source,
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
            profile=adaptive_anchor_profile,
            topm_frames=adaptive_anchor_topm_frames,
            random_seed=adaptive_anchor_random_seed,
            debug=adaptive_anchor_debug,
            debug_dir=adaptive_anchor_debug_dir,
        )
        self.set_progressive_attention(progressive_attention)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    @staticmethod
    def _merge_is_enabled(global_merging: bool, merging: int | None, merge_ratio: float) -> bool:
        return global_merging and merging is not None and merge_ratio > 0.0

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def set_merge_ratio(self, merge_ratio: float) -> None:
        if not 0.0 <= merge_ratio <= 1.0:
            raise ValueError(f"merge_ratio must be between 0.0 and 1.0, got {merge_ratio}")
        if self.sparse_attention and merge_ratio > 0.0:
            raise ValueError("Sparse attention and token merging are mutually exclusive")
        if (
            self.use_adaptive_kv_anchor
            and self.adaptive_anchor_layers
            and self._merge_is_enabled(self.global_merging, self.merging, merge_ratio)
        ):
            raise ValueError(
                "Adaptive K/V anchors and token merging are mutually exclusive; "
                "disable global_merging, set merging=None, or use merge_ratio=0.0"
            )
        self.merge_ratio = merge_ratio
        for block in self.inter_frame_blocks:
            block.attn.merge_ratio = merge_ratio

    def set_sparse_attention(
        self,
        enabled: bool,
        sparse_ratio: float | None = None,
        sparse_cdf_threshold: float | None = None,
        sparse_pool_mode: str = "avg",
    ) -> None:
        if enabled and self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
            raise ValueError("Sparse attention and token merging are mutually exclusive")
        if enabled and self.use_adaptive_kv_anchor and self.adaptive_anchor_layers:
            raise ValueError("Sparse attention and adaptive K/V anchors are mutually exclusive")
        if sparse_pool_mode not in {"avg", "max"}:
            raise ValueError(f"sparse_pool_mode must be 'avg' or 'max', got {sparse_pool_mode!r}")
        self.sparse_attention = enabled
        self.sparse_ratio = sparse_ratio
        self.sparse_cdf_threshold = sparse_cdf_threshold
        self.sparse_pool_mode = sparse_pool_mode

    def set_progressive_attention(
        self,
        config: ProgressiveAttentionConfig | dict | None,
    ) -> None:
        resolved = progressive_config_from_dict(config)
        incompatible = []
        if resolved.enabled and self._merge_is_enabled(
            self.global_merging,
            self.merging,
            self.merge_ratio,
        ):
            incompatible.append("token merging")
        if resolved.enabled and self.sparse_attention:
            incompatible.append("per-layer sparse attention")
        if (
            resolved.enabled
            and self.use_adaptive_kv_anchor
            and self.adaptive_anchor_layers
        ):
            incompatible.append("adaptive K/V anchors")
        if resolved.enabled and self.inter_frame_only_layers:
            incompatible.append("inter-frame-only attention")
        if incompatible:
            raise ValueError(
                "Progressive attention is mutually exclusive with "
                + ", ".join(incompatible)
            )
        self.progressive_attention_config = resolved
        if (
            resolved.enabled
            and resolved.algorithm == "adaptive_pair_scope"
        ):
            adaptive = resolved.adaptive_pair_scope_config
            assert adaptive is not None
            invalid = sorted(
                layer
                for layer in adaptive.enabled_layers
                if layer < 0 or layer >= self.depth
            )
            if invalid:
                raise ValueError(
                    "adaptive pair-scope enabled_layers out of range "
                    f"0..{self.depth - 1}: {invalid}"
                )
            non_global = sorted(
                layer
                for layer in adaptive.enabled_layers
                if self.inter_frame_attention_types[layer] != "global"
            )
            if non_global:
                raise ValueError(
                    "adaptive pair-scope can only run on global inter-frame "
                    f"layers; non-global layers: {non_global}"
                )
        self.progressive_layer_schedule = (
            resolve_progressive_schedule(
                depth=self.depth,
                inter_frame_attention_types=self.inter_frame_attention_types,
                config=resolved,
            )
            if resolved.enabled
            and resolved.algorithm == "legacy_token_scope"
            else {}
        )
        self._progressive_stage_states.clear()
        self.last_progressive_attention_stats.clear()
        self.last_progressive_sample_indices.clear()
        self.last_adaptive_pair_scope_debug.clear()

    def progressive_attention_metadata(self) -> dict[str, object]:
        config = self.progressive_attention_config
        if config.algorithm == "adaptive_pair_scope":
            adaptive = config.adaptive_pair_scope_config
            if adaptive is None:
                raise RuntimeError(
                    "adaptive pair-scope metadata lacks configuration"
                )
            return {
                "enabled": config.enabled,
                "algorithm": config.algorithm,
                "semantics": (
                    "within_layer_adaptive_pair_scope_reference_v1"
                ),
                "enabled_layers": list(adaptive.enabled_layers),
                "coarse_num_anchors": adaptive.coarse_num_anchors,
                "coarse_stride": adaptive.coarse_stride,
                "routing_score_mode": adaptive.routing_score_mode,
                "coarse_selection_mode": (
                    adaptive.coarse_selection_mode
                ),
                "coarse_keep_ratio": adaptive.coarse_keep_ratio,
                "fine_selection_mode": adaptive.fine_selection_mode,
                "fine_keep_ratio": adaptive.fine_keep_ratio,
                "refine_factor": adaptive.refine_factor,
                "query_chunk_size": adaptive.query_chunk_size,
                "attention_backend": adaptive.backend_type,
                "efficient_sparse_kernel": False,
                "cross_layer_scope_inheritance": False,
            }
        return {
            "enabled": config.enabled,
            "algorithm": config.algorithm,
            "semantics": "exact_token_pair_reference_v2",
            "stage_ranges": [list(stage) for stage in config.stage_ranges],
            "enabled_stages": list(config.enabled_stages),
            "scope_schedule": list(config.scope_schedule),
            "reset_at_stage_boundary": config.reset_at_stage_boundary,
            "final_scope_mode": config.final_scope_mode,
            "require_stage_final_full": config.require_stage_final_full,
            "mask_enabled": config.mask_enabled,
            "profile_components": config.profile_components,
            "sampling": {
                "type": config.sampling_type,
                "random_seed": config.sampling_random_seed,
                "resample_each_stage": config.sampling_resample_each_stage,
                "sample_tokens_before_qkv": True,
                "patch_tokens_only": True,
            },
            "mask": {
                "head_aggregation": "mean",
                "self_weight": config.self_weight,
                "row_weight": config.row_weight,
                "column_weight": config.column_weight,
                "local_weight": config.local_weight,
                "query_neighbor_radius": config.query_neighbor_radius,
                "key_neighbor_radius": config.key_neighbor_radius,
                "row_keep_ratio": config.row_keep_ratio,
                "column_keep_ratio": config.column_keep_ratio,
                "min_pairs_per_query": config.min_pairs_per_query,
                "dilation_query": config.dilation_query,
                "dilation_key": config.dilation_key,
                "representation": config.mask_representation,
                "query_chunk_size": config.mask_query_chunk_size,
                "max_reference_pair_elements": (
                    config.max_reference_pair_elements
                ),
            },
            "layer_schedule": {
                str(layer): {
                    "stage_index": spec.stage_index,
                    "stage_name": spec.stage_name,
                    "stage_global_position": spec.global_position,
                    "stage_global_count": spec.global_count,
                    "scope": spec.scope,
                }
                for layer, spec in self.progressive_layer_schedule.items()
            },
        }

    def set_merge_random_seed(self, seed: int) -> None:
        self.merge_random_seed = int(seed)
        for block in self.inter_frame_blocks:
            block.attn.merge_random_seed = self.merge_random_seed

    def set_inter_frame_only_layers(self, layers: tuple[int, ...] | list[int]) -> None:
        layer_set = {int(layer) for layer in layers}
        invalid = sorted(layer for layer in layer_set if layer < 0 or layer >= self.depth)
        if invalid:
            raise ValueError(f"inter-frame-only layer indices out of range 0..{self.depth - 1}: {invalid}")
        non_global = sorted(
            layer for layer in layer_set if self.inter_frame_attention_types[layer] != "global"
        )
        if non_global:
            raise ValueError(
                "inter-frame-only attention can only be applied to global inter-frame blocks; "
                f"non-global layers: {non_global}"
            )
        if getattr(self, "use_adaptive_kv_anchor", False):
            overlap = sorted(layer_set & getattr(self, "adaptive_anchor_layers", set()))
            if overlap:
                raise ValueError(
                    "inter-frame-only attention and adaptive K/V anchors are mutually exclusive; "
                    f"overlapping layers: {overlap}"
                )
        self.inter_frame_only_layers = layer_set

    def set_adaptive_kv_anchor(
        self,
        enabled: bool,
        layers: str | int | tuple[int, ...] | list[int] = "none",
        ratio: float = 0.25,
        total: int | None = None,
        min_per_frame: int = 1,
        tau: float = 1.0,
        uniform_mix: float = 0.0,
        strategy: str = "lifting",
        score_alpha_cross: float = 1.0,
        score_beta_intra: float = 0.2,
        score_mode: str = "intra",
        proxy_quota_ratio: float = 0.0,
        intra_source: str = "cached_frame_qk",
        frame_budget_mode: str = "hybrid",
        frame_budget_top_frac: float = 0.1,
        frame_budget_lambda_intra: float = 0.7,
        frame_budget_lambda_reg: float = 0.3,
        frame_budget_reg_topm: int = 4,
        reg_patch_topk_ratio: float = 0.1,
        reg_patch_topk_min: int = 8,
        reg_patch_topk_max: int = 64,
        reg_patch_conf_power: float = 1.0,
        reg_patch_min_conf: float = 0.05,
        query_conditioned_eta: float = 0.1,
        gated_anchor_ratio_per_key_frame: float = 0.1,
        gated_min_per_key_frame: int = 4,
        gated_max_per_key_frame: int = 64,
        always_include_self_frame: bool = True,
        profile: bool = False,
        topm_frames: int | None = 4,
        random_seed: int = 33,
        debug: bool = False,
        debug_dir: str | Path = "outputs/debug_register_mediated_anchor",
    ) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"adaptive_anchor_ratio must be in [0, 1], got {ratio}")
        if total is not None and total < 0:
            raise ValueError(f"adaptive_anchor_total must be non-negative or None, got {total}")
        if min_per_frame < 0:
            raise ValueError(f"adaptive_anchor_min_per_frame must be non-negative, got {min_per_frame}")
        if tau <= 0.0:
            raise ValueError(f"adaptive_anchor_tau must be positive, got {tau}")
        if not 0.0 <= uniform_mix <= 1.0:
            raise ValueError(f"adaptive_anchor_uniform_mix must be in [0, 1], got {uniform_mix}")
        valid_strategies = {
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
        strategy = strategy.replace("-", "_").lower()
        if strategy not in valid_strategies:
            raise ValueError(f"adaptive_anchor_strategy must be one of {sorted(valid_strategies)}, got {strategy!r}")
        if topm_frames is not None and topm_frames <= 0:
            topm_frames = None
        if score_mode.replace("-", "_").lower() not in {"intra", "proxy", "linear_fusion", "quota_union"}:
            raise ValueError(
                "adaptive_anchor_score_mode must be one of "
                "['intra', 'proxy', 'linear_fusion', 'quota_union']"
            )
        if not 0.0 <= proxy_quota_ratio <= 1.0:
            raise ValueError(f"adaptive_anchor_proxy_quota_ratio must be in [0, 1], got {proxy_quota_ratio}")
        if intra_source not in {"current_inter_qk", "cached_frame_qk"}:
            raise ValueError(
                "adaptive_anchor_intra_source must be one of "
                "['current_inter_qk', 'cached_frame_qk']"
            )
        if frame_budget_mode not in {"uniform", "intra_concentration", "register_importance", "hybrid"}:
            raise ValueError(
                "adaptive_anchor_frame_budget_mode must be one of "
                "['uniform', 'intra_concentration', 'register_importance', 'hybrid']"
            )
        if not 0.0 < frame_budget_top_frac <= 1.0:
            raise ValueError(
                f"adaptive_anchor_frame_budget_top_frac must be in (0, 1], got {frame_budget_top_frac}"
            )
        if frame_budget_reg_topm <= 0:
            raise ValueError(
                f"adaptive_anchor_frame_budget_reg_topm must be positive, got {frame_budget_reg_topm}"
            )
        if not 0.0 <= reg_patch_topk_ratio <= 1.0:
            raise ValueError(f"adaptive_anchor_reg_patch_topk_ratio must be in [0, 1], got {reg_patch_topk_ratio}")
        if reg_patch_topk_min <= 0:
            raise ValueError(f"adaptive_anchor_reg_patch_topk_min must be positive, got {reg_patch_topk_min}")
        if reg_patch_topk_max <= 0:
            raise ValueError(f"adaptive_anchor_reg_patch_topk_max must be positive, got {reg_patch_topk_max}")
        if reg_patch_topk_max < reg_patch_topk_min:
            raise ValueError(
                "adaptive_anchor_reg_patch_topk_max must be >= adaptive_anchor_reg_patch_topk_min"
            )
        if reg_patch_conf_power < 0.0:
            raise ValueError(
                f"adaptive_anchor_reg_patch_conf_power must be non-negative, got {reg_patch_conf_power}"
            )
        if reg_patch_min_conf < 0.0:
            raise ValueError(
                f"adaptive_anchor_reg_patch_min_conf must be non-negative, got {reg_patch_min_conf}"
            )
        if gated_anchor_ratio_per_key_frame < 0.0:
            raise ValueError(
                "adaptive_anchor_gated_anchor_ratio_per_key_frame must be non-negative, "
                f"got {gated_anchor_ratio_per_key_frame}"
            )
        if gated_min_per_key_frame < 0:
            raise ValueError(
                f"adaptive_anchor_gated_min_per_key_frame must be non-negative, got {gated_min_per_key_frame}"
            )
        if gated_max_per_key_frame <= 0:
            raise ValueError(
                f"adaptive_anchor_gated_max_per_key_frame must be positive, got {gated_max_per_key_frame}"
            )

        layer_set = self._parse_adaptive_anchor_layer_spec(layers)
        if enabled and layer_set:
            layer_set = {
                layer
                for layer in layer_set
                if self.inter_frame_attention_types[layer] == "global"
            }
        if enabled and layer_set:
            overlap = sorted(layer_set & self.inter_frame_only_layers)
            if overlap:
                raise ValueError(
                    "Adaptive K/V anchors and inter-frame-only attention are mutually exclusive; "
                    f"overlapping layers: {overlap}"
                )
            if self.sparse_attention:
                raise ValueError("Adaptive K/V anchors and sparse attention are mutually exclusive")
            if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
                raise ValueError(
                    "Adaptive K/V anchors and token merging are mutually exclusive; "
                    "disable global_merging, set merging=None, or use merge_ratio=0.0"
                )

        self.use_adaptive_kv_anchor = bool(enabled)
        self.adaptive_anchor_layers = layer_set
        self.adaptive_anchor_ratio = ratio
        self.adaptive_anchor_total = total
        self.adaptive_anchor_min_per_frame = int(min_per_frame)
        self.adaptive_anchor_tau = float(tau)
        self.adaptive_anchor_uniform_mix = float(uniform_mix)
        self.adaptive_anchor_strategy = strategy
        self.adaptive_anchor_score_alpha_cross = float(score_alpha_cross)
        self.adaptive_anchor_score_beta_intra = float(score_beta_intra)
        self.adaptive_anchor_score_mode = score_mode.replace("-", "_").lower()
        self.adaptive_anchor_proxy_quota_ratio = float(proxy_quota_ratio)
        self.adaptive_anchor_intra_source = intra_source
        self.adaptive_anchor_frame_budget_mode = frame_budget_mode
        self.adaptive_anchor_frame_budget_top_frac = float(frame_budget_top_frac)
        self.adaptive_anchor_frame_budget_lambda_intra = float(frame_budget_lambda_intra)
        self.adaptive_anchor_frame_budget_lambda_reg = float(frame_budget_lambda_reg)
        self.adaptive_anchor_frame_budget_reg_topm = int(frame_budget_reg_topm)
        self.adaptive_anchor_reg_patch_topk_ratio = float(reg_patch_topk_ratio)
        self.adaptive_anchor_reg_patch_topk_min = int(reg_patch_topk_min)
        self.adaptive_anchor_reg_patch_topk_max = int(reg_patch_topk_max)
        self.adaptive_anchor_reg_patch_conf_power = float(reg_patch_conf_power)
        self.adaptive_anchor_reg_patch_min_conf = float(reg_patch_min_conf)
        self.adaptive_anchor_query_conditioned_eta = float(query_conditioned_eta)
        self.adaptive_anchor_gated_anchor_ratio_per_key_frame = float(gated_anchor_ratio_per_key_frame)
        self.adaptive_anchor_gated_min_per_key_frame = int(gated_min_per_key_frame)
        self.adaptive_anchor_gated_max_per_key_frame = int(gated_max_per_key_frame)
        self.adaptive_anchor_always_include_self_frame = bool(always_include_self_frame)
        self.adaptive_anchor_profile = bool(profile)
        self.adaptive_anchor_topm_frames = None if topm_frames is None else int(topm_frames)
        self.adaptive_anchor_random_seed = int(random_seed)
        self.adaptive_anchor_debug = bool(debug)
        self.adaptive_anchor_debug_dir = Path(debug_dir)

    def _parse_adaptive_anchor_layer_spec(
        self,
        layers: str | int | tuple[int, ...] | list[int] | set[int] | None,
    ) -> set[int]:
        if layers is None:
            return set()
        if isinstance(layers, str):
            spec = layers.strip().lower()
            if spec in {"", "none"}:
                return set()
            if spec == "all":
                return {
                    idx
                    for idx, attention_type in enumerate(self.inter_frame_attention_types)
                    if attention_type == "global"
                }
            parsed: set[int] = set()
            for part in spec.split(","):
                item = part.strip()
                if not item:
                    continue
                if "-" in item:
                    bounds = [bound.strip() for bound in item.split("-", 1)]
                    if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                        raise ValueError(f"Invalid adaptive_anchor_layers range: {part!r}")
                    start, end = int(bounds[0]), int(bounds[1])
                    if start > end:
                        raise ValueError(f"Invalid adaptive_anchor_layers range {part!r}: start > end")
                    parsed.update(range(start, end + 1))
                else:
                    parsed.add(int(item))
            layer_set = parsed
        elif isinstance(layers, int):
            layer_set = {int(layers)}
        else:
            layer_set = {int(layer) for layer in layers}

        invalid = sorted(layer for layer in layer_set if layer < 0 or layer >= self.depth)
        if invalid:
            raise ValueError(f"adaptive_anchor_layers out of range 0..{self.depth - 1}: {invalid}")
        return layer_set

    def set_register_patch_inter_frame(
        self,
        mode: str = "none",
        percent: float = 0.0,
        seed: int | None = None,
    ) -> None:
        valid_modes = {"none", "random", "least-register"}
        if mode not in valid_modes:
            raise ValueError(f"register patch inter-frame mode must be one of {sorted(valid_modes)}, got {mode}")
        if not 0.0 <= percent <= 100.0:
            raise ValueError(f"register patch inter-frame percent must be in [0, 100], got {percent}")
        self.register_patch_inter_frame_mode = mode
        self.register_patch_inter_frame_percent = percent
        if seed is not None:
            self.register_patch_inter_frame_seed = seed

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        self._register_patch_selection.clear()
        self._adaptive_intra_scores.clear()
        self._progressive_stage_states.clear()
        self.last_progressive_attention_stats.clear()
        self.last_progressive_sample_indices.clear()
        self.last_adaptive_pair_scope_debug.clear()
        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
                patch_grid_size,
            )
            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        return outputs, self.patch_token_start

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        self._maybe_store_adaptive_intra_scores(
            tokens=tokens,
            batch_size=batch_size,
            num_frames=num_frames,
            num_tokens=num_tokens,
            block_idx=block_idx,
            rope_sincos=rope_sincos,
        )
        self._maybe_store_register_patch_selection(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            block_idx,
            rope_sincos,
        )
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _maybe_store_adaptive_intra_scores(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        if not self.use_adaptive_kv_anchor:
            return
        if self.adaptive_anchor_intra_source != "cached_frame_qk":
            return
        if block_idx not in self.adaptive_anchor_layers:
            return

        patch_count = num_tokens - self.patch_token_start
        if patch_count <= 0:
            return

        block = self.frame_blocks[block_idx]
        normalized = block.norm1(tokens)
        batch_frames, _, hidden = normalized.shape
        num_heads = block.attn.num_heads
        head_dim = hidden // num_heads
        qkv = block.attn.qkv(normalized).reshape(batch_frames, num_tokens, 3, num_heads, head_dim)
        q, k, _ = torch.unbind(qkv, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if block.attn.use_qk_norm:
            q = block.attn.q_norm(q)
            k = block.attn.k_norm(k)
        q, k = block.attn.apply_rope(q, k, rope_sincos)

        q_patch = q[:, :, self.patch_token_start :].float()
        k_patch = k[:, :, self.patch_token_start :].float()
        col_mass = torch.empty(batch_frames, patch_count, device=tokens.device, dtype=torch.float32)
        denom = max(batch_size * num_frames * num_heads * patch_count * patch_count, 1)
        frame_chunk = max(1, min(batch_frames, 8_000_000 // denom))
        for start in range(0, batch_frames, frame_chunk):
            end = min(start + frame_chunk, batch_frames)
            logits = torch.matmul(q_patch[start:end], k_patch[start:end].transpose(-2, -1)) * block.attn.scale
            intra = logits.softmax(dim=-1)
            col_mass[start:end] = intra.sum(dim=-2).mean(dim=1)
        self._adaptive_intra_scores[block_idx] = col_mass.view(batch_size, num_frames, patch_count)

    def _maybe_store_register_patch_selection(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        percent = self.register_patch_inter_frame_percent
        mode = self.register_patch_inter_frame_mode
        if mode == "none" or percent <= 0.0:
            return
        patch_count = num_tokens - self.patch_token_start
        if patch_count <= 0:
            return
        keep_count = max(1, min(patch_count, round(patch_count * percent / 100.0)))

        if mode == "random":
            generator = torch.Generator(device=tokens.device)
            generator.manual_seed(self.register_patch_inter_frame_seed + block_idx)
            scores = torch.rand(batch_size, num_frames, patch_count, device=tokens.device, generator=generator)
        elif mode == "least-register":
            scores = self._register_to_patch_attention_scores(tokens, block_idx, rope_sincos)
            scores = scores.view(batch_size, num_frames, patch_count)
            scores = -scores
        else:
            raise ValueError(f"Unknown register patch inter-frame mode: {mode}")

        selected = scores.topk(keep_count, dim=-1, largest=True, sorted=False).indices
        self._register_patch_selection[block_idx] = selected

    def _register_to_patch_attention_scores(
        self,
        tokens: torch.Tensor,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        block = self.frame_blocks[block_idx]
        normalized = block.norm1(tokens)
        batch_frames, num_tokens, hidden = normalized.shape
        num_heads = block.attn.num_heads
        head_dim = hidden // num_heads
        qkv = block.attn.qkv(normalized).reshape(batch_frames, num_tokens, 3, num_heads, head_dim)
        q, k, _ = torch.unbind(qkv, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if block.attn.use_qk_norm:
            q = block.attn.q_norm(q)
            k = block.attn.k_norm(k)
        q, k = block.attn.apply_rope(q, k, rope_sincos)

        patch_token_start = self.patch_token_start
        register_q = q[:, :, 1:patch_token_start].float()
        logits = torch.matmul(register_q, k.float().transpose(-2, -1)) * block.attn.scale
        probabilities = logits.softmax(dim=-1)
        patch_scores = probabilities[..., patch_token_start:].mean(dim=(1, 2))
        return patch_scores.to(dtype=tokens.dtype)

    def _run_adaptive_pair_scope_attention_block(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        patch_grid_size: tuple[int, int],
    ) -> torch.Tensor:
        """Run complete two-level routing inside one global attention layer."""

        adaptive = (
            self.progressive_attention_config.adaptive_pair_scope_config
        )
        if adaptive is None:
            raise RuntimeError(
                "adaptive pair-scope layer lacks adaptive configuration"
            )
        flat_tokens = tokens.reshape(
            batch_size,
            num_frames * num_tokens,
            embed_dim,
        )
        result = adaptive_pair_scope_attention_block(
            self.inter_frame_blocks[block_idx],
            flat_tokens,
            num_frames=num_frames,
            tokens_per_frame=num_tokens,
            num_special_tokens=self.patch_token_start,
            patch_grid_size=patch_grid_size,
            layer_index=block_idx,
            config=adaptive,
        )
        full_patch_tokens = num_frames * (
            num_tokens - self.patch_token_start
        )
        logical_patch_pairs = int(
            result.stats.get("final_logical_patch_pairs", 0)
        )
        result.stats["patch_attention_pair_retention_vs_full"] = (
            logical_patch_pairs
            / max(full_patch_tokens * full_patch_tokens, 1)
        )
        self.last_progressive_attention_stats[block_idx] = result.stats
        if result.debug:
            self.last_adaptive_pair_scope_debug[block_idx] = result.debug
        return result.output.view(
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
        )

    def _run_progressive_attention_block(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        patch_grid_size: tuple[int, int],
    ) -> torch.Tensor:
        config = self.progressive_attention_config
        spec = self.progressive_layer_schedule[block_idx]
        if spec.is_stage_first:
            self._progressive_stage_states.pop(spec.stage_index, None)
        previous_state = (
            self._progressive_stage_states.get(spec.stage_index)
            if config.mask_enabled
            else None
        )
        flat_tokens = tokens.reshape(
            batch_size,
            num_frames * num_tokens,
            embed_dim,
        )
        if spec.scope == "full" and config.final_scope_mode == "dense":
            flat_tokens = self.inter_frame_blocks[block_idx](flat_tokens, None)
            full_patch_tokens = num_frames * (
                num_tokens - self.patch_token_start
            )
            total_tokens = num_frames * num_tokens
            self.last_progressive_attention_stats[block_idx] = {
                "stage_index": spec.stage_index,
                "stage_name": spec.stage_name,
                "layer_index": block_idx,
                "stage_global_position": spec.global_position,
                "stage_global_count": spec.global_count,
                "scope": "full",
                "sampled_patch_tokens": full_patch_tokens,
                "special_tokens": num_frames * self.patch_token_start,
                "qkv_projection_tokens": total_tokens,
                "full_patch_tokens": full_patch_tokens,
                "patch_sampling_ratio": 1.0,
                "sample_before_qkv": True,
                "mask_inherited": False,
                "mask_generated": False,
                "attention_backend": "original_dense_sdpa",
                "logical_attention_pairs_per_batch": (
                    total_tokens * total_tokens
                ),
                "evaluated_attention_pairs_per_batch": (
                    total_tokens * total_tokens
                ),
                "patch_attention_pairs_per_batch": (
                    full_patch_tokens * full_patch_tokens
                ),
                "patch_attention_pair_retention_vs_full": 1.0,
            }
            self._progressive_stage_states.pop(spec.stage_index, None)
            return flat_tokens.view(
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
            )

        if (
            spec.scope == "full"
            and config.final_scope_mode == "inherited_sparse"
            and previous_state is None
        ):
            raise RuntimeError(
                f"Progressive full sparse layer {block_idx} has no inherited mask"
            )
        result = progressive_attention_block(
            self.inter_frame_blocks[block_idx],
            flat_tokens,
            num_frames=num_frames,
            tokens_per_frame=num_tokens,
            num_special_tokens=self.patch_token_start,
            patch_grid_size=patch_grid_size,
            layer_spec=spec,
            config=config,
            previous_state=previous_state,
            build_next_mask=bool(
                config.mask_enabled and not spec.is_stage_last
            ),
        )
        if spec.is_stage_last:
            self._progressive_stage_states.pop(spec.stage_index, None)
        elif result.next_state is not None:
            self._progressive_stage_states[spec.stage_index] = result.next_state
        full_patch_tokens = num_frames * (
            num_tokens - self.patch_token_start
        )
        patch_pairs = int(
            result.stats.get("patch_attention_pairs_per_batch", 0)
        )
        result.stats["patch_attention_pair_retention_vs_full"] = (
            patch_pairs / max(full_patch_tokens * full_patch_tokens, 1)
        )
        self.last_progressive_attention_stats[block_idx] = result.stats
        if result.sample_coordinates is not None:
            self.last_progressive_sample_indices[block_idx] = (
                result.sample_coordinates
            )
        return result.output.view(
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
        )

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
        patch_grid_size: tuple[int, int],
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            progressive_config = self.progressive_attention_config
            if (
                progressive_config.enabled
                and progressive_config.algorithm
                == "adaptive_pair_scope"
            ):
                adaptive = progressive_config.adaptive_pair_scope_config
                assert adaptive is not None
                if block_idx in adaptive.enabled_layers:
                    return self._run_adaptive_pair_scope_attention_block(
                        tokens,
                        batch_size=batch_size,
                        num_frames=num_frames,
                        num_tokens=num_tokens,
                        embed_dim=embed_dim,
                        block_idx=block_idx,
                        patch_grid_size=patch_grid_size,
                    )
            progressive_spec = self.progressive_layer_schedule.get(block_idx)
            if (
                progressive_config.enabled
                and progressive_config.algorithm == "legacy_token_scope"
                and progressive_spec is not None
            ):
                return self._run_progressive_attention_block(
                    tokens,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    num_tokens=num_tokens,
                    embed_dim=embed_dim,
                    block_idx=block_idx,
                    patch_grid_size=patch_grid_size,
                )
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            self.inter_frame_blocks[block_idx].attn.merge_random_seed = self.merge_random_seed
            self.inter_frame_blocks[block_idx].attn.precomputed_intra_scores = self._adaptive_intra_scores.get(block_idx)
            inter_frame_only_attention = block_idx in self.inter_frame_only_layers
            adaptive_kv_anchor = self.use_adaptive_kv_anchor and block_idx in self.adaptive_anchor_layers
            if adaptive_kv_anchor:
                if inter_frame_only_attention:
                    raise ValueError(
                        "Adaptive K/V anchors and inter-frame-only attention are mutually exclusive; "
                        f"layer {block_idx}"
                    )
                if self.sparse_attention:
                    raise ValueError("Adaptive K/V anchors and sparse attention are mutually exclusive")
                if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio):
                    raise ValueError(
                        "Adaptive K/V anchors and token merging are mutually exclusive; "
                        "disable global_merging, set merging=None, or use merge_ratio=0.0"
                    )
            global_merging = (
                block_idx
                if self._merge_is_enabled(self.global_merging, self.merging, self.merge_ratio)
                and block_idx >= self.merging
                and not inter_frame_only_attention
                and not adaptive_kv_anchor
                else None
            )
            tokens = self.inter_frame_blocks[block_idx](
                tokens,
                None,
                global_merging=global_merging,
                patch_grid_size=patch_grid_size,
                num_special_tokens=self.patch_token_start,
                sparse_attention=self.sparse_attention and not inter_frame_only_attention,
                sparse_ratio=self.sparse_ratio,
                sparse_cdf_threshold=self.sparse_cdf_threshold,
                sparse_pool_mode=self.sparse_pool_mode,
                inter_frame_only_attention=inter_frame_only_attention,
                use_adaptive_kv_anchor=adaptive_kv_anchor,
                adaptive_anchor_ratio=self.adaptive_anchor_ratio,
                adaptive_anchor_total=self.adaptive_anchor_total,
                adaptive_anchor_min_per_frame=self.adaptive_anchor_min_per_frame,
                adaptive_anchor_tau=self.adaptive_anchor_tau,
                adaptive_anchor_uniform_mix=self.adaptive_anchor_uniform_mix,
                adaptive_anchor_strategy=self.adaptive_anchor_strategy,
                adaptive_anchor_score_alpha_cross=self.adaptive_anchor_score_alpha_cross,
                adaptive_anchor_score_beta_intra=self.adaptive_anchor_score_beta_intra,
                adaptive_anchor_score_mode=self.adaptive_anchor_score_mode,
                adaptive_anchor_proxy_quota_ratio=self.adaptive_anchor_proxy_quota_ratio,
                adaptive_anchor_intra_source=self.adaptive_anchor_intra_source,
                adaptive_anchor_frame_budget_mode=self.adaptive_anchor_frame_budget_mode,
                adaptive_anchor_frame_budget_top_frac=self.adaptive_anchor_frame_budget_top_frac,
                adaptive_anchor_frame_budget_lambda_intra=self.adaptive_anchor_frame_budget_lambda_intra,
                adaptive_anchor_frame_budget_lambda_reg=self.adaptive_anchor_frame_budget_lambda_reg,
                adaptive_anchor_frame_budget_reg_topm=self.adaptive_anchor_frame_budget_reg_topm,
                adaptive_anchor_reg_patch_topk_ratio=self.adaptive_anchor_reg_patch_topk_ratio,
                adaptive_anchor_reg_patch_topk_min=self.adaptive_anchor_reg_patch_topk_min,
                adaptive_anchor_reg_patch_topk_max=self.adaptive_anchor_reg_patch_topk_max,
                adaptive_anchor_reg_patch_conf_power=self.adaptive_anchor_reg_patch_conf_power,
                adaptive_anchor_reg_patch_min_conf=self.adaptive_anchor_reg_patch_min_conf,
                adaptive_anchor_query_conditioned_eta=self.adaptive_anchor_query_conditioned_eta,
                adaptive_anchor_gated_anchor_ratio_per_key_frame=self.adaptive_anchor_gated_anchor_ratio_per_key_frame,
                adaptive_anchor_gated_min_per_key_frame=self.adaptive_anchor_gated_min_per_key_frame,
                adaptive_anchor_gated_max_per_key_frame=self.adaptive_anchor_gated_max_per_key_frame,
                adaptive_anchor_always_include_self_frame=self.adaptive_anchor_always_include_self_frame,
                adaptive_anchor_profile=self.adaptive_anchor_profile,
                adaptive_anchor_topm_frames=self.adaptive_anchor_topm_frames,
                adaptive_anchor_random_seed=self.adaptive_anchor_random_seed,
                adaptive_anchor_debug=self.adaptive_anchor_debug,
            )
            self.inter_frame_blocks[block_idx].attn.precomputed_intra_scores = None
            if adaptive_kv_anchor:
                self._maybe_save_adaptive_anchor_debug(block_idx)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:]
        selected_indices = self._register_patch_selection.pop(block_idx, None)

        if selected_indices is None:
            camera_and_register_tokens = self.inter_frame_blocks[block_idx](camera_and_register_tokens, None)
            return torch.cat(
                [
                    camera_and_register_tokens.view(batch_size, num_frames, patch_token_start, embed_dim),
                    patch_tokens,
                ],
                dim=2,
            )

        selected_indices = selected_indices.to(device=tokens.device)
        keep_count = selected_indices.shape[-1]
        gather_indices = selected_indices.unsqueeze(-1).expand(batch_size, num_frames, keep_count, embed_dim)
        selected_patch_tokens = patch_tokens.gather(dim=2, index=gather_indices)
        inter_frame_tokens = torch.cat(
            [
                camera_and_register_tokens,
                selected_patch_tokens.reshape(batch_size, num_frames * keep_count, embed_dim),
            ],
            dim=1,
        )
        inter_frame_tokens = self.inter_frame_blocks[block_idx](inter_frame_tokens, None)

        camera_and_register_tokens = inter_frame_tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        selected_patch_tokens = inter_frame_tokens[:, num_frames * patch_token_start :].view(
            batch_size, num_frames, keep_count, embed_dim
        )
        patch_tokens = patch_tokens.scatter(dim=2, index=gather_indices, src=selected_patch_tokens)
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)

    def _maybe_save_adaptive_anchor_debug(self, block_idx: int) -> None:
        if not self.adaptive_anchor_debug:
            return
        debug_payload = self.inter_frame_blocks[block_idx].attn.last_adaptive_anchor_debug
        if debug_payload is None:
            return
        debug_dir = Path(self.adaptive_anchor_debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"step_{self._adaptive_anchor_debug_step:06d}_"
            f"layer_{block_idx:02d}_{self.adaptive_anchor_strategy}.pt"
        )
        torch.save(debug_payload, debug_dir / filename)
        self._adaptive_anchor_debug_step += 1


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
