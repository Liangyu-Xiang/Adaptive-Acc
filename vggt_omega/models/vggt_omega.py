# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        global_merging: bool = True,
        merging: int | None = 0,
        merge_ratio: float = 0.9,
        merge_random_seed: int = 33,
        first_frame_token_indices: tuple[int, ...] | list[int] | str = (0,),
        register_patch_inter_frame_mode: str = "none",
        register_patch_inter_frame_percent: float = 0.0,
        register_patch_inter_frame_seed: int = 33,
        sparse_attention: bool = False,
        sparse_ratio: float | None = None,
        sparse_cdf_threshold: float | None = None,
        sparse_pool_mode: str = "avg",
        progressive_attention: dict | None = None,
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
        use_register_mediated_anchor: bool | None = None,
        frame_fusion_mode: str = "none",
        frame_fusion_k: int | None = None,
        frame_fusion_max_group_size: int = 5,
        frame_fusion_beta: float = 1.0,
        frame_fusion_start_layer: int = -1,
        frame_fusion_pair_percent: float = 25.0,
        frame_fusion_pool_size: int = 2,
        frame_fusion_group_similarity_threshold: float = 0.0,
        frame_fusion_target_keep_policy: str = "none",
        frame_fusion_target_keep_grid_size: int = 4,
        frame_fusion_target_keep_percent: float = 0.0,
        frame_fusion_target_keep_threshold: float = 0.0,
        frame_fusion_target_keep_seed: int = 33,
        frame_fusion_recompute_each_global: bool = False,
        frame_fusion_recompute_layers: tuple[int, ...] | list[int] | str = (),
        frame_fusion_lambda_cost: float = 0.15,
        frame_fusion_min_keep_ratio: float = 0.4,
        frame_fusion_temporal_window: int = 1,
        frame_fusion_spatial_neighborhood: str = "N8",
        frame_fusion_time_overlap: float = 0.5,
        frame_fusion_reassignment_candidates: int = 8,
        frame_fusion_representative_update: str = "parent",
    ) -> None:
        super().__init__()
        if use_register_mediated_anchor is not None:
            use_adaptive_kv_anchor = bool(use_register_mediated_anchor)

        self.aggregator = Aggregator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            global_merging=global_merging,
            merging=merging,
            merge_ratio=merge_ratio,
            merge_random_seed=merge_random_seed,
            first_frame_token_indices=first_frame_token_indices,
            register_patch_inter_frame_mode=register_patch_inter_frame_mode,
            register_patch_inter_frame_percent=register_patch_inter_frame_percent,
            register_patch_inter_frame_seed=register_patch_inter_frame_seed,
            sparse_attention=sparse_attention,
            sparse_ratio=sparse_ratio,
            sparse_cdf_threshold=sparse_cdf_threshold,
            sparse_pool_mode=sparse_pool_mode,
            progressive_attention=progressive_attention,
            inter_frame_only_layers=inter_frame_only_layers,
            use_adaptive_kv_anchor=use_adaptive_kv_anchor,
            adaptive_anchor_layers=adaptive_anchor_layers,
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
            adaptive_anchor_debug_dir=adaptive_anchor_debug_dir,
            frame_fusion_mode=frame_fusion_mode,
            frame_fusion_k=frame_fusion_k,
            frame_fusion_max_group_size=frame_fusion_max_group_size,
            frame_fusion_beta=frame_fusion_beta,
            frame_fusion_start_layer=frame_fusion_start_layer,
            frame_fusion_pair_percent=frame_fusion_pair_percent,
            frame_fusion_pool_size=frame_fusion_pool_size,
            frame_fusion_group_similarity_threshold=frame_fusion_group_similarity_threshold,
            frame_fusion_target_keep_policy=frame_fusion_target_keep_policy,
            frame_fusion_target_keep_grid_size=frame_fusion_target_keep_grid_size,
            frame_fusion_target_keep_percent=frame_fusion_target_keep_percent,
            frame_fusion_target_keep_threshold=frame_fusion_target_keep_threshold,
            frame_fusion_target_keep_seed=frame_fusion_target_keep_seed,
            frame_fusion_recompute_each_global=frame_fusion_recompute_each_global,
            frame_fusion_recompute_layers=frame_fusion_recompute_layers,
            frame_fusion_lambda_cost=frame_fusion_lambda_cost,
            frame_fusion_min_keep_ratio=frame_fusion_min_keep_ratio,
            frame_fusion_temporal_window=frame_fusion_temporal_window,
            frame_fusion_spatial_neighborhood=frame_fusion_spatial_neighborhood,
            frame_fusion_time_overlap=frame_fusion_time_overlap,
            frame_fusion_reassignment_candidates=frame_fusion_reassignment_candidates,
            frame_fusion_representative_update=frame_fusion_representative_update,
        )
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None

    def set_merge_ratio(self, merge_ratio: float) -> None:
        self.aggregator.set_merge_ratio(merge_ratio)

    def set_sparse_attention(
        self,
        enabled: bool,
        sparse_ratio: float | None = None,
        sparse_cdf_threshold: float | None = None,
        sparse_pool_mode: str = "avg",
    ) -> None:
        self.aggregator.set_sparse_attention(
            enabled=enabled,
            sparse_ratio=sparse_ratio,
            sparse_cdf_threshold=sparse_cdf_threshold,
            sparse_pool_mode=sparse_pool_mode,
        )

    def set_progressive_attention(self, config: dict | None) -> None:
        self.aggregator.set_progressive_attention(config)

    def set_inter_frame_only_layers(self, layers: tuple[int, ...] | list[int]) -> None:
        self.aggregator.set_inter_frame_only_layers(layers)

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
        self.aggregator.set_adaptive_kv_anchor(
            enabled=enabled,
            layers=layers,
            ratio=ratio,
            total=total,
            min_per_frame=min_per_frame,
            tau=tau,
            uniform_mix=uniform_mix,
            strategy=strategy,
            score_alpha_cross=score_alpha_cross,
            score_beta_intra=score_beta_intra,
            score_mode=score_mode,
            proxy_quota_ratio=proxy_quota_ratio,
            intra_source=intra_source,
            frame_budget_mode=frame_budget_mode,
            frame_budget_top_frac=frame_budget_top_frac,
            frame_budget_lambda_intra=frame_budget_lambda_intra,
            frame_budget_lambda_reg=frame_budget_lambda_reg,
            frame_budget_reg_topm=frame_budget_reg_topm,
            reg_patch_topk_ratio=reg_patch_topk_ratio,
            reg_patch_topk_min=reg_patch_topk_min,
            reg_patch_topk_max=reg_patch_topk_max,
            reg_patch_conf_power=reg_patch_conf_power,
            reg_patch_min_conf=reg_patch_min_conf,
            query_conditioned_eta=query_conditioned_eta,
            gated_anchor_ratio_per_key_frame=gated_anchor_ratio_per_key_frame,
            gated_min_per_key_frame=gated_min_per_key_frame,
            gated_max_per_key_frame=gated_max_per_key_frame,
            always_include_self_frame=always_include_self_frame,
            profile=profile,
            topm_frames=topm_frames,
            random_seed=random_seed,
            debug=debug,
            debug_dir=debug_dir,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
