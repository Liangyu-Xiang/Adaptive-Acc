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
        register_patch_inter_frame_mode: str = "none",
        register_patch_inter_frame_percent: float = 0.0,
        register_patch_inter_frame_seed: int = 33,
        sparse_attention: bool = False,
        sparse_ratio: float | None = None,
        sparse_cdf_threshold: float | None = None,
        sparse_pool_mode: str = "avg",
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
        adaptive_anchor_topm_frames: int | None = 4,
        adaptive_anchor_random_seed: int = 33,
        adaptive_anchor_debug: bool = False,
        adaptive_anchor_debug_dir: str | Path = "outputs/debug_register_mediated_anchor",
        use_register_mediated_anchor: bool | None = None,
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
            register_patch_inter_frame_mode=register_patch_inter_frame_mode,
            register_patch_inter_frame_percent=register_patch_inter_frame_percent,
            register_patch_inter_frame_seed=register_patch_inter_frame_seed,
            sparse_attention=sparse_attention,
            sparse_ratio=sparse_ratio,
            sparse_cdf_threshold=sparse_cdf_threshold,
            sparse_pool_mode=sparse_pool_mode,
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
            adaptive_anchor_topm_frames=adaptive_anchor_topm_frames,
            adaptive_anchor_random_seed=adaptive_anchor_random_seed,
            adaptive_anchor_debug=adaptive_anchor_debug,
            adaptive_anchor_debug_dir=adaptive_anchor_debug_dir,
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
