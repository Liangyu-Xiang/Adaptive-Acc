# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
import torch.nn as nn

from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


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
        self.adaptive_anchor_topm_frames = adaptive_anchor_topm_frames
        self.adaptive_anchor_random_seed = int(adaptive_anchor_random_seed)
        self.adaptive_anchor_debug = adaptive_anchor_debug
        self.adaptive_anchor_debug_dir = Path(adaptive_anchor_debug_dir)
        self._adaptive_anchor_debug_step = 0
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
            topm_frames=adaptive_anchor_topm_frames,
            random_seed=adaptive_anchor_random_seed,
            debug=adaptive_anchor_debug,
            debug_dir=adaptive_anchor_debug_dir,
        )

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
            "lifting",
            "frame_pair_gated",
            "hybrid",
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
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            self.inter_frame_blocks[block_idx].attn.merge_random_seed = self.merge_random_seed
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
                adaptive_anchor_topm_frames=self.adaptive_anchor_topm_frames,
                adaptive_anchor_random_seed=self.adaptive_anchor_random_seed,
                adaptive_anchor_debug=self.adaptive_anchor_debug,
            )
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
