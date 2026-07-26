"""Progressive multi-level attention for long VGGT-Omega sequences.

The implementation keeps the original frame-major tensor layout outside the
attention kernel.  Inside a progressive global block, sampled patch tokens are
placed first and every per-frame camera/register token is appended.  The
attention result is scattered back to the original flattened layout before the
residual connection, so output shapes and token identities are unchanged.

The first implementation is deliberately a semantic reference path.  Mask
prediction, row/column Top-K selection, dilation, and parent expansion all
operate on exact patch-token pairs.  Inherited sparse attention is evaluated
with a dense SDPA mask and is therefore diagnostic rather than an acceleration
claim.  A later efficient backend may regularize this exact logical mask into
execution blocks, but execution blocks must not redefine the routing method.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from vggt_omega.models.adaptive_pair_scope_attention import (
    AdaptivePairScopeConfig,
    adaptive_pair_scope_config_from_dict,
    validate_adaptive_pair_scope_config,
)

if TYPE_CHECKING:
    from vggt_omega.models.layers.block import SelfAttentionBlock


Scope = int | str


@dataclass(frozen=True)
class ProgressiveAttentionConfig:
    enabled: bool = False
    algorithm: str = "legacy_token_scope"
    adaptive_pair_scope_config: AdaptivePairScopeConfig | None = None
    stage_ranges: tuple[tuple[int, int], ...] = ((0, 9), (10, 16), (17, 23))
    enabled_stages: tuple[str, ...] = ("early", "middle", "late")
    scope_schedule: tuple[Scope, ...] = (32, 64, "full")
    reset_at_stage_boundary: bool = True
    final_scope_mode: str = "inherited_sparse"
    require_stage_final_full: bool = True
    mask_enabled: bool = True
    self_weight: float = 0.25
    row_weight: float = 0.25
    column_weight: float = 0.25
    local_weight: float = 0.25
    query_neighbor_radius: int = 1
    key_neighbor_radius: int = 1
    row_keep_ratio: float = 0.25
    column_keep_ratio: float = 0.05
    min_pairs_per_query: int = 4
    dilation_query: int = 1
    dilation_key: int = 1
    mask_representation: str = "dense_token_pair_reference"
    mask_query_chunk_size: int = 128
    max_reference_pair_elements: int = 5_000_000_000
    sampling_type: str = "nested_random_balanced"
    sampling_random_seed: int = 0
    sampling_resample_each_stage: bool = True
    profile_components: bool = False
    save_sample_indices: bool = True
    save_mask_statistics: bool = True


@dataclass(frozen=True)
class ProgressiveLayerSpec:
    layer_index: int
    stage_index: int
    stage_name: str
    global_position: int
    global_count: int
    scope: Scope

    @property
    def is_stage_first(self) -> bool:
        return self.global_position == 0

    @property
    def is_stage_last(self) -> bool:
        return self.global_position == self.global_count - 1


@dataclass
class ProgressiveMaskState:
    patch_indices: Tensor
    pair_mask: Tensor
    highest_score_key: Tensor | None = None


@dataclass
class ProgressiveBlockResult:
    output: Tensor
    next_state: ProgressiveMaskState | None
    stats: dict[str, Any]
    sample_coordinates: Tensor | None = None


@dataclass
class _ComponentTimer:
    enabled: bool
    device: torch.device
    elapsed_ms: dict[str, float] = field(default_factory=dict)

    def measure(self, name: str):
        return _TimerContext(self, name)


class _TimerContext:
    def __init__(self, timer: _ComponentTimer, name: str) -> None:
        self.timer = timer
        self.name = name
        self.start_event = None
        self.start_time = 0.0

    def __enter__(self):
        if not self.timer.enabled:
            return self
        if self.timer.device.type == "cuda":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.timer.enabled or exc_type is not None:
            return False
        if self.timer.device.type == "cuda":
            self.end_event.record()
            self.end_event.synchronize()
            elapsed = float(self.start_event.elapsed_time(self.end_event))
        else:
            elapsed = 1000.0 * (time.perf_counter() - self.start_time)
        self.timer.elapsed_ms[self.name] = (
            self.timer.elapsed_ms.get(self.name, 0.0) + elapsed
        )
        return False


def progressive_config_from_dict(
    config: ProgressiveAttentionConfig | dict[str, Any] | None,
) -> ProgressiveAttentionConfig:
    if config is None:
        result = ProgressiveAttentionConfig()
    elif isinstance(config, ProgressiveAttentionConfig):
        result = config
    elif isinstance(config, dict):
        payload = dict(config)
        algorithm = str(
            payload.pop("algorithm", "legacy_token_scope")
        ).strip().lower()
        if algorithm == "adaptive_pair_scope":
            enabled = bool(payload.pop("enabled", False))
            adaptive_config = adaptive_pair_scope_config_from_dict(payload)
            result = ProgressiveAttentionConfig(
                enabled=enabled,
                algorithm=algorithm,
                adaptive_pair_scope_config=adaptive_config,
            )
            _validate_config(result)
            return result
        if algorithm != "legacy_token_scope":
            raise ValueError(
                "progressive_attention.algorithm must be "
                "'legacy_token_scope' or 'adaptive_pair_scope', got "
                f"{algorithm!r}"
            )
        sampling = dict(payload.pop("sampling", {}) or {})
        special_tokens = dict(payload.pop("special_tokens", {}) or {})
        mask = dict(payload.pop("mask", {}) or {})
        debug = dict(payload.pop("debug", {}) or {})
        payload.pop("fallback", None)
        if sampling:
            if not sampling.get("sample_tokens_before_qkv", True):
                raise ValueError("progressive attention requires sampling before QKV")
            if not sampling.get("patch_tokens_only", True):
                raise ValueError("progressive attention currently samples patch tokens only")
            payload["sampling_type"] = sampling.get(
                "type",
                "nested_random_balanced",
            )
            payload["sampling_random_seed"] = sampling.get("random_seed", 0)
            payload["sampling_resample_each_stage"] = sampling.get(
                "resample_each_stage",
                True,
            )
            unknown_sampling = set(sampling).difference(
                {
                    "type",
                    "random_seed",
                    "resample_each_stage",
                    "sample_tokens_before_qkv",
                    "patch_tokens_only",
                }
            )
            if unknown_sampling:
                raise ValueError(
                    "Unknown progressive sampling keys: "
                    f"{sorted(unknown_sampling)}"
                )
        if special_tokens and (
            not special_tokens.get("keep_camera_tokens", True)
            or not special_tokens.get("keep_register_tokens", True)
        ):
            raise ValueError(
                "progressive attention currently requires all camera/register tokens"
            )
        aliases = {
            "head_aggregation": None,
            "dilation_query": "dilation_query",
            "dilation_key": "dilation_key",
            "representation": "mask_representation",
            "query_chunk_size": "mask_query_chunk_size",
        }
        for key, value in mask.items():
            target = aliases.get(key, key)
            if key == "head_aggregation":
                if value != "mean":
                    raise ValueError(
                        "progressive mask head_aggregation must be 'mean'"
                    )
                continue
            payload[target] = value
        for key in (
            "save_sample_indices",
            "save_mask_statistics",
        ):
            if key in debug:
                payload[key] = debug[key]
        if "stage_ranges" in payload:
            payload["stage_ranges"] = tuple(
                tuple(int(value) for value in pair)
                for pair in payload["stage_ranges"]
            )
        if "enabled_stages" in payload:
            payload["enabled_stages"] = tuple(payload["enabled_stages"])
        if "scope_schedule" in payload:
            payload["scope_schedule"] = tuple(
                _normalize_scope(scope) for scope in payload["scope_schedule"]
            )
        result = ProgressiveAttentionConfig(
            algorithm=algorithm,
            **payload,
        )
    else:
        raise TypeError(
            "progressive_attention must be a mapping, "
            f"ProgressiveAttentionConfig, or None; got {type(config)!r}"
        )
    _validate_config(result)
    return result


def _normalize_scope(scope: Any) -> Scope:
    if isinstance(scope, str):
        normalized = scope.strip().lower()
        if normalized == "full":
            return "full"
        if normalized.isdigit():
            return int(normalized)
    if isinstance(scope, int) and not isinstance(scope, bool):
        return scope
    raise ValueError(f"Invalid progressive attention scope {scope!r}")


def _validate_config(config: ProgressiveAttentionConfig) -> None:
    if config.algorithm not in {
        "legacy_token_scope",
        "adaptive_pair_scope",
    }:
        raise ValueError(
            "progressive attention algorithm must be "
            "'legacy_token_scope' or 'adaptive_pair_scope'"
        )
    if config.algorithm == "adaptive_pair_scope":
        if config.adaptive_pair_scope_config is None:
            raise ValueError(
                "adaptive_pair_scope requires adaptive configuration"
            )
        validate_adaptive_pair_scope_config(
            config.adaptive_pair_scope_config
        )
        return
    if config.adaptive_pair_scope_config is not None:
        raise ValueError(
            "legacy_token_scope must not carry adaptive pair-scope config"
        )
    if not config.scope_schedule:
        raise ValueError("progressive scope_schedule must not be empty")
    for scope in config.scope_schedule:
        if scope != "full" and (not isinstance(scope, int) or scope < 1):
            raise ValueError(f"Invalid progressive scope {scope!r}")
    numeric = [
        math.inf if scope == "full" else int(scope)
        for scope in config.scope_schedule
    ]
    if any(right < left for left, right in zip(numeric, numeric[1:])):
        raise ValueError("progressive scope_schedule must be monotonic")
    if config.require_stage_final_full and config.scope_schedule[-1] != "full":
        raise ValueError(
            "progressive scope_schedule must end in 'full' when "
            "require_stage_final_full is true"
        )
    if config.final_scope_mode not in {
        "dense",
        "inherited_sparse",
        "sampled",
    }:
        raise ValueError(
            "progressive final_scope_mode must be 'dense', "
            "'inherited_sparse', or 'sampled'"
        )
    if not config.reset_at_stage_boundary:
        raise ValueError(
            "the first progressive implementation requires stage-boundary reset"
        )
    if config.mask_representation != "dense_token_pair_reference":
        raise ValueError(
            "the corrected first implementation requires "
            "mask.representation='dense_token_pair_reference'"
        )
    if config.mask_query_chunk_size < 1:
        raise ValueError("mask.query_chunk_size must be positive")
    if config.max_reference_pair_elements < 1:
        raise ValueError("max_reference_pair_elements must be positive")
    if config.sampling_type not in {
        "nested_uniform_flat",
        "nested_random_balanced",
    }:
        raise ValueError(
            "progressive sampling.type must be 'nested_uniform_flat' or "
            f"'nested_random_balanced', got {config.sampling_type!r}"
        )
    if (
        isinstance(config.sampling_random_seed, bool)
        or not isinstance(config.sampling_random_seed, int)
        or not 0 <= config.sampling_random_seed < 2**63
    ):
        raise ValueError(
            "progressive sampling.random_seed must be an integer in "
            f"[0, 2^63), got {config.sampling_random_seed!r}"
        )
    if not isinstance(config.sampling_resample_each_stage, bool):
        raise ValueError(
            "progressive sampling.resample_each_stage must be a boolean"
        )
    for name, value in (
        ("row_keep_ratio", config.row_keep_ratio),
        ("column_keep_ratio", config.column_keep_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if config.row_keep_ratio == 0.0 and config.column_keep_ratio == 0.0:
        raise ValueError("at least one progressive keep ratio must be positive")
    if config.min_pairs_per_query < 1:
        raise ValueError("min_pairs_per_query must be positive")
    for name, value in (
        ("query_neighbor_radius", config.query_neighbor_radius),
        ("key_neighbor_radius", config.key_neighbor_radius),
        ("dilation_query", config.dilation_query),
        ("dilation_key", config.dilation_key),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    weight_sum = (
        config.self_weight
        + config.row_weight
        + config.column_weight
        + config.local_weight
    )
    if weight_sum <= 0.0:
        raise ValueError("progressive mask weights must have a positive sum")


def resolve_progressive_schedule(
    *,
    depth: int,
    inter_frame_attention_types: list[str] | tuple[str, ...],
    config: ProgressiveAttentionConfig,
) -> dict[int, ProgressiveLayerSpec]:
    if config.algorithm != "legacy_token_scope":
        raise ValueError(
            "cross-layer progressive schedule is only defined for "
            "legacy_token_scope"
        )
    if len(inter_frame_attention_types) != depth:
        raise ValueError(
            "inter-frame attention schedule length does not match depth: "
            f"{len(inter_frame_attention_types)} versus {depth}"
        )
    stage_names = ("early", "middle", "late")
    if len(config.stage_ranges) != len(stage_names):
        raise ValueError(
            "the configured progressive stage_ranges must define early, middle, "
            "and late stages"
        )
    enabled = set(config.enabled_stages)
    unknown = enabled.difference(stage_names)
    if unknown:
        raise ValueError(f"Unknown progressive stages: {sorted(unknown)}")

    schedule: dict[int, ProgressiveLayerSpec] = {}
    covered: set[int] = set()
    for stage_index, ((start, end), stage_name) in enumerate(
        zip(config.stage_ranges, stage_names)
    ):
        if start < 0 or end < start or end >= depth:
            raise ValueError(
                f"Invalid progressive stage range {(start, end)} for depth {depth}"
            )
        overlap = covered.intersection(range(start, end + 1))
        if overlap:
            raise ValueError(
                f"Progressive stage ranges overlap at layers {sorted(overlap)}"
            )
        covered.update(range(start, end + 1))
        if stage_name not in enabled:
            continue
        global_layers = [
            layer
            for layer in range(start, end + 1)
            if inter_frame_attention_types[layer] == "global"
        ]
        if not global_layers:
            raise ValueError(
                f"Progressive stage {stage_name!r} contains no global layers"
            )
        scope_count = len(config.scope_schedule)
        global_count = len(global_layers)
        for position, layer in enumerate(global_layers):
            if global_count == 1:
                scope_index = scope_count - 1
            else:
                scope_index = (position * (scope_count - 1)) // (
                    global_count - 1
                )
            scope = config.scope_schedule[scope_index]
            schedule[layer] = ProgressiveLayerSpec(
                layer_index=layer,
                stage_index=stage_index,
                stage_name=stage_name,
                global_position=position,
                global_count=global_count,
                scope=scope,
            )
        if config.require_stage_final_full and schedule[global_layers[-1]].scope != "full":
            raise RuntimeError(
                f"Progressive stage {stage_name!r} did not resolve to a final full scope"
            )
    return schedule


@lru_cache(maxsize=32)
def nested_uniform_order(token_count: int) -> Tensor:
    """Return a deterministic bit-reversal order over one flat patch axis."""
    if token_count < 1:
        raise ValueError(f"token_count must be positive, got {token_count}")
    bit_count = max(1, (token_count - 1).bit_length())
    padded_count = 1 << bit_count
    values = torch.arange(padded_count, dtype=torch.long)
    work = values.clone()
    reversed_values = torch.zeros_like(values)
    for _ in range(bit_count):
        reversed_values = (reversed_values << 1) | (work & 1)
        work >>= 1
    order = reversed_values[reversed_values < token_count]
    if order.numel() != token_count or torch.unique(order).numel() != token_count:
        raise RuntimeError("Failed to construct a complete nested patch ordering")
    return order


@lru_cache(maxsize=32)
def _nested_frame_patch_order(
    num_frames: int,
    patches_per_frame: int,
) -> Tensor:
    """Balance every prefix over frames while rotating patch locations."""
    if num_frames < 1 or patches_per_frame < 1:
        raise ValueError("num_frames and patches_per_frame must be positive")
    total = num_frames * patches_per_frame
    rank = torch.arange(total, dtype=torch.long)
    cycle = rank // num_frames
    within_cycle = rank % num_frames

    frame_stride = max(1, int(num_frames * 0.6180339887498949))
    while math.gcd(frame_stride, num_frames) != 1:
        frame_stride -= 1
    patch_stride = max(1, int(patches_per_frame * 0.6180339887498949))
    while math.gcd(patch_stride, patches_per_frame) != 1:
        patch_stride -= 1

    frame = (within_cycle * frame_stride + cycle) % num_frames
    patch = (cycle + frame * patch_stride) % patches_per_frame
    order = frame * patches_per_frame + patch
    if torch.unique(order).numel() != total:
        raise RuntimeError("Failed to construct nested frame/patch sampling order")
    return order


@lru_cache(maxsize=128)
def _nested_random_frame_patch_order(
    num_frames: int,
    patches_per_frame: int,
    random_seed: int,
) -> Tensor:
    """Build a seeded random order with balanced frame-prefix coverage."""
    if num_frames < 1 or patches_per_frame < 1:
        raise ValueError("num_frames and patches_per_frame must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))

    # Every cycle contributes exactly one unused patch from every frame. Frame
    # order and each frame's patch permutation are randomized independently.
    patch_orders = torch.rand(
        num_frames,
        patches_per_frame,
        generator=generator,
    ).argsort(dim=-1, stable=True)
    frame_orders = torch.rand(
        patches_per_frame,
        num_frames,
        generator=generator,
    ).argsort(dim=-1, stable=True)
    cycles = (
        torch.arange(patches_per_frame, dtype=torch.long)[:, None]
        .expand(-1, num_frames)
        .reshape(-1)
    )
    frames = frame_orders.reshape(-1)
    patches = patch_orders[frames, cycles]
    order = frames * patches_per_frame + patches
    total = num_frames * patches_per_frame
    if order.numel() != total or torch.unique(order).numel() != total:
        raise RuntimeError("Failed to construct a complete random patch ordering")
    return order


def nested_patch_indices(
    *,
    num_frames: int,
    patches_per_frame: int,
    equivalent_scope: Scope,
    sampling_type: str = "nested_random_balanced",
    random_seed: int = 0,
    device: torch.device | str | None = None,
) -> Tensor:
    total = int(num_frames) * int(patches_per_frame)
    if equivalent_scope == "full":
        count = total
    else:
        count = min(int(equivalent_scope), int(num_frames)) * int(
            patches_per_frame
        )
    if sampling_type == "nested_random_balanced":
        order = _nested_random_frame_patch_order(
            int(num_frames),
            int(patches_per_frame),
            int(random_seed),
        )
    elif sampling_type == "nested_uniform_flat":
        order = _nested_frame_patch_order(
            int(num_frames),
            int(patches_per_frame),
        )
    else:
        raise ValueError(f"Unknown progressive sampling type {sampling_type!r}")
    selected = order[:count].sort().values
    return selected.to(device=device) if device is not None else selected


def nested_uniform_indices(
    *,
    num_frames: int,
    patches_per_frame: int,
    equivalent_scope: Scope,
    device: torch.device | str | None = None,
) -> Tensor:
    """Backward-compatible deterministic sampler."""
    return nested_patch_indices(
        num_frames=num_frames,
        patches_per_frame=patches_per_frame,
        equivalent_scope=equivalent_scope,
        sampling_type="nested_uniform_flat",
        device=device,
    )


def _effective_sampling_seed(
    config: ProgressiveAttentionConfig,
    layer_spec: ProgressiveLayerSpec,
) -> int:
    seed = int(config.sampling_random_seed)
    if config.sampling_resample_each_stage:
        seed += 1_000_003 * int(layer_spec.stage_index)
    return seed % (2**63)


def patch_sample_coordinates(
    patch_indices: Tensor,
    *,
    patches_per_frame: int,
    patch_grid_size: tuple[int, int],
) -> Tensor:
    patch_h, patch_w = map(int, patch_grid_size)
    if patches_per_frame != patch_h * patch_w:
        raise ValueError(
            "patches_per_frame does not match patch_grid_size: "
            f"{patches_per_frame} versus {patch_grid_size}"
        )
    flat = patch_indices.to(dtype=torch.long)
    frame = flat // patches_per_frame
    patch = flat % patches_per_frame
    patch_u = patch // patch_w
    patch_v = patch % patch_w
    return torch.stack((frame, patch_u, patch_v), dim=-1)


def selected_original_token_indices(
    patch_indices: Tensor,
    *,
    num_frames: int,
    tokens_per_frame: int,
    num_special_tokens: int,
) -> Tensor:
    patches_per_frame = tokens_per_frame - num_special_tokens
    patch_frames = patch_indices // patches_per_frame
    patch_offsets = patch_indices % patches_per_frame
    original_patch_indices = (
        patch_frames * tokens_per_frame
        + num_special_tokens
        + patch_offsets
    )
    special = (
        torch.arange(num_frames, device=patch_indices.device)[:, None]
        * tokens_per_frame
        + torch.arange(num_special_tokens, device=patch_indices.device)[None, :]
    ).reshape(-1)
    # The working layout is patches first, then every special token.  The
    # returned indices restore all outputs to the original frame-major layout.
    return torch.cat((original_patch_indices, special), dim=0)


def nearest_parent_positions(child: Tensor, parent: Tensor) -> Tensor:
    if parent.numel() < 1:
        raise ValueError("parent indices must not be empty")
    positions = torch.searchsorted(parent, child)
    right = positions.clamp(max=parent.numel() - 1)
    left = (positions - 1).clamp(min=0)
    left_distance = (child - parent[left]).abs()
    right_distance = (parent[right] - child).abs()
    return torch.where(left_distance <= right_distance, left, right)


def progressive_attention_block(
    block: "SelfAttentionBlock",
    x: Tensor,
    *,
    num_frames: int,
    tokens_per_frame: int,
    num_special_tokens: int,
    patch_grid_size: tuple[int, int],
    layer_spec: ProgressiveLayerSpec,
    config: ProgressiveAttentionConfig,
    previous_state: ProgressiveMaskState | None,
    build_next_mask: bool,
) -> ProgressiveBlockResult:
    if block.training:
        raise NotImplementedError("progressive attention is currently inference-only")
    if x.ndim != 3:
        raise ValueError(
            "progressive attention expects [batch, tokens, channels], got "
            f"{tuple(x.shape)}"
        )
    batch_size, total_tokens, embed_dim = x.shape
    expected_tokens = num_frames * tokens_per_frame
    if total_tokens != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} flattened tokens, got {total_tokens}"
        )
    patches_per_frame = tokens_per_frame - num_special_tokens
    if patches_per_frame != patch_grid_size[0] * patch_grid_size[1]:
        raise ValueError("progressive token layout does not match the patch grid")

    timer = _ComponentTimer(config.profile_components, x.device)
    with timer.measure("sampling"):
        effective_sampling_seed = _effective_sampling_seed(config, layer_spec)
        patch_indices = nested_patch_indices(
            num_frames=num_frames,
            patches_per_frame=patches_per_frame,
            equivalent_scope=layer_spec.scope,
            sampling_type=config.sampling_type,
            random_seed=effective_sampling_seed,
            device=x.device,
        )
        selected_indices = selected_original_token_indices(
            patch_indices,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            num_special_tokens=num_special_tokens,
        )
        selected_tokens = block.norm1(
            x.index_select(1, selected_indices)
        )

    patch_count = int(patch_indices.numel())
    special_count = num_frames * num_special_tokens
    selected_count = patch_count + special_count
    if selected_count != selected_tokens.shape[1]:
        raise RuntimeError("progressive selected-token count mismatch")

    with timer.measure("qkv_projection"):
        qkv = block.attn.qkv(selected_tokens).reshape(
            batch_size,
            selected_count,
            3,
            block.attn.num_heads,
            embed_dim // block.attn.num_heads,
        )
        q, k, v = torch.unbind(qkv, dim=2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        if block.attn.use_qk_norm:
            q = block.attn.q_norm(q)
            k = block.attn.k_norm(k)

    parent_positions = None
    if previous_state is not None:
        if previous_state.patch_indices.device != patch_indices.device:
            previous_state.patch_indices = previous_state.patch_indices.to(
                patch_indices.device
            )
        parent_positions = nearest_parent_positions(
            patch_indices,
            previous_state.patch_indices,
        )

    attended, attention_stats = _progressive_attention(
        q,
        k,
        v,
        patch_count=patch_count,
        previous_state=previous_state,
        parent_positions=parent_positions,
        config=config,
        scale=block.attn.scale,
        timer=timer,
    )
    with timer.measure("attention_output_projection"):
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            selected_count,
            embed_dim,
        )
        attended = block.attn.proj_drop(block.attn.proj(attended))

    with timer.measure("scatter"):
        attention_update = torch.zeros_like(x)
        attention_update.index_copy_(
            1,
            selected_indices,
            attended.to(dtype=attention_update.dtype),
        )
        x_attn = x + block.ls1(attention_update)
    with timer.measure("residual_mlp"):
        output = x_attn + block.ls2(block.mlp(block.norm2(x_attn)))

    next_state = None
    mask_stats: dict[str, Any] = {}
    if build_next_mask:
        with timer.measure("mask_generation"):
            pair_mask, highest_score_key, mask_stats = (
                _build_next_token_pair_mask(
                    q,
                    k,
                    patch_indices=patch_indices,
                    previous_state=previous_state,
                    parent_positions=parent_positions,
                    config=config,
                    scale=block.attn.scale,
                )
            )
            next_state = ProgressiveMaskState(
                patch_indices=patch_indices,
                pair_mask=pair_mask,
                highest_score_key=highest_score_key,
            )

    sample_coordinates = None
    if config.save_sample_indices:
        sample_coordinates = patch_sample_coordinates(
            patch_indices.detach().cpu(),
            patches_per_frame=patches_per_frame,
            patch_grid_size=patch_grid_size,
        )

    stats: dict[str, Any] = {
        "stage_index": layer_spec.stage_index,
        "stage_name": layer_spec.stage_name,
        "layer_index": layer_spec.layer_index,
        "stage_global_position": layer_spec.global_position,
        "stage_global_count": layer_spec.global_count,
        "scope": layer_spec.scope,
        "num_frames": num_frames,
        "patches_per_frame": patches_per_frame,
        "sampled_patch_tokens": patch_count,
        "special_tokens": special_count,
        "qkv_projection_tokens": selected_count,
        "full_patch_tokens": num_frames * patches_per_frame,
        "patch_sampling_ratio": patch_count / (num_frames * patches_per_frame),
        "sampling_type": config.sampling_type,
        "sampling_random_seed": config.sampling_random_seed,
        "sampling_effective_seed": effective_sampling_seed,
        "sampling_resample_each_stage": config.sampling_resample_each_stage,
        "progressive_attention_semantics": "exact_token_pair_reference_v2",
        "attention_working_layout": "sampled_patches_then_all_special_tokens",
        "scatter_restores_frame_major_layout": True,
        "sample_before_norm_and_qkv": True,
        "sample_before_qkv": True,
        "mask_inherited": previous_state is not None,
        "mask_generated": next_state is not None,
        **attention_stats,
        **mask_stats,
    }
    if timer.enabled:
        stats["component_time_ms"] = dict(timer.elapsed_ms)
    return ProgressiveBlockResult(
        output=output,
        next_state=next_state,
        stats=stats,
        sample_coordinates=sample_coordinates,
    )


def _progressive_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    patch_count: int,
    previous_state: ProgressiveMaskState | None,
    parent_positions: Tensor | None,
    config: ProgressiveAttentionConfig,
    scale: float,
    timer: _ComponentTimer,
) -> tuple[Tensor, dict[str, Any]]:
    _, _, selected_count, _ = q.shape
    special_count = selected_count - patch_count
    q_attention = q.to(dtype=v.dtype)
    k_attention = k.to(dtype=v.dtype)
    if previous_state is None:
        with timer.measure("attention_kernel"):
            output = F.scaled_dot_product_attention(
                q_attention,
                k_attention,
                v,
                scale=scale,
            )
        logical_pairs = selected_count * selected_count
        patch_pairs = patch_count * patch_count
        return output, {
            "attention_backend": "sdpa_dense_sampled",
            "logical_attention_pairs_per_batch": logical_pairs,
            "evaluated_attention_pairs_per_batch": logical_pairs,
            "patch_attention_pairs_per_batch": patch_pairs,
            "inherited_mask_representation": None,
        }

    if parent_positions is None:
        raise RuntimeError("inherited progressive attention lacks parent positions")
    q_patch = q_attention[:, :, :patch_count]
    q_special = q_attention[:, :, patch_count:]
    _check_reference_pair_budget(
        previous_state.pair_mask.numel(),
        max_pair_elements=config.max_reference_pair_elements,
        operation="stored inherited token-pair mask",
    )
    query_chunk = min(config.mask_query_chunk_size, patch_count)
    patch_output = torch.empty_like(q_patch)
    logical_patch_pairs = 0
    fallback_parent_rows = 0
    fallback_highest_anchor_rows = 0
    fallback_special_only_rows = 0
    for query_start in range(0, patch_count, query_chunk):
        query_end = min(query_start + query_chunk, patch_count)
        with timer.measure("mask_expansion"):
            allowed_patch, fallback = _inherited_patch_mask_chunk(
                previous_state,
                parent_query_positions=parent_positions[
                    query_start:query_end
                ],
                parent_key_positions=parent_positions,
            )
            fallback_parent_rows += fallback["fallback_parent_rows"]
            fallback_highest_anchor_rows += fallback[
                "fallback_highest_anchor_rows"
            ]
            fallback_special_only_rows += fallback[
                "fallback_special_only_rows"
            ]
            if fallback["fallback_special_only_rows"] and not special_count:
                raise RuntimeError(
                    "inherited progressive attention has an empty patch-key "
                    "row and no camera/register fallback keys"
                )
            logical_patch_pairs += int(allowed_patch.sum().item())
            if special_count:
                allowed = torch.cat(
                    (
                        allowed_patch,
                        torch.ones(
                            allowed_patch.shape[0],
                            query_end - query_start,
                            special_count,
                            device=q.device,
                            dtype=torch.bool,
                        ),
                    ),
                    dim=-1,
                )
            else:
                allowed = allowed_patch
        with timer.measure("attention_kernel"):
            patch_output[:, :, query_start:query_end] = (
                F.scaled_dot_product_attention(
                    q_patch[:, :, query_start:query_end],
                    k_attention,
                    v,
                    attn_mask=allowed[:, None],
                    scale=scale,
                )
            )
    with timer.measure("attention_kernel"):
        special_output = (
            F.scaled_dot_product_attention(
                q_special,
                k_attention,
                v,
                scale=scale,
            )
            if special_count
            else q_special
        )
    dense_special_pairs = (
        patch_count * special_count
        + special_count * selected_count
    )
    return torch.cat((patch_output, special_output), dim=2), {
        "attention_backend": "sdpa_dense_token_pair_reference_mask",
        "inherited_mask_representation": "exact_token_pair",
        "efficient_sparse_kernel": False,
        "logical_attention_pairs_per_batch": (
            logical_patch_pairs // max(q.shape[0], 1)
            + dense_special_pairs
        ),
        "evaluated_attention_pairs_per_batch": selected_count * selected_count,
        "patch_attention_pairs_per_batch": (
            logical_patch_pairs // max(q.shape[0], 1)
        ),
        "expanded_mask_logical_density": (
            logical_patch_pairs
            / max(q.shape[0] * patch_count * patch_count, 1)
        ),
        "mask_expansion_fallback_parent_rows": fallback_parent_rows,
        "mask_expansion_fallback_highest_anchor_rows": (
            fallback_highest_anchor_rows
        ),
        "mask_expansion_fallback_special_only_rows": (
            fallback_special_only_rows
        ),
        "mask_expansion_query_chunk_size": query_chunk,
    }


def _inherited_patch_mask_chunk(
    state: ProgressiveMaskState,
    *,
    parent_query_positions: Tensor,
    parent_key_positions: Tensor,
) -> tuple[Tensor, dict[str, int]]:
    patch_mask = (
        state.pair_mask.index_select(1, parent_query_positions)
        .index_select(2, parent_key_positions)
    )
    empty_after_parent = ~patch_mask.any(dim=-1)
    parent_fallback_rows = int(empty_after_parent.sum().item())
    highest_anchor_rows = 0
    if parent_fallback_rows and state.highest_score_key is not None:
        best_parent_key = state.highest_score_key.index_select(
            1,
            parent_query_positions,
        )
        highest_region = (
            best_parent_key[:, :, None]
            == parent_key_positions[None, None, :]
        )
        patch_mask |= empty_after_parent[:, :, None] & highest_region
        highest_anchor_rows = int(
            (empty_after_parent & patch_mask.any(dim=-1)).sum().item()
        )
    special_only_rows = int((~patch_mask.any(dim=-1)).sum().item())
    return patch_mask, {
        "fallback_parent_rows": parent_fallback_rows,
        "fallback_highest_anchor_rows": highest_anchor_rows,
        "fallback_special_only_rows": special_only_rows,
    }


def _dense_inherited_mask(
    state: ProgressiveMaskState,
    parent_positions: Tensor,
    *,
    patch_count: int,
    special_count: int,
    max_pair_elements: int,
) -> tuple[Tensor, dict[str, Any]]:
    pair_elements = (
        state.pair_mask.shape[0] * patch_count * patch_count
    )
    _check_reference_pair_budget(
        pair_elements,
        max_pair_elements=max_pair_elements,
        operation="inherited token-pair mask expansion",
    )
    patch_mask, fallback = _inherited_patch_mask_chunk(
        state,
        parent_query_positions=parent_positions,
        parent_key_positions=parent_positions,
    )
    logical_patch_pairs = patch_mask.sum(dim=(-2, -1), dtype=torch.int64)
    if special_count:
        batch_size = patch_mask.shape[0]
        patch_mask = torch.cat(
            (
                patch_mask,
                torch.ones(
                    batch_size,
                    patch_count,
                    special_count,
                    device=patch_mask.device,
                    dtype=torch.bool,
                ),
            ),
            dim=-1,
        )
    selected_count = patch_count + special_count
    dense_special_pairs = (
        patch_count * special_count
        + special_count * selected_count
    )
    logical_total = logical_patch_pairs + dense_special_pairs
    return patch_mask, {
        "logical_attention_pairs_per_batch": int(
            logical_total.to(dtype=torch.float64).mean().item()
        ),
        "evaluated_attention_pairs_per_batch": selected_count * selected_count,
        "patch_attention_pairs_per_batch": int(
            logical_patch_pairs.to(dtype=torch.float64).mean().item()
        ),
        "expanded_mask_logical_density": float(
            logical_patch_pairs.to(dtype=torch.float64).mean().item()
            / max(patch_count * patch_count, 1)
        ),
        "mask_expansion_fallback_parent_rows": fallback[
            "fallback_parent_rows"
        ],
        "mask_expansion_fallback_highest_anchor_rows": fallback[
            "fallback_highest_anchor_rows"
        ],
        "mask_expansion_fallback_special_only_rows": fallback[
            "fallback_special_only_rows"
        ],
    }


def _check_reference_pair_budget(
    pair_elements: int,
    *,
    max_pair_elements: int,
    operation: str,
) -> None:
    if pair_elements <= max_pair_elements:
        return
    raise RuntimeError(
        f"{operation} requires {pair_elements:,} logical pair elements, "
        f"exceeding max_reference_pair_elements={max_pair_elements:,}. "
        "This corrected backend is the exact token-pair diagnostic path; it "
        "will not silently replace token-pair routing with execution-block "
        "routing. Reduce the diagnostic scope/resolution or implement and "
        "validate a semantics-preserving mask-regularization backend."
    )


def _build_next_token_pair_mask(
    q: Tensor,
    k: Tensor,
    *,
    patch_indices: Tensor,
    previous_state: ProgressiveMaskState | None,
    parent_positions: Tensor | None,
    config: ProgressiveAttentionConfig,
    scale: float,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Build exact logical ``B_S[i, j]`` without block-score reduction."""
    patch_count = int(patch_indices.numel())
    batch_size = q.shape[0]
    selected_count = q.shape[-2]
    pair_elements = batch_size * patch_count * patch_count
    _check_reference_pair_budget(
        pair_elements,
        max_pair_elements=config.max_reference_pair_elements,
        operation="token-pair mask generation",
    )
    if patch_count < 1:
        raise ValueError("progressive mask generation requires patch tokens")
    if previous_state is not None and parent_positions is None:
        raise RuntimeError("mask generation lacks inherited parent positions")

    binary = torch.zeros(
        batch_size,
        patch_count,
        patch_count,
        device=q.device,
        dtype=torch.bool,
    )
    highest_score_key = torch.empty(
        batch_size,
        patch_count,
        device=q.device,
        dtype=torch.long,
    )
    row_keep = min(
        patch_count,
        max(
            config.min_pairs_per_query,
            math.ceil(config.row_keep_ratio * patch_count),
        ),
    )
    column_keep = (
        min(
            patch_count,
            max(1, math.ceil(config.column_keep_ratio * patch_count)),
        )
        if config.column_keep_ratio > 0.0
        else 0
    )
    column_values = (
        torch.full(
            (batch_size, patch_count, column_keep),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        if column_keep
        else None
    )
    column_queries = (
        torch.zeros(
            batch_size,
            patch_count,
            column_keep,
            device=q.device,
            dtype=torch.int32,
        )
        if column_keep
        else None
    )

    q_patch = q[:, :, :patch_count].float()
    k_all = k.float()
    query_radius = config.query_neighbor_radius
    key_radius = config.key_neighbor_radius
    query_chunk = min(config.mask_query_chunk_size, patch_count)
    inherited_active_pairs = 0
    finite_score_pairs = 0

    for query_start in range(0, patch_count, query_chunk):
        query_end = min(query_start + query_chunk, patch_count)
        halo_start = max(0, query_start - query_radius)
        halo_end = min(patch_count, query_end + query_radius)
        if previous_state is None:
            allowed_halo = None
        else:
            allowed_halo, _ = _inherited_patch_mask_chunk(
                previous_state,
                parent_query_positions=parent_positions[
                    halo_start:halo_end
                ],
                parent_key_positions=parent_positions,
            )
            inherited_active_pairs += int(
                allowed_halo[
                    :,
                    query_start - halo_start : query_end - halo_start,
                ].sum().item()
            )

        logits = torch.matmul(
            q_patch[:, :, halo_start:halo_end],
            k_all.transpose(-2, -1),
        )
        logits.mul_(scale)
        if allowed_halo is not None:
            logits[..., :patch_count].masked_fill_(
                ~allowed_halo[:, None],
                float("-inf"),
            )
        probabilities = logits.softmax(dim=-1, dtype=torch.float32).mean(dim=1)
        patch_probabilities = probabilities[..., :patch_count]
        local_input = patch_probabilities[:, None]
        row_response = F.avg_pool2d(
            local_input,
            kernel_size=(1, 2 * key_radius + 1),
            stride=1,
            padding=(0, key_radius),
            count_include_pad=False,
        )[:, 0]
        column_response = F.avg_pool2d(
            local_input,
            kernel_size=(2 * query_radius + 1, 1),
            stride=1,
            padding=(query_radius, 0),
            count_include_pad=False,
        )[:, 0]
        local_response = F.avg_pool2d(
            local_input,
            kernel_size=(2 * query_radius + 1, 2 * key_radius + 1),
            stride=1,
            padding=(query_radius, key_radius),
            count_include_pad=False,
        )[:, 0]
        center_start = query_start - halo_start
        center_end = center_start + (query_end - query_start)
        scores = (
            config.self_weight
            * patch_probabilities[:, center_start:center_end]
            + config.row_weight * row_response[:, center_start:center_end]
            + config.column_weight
            * column_response[:, center_start:center_end]
            + config.local_weight
            * local_response[:, center_start:center_end]
        )
        if allowed_halo is None:
            candidates = torch.ones_like(scores, dtype=torch.bool)
            inherited_active_pairs += candidates.numel()
        else:
            candidates = allowed_halo[:, center_start:center_end]
        scores.masked_fill_(~candidates, float("-inf"))
        finite_score_pairs += int(torch.isfinite(scores).sum().item())
        highest_score_key[:, query_start:query_end] = scores.argmax(dim=-1)

        row_indices = torch.topk(
            scores,
            k=row_keep,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        selected_rows = torch.zeros_like(candidates)
        selected_rows.scatter_(dim=-1, index=row_indices, value=True)
        binary[:, query_start:query_end] |= selected_rows & candidates

        if column_keep:
            assert column_values is not None
            assert column_queries is not None
            incoming = scores.transpose(-2, -1)
            combined_values = torch.cat((column_values, incoming), dim=-1)
            next_values, next_sources = torch.topk(
                combined_values,
                k=column_keep,
                dim=-1,
                largest=True,
                sorted=False,
            )
            old_sources = next_sources.clamp_max(column_keep - 1)
            old_queries = torch.gather(
                column_queries,
                dim=-1,
                index=old_sources,
            )
            new_queries = (
                query_start + next_sources - column_keep
            ).to(dtype=torch.int32)
            column_queries = torch.where(
                next_sources < column_keep,
                old_queries,
                new_queries,
            )
            column_values = next_values

    if column_keep:
        assert column_values is not None
        assert column_queries is not None
        key_chunk = max(1, min(config.mask_query_chunk_size, patch_count))
        for key_start in range(0, patch_count, key_chunk):
            key_end = min(key_start + key_chunk, patch_count)
            queries = column_queries[:, key_start:key_end].long()
            valid = torch.isfinite(column_values[:, key_start:key_end])
            batch = torch.arange(batch_size, device=q.device)[:, None, None]
            keys = torch.arange(
                key_start,
                key_end,
                device=q.device,
            )[None, :, None]
            binary[batch, queries, keys] |= valid
        del column_values, column_queries

    empty = ~binary.any(dim=-1)
    fallback_parent_rows = int(empty.sum().item())
    fallback_special_rows = 0
    if fallback_parent_rows:
        # A valid inherited parent row normally prevents this branch.  If the
        # logical patch mask is empty, special-token keys still make attention
        # safe; retain a diagonal anchor solely so the next level has a parent.
        diagonal = torch.arange(patch_count, device=q.device)
        binary[:, diagonal, diagonal] |= empty
        fallback_special_rows = fallback_parent_rows

    binary_density = float(binary.float().mean().item())
    dilated = _dilate_token_pair_mask(
        binary,
        query_radius=config.dilation_query,
        key_radius=config.dilation_key,
    )
    dilated_density = float(dilated.float().mean().item())
    active_keys = dilated.sum(dim=-1, dtype=torch.int64)
    flattened_active = active_keys.flatten().float()
    return dilated, highest_score_key, {
        "mask_selection_granularity": "exact_patch_token_pair",
        "mask_representation": config.mask_representation,
        "mask_head_aggregation": "mean",
        "mask_inherited_density_before_selection": (
            inherited_active_pairs / max(pair_elements, 1)
        ),
        "mask_pre_binarization_finite_density": (
            finite_score_pairs / max(pair_elements, 1)
        ),
        "mask_binary_density": binary_density,
        "mask_dilated_density": dilated_density,
        "mask_binary_patch_pairs_per_batch": int(
            binary.sum(dim=(-2, -1), dtype=torch.int64)
            .to(dtype=torch.float64)
            .mean()
            .item()
        ),
        "mask_dilated_patch_pairs_per_batch": int(
            active_keys.sum(dim=-1, dtype=torch.int64)
            .to(dtype=torch.float64)
            .mean()
            .item()
        ),
        "mask_routing_dense_qk_pairs_per_batch_head": (
            patch_count * selected_count
        ),
        "mask_query_chunk_size": query_chunk,
        "mask_row_topk_per_query": row_keep,
        "mask_column_topk_per_key": column_keep,
        "per_query_active_key_count": {
            "mean": float(flattened_active.mean().item()),
            "median": float(flattened_active.median().item()),
            "minimum": int(flattened_active.min().item()),
            "maximum": int(flattened_active.max().item()),
            "p90": float(torch.quantile(flattened_active, 0.90).item()),
            "p99": float(torch.quantile(flattened_active, 0.99).item()),
        },
        "mask_fallback_parent_rows": fallback_parent_rows,
        "mask_fallback_special_rows": fallback_special_rows,
    }


def _dilate_token_pair_mask(
    pair_mask: Tensor,
    *,
    query_radius: int,
    key_radius: int,
) -> Tensor:
    if query_radius == 0 and key_radius == 0:
        return pair_mask
    dilated = torch.zeros_like(pair_mask)
    query_count, key_count = pair_mask.shape[-2:]
    for query_offset in range(-query_radius, query_radius + 1):
        if query_offset >= 0:
            query_source = slice(0, query_count - query_offset)
            query_target = slice(query_offset, query_count)
        else:
            query_source = slice(-query_offset, query_count)
            query_target = slice(0, query_count + query_offset)
        for key_offset in range(-key_radius, key_radius + 1):
            if key_offset >= 0:
                key_source = slice(0, key_count - key_offset)
                key_target = slice(key_offset, key_count)
            else:
                key_source = slice(-key_offset, key_count)
                key_target = slice(0, key_count + key_offset)
            dilated[:, query_target, key_target] |= pair_mask[
                :,
                query_source,
                key_source,
            ]
    return dilated
