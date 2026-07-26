"""Scope-driven hierarchical routing over one layer's full Q-K patch matrix.

This module is the semantic-reference implementation of adaptive pair-scope
attention.  A global patch axis is sampled uniformly, the sampled Cartesian
grid partitions the complete patch-to-patch matrix into non-overlapping
rectangles, and only active rectangles receive a second level of local probes.
The resulting leaf rectangles are expanded per query chunk for a final
full-token masked attention.  Coarse routing, fine routing, and final attention
all reuse one normalization and one QKV projection from the same Transformer
layer.

All intervals use half-open ``[start, end)`` semantics.  The reference backend
uses masked dense PyTorch SDPA and therefore does not claim sparse-kernel
acceleration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from vggt_omega.models.layers.block import SelfAttentionBlock


ROUTING_SCORE_MODES = {
    "mean_head_logit",
    "mean_head_sampled_probability",
}
SELECTION_MODES = {
    "row_topk_ratio",
    "row_topk_count",
    "global_quantile",
}


@dataclass(frozen=True)
class AxisScope:
    """One non-empty interval on the original flattened patch-token axis."""

    start: int
    end: int
    anchor: int
    level: int


@dataclass
class PairScope:
    """One rectangular Q-K scope represented by a real token-anchor pair."""

    q_scope: AxisScope
    k_scope: AxisScope
    representative_q: int
    representative_k: int
    level: int
    score: Tensor | None = None
    refine: Tensor | bool = False
    parent_id: int | None = None
    scope_id: int | None = None


@dataclass(frozen=True)
class LeafPairScope:
    """One final rectangular leaf for a single batch element."""

    q_start: int
    q_end: int
    k_start: int
    k_end: int
    allowed: bool
    level: int
    parent_id: int | None


@dataclass(frozen=True)
class AdaptivePairScopeConfig:
    """Configuration for within-layer adaptive pair-scope routing."""

    enabled_layers: tuple[int, ...] = (10, 11, 12, 13, 15, 16)
    coarse_num_anchors: int | None = 128
    coarse_stride: int | None = None
    routing_score_mode: str = "mean_head_sampled_probability"
    coarse_selection_mode: str = "row_topk_ratio"
    coarse_keep_ratio: float = 0.25
    coarse_keep_count: int | None = None
    refine_factor: int = 2
    fine_selection_mode: str = "row_topk_ratio"
    fine_keep_ratio: float = 0.25
    fine_keep_count: int | None = None
    min_active_key_scopes_per_query_scope: int = 1
    keep_camera_tokens: bool = True
    keep_register_tokens: bool = True
    dense_special_queries: bool = True
    dense_special_keys: bool = True
    backend_type: str = "query_chunked_dense_mask_reference"
    query_chunk_size: int = 128
    save_anchor_indices: bool = True
    save_pair_scopes: bool = True
    save_mask_statistics: bool = True
    save_materialized_full_mask: bool = False
    materialize_full_mask_max_patch_tokens: int = 512
    profile_components: bool = True


@dataclass
class AdaptivePairRoutingResult:
    """Geometry, decisions, and leaf rectangles produced by two-level routing."""

    coarse_anchor_indices: Tensor
    coarse_axis_scopes: tuple[AxisScope, ...]
    coarse_pair_scopes: tuple[PairScope, ...]
    coarse_scores: Tensor
    coarse_refine: Tensor
    fine_pair_scopes: tuple[tuple[PairScope, ...], ...]
    leaf_scopes: tuple[tuple[LeafPairScope, ...], ...]
    fine_probe_pairs_per_batch: tuple[int, ...]
    fine_kept_scope_count_per_batch: tuple[int, ...]
    final_logical_patch_pairs_per_batch: tuple[int, ...]
    scope_partition_valid: bool


@dataclass(frozen=True)
class CompiledPatchMask:
    """Reusable row templates compiled from one batch of leaf rectangles.

    Storage is ``O(S × N)`` rather than ``O(N²)``, where ``S`` is the number
    of distinct Query slabs induced by allowed leaf boundaries. Query chunks
    can gather exact mask rows without traversing every leaf again.
    """

    query_boundaries: tuple[int, ...]
    query_scope_by_row: Tensor
    allowed_key_mask_by_query_scope: Tensor


@dataclass
class AdaptivePairBlockResult:
    """Full block output plus routing statistics and optional debug artifacts."""

    output: Tensor
    stats: dict[str, Any]
    routing: AdaptivePairRoutingResult
    debug: dict[str, Any] = field(default_factory=dict)


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
        self.end_event = None
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
            assert self.start_event is not None and self.end_event is not None
            self.end_event.record()
            self.end_event.synchronize()
            elapsed = float(self.start_event.elapsed_time(self.end_event))
        else:
            elapsed = 1000.0 * (time.perf_counter() - self.start_time)
        self.timer.elapsed_ms[self.name] = (
            self.timer.elapsed_ms.get(self.name, 0.0) + elapsed
        )
        return False


def adaptive_pair_scope_config_from_dict(
    payload: dict[str, Any],
) -> AdaptivePairScopeConfig:
    """Parse the nested ``adaptive_pair_scope`` progressive configuration."""

    top = dict(payload)
    enabled_layers = tuple(int(value) for value in top.pop("enabled_layers", ()))
    top_num_anchors = top.pop("coarse_num_anchors", None)
    top_stride = top.pop("coarse_stride", None)
    coarse_sampling = dict(top.pop("coarse_sampling", {}) or {})
    routing = dict(top.pop("routing", {}) or {})
    special_tokens = dict(top.pop("special_tokens", {}) or {})
    backend = dict(top.pop("backend", {}) or {})
    debug = dict(top.pop("debug", {}) or {})
    profile_override_present = "profile_components" in top
    profile_components = bool(top.pop("profile_components", True))
    if top:
        raise ValueError(
            "Unknown adaptive pair-scope configuration keys: "
            f"{sorted(top)}"
        )

    sampling_type = coarse_sampling.pop("type", "uniform_flat")
    if sampling_type != "uniform_flat":
        raise ValueError(
            "adaptive coarse_sampling.type must be 'uniform_flat', got "
            f"{sampling_type!r}"
        )
    nested_num_anchors = coarse_sampling.pop("num_anchors", None)
    alias_num_anchors = coarse_sampling.pop("coarse_num_anchors", None)
    nested_stride = coarse_sampling.pop("stride", None)
    alias_stride = coarse_sampling.pop("coarse_stride", None)
    num_values = [
        value
        for value in (
            top_num_anchors,
            nested_num_anchors,
            alias_num_anchors,
        )
        if value is not None
    ]
    stride_values = [
        value
        for value in (top_stride, nested_stride, alias_stride)
        if value is not None
    ]
    if len(num_values) > 1:
        raise ValueError("coarse anchor count was specified more than once")
    if len(stride_values) > 1:
        raise ValueError("coarse stride was specified more than once")
    num_anchors = num_values[0] if num_values else None
    stride = stride_values[0] if stride_values else None
    if num_anchors is None and stride is None:
        num_anchors = 128
    if coarse_sampling:
        raise ValueError(
            "Unknown adaptive coarse_sampling keys: "
            f"{sorted(coarse_sampling)}"
        )

    config = AdaptivePairScopeConfig(
        enabled_layers=enabled_layers,
        coarse_num_anchors=(
            None if num_anchors is None else int(num_anchors)
        ),
        coarse_stride=None if stride is None else int(stride),
        routing_score_mode=routing.pop(
            "score_mode",
            "mean_head_sampled_probability",
        ),
        coarse_selection_mode=routing.pop(
            "coarse_selection_mode",
            "row_topk_ratio",
        ),
        coarse_keep_ratio=float(routing.pop("coarse_keep_ratio", 0.25)),
        coarse_keep_count=_optional_int(
            routing.pop("coarse_keep_count", None)
        ),
        refine_factor=int(routing.pop("refine_factor", 2)),
        fine_selection_mode=routing.pop(
            "fine_selection_mode",
            "row_topk_ratio",
        ),
        fine_keep_ratio=float(routing.pop("fine_keep_ratio", 0.25)),
        fine_keep_count=_optional_int(
            routing.pop("fine_keep_count", None)
        ),
        min_active_key_scopes_per_query_scope=int(
            routing.pop("min_active_key_scopes_per_query_scope", 1)
        ),
        keep_camera_tokens=bool(
            special_tokens.pop("keep_camera_tokens", True)
        ),
        keep_register_tokens=bool(
            special_tokens.pop("keep_register_tokens", True)
        ),
        dense_special_queries=bool(
            special_tokens.pop("dense_special_queries", True)
        ),
        dense_special_keys=bool(
            special_tokens.pop("dense_special_keys", True)
        ),
        backend_type=backend.pop(
            "type",
            "query_chunked_dense_mask_reference",
        ),
        query_chunk_size=int(backend.pop("query_chunk_size", 128)),
        save_anchor_indices=bool(
            debug.pop("save_anchor_indices", True)
        ),
        save_pair_scopes=bool(debug.pop("save_pair_scopes", True)),
        save_mask_statistics=bool(
            debug.pop("save_mask_statistics", True)
        ),
        save_materialized_full_mask=bool(
            debug.pop("save_materialized_full_mask", False)
        ),
        materialize_full_mask_max_patch_tokens=int(
            debug.pop("materialize_full_mask_max_patch_tokens", 512)
        ),
        profile_components=(
            profile_components
            if profile_override_present
            else bool(debug.pop("profile_components", profile_components))
        ),
    )
    if profile_override_present:
        debug.pop("profile_components", None)
    if routing:
        raise ValueError(
            f"Unknown adaptive routing keys: {sorted(routing)}"
        )
    if special_tokens:
        raise ValueError(
            "Unknown adaptive special_tokens keys: "
            f"{sorted(special_tokens)}"
        )
    if backend:
        raise ValueError(
            f"Unknown adaptive backend keys: {sorted(backend)}"
        )
    if debug:
        raise ValueError(
            f"Unknown adaptive debug keys: {sorted(debug)}"
        )
    validate_adaptive_pair_scope_config(config)
    return config


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def validate_adaptive_pair_scope_config(
    config: AdaptivePairScopeConfig,
) -> None:
    """Validate values that do not depend on the runtime token count."""

    if len(set(config.enabled_layers)) != len(config.enabled_layers):
        raise ValueError("adaptive enabled_layers must not contain duplicates")
    if any(layer < 0 for layer in config.enabled_layers):
        raise ValueError("adaptive enabled_layers must be non-negative")
    specified = sum(
        value is not None
        for value in (config.coarse_num_anchors, config.coarse_stride)
    )
    if specified != 1:
        raise ValueError(
            "exactly one of coarse_num_anchors and coarse_stride must be set"
        )
    if (
        config.coarse_num_anchors is not None
        and config.coarse_num_anchors < 1
    ):
        raise ValueError("coarse_num_anchors must be positive")
    if config.coarse_stride is not None and config.coarse_stride < 1:
        raise ValueError("coarse_stride must be positive")
    if config.routing_score_mode not in ROUTING_SCORE_MODES:
        raise ValueError(
            "routing_score_mode must be one of "
            f"{sorted(ROUTING_SCORE_MODES)}, got "
            f"{config.routing_score_mode!r}"
        )
    for label, mode in (
        ("coarse", config.coarse_selection_mode),
        ("fine", config.fine_selection_mode),
    ):
        if mode not in SELECTION_MODES:
            raise ValueError(
                f"{label}_selection_mode must be one of "
                f"{sorted(SELECTION_MODES)}, got {mode!r}"
            )
    for label, ratio in (
        ("coarse_keep_ratio", config.coarse_keep_ratio),
        ("fine_keep_ratio", config.fine_keep_ratio),
    ):
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"{label} must be in (0, 1], got {ratio}")
    if (
        config.coarse_selection_mode == "row_topk_count"
        and config.coarse_keep_count is None
    ):
        raise ValueError(
            "coarse_keep_count is required for row_topk_count"
        )
    if (
        config.fine_selection_mode == "row_topk_count"
        and config.fine_keep_count is None
    ):
        raise ValueError("fine_keep_count is required for row_topk_count")
    for label, count in (
        ("coarse_keep_count", config.coarse_keep_count),
        ("fine_keep_count", config.fine_keep_count),
    ):
        if count is not None and count < 1:
            raise ValueError(f"{label} must be positive")
    if config.refine_factor < 2:
        raise ValueError("refine_factor must be at least 2")
    if config.min_active_key_scopes_per_query_scope < 1:
        raise ValueError(
            "min_active_key_scopes_per_query_scope must be positive"
        )
    if not all(
        (
            config.keep_camera_tokens,
            config.keep_register_tokens,
            config.dense_special_queries,
            config.dense_special_keys,
        )
    ):
        raise ValueError(
            "adaptive_pair_scope reference requires all camera/register "
            "tokens and dense special-token queries and keys"
        )
    if config.backend_type != "query_chunked_dense_mask_reference":
        raise ValueError(
            "adaptive backend.type must be "
            "'query_chunked_dense_mask_reference'"
        )
    if config.query_chunk_size < 1:
        raise ValueError("adaptive backend query_chunk_size must be positive")
    if config.materialize_full_mask_max_patch_tokens < 1:
        raise ValueError(
            "materialize_full_mask_max_patch_tokens must be positive"
        )


def uniform_flat_anchor_indices(
    token_count: int,
    *,
    num_anchors: int | None = None,
    stride: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return sorted deterministic representatives over ``[0, token_count)``.

    Exactly one of ``num_anchors`` or ``stride`` must be supplied.  Count-based
    anchors are centers of equally spaced strata.  If the requested count
    exceeds ``token_count``, every token is used once.
    """

    token_count = int(token_count)
    if token_count < 1:
        raise ValueError(f"token_count must be positive, got {token_count}")
    if (num_anchors is None) == (stride is None):
        raise ValueError(
            "exactly one of num_anchors and stride must be provided"
        )
    if num_anchors is not None:
        if int(num_anchors) < 1:
            raise ValueError("num_anchors must be positive")
        count = min(int(num_anchors), token_count)
        ranks = torch.arange(count, dtype=torch.long)
        anchors = ((2 * ranks + 1) * token_count) // (2 * count)
    else:
        assert stride is not None
        stride = int(stride)
        if stride < 1:
            raise ValueError("stride must be positive")
        first = min(stride // 2, token_count - 1)
        anchors = torch.arange(first, token_count, stride, dtype=torch.long)
    if anchors.numel() < 1:
        raise RuntimeError("uniform anchor construction produced no anchor")
    if (
        not torch.all(anchors[1:] > anchors[:-1])
        or int(anchors[0]) < 0
        or int(anchors[-1]) >= token_count
    ):
        raise RuntimeError("uniform anchors are not sorted unique in range")
    return anchors.to(device=device) if device is not None else anchors


def build_axis_scopes_from_anchors(
    anchors: Tensor | Sequence[int],
    *,
    token_count: int,
    level: int = 0,
) -> tuple[AxisScope, ...]:
    """Partition ``[0, token_count)`` by midpoints between sorted anchors."""

    token_count = int(token_count)
    values = _integer_list(anchors)
    if token_count < 1:
        raise ValueError("token_count must be positive")
    if not values:
        raise ValueError("at least one anchor is required")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("anchors must be strictly increasing")
    if values[0] < 0 or values[-1] >= token_count:
        raise ValueError("anchors must lie in [0, token_count)")

    boundaries = [0]
    boundaries.extend(
        (left + right + 1) // 2
        for left, right in zip(values, values[1:])
    )
    boundaries.append(token_count)
    scopes = tuple(
        AxisScope(
            start=boundaries[index],
            end=boundaries[index + 1],
            anchor=anchor,
            level=int(level),
        )
        for index, anchor in enumerate(values)
    )
    validate_axis_scope_partition(scopes, token_count=token_count)
    return scopes


def _integer_list(values: Tensor | Sequence[int]) -> list[int]:
    if isinstance(values, Tensor):
        if values.ndim != 1:
            raise ValueError("anchor tensor must be one-dimensional")
        return [int(value) for value in values.detach().cpu().tolist()]
    return [int(value) for value in values]


def validate_axis_scope_partition(
    scopes: Sequence[AxisScope],
    *,
    token_count: int,
) -> bool:
    """Raise unless scopes are non-empty and exactly partition one axis."""

    cursor = 0
    for scope in scopes:
        if scope.start != cursor:
            raise ValueError(
                f"axis partition has a gap or overlap at {cursor}"
            )
        if scope.end <= scope.start:
            raise ValueError("axis scopes must be non-empty")
        if not scope.start <= scope.anchor < scope.end:
            raise ValueError("axis anchor must belong to its scope")
        cursor = scope.end
    if cursor != int(token_count):
        raise ValueError(
            f"axis partition ends at {cursor}, expected {token_count}"
        )
    return True


def build_coarse_pair_scopes(
    axis_scopes: Sequence[AxisScope],
) -> tuple[PairScope, ...]:
    """Build the complete Cartesian partition of coarse Q and K scopes."""

    pair_scopes: list[PairScope] = []
    width = len(axis_scopes)
    for q_index, q_scope in enumerate(axis_scopes):
        for k_index, k_scope in enumerate(axis_scopes):
            scope_id = q_index * width + k_index
            pair_scopes.append(
                PairScope(
                    q_scope=q_scope,
                    k_scope=k_scope,
                    representative_q=q_scope.anchor,
                    representative_k=k_scope.anchor,
                    level=q_scope.level,
                    scope_id=scope_id,
                )
            )
    return tuple(pair_scopes)


def validate_scope_partition(
    scopes: Sequence[PairScope | LeafPairScope],
    *,
    token_count: int,
) -> bool:
    """Validate a non-overlapping rectangular partition of ``N × N``.

    ``allowed`` values on leaf scopes are deliberately ignored: both allowed
    and rejected leaves participate in the geometric partition.
    """

    token_count = int(token_count)
    if token_count < 1:
        raise ValueError("token_count must be positive")
    if not scopes:
        raise ValueError("pair-scope partition must not be empty")
    rectangles = [_scope_rectangle(scope) for scope in scopes]
    for q_start, q_end, k_start, k_end in rectangles:
        if not (
            0 <= q_start < q_end <= token_count
            and 0 <= k_start < k_end <= token_count
        ):
            raise ValueError("pair scope lies outside the full Q-K matrix")
    q_boundaries = sorted(
        {0, token_count}
        | {value for rect in rectangles for value in rect[:2]}
    )
    for slab_start, slab_end in zip(q_boundaries, q_boundaries[1:]):
        if slab_end <= slab_start:
            raise ValueError("invalid query slab in pair partition")
        intervals = sorted(
            (k_start, k_end)
            for q_start, q_end, k_start, k_end in rectangles
            if q_start <= slab_start and q_end >= slab_end
        )
        cursor = 0
        for k_start, k_end in intervals:
            if k_start != cursor:
                raise ValueError(
                    "pair scopes leave a gap or overlap on a query slab"
                )
            cursor = k_end
        if cursor != token_count:
            raise ValueError(
                "pair scopes do not cover the full key axis"
            )
    return True


def _scope_rectangle(
    scope: PairScope | LeafPairScope,
) -> tuple[int, int, int, int]:
    if isinstance(scope, PairScope):
        return (
            scope.q_scope.start,
            scope.q_scope.end,
            scope.k_scope.start,
            scope.k_scope.end,
        )
    return (
        scope.q_start,
        scope.q_end,
        scope.k_start,
        scope.k_end,
    )


def subdivide_axis_scope(
    scope: AxisScope,
    *,
    refine_factor: int,
) -> tuple[AxisScope, ...]:
    """Uniformly split one interval into at most ``refine_factor`` children."""

    if refine_factor < 1:
        raise ValueError("refine_factor must be positive")
    length = scope.end - scope.start
    if length < 1:
        raise ValueError("cannot subdivide an empty axis scope")
    child_count = min(int(refine_factor), length)
    boundaries = [
        scope.start + (index * length) // child_count
        for index in range(child_count)
    ]
    boundaries.append(scope.end)
    children = tuple(
        AxisScope(
            start=boundaries[index],
            end=boundaries[index + 1],
            anchor=(
                boundaries[index] + boundaries[index + 1] - 1
            )
            // 2,
            level=scope.level + 1,
        )
        for index in range(child_count)
    )
    if children[0].start != scope.start or children[-1].end != scope.end:
        raise RuntimeError("child scopes do not cover their parent")
    cursor = scope.start
    for child in children:
        if child.start != cursor or child.end <= child.start:
            raise RuntimeError("child scopes overlap or contain a gap")
        if not child.start <= child.anchor < child.end:
            raise RuntimeError("child anchor is outside its interval")
        cursor = child.end
    return children


def compute_sampled_pair_scores(
    q: Tensor,
    k: Tensor,
    *,
    query_anchor_indices: Tensor,
    key_anchor_indices: Tensor | None = None,
    score_mode: str,
    scale: float,
) -> Tensor:
    """Score sampled Q-K pairs.

    Args:
        q: Full patch queries with shape ``[B, H, N, D]``.
        k: Full patch keys with shape ``[B, H, N, D]``.
        query_anchor_indices: Original patch indices with shape ``[Mq]``.
        key_anchor_indices: Original patch indices with shape ``[Mk]``.
        score_mode: One of :data:`ROUTING_SCORE_MODES`.
        scale: Q-K logit scale.

    Returns:
        Scores with shape ``[B, Mq, Mk]``.  Sampled probabilities normalize
        only over the sampled ``Mk`` keys, not the original full key axis.
    """

    _validate_qk(q, k)
    if key_anchor_indices is None:
        key_anchor_indices = query_anchor_indices
    q_selected = q.index_select(2, query_anchor_indices.to(q.device))
    k_selected = k.index_select(2, key_anchor_indices.to(k.device))
    return _scores_from_selected_qk(
        q_selected,
        k_selected,
        score_mode=score_mode,
        scale=scale,
    )


def _validate_qk(q: Tensor, k: Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("Q and K must have shape [B, H, N, D]")
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K batch/head/channel dimensions must match")


def _scores_from_selected_qk(
    q_selected: Tensor,
    k_selected: Tensor,
    *,
    score_mode: str,
    scale: float,
) -> Tensor:
    if score_mode not in ROUTING_SCORE_MODES:
        raise ValueError(f"Unknown routing score mode {score_mode!r}")
    logits = torch.matmul(
        q_selected.float(),
        k_selected.float().transpose(-2, -1),
    )
    logits.mul_(float(scale))
    if score_mode == "mean_head_logit":
        return logits.mean(dim=1)
    return logits.softmax(dim=-1, dtype=torch.float32).mean(dim=1)


def select_active_pair_scopes(
    scores: Tensor,
    *,
    selection_mode: str,
    keep_ratio: float,
    keep_count: int | None = None,
    min_active_key_scopes_per_query_scope: int = 1,
) -> Tensor:
    """Select pair scopes while guaranteeing a non-empty key set per row."""

    if scores.ndim != 3:
        raise ValueError("pair scores must have shape [B, Q, K]")
    if scores.shape[-1] < 1:
        raise ValueError("pair scores require at least one key scope")
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode {selection_mode!r}")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    key_count = scores.shape[-1]
    minimum = min(
        key_count,
        max(1, int(min_active_key_scopes_per_query_scope)),
    )
    selected = torch.zeros_like(scores, dtype=torch.bool)
    if selection_mode == "row_topk_ratio":
        count = max(minimum, math.ceil(float(keep_ratio) * key_count))
        indices = scores.topk(
            min(count, key_count),
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        selected.scatter_(-1, indices, True)
    elif selection_mode == "row_topk_count":
        if keep_count is None or int(keep_count) < 1:
            raise ValueError(
                "keep_count must be positive for row_topk_count"
            )
        count = min(key_count, max(minimum, int(keep_count)))
        indices = scores.topk(
            count,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        selected.scatter_(-1, indices, True)
    else:
        flat = scores.flatten(start_dim=1)
        threshold = torch.quantile(
            flat,
            max(0.0, min(1.0, 1.0 - float(keep_ratio))),
            dim=-1,
            keepdim=True,
        )
        selected = scores >= threshold[:, None]

    missing = selected.sum(dim=-1) < minimum
    if missing.any():
        fallback = scores.topk(
            minimum,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        fallback_mask = torch.zeros_like(selected)
        fallback_mask.scatter_(-1, fallback, True)
        selected |= missing[..., None] & fallback_mask
    if not selected.any(dim=-1).all():
        raise RuntimeError("pair-scope selection produced an empty query row")
    return selected


@dataclass(frozen=True)
class _ActiveParent:
    batch_index: int
    q_scope_index: int
    k_scope_index: int
    parent_id: int
    q_children: tuple[AxisScope, ...]
    k_children: tuple[AxisScope, ...]


def refine_active_pair_scopes(
    q_patch: Tensor,
    k_patch: Tensor,
    *,
    coarse_axis_scopes: Sequence[AxisScope],
    coarse_refine: Tensor,
    config: AdaptivePairScopeConfig,
    scale: float,
) -> tuple[
    tuple[tuple[PairScope, ...], ...],
    tuple[tuple[LeafPairScope, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Probe only active parents using independent local child products.

    Active parents with the same local child shape are batched for execution,
    but each parent remains an independent batch item.  No global Cartesian
    product is formed between anchors from different parents.
    """

    _validate_qk(q_patch, k_patch)
    batch_size, _, token_count, _ = q_patch.shape
    coarse_count = len(coarse_axis_scopes)
    if coarse_refine.shape != (batch_size, coarse_count, coarse_count):
        raise ValueError(
            "coarse_refine shape must be [B, M0, M0], got "
            f"{tuple(coarse_refine.shape)}"
        )
    q_children = tuple(
        subdivide_axis_scope(
            scope,
            refine_factor=config.refine_factor,
        )
        for scope in coarse_axis_scopes
    )
    k_children = q_children
    refine_cpu = coarse_refine.detach().cpu()
    leaf_lists: list[list[LeafPairScope]] = [
        [] for _ in range(batch_size)
    ]
    fine_lists: list[list[PairScope]] = [
        [] for _ in range(batch_size)
    ]
    active_records: list[_ActiveParent] = []
    for batch_index in range(batch_size):
        for q_index, q_scope in enumerate(coarse_axis_scopes):
            for k_index, k_scope in enumerate(coarse_axis_scopes):
                parent_id = q_index * coarse_count + k_index
                if not bool(refine_cpu[batch_index, q_index, k_index]):
                    leaf_lists[batch_index].append(
                        LeafPairScope(
                            q_start=q_scope.start,
                            q_end=q_scope.end,
                            k_start=k_scope.start,
                            k_end=k_scope.end,
                            allowed=False,
                            level=q_scope.level,
                            parent_id=parent_id,
                        )
                    )
                    continue
                active_records.append(
                    _ActiveParent(
                        batch_index=batch_index,
                        q_scope_index=q_index,
                        k_scope_index=k_index,
                        parent_id=parent_id,
                        q_children=q_children[q_index],
                        k_children=k_children[k_index],
                    )
                )

    grouped: dict[tuple[int, int], list[_ActiveParent]] = {}
    for record in active_records:
        grouped.setdefault(
            (len(record.q_children), len(record.k_children)),
            [],
        ).append(record)

    fine_probe_pairs = [0] * batch_size
    fine_kept_count = [0] * batch_size
    for (query_count, key_count), records in grouped.items():
        batch_indices = torch.tensor(
            [record.batch_index for record in records],
            device=q_patch.device,
            dtype=torch.long,
        )
        query_indices = torch.tensor(
            [
                [child.anchor for child in record.q_children]
                for record in records
            ],
            device=q_patch.device,
            dtype=torch.long,
        )
        key_indices = torch.tensor(
            [
                [child.anchor for child in record.k_children]
                for record in records
            ],
            device=k_patch.device,
            dtype=torch.long,
        )
        q_local = _batched_gather_tokens(
            q_patch,
            batch_indices=batch_indices,
            token_indices=query_indices,
        )
        k_local = _batched_gather_tokens(
            k_patch,
            batch_indices=batch_indices,
            token_indices=key_indices,
        )
        scores = _scores_from_selected_qk(
            q_local,
            k_local,
            score_mode=config.routing_score_mode,
            scale=scale,
        )
        keep = select_active_pair_scopes(
            scores,
            selection_mode=config.fine_selection_mode,
            keep_ratio=config.fine_keep_ratio,
            keep_count=config.fine_keep_count,
            min_active_key_scopes_per_query_scope=(
                config.min_active_key_scopes_per_query_scope
            ),
        )
        scores_cpu = scores.detach().cpu()
        keep_cpu = keep.detach().cpu()
        for local_index, record in enumerate(records):
            batch_index = record.batch_index
            fine_probe_pairs[batch_index] += query_count * key_count
            for q_child_index, child_q in enumerate(record.q_children):
                for k_child_index, child_k in enumerate(record.k_children):
                    allowed = bool(
                        keep_cpu[
                            local_index,
                            q_child_index,
                            k_child_index,
                        ]
                    )
                    fine_kept_count[batch_index] += int(allowed)
                    score = (
                        scores_cpu[
                            local_index,
                            q_child_index,
                            k_child_index,
                        ].clone()
                        if config.save_pair_scopes
                        else None
                    )
                    fine_scope = PairScope(
                        q_scope=child_q,
                        k_scope=child_k,
                        representative_q=child_q.anchor,
                        representative_k=child_k.anchor,
                        level=child_q.level,
                        score=score,
                        refine=allowed,
                        parent_id=record.parent_id,
                    )
                    if config.save_pair_scopes:
                        fine_lists[batch_index].append(fine_scope)
                    leaf_lists[batch_index].append(
                        LeafPairScope(
                            q_start=child_q.start,
                            q_end=child_q.end,
                            k_start=child_k.start,
                            k_end=child_k.end,
                            allowed=allowed,
                            level=child_q.level,
                            parent_id=record.parent_id,
                        )
                    )

    leaf_scopes = tuple(tuple(items) for items in leaf_lists)
    for batch_leaves in leaf_scopes:
        validate_scope_partition(
            batch_leaves,
            token_count=token_count,
        )
    return (
        tuple(tuple(items) for items in fine_lists),
        leaf_scopes,
        tuple(fine_probe_pairs),
        tuple(fine_kept_count),
    )


def _batched_gather_tokens(
    tensor: Tensor,
    *,
    batch_indices: Tensor,
    token_indices: Tensor,
) -> Tensor:
    """Gather ``[A, H, M, D]`` from ``[B, H, N, D]`` without cross-products."""

    if batch_indices.ndim != 1 or token_indices.ndim != 2:
        raise ValueError(
            "batch_indices must be [A] and token_indices must be [A, M]"
        )
    if token_indices.shape[0] != batch_indices.numel():
        raise ValueError("active-parent batch and token rows must match")
    # Index B and N simultaneously.  Selecting B first would replicate the
    # complete [H, N, D] tensor once per active parent, which is catastrophic
    # for long sequences (for example thousands of parents at 300 frames).
    selected = tensor.permute(0, 2, 1, 3)[
        batch_indices[:, None],
        token_indices,
    ]
    return selected.permute(0, 2, 1, 3).contiguous()


def route_adaptive_pair_scopes(
    q_patch: Tensor,
    k_patch: Tensor,
    *,
    config: AdaptivePairScopeConfig,
    scale: float,
) -> AdaptivePairRoutingResult:
    """Run coarse-grid routing and active-parent-only local refinement."""

    _validate_qk(q_patch, k_patch)
    if q_patch.shape[2] != k_patch.shape[2]:
        raise ValueError("adaptive routing requires the same patch Q/K axis")
    token_count = q_patch.shape[2]
    anchors = uniform_flat_anchor_indices(
        token_count,
        num_anchors=config.coarse_num_anchors,
        stride=config.coarse_stride,
        device=q_patch.device,
    )
    axis_scopes = build_axis_scopes_from_anchors(
        anchors,
        token_count=token_count,
        level=0,
    )
    coarse_pair_scopes = build_coarse_pair_scopes(axis_scopes)
    validate_scope_partition(
        coarse_pair_scopes,
        token_count=token_count,
    )
    coarse_scores = compute_sampled_pair_scores(
        q_patch,
        k_patch,
        query_anchor_indices=anchors,
        score_mode=config.routing_score_mode,
        scale=scale,
    )
    coarse_refine = select_active_pair_scopes(
        coarse_scores,
        selection_mode=config.coarse_selection_mode,
        keep_ratio=config.coarse_keep_ratio,
        keep_count=config.coarse_keep_count,
        min_active_key_scopes_per_query_scope=(
            config.min_active_key_scopes_per_query_scope
        ),
    )
    for scope in coarse_pair_scopes:
        assert scope.scope_id is not None
        q_index = scope.scope_id // len(axis_scopes)
        k_index = scope.scope_id % len(axis_scopes)
        scope.score = coarse_scores[:, q_index, k_index].detach()
        scope.refine = coarse_refine[:, q_index, k_index].detach()
    fine_scopes, leaves, fine_probes, fine_kept = (
        refine_active_pair_scopes(
            q_patch,
            k_patch,
            coarse_axis_scopes=axis_scopes,
            coarse_refine=coarse_refine,
            config=config,
            scale=scale,
        )
    )
    logical_pairs = tuple(
        sum(
            (leaf.q_end - leaf.q_start)
            * (leaf.k_end - leaf.k_start)
            for leaf in batch_leaves
            if leaf.allowed
        )
        for batch_leaves in leaves
    )
    return AdaptivePairRoutingResult(
        coarse_anchor_indices=anchors,
        coarse_axis_scopes=axis_scopes,
        coarse_pair_scopes=coarse_pair_scopes,
        coarse_scores=coarse_scores,
        coarse_refine=coarse_refine,
        fine_pair_scopes=fine_scopes,
        leaf_scopes=leaves,
        fine_probe_pairs_per_batch=fine_probes,
        fine_kept_scope_count_per_batch=fine_kept,
        final_logical_patch_pairs_per_batch=logical_pairs,
        scope_partition_valid=True,
    )


def build_patch_mask_chunk(
    query_start: int,
    query_end: int,
    leaf_scopes: Sequence[Sequence[LeafPairScope]],
    *,
    token_count: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Expand leaf rectangles for contiguous patch queries ``[start, end)``."""

    query_start = int(query_start)
    query_end = int(query_end)
    token_count = int(token_count)
    if not 0 <= query_start < query_end <= token_count:
        raise ValueError("invalid patch query chunk")
    query_indices = torch.arange(
        query_start,
        query_end,
        device=device,
        dtype=torch.long,
    )
    return build_patch_mask_rows(
        query_indices,
        leaf_scopes,
        token_count=token_count,
        device=device,
    )


def build_patch_mask_rows(
    query_indices: Tensor | Sequence[int],
    leaf_scopes: Sequence[Sequence[LeafPairScope]],
    *,
    token_count: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Expand leaf rectangles for arbitrary original patch-query indices."""

    if isinstance(query_indices, Tensor):
        rows = query_indices.to(device=device, dtype=torch.long)
    else:
        rows = torch.tensor(
            list(query_indices),
            device=device,
            dtype=torch.long,
        )
    if rows.ndim != 1 or rows.numel() < 1:
        raise ValueError("query_indices must be a non-empty 1-D sequence")
    if int(rows.min()) < 0 or int(rows.max()) >= int(token_count):
        raise ValueError("patch query index is out of range")
    batch_size = len(leaf_scopes)
    mask = torch.zeros(
        batch_size,
        rows.numel(),
        int(token_count),
        device=rows.device,
        dtype=torch.bool,
    )
    for batch_index, batch_leaves in enumerate(leaf_scopes):
        for leaf in batch_leaves:
            if not leaf.allowed:
                continue
            active_rows = (
                (rows >= leaf.q_start) & (rows < leaf.q_end)
            ).nonzero(as_tuple=False).flatten()
            if active_rows.numel():
                mask[
                    batch_index,
                    active_rows,
                    leaf.k_start : leaf.k_end,
                ] = True
    return mask


def compile_leaf_patch_mask(
    leaf_scopes: Sequence[Sequence[LeafPairScope]]
    | Sequence[LeafPairScope],
    *,
    token_count: int,
    device: torch.device | str | None = None,
) -> CompiledPatchMask:
    """Compile exact leaf semantics into reusable Query-slab row templates."""

    token_count = int(token_count)
    if token_count < 1:
        raise ValueError("token_count must be positive")
    nested = _normalize_leaf_batches(leaf_scopes)
    boundaries = sorted(
        {0, token_count}
        | {
            value
            for batch_leaves in nested
            for leaf in batch_leaves
            if leaf.allowed
            for value in (leaf.q_start, leaf.q_end)
        }
    )
    if boundaries[0] != 0 or boundaries[-1] != token_count:
        raise RuntimeError("compiled Query boundaries do not cover the axis")
    boundary_to_start = {
        value: index for index, value in enumerate(boundaries[:-1])
    }
    boundary_to_end = {
        value: index for index, value in enumerate(boundaries)
    }
    query_scope_count = len(boundaries) - 1
    key_masks = torch.zeros(
        len(nested),
        query_scope_count,
        token_count,
        device=device,
        dtype=torch.bool,
    )
    for batch_index, batch_leaves in enumerate(nested):
        for leaf in batch_leaves:
            if not leaf.allowed:
                continue
            first_scope = boundary_to_start[leaf.q_start]
            end_scope = boundary_to_end[leaf.q_end]
            key_masks[
                batch_index,
                first_scope:end_scope,
                leaf.k_start : leaf.k_end,
            ] = True

    row_to_scope = torch.empty(
        token_count,
        device=key_masks.device,
        dtype=torch.long,
    )
    for scope_index, (start, end) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        if end <= start:
            raise RuntimeError("compiled Query scope must be non-empty")
        row_to_scope[start:end] = scope_index
    return CompiledPatchMask(
        query_boundaries=tuple(boundaries),
        query_scope_by_row=row_to_scope,
        allowed_key_mask_by_query_scope=key_masks,
    )


def gather_compiled_patch_mask_rows(
    query_indices: Tensor | Sequence[int],
    compiled: CompiledPatchMask,
) -> Tensor:
    """Gather exact ``[B, Q, N]`` patch-mask rows from compiled templates."""

    device = compiled.allowed_key_mask_by_query_scope.device
    if isinstance(query_indices, Tensor):
        rows = query_indices.to(device=device, dtype=torch.long)
    else:
        rows = torch.tensor(
            list(query_indices),
            device=device,
            dtype=torch.long,
        )
    token_count = compiled.query_scope_by_row.numel()
    if rows.ndim != 1 or rows.numel() < 1:
        raise ValueError("query_indices must be a non-empty 1-D sequence")
    if int(rows.min()) < 0 or int(rows.max()) >= token_count:
        raise ValueError("patch query index is out of range")
    query_scopes = compiled.query_scope_by_row.index_select(0, rows)
    return compiled.allowed_key_mask_by_query_scope.index_select(
        1,
        query_scopes,
    )


def materialize_full_patch_mask(
    leaf_scopes: Sequence[Sequence[LeafPairScope]]
    | Sequence[LeafPairScope],
    *,
    token_count: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Materialize ``[B, N, N]`` only for small tests and diagnostics."""

    nested = _normalize_leaf_batches(leaf_scopes)
    return build_patch_mask_chunk(
        0,
        int(token_count),
        nested,
        token_count=int(token_count),
        device=device,
    )


def _normalize_leaf_batches(
    leaf_scopes: Sequence[Sequence[LeafPairScope]]
    | Sequence[LeafPairScope],
) -> tuple[tuple[LeafPairScope, ...], ...]:
    if not leaf_scopes:
        raise ValueError("leaf_scopes must not be empty")
    first = leaf_scopes[0]
    if isinstance(first, LeafPairScope):
        return (tuple(leaf_scopes),)  # type: ignore[arg-type]
    return tuple(tuple(batch) for batch in leaf_scopes)  # type: ignore[arg-type]


def original_patch_token_indices(
    *,
    num_frames: int,
    tokens_per_frame: int,
    num_special_tokens: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Map frame-major patch indices to the original flattened token layout."""

    patches_per_frame = int(tokens_per_frame) - int(num_special_tokens)
    if num_frames < 1 or patches_per_frame < 1:
        raise ValueError("frame and patch counts must be positive")
    patch_axis = torch.arange(
        int(num_frames) * patches_per_frame,
        device=device,
        dtype=torch.long,
    )
    return (
        patch_axis.div(patches_per_frame, rounding_mode="floor")
        * int(tokens_per_frame)
        + int(num_special_tokens)
        + patch_axis.remainder(patches_per_frame)
    )


def adaptive_pair_scope_attention_block(
    block: "SelfAttentionBlock",
    x: Tensor,
    *,
    num_frames: int,
    tokens_per_frame: int,
    num_special_tokens: int,
    patch_grid_size: tuple[int, int],
    layer_index: int,
    config: AdaptivePairScopeConfig,
) -> AdaptivePairBlockResult:
    """Execute one full residual block with same-layer hierarchical routing."""

    if block.training:
        raise NotImplementedError(
            "adaptive pair-scope attention is currently inference-only"
        )
    if x.ndim != 3:
        raise ValueError(
            "adaptive pair-scope attention expects [B, T, C], got "
            f"{tuple(x.shape)}"
        )
    validate_adaptive_pair_scope_config(config)
    batch_size, total_tokens, embed_dim = x.shape
    expected_tokens = int(num_frames) * int(tokens_per_frame)
    if total_tokens != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} flattened tokens, got {total_tokens}"
        )
    patches_per_frame = int(tokens_per_frame) - int(num_special_tokens)
    if patches_per_frame != int(patch_grid_size[0]) * int(
        patch_grid_size[1]
    ):
        raise ValueError(
            "adaptive pair-scope token layout does not match patch grid"
        )
    patch_count = int(num_frames) * patches_per_frame
    special_count = int(num_frames) * int(num_special_tokens)
    patch_token_indices = original_patch_token_indices(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        num_special_tokens=num_special_tokens,
        device=x.device,
    )
    token_to_patch = torch.full(
        (total_tokens,),
        -1,
        device=x.device,
        dtype=torch.long,
    )
    token_to_patch[patch_token_indices] = torch.arange(
        patch_count,
        device=x.device,
    )

    timer = _ComponentTimer(config.profile_components, x.device)
    with timer.measure("normalization"):
        normalized = block.norm1(x)
    with timer.measure("qkv_projection"):
        qkv = block.attn.qkv(normalized).reshape(
            batch_size,
            total_tokens,
            3,
            block.attn.num_heads,
            embed_dim // block.attn.num_heads,
        )
        q, k, v = torch.unbind(qkv, dim=2)
        q, k, v = (
            tensor.transpose(1, 2) for tensor in (q, k, v)
        )
        if block.attn.use_qk_norm:
            q = block.attn.q_norm(q)
            k = block.attn.k_norm(k)

    q_patch = q.index_select(2, patch_token_indices)
    k_patch = k.index_select(2, patch_token_indices)
    with timer.measure("routing"):
        routing = route_adaptive_pair_scopes(
            q_patch,
            k_patch,
            config=config,
            scale=block.attn.scale,
        )
    with timer.measure("mask_compilation"):
        compiled_patch_mask = compile_leaf_patch_mask(
            routing.leaf_scopes,
            token_count=patch_count,
            device=x.device,
        )

    q_attention = q.to(dtype=v.dtype)
    k_attention = k.to(dtype=v.dtype)
    attended = torch.empty_like(q_attention)
    all_query_rows_nonempty = True
    query_chunk_size = min(config.query_chunk_size, total_tokens)
    with timer.measure("attention_kernel"):
        for query_start in range(0, total_tokens, query_chunk_size):
            query_end = min(
                query_start + query_chunk_size,
                total_tokens,
            )
            patch_rows = token_to_patch[query_start:query_end]
            patch_positions = (patch_rows >= 0).nonzero(
                as_tuple=False
            ).flatten()
            full_mask = torch.ones(
                batch_size,
                query_end - query_start,
                total_tokens,
                device=x.device,
                dtype=torch.bool,
            )
            if patch_positions.numel():
                queried_patch_indices = patch_rows.index_select(
                    0,
                    patch_positions,
                )
                patch_mask = gather_compiled_patch_mask_rows(
                    queried_patch_indices,
                    compiled_patch_mask,
                )
                all_query_rows_nonempty = bool(
                    all_query_rows_nonempty
                    and patch_mask.any(dim=-1).all().item()
                )
                full_mask[
                    :,
                    patch_positions[:, None],
                    patch_token_indices[None, :],
                ] = patch_mask
            attended[:, :, query_start:query_end] = (
                F.scaled_dot_product_attention(
                    q_attention[:, :, query_start:query_end],
                    k_attention,
                    v,
                    attn_mask=full_mask[:, None],
                    scale=block.attn.scale,
                )
            )

    with timer.measure("attention_output_projection"):
        attended_tokens = attended.transpose(1, 2).reshape(
            batch_size,
            total_tokens,
            embed_dim,
        )
        attention_update = block.attn.proj_drop(
            block.attn.proj(attended_tokens)
        )
    with timer.measure("residual_mlp"):
        x_attn = x + block.ls1(attention_update)
        output = x_attn + block.ls2(
            block.mlp(block.norm2(x_attn))
        )

    coarse_count = int(routing.coarse_anchor_indices.numel())
    coarse_active_per_batch = tuple(
        int(value)
        for value in routing.coarse_refine.sum(
            dim=(-2, -1),
            dtype=torch.int64,
        )
        .detach()
        .cpu()
        .tolist()
    )
    logical_patch_pairs = routing.final_logical_patch_pairs_per_batch
    dense_special_pairs = (
        patch_count * special_count
        + special_count * total_tokens
    )
    stats: dict[str, Any] = {
        "layer_index": int(layer_index),
        "algorithm": "adaptive_pair_scope",
        "progressive_attention_semantics": (
            "within_layer_adaptive_pair_scope_reference_v1"
        ),
        "num_frames": int(num_frames),
        "patches_per_frame": patches_per_frame,
        "full_patch_tokens": patch_count,
        "special_tokens": special_count,
        "coarse_anchor_count": coarse_count,
        "coarse_requested_num_anchors": config.coarse_num_anchors,
        "coarse_stride": config.coarse_stride,
        "coarse_probe_pairs": coarse_count * coarse_count,
        "coarse_active_scope_count": _mean_int(
            coarse_active_per_batch
        ),
        "coarse_active_scope_count_per_batch": list(
            coarse_active_per_batch
        ),
        "coarse_active_scope_ratio": (
            sum(coarse_active_per_batch)
            / max(
                batch_size * coarse_count * coarse_count,
                1,
            )
        ),
        "fine_probe_pairs": _mean_int(
            routing.fine_probe_pairs_per_batch
        ),
        "fine_probe_pairs_per_batch": list(
            routing.fine_probe_pairs_per_batch
        ),
        "fine_kept_scope_count": _mean_int(
            routing.fine_kept_scope_count_per_batch
        ),
        "fine_kept_scope_count_per_batch": list(
            routing.fine_kept_scope_count_per_batch
        ),
        "final_logical_patch_pairs": _mean_int(
            logical_patch_pairs
        ),
        "final_logical_patch_pairs_per_batch": list(
            logical_patch_pairs
        ),
        "final_patch_pair_density": (
            sum(logical_patch_pairs)
            / max(batch_size * patch_count * patch_count, 1)
        ),
        "logical_attention_pairs_per_batch": (
            _mean_int(logical_patch_pairs) + dense_special_pairs
        ),
        "evaluated_attention_pairs_per_batch": (
            total_tokens * total_tokens
        ),
        "routing_score_mode": config.routing_score_mode,
        "sampled_probability_normalization": (
            "sampled_key_anchors_only"
            if config.routing_score_mode
            == "mean_head_sampled_probability"
            else None
        ),
        "sampled_probability_is_full_attention_probability": False,
        "coarse_selection_mode": config.coarse_selection_mode,
        "fine_selection_mode": config.fine_selection_mode,
        "refine_factor": config.refine_factor,
        "attention_backend": config.backend_type,
        "efficient_sparse_kernel": False,
        "qkv_projection_count": 1,
        "scope_partition_valid": routing.scope_partition_valid,
        "all_query_rows_nonempty": all_query_rows_nonempty,
        "all_patch_queries_receive_attention_output": True,
        "sampled_token_scatter": False,
        "dense_special_queries": True,
        "dense_special_keys": True,
        "cross_layer_scope_inheritance": False,
        "component_time_ms": dict(timer.elapsed_ms),
    }
    debug = _routing_debug_payload(
        routing,
        config=config,
        patch_count=patch_count,
    )
    return AdaptivePairBlockResult(
        output=output,
        stats=stats,
        routing=routing,
        debug=debug,
    )


def _mean_int(values: Iterable[int]) -> int:
    items = [int(value) for value in values]
    if not items:
        return 0
    return int(round(sum(items) / len(items)))


def _routing_debug_payload(
    routing: AdaptivePairRoutingResult,
    *,
    config: AdaptivePairScopeConfig,
    patch_count: int,
) -> dict[str, Any]:
    debug: dict[str, Any] = {}
    if config.save_anchor_indices:
        debug["coarse_anchor_indices"] = (
            routing.coarse_anchor_indices.detach().cpu()
        )
    if config.save_pair_scopes:
        debug["coarse_axis_scopes"] = [
            _axis_scope_dict(scope)
            for scope in routing.coarse_axis_scopes
        ]
        debug["coarse_pair_scopes"] = [
            _pair_scope_dict(scope)
            for scope in routing.coarse_pair_scopes
        ]
        debug["coarse_scores"] = routing.coarse_scores.detach().cpu()
        debug["coarse_refine"] = routing.coarse_refine.detach().cpu()
        debug["fine_pair_scopes"] = [
            [_pair_scope_dict(scope) for scope in scopes]
            for scopes in routing.fine_pair_scopes
        ]
        debug["final_leaf_scopes"] = [
            [_leaf_scope_dict(scope) for scope in scopes]
            for scopes in routing.leaf_scopes
        ]
    if (
        config.save_materialized_full_mask
        and patch_count <= config.materialize_full_mask_max_patch_tokens
    ):
        debug["materialized_full_patch_mask"] = (
            materialize_full_patch_mask(
                routing.leaf_scopes,
                token_count=patch_count,
                device="cpu",
            )
        )
    return debug


def _axis_scope_dict(scope: AxisScope) -> dict[str, int]:
    return {
        "start": scope.start,
        "end": scope.end,
        "anchor": scope.anchor,
        "level": scope.level,
    }


def _pair_scope_dict(scope: PairScope) -> dict[str, Any]:
    if isinstance(scope.refine, Tensor):
        refine: bool | list[bool] = [
            bool(value)
            for value in scope.refine.detach().cpu().flatten().tolist()
        ]
    else:
        refine = bool(scope.refine)
    return {
        "q_scope": _axis_scope_dict(scope.q_scope),
        "k_scope": _axis_scope_dict(scope.k_scope),
        "representative_q": scope.representative_q,
        "representative_k": scope.representative_k,
        "level": scope.level,
        "score": _score_debug_value(scope.score),
        "refine": refine,
        "parent_id": scope.parent_id,
        "scope_id": scope.scope_id,
    }


def _score_debug_value(score: Tensor | None) -> float | list[float] | None:
    if score is None:
        return None
    values = score.detach().cpu().flatten()
    if values.numel() == 1:
        return float(values.item())
    return [float(value) for value in values.tolist()]


def _leaf_scope_dict(scope: LeafPairScope) -> dict[str, Any]:
    return {
        "q_start": scope.q_start,
        "q_end": scope.q_end,
        "k_start": scope.k_start,
        "k_end": scope.k_end,
        "allowed": scope.allowed,
        "level": scope.level,
        "parent_id": scope.parent_id,
    }
