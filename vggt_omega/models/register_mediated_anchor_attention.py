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
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


REGISTER_MEDIATED_STRATEGIES = {
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
}
ANCHOR_SCORE_MODES = {"intra", "proxy", "linear_fusion", "quota_union"}
INTRA_SOURCES = {"current_inter_qk", "cached_frame_qk"}
FRAME_BUDGET_MODES = {"uniform", "intra_concentration", "register_importance", "hybrid"}


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


class SegmentProfiler:
    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = enabled
        self.device = device
        self.timings: dict[str, float] = {}

    def measure(self, name: str, fn):
        if not self.enabled:
            return fn()
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize(self.device)
            elapsed_ms = float(start.elapsed_time(end))
        else:
            start_time = time.perf_counter()
            result = fn()
            elapsed_ms = float((time.perf_counter() - start_time) * 1000.0)
        self.timings[name] = self.timings.get(name, 0.0) + elapsed_ms
        return result


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
    score_mode: str,
    proxy_quota_ratio: float,
    intra_source: str,
    precomputed_intra_scores: Tensor | None,
    frame_budget_mode: str,
    frame_budget_top_frac: float,
    frame_budget_lambda_intra: float,
    frame_budget_lambda_reg: float,
    frame_budget_reg_topm: int,
    reg_patch_topk_ratio: float,
    reg_patch_topk_min: int,
    reg_patch_topk_max: int,
    reg_patch_conf_power: float,
    reg_patch_min_conf: float,
    query_conditioned_eta: float,
    gated_anchor_ratio_per_key_frame: float,
    gated_min_per_key_frame: int,
    gated_max_per_key_frame: int,
    always_include_self_frame: bool,
    alpha_cross: float,
    beta_intra: float,
    topm_frames: int | None,
    random_seed: int,
    scale: float,
    profile: bool = False,
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
    if not 0.0 <= proxy_quota_ratio <= 1.0:
        raise ValueError(f"proxy_quota_ratio must be in [0, 1], got {proxy_quota_ratio}")
    if intra_source not in INTRA_SOURCES:
        raise ValueError(f"intra_source must be one of {sorted(INTRA_SOURCES)}, got {intra_source!r}")
    if frame_budget_mode not in FRAME_BUDGET_MODES:
        raise ValueError(
            f"frame_budget_mode must be one of {sorted(FRAME_BUDGET_MODES)}, got {frame_budget_mode!r}"
        )
    if not 0.0 < frame_budget_top_frac <= 1.0:
        raise ValueError(f"frame_budget_top_frac must be in (0, 1], got {frame_budget_top_frac}")
    if frame_budget_reg_topm <= 0:
        raise ValueError(f"frame_budget_reg_topm must be positive, got {frame_budget_reg_topm}")
    if not 0.0 <= reg_patch_topk_ratio <= 1.0:
        raise ValueError(f"reg_patch_topk_ratio must be in [0, 1], got {reg_patch_topk_ratio}")
    if reg_patch_topk_min <= 0:
        raise ValueError(f"reg_patch_topk_min must be positive, got {reg_patch_topk_min}")
    if reg_patch_topk_max <= 0:
        raise ValueError(f"reg_patch_topk_max must be positive, got {reg_patch_topk_max}")
    if reg_patch_topk_max < reg_patch_topk_min:
        raise ValueError(
            "reg_patch_topk_max must be >= reg_patch_topk_min, "
            f"got {reg_patch_topk_max} < {reg_patch_topk_min}"
        )
    if reg_patch_conf_power < 0.0:
        raise ValueError(f"reg_patch_conf_power must be non-negative, got {reg_patch_conf_power}")
    if reg_patch_min_conf < 0.0:
        raise ValueError(f"reg_patch_min_conf must be non-negative, got {reg_patch_min_conf}")
    if gated_anchor_ratio_per_key_frame < 0.0:
        raise ValueError(
            "gated_anchor_ratio_per_key_frame must be non-negative, "
            f"got {gated_anchor_ratio_per_key_frame}"
        )
    if gated_min_per_key_frame < 0:
        raise ValueError(f"gated_min_per_key_frame must be non-negative, got {gated_min_per_key_frame}")
    if gated_max_per_key_frame <= 0:
        raise ValueError(f"gated_max_per_key_frame must be positive, got {gated_max_per_key_frame}")

    strategy = anchor_mode.replace("-", "_").lower()
    if strategy not in REGISTER_MEDIATED_STRATEGIES:
        raise ValueError(
            "anchor_mode must be one of "
            f"{sorted(REGISTER_MEDIATED_STRATEGIES)}, got {anchor_mode!r}"
        )
    score_mode = _resolve_score_mode(strategy=strategy, requested_mode=score_mode)

    batch_size, num_heads, total_tokens, head_dim = q.shape
    layout = get_token_layout(total_tokens, patch_grid_size, num_special_tokens, q.device)
    profile_enabled = bool(profile or debug)
    profiler = SegmentProfiler(enabled=profile_enabled, device=q.device)

    total_start_time = time.perf_counter() if profiler.enabled and q.device.type != "cuda" else None
    total_start_event = None
    total_end_event = None
    if profiler.enabled and q.device.type == "cuda":
        total_start_event = torch.cuda.Event(enable_timing=True)
        total_end_event = torch.cuda.Event(enable_timing=True)
        total_start_event.record()

    with torch.no_grad():
        score_payload = _compute_register_mediated_scores(
            q=q.detach(),
            k=k.detach(),
            layout=layout,
            scale=scale,
            strategy=strategy,
            score_mode=score_mode,
            alpha_cross=alpha_cross,
            beta_intra=beta_intra,
            intra_source_requested=intra_source,
            precomputed_intra_scores=precomputed_intra_scores,
            frame_budget_mode=frame_budget_mode,
            frame_budget_top_frac=frame_budget_top_frac,
            frame_budget_lambda_intra=frame_budget_lambda_intra,
            frame_budget_lambda_reg=frame_budget_lambda_reg,
            frame_budget_reg_topm=frame_budget_reg_topm,
            random_seed=random_seed,
            reg_patch_topk_ratio=reg_patch_topk_ratio,
            reg_patch_topk_min=reg_patch_topk_min,
            reg_patch_topk_max=reg_patch_topk_max,
            reg_patch_conf_power=reg_patch_conf_power,
            reg_patch_min_conf=reg_patch_min_conf,
            profiler=profiler,
        )
        selection = profiler.measure(
            "selection_time_ms",
            lambda: _select_patch_anchors(
                layout=layout,
                frame_scores=score_payload["frame_scores"],
                primary_patch_scores=score_payload["primary_patch_scores"],
                intra_patch_scores=score_payload["intra_patch_scores"],
                proxy_patch_scores=score_payload["proxy_patch_scores"],
                anchor_ratio=anchor_ratio,
                anchor_total=anchor_total,
                anchor_min_per_frame=anchor_min_per_frame,
                anchor_tau=anchor_tau,
                anchor_uniform_mix=anchor_uniform_mix,
                score_mode=score_mode,
                proxy_quota_ratio=proxy_quota_ratio,
            ),
        )

    kv_indices = selection["kv_indices"]
    anchor_budget = int(selection["anchor_budget"])
    full_patch_kv = anchor_budget >= layout.num_frames * layout.patch_count
    use_gating = strategy in {
        "frame_pair_gated",
        "hybrid",
        "random_frame_intra",
        "register_gated_intra",
        "register_gated_intra_query",
        "temporal_neighbor_intra",
        "oracle_frame_intra",
    }
    if use_gating and topm_frames is not None and topm_frames <= 0:
        topm_frames = None

    topm_key_frames = None
    gated_cost = None
    gather_time_ms = 0.0
    sdpa_time_ms = 0.0

    if strategy == "register_gated_intra_query":
        if score_payload["frame_pair_graph"] is None:
            raise RuntimeError("register_gated_intra_query requires frame_pair_graph")
        if score_payload["col_mass"] is None:
            raise RuntimeError("register_gated_intra_query requires intra scores")
        if score_payload["reg_to_patch"] is None:
            raise RuntimeError("register_gated_intra_query requires reg_to_patch affinity")
        if score_payload["register_conditioning"] is None:
            raise RuntimeError("register_gated_intra_query requires register conditioning weights")
        (
            output,
            topm_key_frames,
            gated_cost,
            gather_time_ms,
            sdpa_time_ms,
        ) = _query_conditioned_frame_pair_gated_attention(
            q=q,
            k=k,
            v=v,
            layout=layout,
            selected_by_frame=selection["selected_by_frame"],
            anchor_counts=selection["anchor_counts"],
            frame_pair_graph=score_payload["frame_pair_graph"],
            col_mass=score_payload["col_mass"],
            reg_to_patch=score_payload["reg_to_patch"],
            register_conditioning=score_payload["register_conditioning"],
            topm_frames=topm_frames,
            always_include_self_frame=always_include_self_frame,
            query_conditioned_eta=query_conditioned_eta,
            gated_anchor_ratio_per_key_frame=gated_anchor_ratio_per_key_frame,
            gated_min_per_key_frame=gated_min_per_key_frame,
            gated_max_per_key_frame=gated_max_per_key_frame,
            profiler=profiler,
        )
    elif full_patch_kv or not use_gating:
        def _gather_dense_kv() -> tuple[Tensor, Tensor]:
            gather_index = kv_indices[:, None, :, None].expand(batch_size, num_heads, kv_indices.shape[1], head_dim)
            compressed_k = k.gather(dim=2, index=gather_index)
            compressed_v = v.gather(dim=2, index=gather_index)
            return compressed_k, compressed_v

        compressed_k, compressed_v = profiler.measure("gather_time_ms", _gather_dense_kv)
        gather_time_ms = profiler.timings.get("gather_time_ms", 0.0)
        output = profiler.measure("sdpa_time_ms", lambda: F.scaled_dot_product_attention(q, compressed_k, compressed_v))
        sdpa_time_ms = profiler.timings.get("sdpa_time_ms", 0.0)
    else:
        if score_payload["frame_pair_graph"] is None:
            raise RuntimeError(f"{strategy} requires frame_pair_graph")
        (
            output,
            topm_key_frames,
            gated_cost,
            gather_time_ms,
            sdpa_time_ms,
        ) = _frame_pair_gated_attention(
            q=q,
            k=k,
            v=v,
            layout=layout,
            selected_by_frame=selection["selected_by_frame"],
            anchor_counts=selection["anchor_counts"],
            frame_pair_graph=score_payload["frame_pair_graph"],
            topm_frames=topm_frames,
            always_include_self_frame=always_include_self_frame,
            gated_anchor_ratio_per_key_frame=gated_anchor_ratio_per_key_frame,
            gated_min_per_key_frame=gated_min_per_key_frame,
            gated_max_per_key_frame=gated_max_per_key_frame,
            profiler=profiler,
        )

    if profiler.enabled and q.device.type == "cuda":
        assert total_end_event is not None and total_start_event is not None
        total_end_event.record()
        torch.cuda.synchronize(q.device)
        total_time_ms = float(total_start_event.elapsed_time(total_end_event))
    elif profiler.enabled:
        assert total_start_time is not None
        total_time_ms = float((time.perf_counter() - total_start_time) * 1000.0)
    else:
        total_time_ms = 0.0

    n_reg = int(layout.all_reg_indices.numel())
    original_cost = int(total_tokens * total_tokens)
    compressed_cost_lifting = int(total_tokens * (n_reg + anchor_budget))
    compressed_cost_gated = gated_cost
    if compressed_cost_gated is None:
        relative_cost = float(compressed_cost_lifting) / float(max(original_cost, 1))
    else:
        relative_cost = [float(cost) / float(max(original_cost, 1)) for cost in compressed_cost_gated]

    debug_payload: dict[str, object] = {
        "strategy": strategy,
        "mode": strategy,
        "score_mode": score_mode,
        "intra_source_requested": intra_source,
        "intra_source_used": score_payload["intra_source_used"],
        "cached_intra_available": score_payload["cached_intra_available"],
        "num_frames": int(layout.num_frames),
        "tokens_per_frame": int(layout.tokens_per_frame),
        "num_special_tokens": int(layout.num_special_tokens),
        "patch_count": int(layout.patch_count),
        "anchor_budget": int(anchor_budget),
        "kv_token_count": int(kv_indices.shape[1]),
        "anchor_counts": selection["anchor_counts"],
        "kv_indices": kv_indices,
        "selected_patch_anchor_indices_by_frame": selection["selected_by_frame"],
        "frame_scores": score_payload["frame_scores"],
        "frame_budget_mode": frame_budget_mode,
        "frame_budget_probs": selection["frame_budget_probs"],
        "frame_budget_entropy": selection["frame_budget_entropy"],
        "frame_budget_gini": selection["frame_budget_gini"],
        "top20_budget_ratio": selection["top20_budget_ratio"],
        "topm_key_frames": topm_key_frames,
        "theoretical_cost": {
            "original_cost": original_cost,
            "compressed_cost_lifting": compressed_cost_lifting,
            "compressed_cost_gated": compressed_cost_gated,
            "relative_attention_pair_cost": relative_cost,
        },
        "profile": {
            "score_time_ms": profiler.timings.get("reg_attention_time_ms", 0.0)
            + profiler.timings.get("reg_to_patch_time_ms", 0.0)
            + profiler.timings.get("intra_score_time_ms", 0.0),
            "reg_attention_time_ms": profiler.timings.get("reg_attention_time_ms", 0.0),
            "reg_to_patch_time_ms": profiler.timings.get("reg_to_patch_time_ms", 0.0),
            "intra_score_time_ms": profiler.timings.get("intra_score_time_ms", 0.0),
            "selection_time_ms": profiler.timings.get("selection_time_ms", 0.0),
            "gather_time_ms": gather_time_ms,
            "sdpa_time_ms": sdpa_time_ms,
            "total_time_ms": total_time_ms,
        },
        "warnings": score_payload["warnings"] + selection["warnings"],
    }
    if score_mode == "quota_union":
        debug_payload.update(
            {
                "selected_by_intra": selection["selected_by_intra"],
                "selected_by_proxy": selection["selected_by_proxy"],
                "proxy_quota_ratio": float(proxy_quota_ratio),
                "overlap_between_intra_and_proxy": selection["overlap_between_intra_and_proxy"],
                "final_selected_indices": selection["selected_by_frame"],
            }
        )
    if debug:
        debug_payload.update(
            {
                "frame_pair_graph": score_payload["frame_pair_graph"],
                "col_mass": score_payload["col_mass"],
                "s_cross": score_payload["s_cross"],
                "s_anchor": score_payload["primary_patch_scores"],
                "intra_patch_scores": score_payload["intra_patch_scores"],
                "proxy_patch_scores": score_payload["proxy_patch_scores"],
                "reg_patch_confidence": score_payload["reg_patch_confidence"],
            }
        )
    return output, debug_payload


def _resolve_score_mode(strategy: str, requested_mode: str) -> str:
    mode = requested_mode.replace("-", "_").lower()
    if mode not in ANCHOR_SCORE_MODES:
        raise ValueError(f"score_mode must be one of {sorted(ANCHOR_SCORE_MODES)}, got {requested_mode!r}")
    if strategy == "register_gated_intra":
        return "intra"
    if strategy == "register_gated_intra_query":
        return "intra"
    if strategy in {"all_frame_intra", "random_frame_intra", "temporal_neighbor_intra", "oracle_frame_intra"}:
        return "intra"
    if strategy == "quota_intra_proxy":
        return "quota_union"
    return mode


def _compute_register_mediated_scores(
    q: Tensor,
    k: Tensor,
    layout: TokenLayout,
    scale: float,
    strategy: str,
    score_mode: str,
    alpha_cross: float,
    beta_intra: float,
    intra_source_requested: str,
    precomputed_intra_scores: Tensor | None,
    frame_budget_mode: str,
    frame_budget_top_frac: float,
    frame_budget_lambda_intra: float,
    frame_budget_lambda_reg: float,
    frame_budget_reg_topm: int,
    random_seed: int,
    reg_patch_topk_ratio: float,
    reg_patch_topk_min: int,
    reg_patch_topk_max: int,
    reg_patch_conf_power: float,
    reg_patch_min_conf: float,
    profiler: SegmentProfiler,
) -> dict[str, object]:
    batch_size, num_heads, _, head_dim = q.shape
    num_frames = layout.num_frames
    reg_count = layout.num_special_tokens
    patch_count = layout.patch_count
    tokens_per_frame = layout.tokens_per_frame
    warnings: list[str] = []

    q_frames = q.reshape(batch_size, num_heads, num_frames, tokens_per_frame, head_dim)
    k_frames = k.reshape(batch_size, num_heads, num_frames, tokens_per_frame, head_dim)
    q_reg = q_frames[:, :, :, :reg_count]
    k_reg = k_frames[:, :, :, :reg_count]
    q_patch = q_frames[:, :, :, reg_count:]
    k_patch = k_frames[:, :, :, reg_count:]

    need_intra = (
        score_mode in {"intra", "linear_fusion", "quota_union"}
        or strategy in {
            "all_frame_intra",
            "random_frame_intra",
            "register_gated_intra",
            "register_gated_intra_query",
            "temporal_neighbor_intra",
            "oracle_frame_intra",
        }
        or frame_budget_mode in {"intra_concentration", "hybrid"}
    )
    need_frame_pair_graph = (
        strategy in {
            "frame_pair_gated",
            "hybrid",
            "random_frame_intra",
            "register_gated_intra",
            "register_gated_intra_query",
            "temporal_neighbor_intra",
            "oracle_frame_intra",
        }
        or frame_budget_mode in {"register_importance", "hybrid"}
    )
    need_proxy = score_mode in {"proxy", "linear_fusion", "quota_union"}
    need_query_conditioning = strategy == "register_gated_intra_query"
    synthetic_frame_graph = strategy in {"random_frame_intra", "temporal_neighbor_intra", "oracle_frame_intra"}
    need_register_attention = (
        (need_frame_pair_graph and not synthetic_frame_graph)
        or need_proxy
        or need_query_conditioning
        or frame_budget_mode in {"register_importance", "hybrid"}
    )
    need_reg_to_patch = need_proxy or need_query_conditioning

    col_mass = None
    cached_intra_available = False
    intra_source_used = "current_inter_qk"
    if need_intra and intra_source_requested == "cached_frame_qk":
        cached_intra_available = _precomputed_intra_is_valid(precomputed_intra_scores, batch_size, num_frames, patch_count)
        if cached_intra_available:
            col_mass = torch.nan_to_num(precomputed_intra_scores.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            intra_source_used = "cached_frame_qk"
        else:
            intra_source_used = "current_inter_qk_fallback"
            warnings.append("cached_frame_qk requested but unavailable; falling back to current_inter_qk")

    if need_intra and col_mass is None:
        col_mass = profiler.measure(
            "intra_score_time_ms",
            lambda: _compute_current_intra_col_mass(
                q_patch=q_patch,
                k_patch=k_patch,
                scale=scale,
                batch_size=batch_size,
                num_heads=num_heads,
                num_frames=num_frames,
                patch_count=patch_count,
                device=q.device,
            ),
        )
    elif not need_intra:
        intra_source_used = intra_source_requested

    frame_pair_graph = None
    reg_recv = None
    register_conditioning = None
    if need_register_attention:
        reg_attention = profiler.measure(
            "reg_attention_time_ms",
            lambda: _compute_register_attention_payload(q_reg=q_reg, k_reg=k_reg, scale=scale),
        )
        frame_pair_graph = reg_attention["frame_pair_graph"]
        reg_recv = reg_attention["reg_recv"]
        register_conditioning = reg_attention["register_conditioning"]
    if strategy == "random_frame_intra":
        frame_pair_graph = _compute_random_frame_pair_graph(
            batch_size=batch_size,
            num_frames=num_frames,
            device=q.device,
            seed=int(random_seed),
        )
    elif strategy == "temporal_neighbor_intra":
        frame_pair_graph = _compute_temporal_neighbor_frame_pair_graph(
            batch_size=batch_size,
            num_frames=num_frames,
            device=q.device,
        )
    elif strategy == "oracle_frame_intra":
        frame_pair_graph = profiler.measure(
            "oracle_frame_graph_time_ms",
            lambda: _compute_oracle_patch_frame_pair_graph(
                q_patch=q_patch,
                k_patch=k_patch,
                scale=scale,
                batch_size=batch_size,
                num_heads=num_heads,
                num_frames=num_frames,
                patch_count=patch_count,
                device=q.device,
            ),
        )

    reg_to_patch = None
    reg_patch_confidence = None
    s_cross = None
    if need_reg_to_patch:
        reg_to_patch, reg_patch_confidence = profiler.measure(
            "reg_to_patch_time_ms",
            lambda: _compute_reg_to_patch_affinity(
                q_reg=q_reg,
                k_patch=k_patch,
                scale=scale,
                topk_ratio=reg_patch_topk_ratio,
                topk_min=reg_patch_topk_min,
                topk_max=reg_patch_topk_max,
                conf_power=reg_patch_conf_power,
                min_conf=reg_patch_min_conf,
            ),
        )
    if need_proxy:
        if reg_recv is None or reg_to_patch is None:
            raise RuntimeError("Proxy score requested without register attention or reg_to_patch affinity")
        s_cross = (reg_recv[:, :, :, None] * reg_to_patch).sum(dim=2)
        s_cross = torch.nan_to_num(s_cross, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    anchor_scores = _compute_anchor_scores(
        s_cross=s_cross,
        col_mass=col_mass,
        score_mode=score_mode,
        alpha_cross=alpha_cross,
        beta_intra=beta_intra,
    )
    frame_scores = _compute_frame_scores(
        col_mass=col_mass,
        s_cross=s_cross,
        frame_pair_graph=frame_pair_graph,
        mode=frame_budget_mode,
        top_frac=frame_budget_top_frac,
        lambda_intra=frame_budget_lambda_intra,
        lambda_reg=frame_budget_lambda_reg,
        reg_topm=frame_budget_reg_topm,
        batch_size=batch_size,
        num_frames=num_frames,
        device=q.device,
    )

    return {
        "frame_pair_graph": frame_pair_graph,
        "reg_recv": reg_recv,
        "register_conditioning": register_conditioning,
        "reg_to_patch": reg_to_patch,
        "reg_patch_confidence": reg_patch_confidence,
        "s_cross": s_cross,
        "col_mass": col_mass,
        "primary_patch_scores": anchor_scores["primary"],
        "intra_patch_scores": anchor_scores["intra"],
        "proxy_patch_scores": anchor_scores["proxy"],
        "frame_scores": frame_scores,
        "intra_source_used": intra_source_used,
        "cached_intra_available": cached_intra_available,
        "warnings": warnings,
    }


def _precomputed_intra_is_valid(
    precomputed_intra_scores: Tensor | None,
    batch_size: int,
    num_frames: int,
    patch_count: int,
) -> bool:
    if precomputed_intra_scores is None:
        return False
    return tuple(precomputed_intra_scores.shape) == (batch_size, num_frames, patch_count)


def _compute_register_attention_payload(q_reg: Tensor, k_reg: Tensor, scale: float) -> dict[str, Tensor]:
    batch_size, num_heads, num_frames, reg_count, head_dim = q_reg.shape
    q_reg_flat = q_reg.reshape(batch_size, num_heads, num_frames * reg_count, head_dim).float()
    k_reg_flat = k_reg.reshape(batch_size, num_heads, num_frames * reg_count, head_dim).float()
    reg_logits = torch.matmul(q_reg_flat, k_reg_flat.transpose(-2, -1)) * scale
    a_reg = reg_logits.softmax(dim=-1)
    a_reg_by_frame = a_reg.reshape(batch_size, num_heads, num_frames, reg_count, num_frames, reg_count)
    frame_pair_graph = a_reg_by_frame.mean(dim=(1, 3, 5))
    cross_frame_mask = ~torch.eye(num_frames, device=q_reg.device, dtype=torch.bool)
    reg_recv = (
        a_reg_by_frame
        * cross_frame_mask.view(1, 1, num_frames, 1, num_frames, 1).to(dtype=a_reg_by_frame.dtype)
    ).sum(dim=(2, 3)).mean(dim=1)
    register_conditioning = a_reg_by_frame.mean(dim=(1, 3))
    return {
        "frame_pair_graph": frame_pair_graph,
        "reg_recv": torch.nan_to_num(reg_recv, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0),
        "register_conditioning": torch.nan_to_num(
            register_conditioning, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0),
    }


def _compute_current_intra_col_mass(
    q_patch: Tensor,
    k_patch: Tensor,
    scale: float,
    batch_size: int,
    num_heads: int,
    num_frames: int,
    patch_count: int,
    device: torch.device,
) -> Tensor:
    col_mass = torch.empty(batch_size, num_frames, patch_count, device=device, dtype=torch.float32)
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
    return torch.nan_to_num(col_mass, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _compute_reg_to_patch_affinity(
    q_reg: Tensor,
    k_patch: Tensor,
    scale: float,
    topk_ratio: float,
    topk_min: int,
    topk_max: int,
    conf_power: float,
    min_conf: float,
) -> tuple[Tensor, Tensor]:
    batch_size, num_heads, num_frames, reg_count, head_dim = q_reg.shape
    patch_count = k_patch.shape[-2]
    affinity = torch.zeros(
        batch_size,
        num_frames,
        reg_count,
        patch_count,
        device=q_reg.device,
        dtype=torch.float32,
    )
    confidence_out = torch.zeros(batch_size, num_frames, reg_count, device=q_reg.device, dtype=torch.float32)
    k_keep = min(topk_max, max(topk_min, int(math.ceil(max(topk_ratio, 0.0) * patch_count))))
    k_keep = min(k_keep, patch_count)
    denom = max(batch_size * num_heads * reg_count * patch_count, 1)
    frame_chunk = max(1, min(num_frames, 16_000_000 // denom))
    log_patch_count = math.log(max(patch_count, 2))
    for start in range(0, num_frames, frame_chunk):
        end = min(start + frame_chunk, num_frames)
        logits = (
            torch.matmul(
                q_reg[:, :, start:end].float(),
                k_patch[:, :, start:end].float().transpose(-2, -1),
            )
            * scale
        )
        base_prob = logits.softmax(dim=-1)
        entropy = -(base_prob * (base_prob.clamp_min(1.0e-12).log())).sum(dim=-1)
        conf = (1.0 - entropy / log_patch_count).clamp_min(0.0)
        conf_weight = torch.where(
            conf >= float(min_conf),
            conf.clamp_min(0.0) ** float(conf_power),
            torch.zeros_like(conf),
        )
        weighted_prob = base_prob * conf_weight[..., None]
        if k_keep >= patch_count:
            sparse_prob = weighted_prob
        else:
            topk_vals, topk_idx = weighted_prob.topk(k_keep, dim=-1, largest=True, sorted=False)
            topk_sum = topk_vals.sum(dim=-1, keepdim=True)
            normalized_vals = torch.where(topk_sum > 1.0e-12, topk_vals / topk_sum, torch.zeros_like(topk_vals))
            sparse_prob = torch.zeros_like(weighted_prob)
            sparse_prob.scatter_(-1, topk_idx, normalized_vals)

            fallback_mask = (conf_weight > 0.0) & (topk_sum.squeeze(-1) <= 1.0e-12)
            if bool(fallback_mask.any().item()):
                fb_vals, fb_idx = base_prob.topk(k_keep, dim=-1, largest=True, sorted=False)
                fb_vals = fb_vals / fb_vals.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
                fallback_sparse = torch.zeros_like(weighted_prob)
                fallback_sparse.scatter_(-1, fb_idx, fb_vals * conf_weight[..., None])
                sparse_prob = torch.where(
                    fallback_mask[..., None],
                    fallback_sparse,
                    sparse_prob,
                )

        affinity[:, start:end] = sparse_prob.mean(dim=1)
        confidence_out[:, start:end] = conf.mean(dim=1)
    return affinity, confidence_out


def _compute_anchor_scores(
    s_cross: Tensor | None,
    col_mass: Tensor | None,
    score_mode: str,
    alpha_cross: float,
    beta_intra: float,
) -> dict[str, Tensor | None]:
    intra_scores = _normalize_per_frame(col_mass) if col_mass is not None else None
    proxy_scores = _normalize_per_frame(s_cross) if s_cross is not None else None
    if score_mode == "intra":
        return {
            "primary": intra_scores,
            "intra": intra_scores,
            "proxy": proxy_scores,
        }
    if score_mode == "proxy":
        return {
            "primary": proxy_scores,
            "intra": intra_scores,
            "proxy": proxy_scores,
        }
    if score_mode == "linear_fusion":
        if intra_scores is None or proxy_scores is None:
            raise RuntimeError("linear_fusion requires both intra and proxy scores")
        return {
            "primary": float(alpha_cross) * proxy_scores + float(beta_intra) * intra_scores,
            "intra": intra_scores,
            "proxy": proxy_scores,
        }
    if score_mode == "quota_union":
        return {
            "primary": intra_scores,
            "intra": intra_scores,
            "proxy": proxy_scores,
        }
    raise ValueError(f"Unsupported score_mode: {score_mode!r}")


def _compute_frame_scores(
    col_mass: Tensor | None,
    s_cross: Tensor | None,
    frame_pair_graph: Tensor | None,
    mode: str,
    top_frac: float,
    lambda_intra: float,
    lambda_reg: float,
    reg_topm: int,
    batch_size: int,
    num_frames: int,
    device: torch.device,
) -> Tensor:
    if mode == "uniform":
        return torch.ones(batch_size, num_frames, device=device, dtype=torch.float32)
    if mode == "intra_concentration":
        if col_mass is None:
            raise RuntimeError("intra_concentration frame budget requires col_mass")
        return _compute_intra_concentration(col_mass=col_mass, top_frac=top_frac)
    if mode == "register_importance":
        if frame_pair_graph is None:
            raise RuntimeError("register_importance frame budget requires frame_pair_graph")
        return _compute_register_importance(frame_pair_graph=frame_pair_graph, reg_topm=reg_topm)
    if mode == "hybrid":
        if col_mass is None:
            raise RuntimeError("hybrid frame budget requires col_mass")
        if frame_pair_graph is None:
            raise RuntimeError("hybrid frame budget requires frame_pair_graph")
        intra = _compute_intra_concentration(col_mass=col_mass, top_frac=top_frac)
        reg = _compute_register_importance(frame_pair_graph=frame_pair_graph, reg_topm=reg_topm)
        return float(lambda_intra) * intra + float(lambda_reg) * reg
    raise ValueError(f"Unsupported frame budget mode: {mode!r}")


def _compute_intra_concentration(col_mass: Tensor, top_frac: float) -> Tensor:
    patch_count = col_mass.shape[-1]
    topk = max(1, int(math.ceil(patch_count * top_frac)))
    top_mass = col_mass.topk(topk, dim=-1).values.sum(dim=-1)
    total_mass = col_mass.sum(dim=-1).clamp_min(1.0e-12)
    return torch.nan_to_num(top_mass / total_mass, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_register_importance(frame_pair_graph: Tensor, reg_topm: int) -> Tensor:
    batch_size, num_frames, _ = frame_pair_graph.shape
    device = frame_pair_graph.device
    eye = torch.eye(num_frames, device=device, dtype=torch.bool)
    off_diag = frame_pair_graph.masked_fill(eye.unsqueeze(0), float("-inf"))
    topm = max(1, min(int(reg_topm), max(num_frames - 1, 1)))
    outgoing_vals = off_diag.topk(topm, dim=-1, largest=True, sorted=False).values
    outgoing = outgoing_vals.masked_fill(~torch.isfinite(outgoing_vals), 0.0).mean(dim=-1)
    incoming_vals = off_diag.transpose(1, 2).topk(topm, dim=-1, largest=True, sorted=False).values
    incoming = incoming_vals.masked_fill(~torch.isfinite(incoming_vals), 0.0).mean(dim=-1)
    return torch.nan_to_num(incoming + outgoing, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_random_frame_pair_graph(
    batch_size: int,
    num_frames: int,
    device: torch.device,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    graph = torch.rand(batch_size, num_frames, num_frames, device=device, generator=generator, dtype=torch.float32)
    return graph


def _compute_temporal_neighbor_frame_pair_graph(
    batch_size: int,
    num_frames: int,
    device: torch.device,
) -> Tensor:
    frame_ids = torch.arange(num_frames, device=device, dtype=torch.float32)
    distance = (frame_ids[:, None] - frame_ids[None, :]).abs()
    graph = 1.0 / (1.0 + distance)
    return graph.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def _compute_oracle_patch_frame_pair_graph(
    q_patch: Tensor,
    k_patch: Tensor,
    scale: float,
    batch_size: int,
    num_heads: int,
    num_frames: int,
    patch_count: int,
    device: torch.device,
) -> Tensor:
    graph = torch.empty(batch_size, num_frames, num_frames, device=device, dtype=torch.float32)
    key_patch_flat = k_patch.float().reshape(batch_size, num_heads, num_frames * patch_count, -1)
    for query_frame in range(num_frames):
        q_frame = q_patch[:, :, query_frame].float()
        logits = torch.matmul(q_frame, key_patch_flat.transpose(-2, -1)) * scale
        prob = logits.softmax(dim=-1).reshape(batch_size, num_heads, patch_count, num_frames, patch_count)
        graph[:, query_frame] = prob.sum(dim=-1).mean(dim=(1, 2))
    return torch.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _normalize_per_frame(scores: Tensor | None) -> Tensor | None:
    if scores is None:
        return None
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = scores.amin(dim=-1, keepdim=True)
    max_value = scores.amax(dim=-1, keepdim=True)
    denom = max_value - min_value
    return torch.where(denom > 1.0e-12, (scores - min_value) / denom.clamp_min(1.0e-12), torch.zeros_like(scores))


def _select_patch_anchors(
    layout: TokenLayout,
    frame_scores: Tensor,
    primary_patch_scores: Tensor | None,
    intra_patch_scores: Tensor | None,
    proxy_patch_scores: Tensor | None,
    anchor_ratio: float,
    anchor_total: int | None,
    anchor_min_per_frame: int,
    anchor_tau: float,
    anchor_uniform_mix: float,
    score_mode: str,
    proxy_quota_ratio: float,
) -> dict[str, object]:
    batch_size = frame_scores.shape[0]
    total_patch_tokens = layout.num_frames * layout.patch_count
    if anchor_total is None:
        anchor_budget = int(math.ceil(total_patch_tokens * anchor_ratio))
    else:
        anchor_budget = int(anchor_total)
    anchor_budget = max(0, min(anchor_budget, total_patch_tokens))

    min_per_frame = min(int(anchor_min_per_frame), layout.patch_count)
    if layout.num_frames > 0 and anchor_budget < layout.num_frames * min_per_frame:
        min_per_frame = anchor_budget // layout.num_frames

    frame_budget_probs = torch.zeros_like(frame_scores, dtype=torch.float32)
    frame_budget_entropy = torch.zeros(batch_size, device=frame_scores.device, dtype=torch.float32)
    frame_budget_gini = torch.zeros(batch_size, device=frame_scores.device, dtype=torch.float32)
    top20_budget_ratio = torch.zeros(batch_size, device=frame_scores.device, dtype=torch.float32)
    warnings: list[str] = []

    if score_mode == "quota_union":
        if intra_patch_scores is None or proxy_patch_scores is None:
            raise RuntimeError("quota_union requires both intra and proxy scores")
    elif primary_patch_scores is None:
        raise RuntimeError(f"{score_mode} requires a primary patch score tensor")

    max_count = layout.patch_count if anchor_budget >= total_patch_tokens else min(layout.patch_count, anchor_budget)
    anchor_indices = torch.empty(batch_size, anchor_budget, device=frame_scores.device, dtype=torch.long)
    anchor_counts = torch.zeros(batch_size, layout.num_frames, device=frame_scores.device, dtype=torch.long)
    selected_by_frame = torch.full(
        (batch_size, layout.num_frames, max_count),
        -1,
        device=frame_scores.device,
        dtype=torch.long,
    )
    selected_by_intra = torch.full_like(selected_by_frame, -1)
    selected_by_proxy = torch.full_like(selected_by_frame, -1)
    overlap_between_intra_and_proxy = torch.zeros(
        batch_size,
        layout.num_frames,
        device=frame_scores.device,
        dtype=torch.long,
    )

    for batch_idx in range(batch_size):
        counts, probs = _allocate_anchor_counts(
            frame_scores=frame_scores[batch_idx],
            num_frames=layout.num_frames,
            patch_count=layout.patch_count,
            anchor_budget=anchor_budget,
            min_per_frame=min_per_frame,
            tau=anchor_tau,
            uniform_mix=anchor_uniform_mix,
        )
        anchor_counts[batch_idx] = counts
        frame_budget_probs[batch_idx] = probs
        frame_budget_entropy[batch_idx] = _normalized_entropy(probs)
        frame_budget_gini[batch_idx] = _gini(probs)
        top20_budget_ratio[batch_idx] = _topk_ratio(counts.float(), 0.2)
        if float(frame_budget_entropy[batch_idx].item()) > 0.98 and float(top20_budget_ratio[batch_idx].item()) < 0.25:
            warnings.append(
                f"batch {batch_idx}: Frame budget is close to uniform; adaptive allocation may be weak for this sample."
            )

        selected_frames = []
        for frame_idx, keep_count_tensor in enumerate(counts):
            keep_count = int(keep_count_tensor.item())
            if keep_count == 0:
                continue
            if score_mode == "quota_union":
                final_local, intra_local, proxy_local, overlap = _quota_union_select_indices(
                    intra_scores=intra_patch_scores[batch_idx, frame_idx],
                    proxy_scores=proxy_patch_scores[batch_idx, frame_idx],
                    keep_count=keep_count,
                    proxy_quota_ratio=proxy_quota_ratio,
                )
                overlap_between_intra_and_proxy[batch_idx, frame_idx] = int(overlap)
                if intra_local.numel() > 0:
                    selected_by_intra[batch_idx, frame_idx, : intra_local.numel()] = layout.patch_indices_by_frame[
                        frame_idx, intra_local
                    ]
                if proxy_local.numel() > 0:
                    selected_by_proxy[batch_idx, frame_idx, : proxy_local.numel()] = layout.patch_indices_by_frame[
                        frame_idx, proxy_local
                    ]
                local_patch = final_local
            else:
                if keep_count >= layout.patch_count:
                    local_patch = torch.arange(layout.patch_count, device=frame_scores.device, dtype=torch.long)
                else:
                    local_patch = primary_patch_scores[batch_idx, frame_idx].topk(
                        keep_count,
                        dim=-1,
                        largest=True,
                        sorted=True,
                    ).indices
            global_patch = layout.patch_indices_by_frame[frame_idx, local_patch]
            selected_by_frame[batch_idx, frame_idx, :keep_count] = global_patch
            selected_frames.append(global_patch)

        if selected_frames:
            selected = torch.cat(selected_frames, dim=0)
        else:
            selected = torch.empty(0, device=frame_scores.device, dtype=torch.long)
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
        "selected_by_intra": selected_by_intra,
        "selected_by_proxy": selected_by_proxy,
        "overlap_between_intra_and_proxy": overlap_between_intra_and_proxy,
        "anchor_budget": anchor_budget,
        "frame_budget_probs": frame_budget_probs,
        "frame_budget_entropy": frame_budget_entropy,
        "frame_budget_gini": frame_budget_gini,
        "top20_budget_ratio": top20_budget_ratio,
        "warnings": warnings,
    }


def _quota_union_select_indices(
    intra_scores: Tensor,
    proxy_scores: Tensor,
    keep_count: int,
    proxy_quota_ratio: float,
) -> tuple[Tensor, Tensor, Tensor, int]:
    patch_count = intra_scores.shape[-1]
    if keep_count >= patch_count:
        all_idx = torch.arange(patch_count, device=intra_scores.device, dtype=torch.long)
        return all_idx, all_idx, torch.empty(0, device=intra_scores.device, dtype=torch.long), patch_count

    n_proxy = int(round(keep_count * proxy_quota_ratio))
    n_proxy = max(0, min(keep_count, n_proxy))
    n_intra = keep_count - n_proxy

    if n_intra > 0:
        intra_top_keep = intra_scores.topk(min(keep_count, patch_count), dim=-1, largest=True, sorted=True).indices
        intra_selected = intra_top_keep[:n_intra]
    else:
        intra_top_keep = intra_scores.topk(min(keep_count, patch_count), dim=-1, largest=True, sorted=True).indices
        intra_selected = torch.empty(0, device=intra_scores.device, dtype=torch.long)

    if n_proxy <= 0:
        return intra_selected, intra_selected, torch.empty(0, device=intra_scores.device, dtype=torch.long), 0

    proxy_top_keep = proxy_scores.topk(min(keep_count, patch_count), dim=-1, largest=True, sorted=True).indices
    overlap = int(torch.isin(proxy_top_keep[:n_proxy], intra_selected).sum().item())
    remaining_mask = torch.ones(patch_count, device=intra_scores.device, dtype=torch.bool)
    if intra_selected.numel() > 0:
        remaining_mask[intra_selected] = False
    remaining_proxy = proxy_top_keep[remaining_mask[proxy_top_keep]]
    proxy_selected = remaining_proxy[:n_proxy]

    if proxy_selected.numel() < n_proxy:
        needed = n_proxy - proxy_selected.numel()
        remaining_intra = intra_top_keep[remaining_mask[intra_top_keep]]
        backfill = remaining_intra[:needed]
        proxy_selected = torch.cat([proxy_selected, backfill], dim=0)
        if backfill.numel() > 0:
            remaining_mask[backfill] = False
    if proxy_selected.numel() < n_proxy:
        remaining_all = torch.arange(patch_count, device=intra_scores.device, dtype=torch.long)[remaining_mask]
        proxy_selected = torch.cat([proxy_selected, remaining_all[: n_proxy - proxy_selected.numel()]], dim=0)

    final_selected = torch.cat([intra_selected, proxy_selected], dim=0)
    return final_selected, intra_selected, proxy_selected, overlap


def _allocate_anchor_counts(
    frame_scores: Tensor,
    num_frames: int,
    patch_count: int,
    anchor_budget: int,
    min_per_frame: int,
    tau: float,
    uniform_mix: float,
) -> tuple[Tensor, Tensor]:
    device = frame_scores.device
    counts = torch.full((num_frames,), min_per_frame, device=device, dtype=torch.long)
    remaining = anchor_budget - int(counts.sum().item())
    if num_frames == 0:
        return counts, torch.empty(0, device=device, dtype=torch.float32)

    scores = torch.nan_to_num(frame_scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if remaining <= 0:
        probs = torch.full((num_frames,), 1.0 / num_frames, device=device, dtype=torch.float32)
        return counts, probs

    capacity = torch.full((num_frames,), patch_count - min_per_frame, device=device, dtype=torch.long)
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
        chosen = int(priority.argmax().item())
        extra[chosen] += 1
        leftover -= 1
        fractional[chosen] = -float("inf")
    return counts + extra, probs


def _normalized_entropy(probs: Tensor) -> Tensor:
    if probs.numel() <= 1:
        return torch.zeros((), device=probs.device, dtype=torch.float32)
    probs = probs / probs.sum().clamp_min(1.0e-12)
    entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum()
    return torch.nan_to_num(entropy / math.log(probs.numel()), nan=0.0, posinf=0.0, neginf=0.0)


def _gini(values: Tensor) -> Tensor:
    if values.numel() <= 1:
        return torch.zeros((), device=values.device, dtype=torch.float32)
    sorted_values = values.float().sort().values
    total = sorted_values.sum()
    if float(total.item()) <= 1.0e-12:
        return torch.zeros((), device=values.device, dtype=torch.float32)
    n = sorted_values.numel()
    index = torch.arange(1, n + 1, device=values.device, dtype=torch.float32)
    gini = (2.0 * (index * sorted_values).sum() / (n * total)) - (n + 1.0) / n
    return torch.nan_to_num(gini, nan=0.0, posinf=0.0, neginf=0.0)


def _topk_ratio(values: Tensor, frac: float) -> Tensor:
    if values.numel() == 0:
        return torch.zeros((), device=values.device, dtype=torch.float32)
    k = max(1, int(math.ceil(values.numel() * frac)))
    return values.topk(k, dim=0).values.sum() / values.sum().clamp_min(1.0e-12)


def _frame_pair_gated_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    layout: TokenLayout,
    selected_by_frame: Tensor,
    anchor_counts: Tensor,
    frame_pair_graph: Tensor,
    topm_frames: int | None,
    always_include_self_frame: bool,
    gated_anchor_ratio_per_key_frame: float,
    gated_min_per_key_frame: int,
    gated_max_per_key_frame: int,
    profiler: SegmentProfiler,
) -> tuple[Tensor, Tensor, list[int], float, float]:
    batch_size, num_heads, _, head_dim = q.shape
    output = torch.empty_like(q)
    topm = _resolve_topm(num_frames=layout.num_frames, topm_frames=topm_frames)
    topm_key_frames = torch.empty(batch_size, layout.num_frames, topm, device=q.device, dtype=torch.long)
    gated_cost: list[int] = []
    gather_time_ms = 0.0
    sdpa_time_ms = 0.0
    per_key_cap = _resolve_gated_per_key_cap(
        patch_count=layout.patch_count,
        ratio=gated_anchor_ratio_per_key_frame,
        min_keep=gated_min_per_key_frame,
        max_keep=gated_max_per_key_frame,
    )

    for batch_idx in range(batch_size):
        all_anchor = selected_by_frame[batch_idx]
        all_anchor = all_anchor[all_anchor >= 0]
        lifting_kv = torch.cat([layout.all_reg_indices, all_anchor], dim=0)
        k_lift, v_lift = profiler.measure(
            "gather_time_ms",
            lambda: _gather_sample_kv(k, v, batch_idx, lifting_kv),
        )
        gather_time_ms = profiler.timings.get("gather_time_ms", 0.0)
        q_special = q[batch_idx : batch_idx + 1, :, layout.all_reg_indices]
        special_out = profiler.measure(
            "sdpa_time_ms",
            lambda: F.scaled_dot_product_attention(q_special, k_lift, v_lift),
        )
        sdpa_time_ms = profiler.timings.get("sdpa_time_ms", 0.0)
        output[batch_idx, :, layout.all_reg_indices] = special_out[0].to(dtype=output.dtype)

        sample_cost = int(layout.all_reg_indices.numel() * lifting_kv.numel())
        for query_frame in range(layout.num_frames):
            key_frames = _select_topm_key_frames(
                frame_scores=frame_pair_graph[batch_idx, query_frame],
                query_frame=query_frame,
                topm=topm,
                always_include_self_frame=always_include_self_frame,
            )
            topm_key_frames[batch_idx, query_frame] = key_frames

            frame_anchors = []
            for key_frame_tensor in key_frames:
                key_frame = int(key_frame_tensor.item())
                keep = int(anchor_counts[batch_idx, key_frame].item())
                if keep <= 0:
                    continue
                keep = min(keep, per_key_cap)
                frame_anchors.append(selected_by_frame[batch_idx, key_frame, :keep])
            if frame_anchors:
                anchor_indices = torch.cat(frame_anchors, dim=0)
                kv_indices = torch.cat([layout.all_reg_indices, anchor_indices], dim=0)
            else:
                kv_indices = layout.all_reg_indices
            k_gated, v_gated = profiler.measure(
                "gather_time_ms",
                lambda: _gather_sample_kv(k, v, batch_idx, kv_indices),
            )
            patch_indices = layout.patch_indices_by_frame[query_frame]
            q_patch = q[batch_idx : batch_idx + 1, :, patch_indices]
            patch_out = profiler.measure(
                "sdpa_time_ms",
                lambda: F.scaled_dot_product_attention(q_patch, k_gated, v_gated),
            )
            output[batch_idx, :, patch_indices] = patch_out[0].to(dtype=output.dtype)
            sample_cost += int(layout.patch_count * kv_indices.numel())
        gated_cost.append(sample_cost)

    return output, topm_key_frames, gated_cost, gather_time_ms, sdpa_time_ms


def _query_conditioned_frame_pair_gated_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    layout: TokenLayout,
    selected_by_frame: Tensor,
    anchor_counts: Tensor,
    frame_pair_graph: Tensor,
    col_mass: Tensor,
    reg_to_patch: Tensor,
    register_conditioning: Tensor,
    topm_frames: int | None,
    always_include_self_frame: bool,
    query_conditioned_eta: float,
    gated_anchor_ratio_per_key_frame: float,
    gated_min_per_key_frame: int,
    gated_max_per_key_frame: int,
    profiler: SegmentProfiler,
) -> tuple[Tensor, Tensor, list[int], float, float]:
    batch_size, num_heads, _, head_dim = q.shape
    output = torch.empty_like(q)
    topm = _resolve_topm(num_frames=layout.num_frames, topm_frames=topm_frames)
    topm_key_frames = torch.empty(batch_size, layout.num_frames, topm, device=q.device, dtype=torch.long)
    gated_cost: list[int] = []
    gather_time_ms = 0.0
    sdpa_time_ms = 0.0
    per_key_cap = _resolve_gated_per_key_cap(
        patch_count=layout.patch_count,
        ratio=gated_anchor_ratio_per_key_frame,
        min_keep=gated_min_per_key_frame,
        max_keep=gated_max_per_key_frame,
    )
    intra_norm = _normalize_per_frame(col_mass)

    for batch_idx in range(batch_size):
        all_anchor = selected_by_frame[batch_idx]
        all_anchor = all_anchor[all_anchor >= 0]
        lifting_kv = torch.cat([layout.all_reg_indices, all_anchor], dim=0)
        k_lift, v_lift = profiler.measure(
            "gather_time_ms",
            lambda: _gather_sample_kv(k, v, batch_idx, lifting_kv),
        )
        q_special = q[batch_idx : batch_idx + 1, :, layout.all_reg_indices]
        special_out = profiler.measure(
            "sdpa_time_ms",
            lambda: F.scaled_dot_product_attention(q_special, k_lift, v_lift),
        )
        output[batch_idx, :, layout.all_reg_indices] = special_out[0].to(dtype=output.dtype)

        sample_cost = int(layout.all_reg_indices.numel() * lifting_kv.numel())
        for query_frame in range(layout.num_frames):
            key_frames = _select_topm_key_frames(
                frame_scores=frame_pair_graph[batch_idx, query_frame],
                query_frame=query_frame,
                topm=topm,
                always_include_self_frame=always_include_self_frame,
            )
            topm_key_frames[batch_idx, query_frame] = key_frames

            frame_anchors = []
            for key_frame_tensor in key_frames:
                key_frame = int(key_frame_tensor.item())
                keep = int(anchor_counts[batch_idx, key_frame].item())
                if keep <= 0:
                    continue
                keep = min(keep, per_key_cap)
                if keep <= 0:
                    continue
                if float(query_conditioned_eta) == 0.0:
                    local_patch = intra_norm[batch_idx, key_frame].topk(keep, dim=-1, largest=True, sorted=True).indices
                else:
                    proxy_cond = torch.matmul(
                        register_conditioning[batch_idx, query_frame, key_frame].float().unsqueeze(0),
                        reg_to_patch[batch_idx, key_frame].float(),
                    ).squeeze(0)
                    query_score = intra_norm[batch_idx, key_frame] + float(query_conditioned_eta) * _normalize_vector(
                        proxy_cond
                    )
                    local_patch = query_score.topk(keep, dim=-1, largest=True, sorted=True).indices
                frame_anchors.append(layout.patch_indices_by_frame[key_frame, local_patch])
            if frame_anchors:
                anchor_indices = torch.cat(frame_anchors, dim=0)
                kv_indices = torch.cat([layout.all_reg_indices, anchor_indices], dim=0)
            else:
                kv_indices = layout.all_reg_indices
            k_gated, v_gated = profiler.measure(
                "gather_time_ms",
                lambda: _gather_sample_kv(k, v, batch_idx, kv_indices),
            )
            patch_indices = layout.patch_indices_by_frame[query_frame]
            q_patch = q[batch_idx : batch_idx + 1, :, patch_indices]
            patch_out = profiler.measure(
                "sdpa_time_ms",
                lambda: F.scaled_dot_product_attention(q_patch, k_gated, v_gated),
            )
            output[batch_idx, :, patch_indices] = patch_out[0].to(dtype=output.dtype)
            sample_cost += int(layout.patch_count * kv_indices.numel())
        gated_cost.append(sample_cost)

    gather_time_ms = profiler.timings.get("gather_time_ms", 0.0)
    sdpa_time_ms = profiler.timings.get("sdpa_time_ms", 0.0)
    return output, topm_key_frames, gated_cost, gather_time_ms, sdpa_time_ms


def _resolve_topm(num_frames: int, topm_frames: int | None) -> int:
    if topm_frames is None or topm_frames <= 0:
        return max(1, num_frames)
    return max(1, min(int(topm_frames), num_frames))


def _resolve_gated_per_key_cap(
    patch_count: int,
    ratio: float,
    min_keep: int,
    max_keep: int,
) -> int:
    if ratio <= 0.0:
        keep = patch_count
    else:
        keep = int(math.ceil(patch_count * ratio))
    keep = max(int(min_keep), keep)
    keep = min(int(max_keep), keep)
    return max(1, min(patch_count, keep))


def _select_topm_key_frames(
    frame_scores: Tensor,
    query_frame: int,
    topm: int,
    always_include_self_frame: bool,
) -> Tensor:
    scores = frame_scores.float().clone()
    num_frames = scores.numel()
    if topm >= num_frames:
        return torch.arange(num_frames, device=scores.device, dtype=torch.long)
    if always_include_self_frame:
        if topm == 1:
            return torch.tensor([query_frame], device=scores.device, dtype=torch.long)
        scores[query_frame] = -float("inf")
        other = scores.topk(topm - 1, dim=0, largest=True, sorted=False).indices
        return torch.cat(
            [
                torch.tensor([query_frame], device=scores.device, dtype=torch.long),
                other,
            ],
            dim=0,
        ).sort().values
    return scores.topk(topm, dim=0, largest=True, sorted=False).indices.sort().values


def _normalize_vector(values: Tensor) -> Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = values.min()
    max_value = values.max()
    denom = max_value - min_value
    if float(denom.item()) <= 1.0e-12:
        return torch.zeros_like(values)
    return (values - min_value) / denom.clamp_min(1.0e-12)


def _gather_sample_kv(k: Tensor, v: Tensor, batch_idx: int, indices: Tensor) -> tuple[Tensor, Tensor]:
    gather_index = indices.view(1, 1, -1, 1).expand(1, k.shape[1], indices.numel(), k.shape[-1])
    k_sample = k[batch_idx : batch_idx + 1].gather(dim=2, index=gather_index)
    v_sample = v[batch_idx : batch_idx + 1].gather(dim=2, index=gather_index)
    return k_sample, v_sample
