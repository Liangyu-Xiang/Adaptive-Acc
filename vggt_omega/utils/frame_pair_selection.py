from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RankedFramePair:
    frame_a: int
    frame_b: int
    similarity: float
    rank: int
    percentile: float
    representative_bucket: str = ""


def ranked_undirected_frame_pairs(
    similarity: np.ndarray,
    *,
    exclude_frames: Iterable[int] = (),
) -> list[RankedFramePair]:
    """Return all finite off-diagonal frame pairs, sorted by descending similarity."""

    matrix = np.asarray(similarity, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"similarity must be a square matrix, got shape {matrix.shape}")
    num_frames = int(matrix.shape[0])
    excluded = {int(frame) for frame in exclude_frames}
    invalid = sorted(frame for frame in excluded if frame < 0 or frame >= num_frames)
    if invalid:
        raise ValueError(f"exclude_frames contains out-of-range indices: {invalid}")

    candidates: list[tuple[int, int, float]] = []
    for frame_a, frame_b in zip(*np.triu_indices(num_frames, k=1)):
        if int(frame_a) in excluded or int(frame_b) in excluded:
            continue
        value = float(matrix[frame_a, frame_b])
        if math.isfinite(value):
            candidates.append((int(frame_a), int(frame_b), value))
    candidates.sort(key=lambda item: item[2], reverse=True)

    total = len(candidates)
    if total == 0:
        return []
    return [
        RankedFramePair(
            frame_a=frame_a,
            frame_b=frame_b,
            similarity=value,
            rank=index + 1,
            percentile=100.0 * float(index + 1) / float(total),
        )
        for index, (frame_a, frame_b, value) in enumerate(candidates)
    ]


def select_top_percent_disjoint_frame_pairs(
    similarity: np.ndarray,
    *,
    top_percent: float,
    exclude_frames: Iterable[int] = (),
    max_pairs: int | None = None,
) -> tuple[list[RankedFramePair], int, int]:
    """Select top-percent frame pairs, then remove redundant overlapping pairs."""

    top_percent = float(top_percent)
    if not 0.0 < top_percent <= 100.0:
        raise ValueError(f"top_percent must be in (0, 100], got {top_percent}")
    if max_pairs is not None:
        max_pairs = max(0, int(max_pairs))

    candidates = ranked_undirected_frame_pairs(
        similarity,
        exclude_frames=exclude_frames,
    )
    candidate_count = len(candidates)
    if candidate_count == 0:
        return [], 0, 0

    top_candidate_count = max(
        1,
        min(candidate_count, int(math.ceil(candidate_count * top_percent / 100.0))),
    )
    selected: list[RankedFramePair] = []
    used_frames: set[int] = set()
    for pair in candidates[:top_candidate_count]:
        if pair.frame_a in used_frames or pair.frame_b in used_frames:
            continue
        selected.append(pair)
        used_frames.add(pair.frame_a)
        used_frames.add(pair.frame_b)
        if max_pairs is not None and len(selected) >= max_pairs:
            break
    return selected, candidate_count, top_candidate_count


def representative_frame_pairs(
    pairs: Sequence[RankedFramePair],
    *,
    buckets: Sequence[float] = (1.0, 10.0, 25.0, 50.0),
    per_bucket: int = 2,
    max_pairs: int = 8,
) -> list[RankedFramePair]:
    """Choose a small set nearest to requested top-similarity percentile buckets."""

    if not pairs:
        return []
    per_bucket = max(1, int(per_bucket))
    max_pairs = max(0, int(max_pairs))
    if max_pairs == 0:
        return []

    selected: list[RankedFramePair] = []
    selected_indices: set[int] = set()
    for bucket in buckets:
        bucket_value = float(bucket)
        if not math.isfinite(bucket_value):
            raise ValueError(f"representative bucket must be finite, got {bucket!r}")
        ordered = sorted(
            enumerate(pairs),
            key=lambda item: (abs(item[1].percentile - bucket_value), item[1].rank),
        )
        picked_for_bucket = 0
        for pair_index, pair in ordered:
            if pair_index in selected_indices:
                continue
            selected_indices.add(pair_index)
            selected.append(
                replace(
                    pair,
                    representative_bucket=f"top_{bucket_value:g}pct",
                )
            )
            picked_for_bucket += 1
            if picked_for_bucket >= per_bucket or len(selected) >= max_pairs:
                break
        if len(selected) >= max_pairs:
            break
    return selected
