from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np


T = TypeVar("T")
SAMPLING_STRATEGIES = ("random", "uniform")


def uniform_first_last_indices(pool_size: int, num_frames: int) -> list[int]:
    """Select evenly spaced frame indices while preserving the first and last frame."""
    _validate_pool_size(pool_size, num_frames)
    indices = np.rint(np.linspace(0, pool_size - 1, num_frames)).astype(np.int64).tolist()
    indices[0] = 0
    indices[-1] = pool_size - 1
    if len(set(indices)) != num_frames:
        raise RuntimeError(
            "Uniform first/last sampling produced duplicate indices; "
            f"pool_size={pool_size}, num_frames={num_frames}"
        )
    return indices


def sample_record_pools(
    pools: Mapping[str, Sequence[T]],
    num_frames: int,
    seed: int,
    strategy: str = "uniform",
) -> tuple[dict[str, list[T]], dict[str, list[int]]]:
    """Sample records from each pool using the requested deterministic strategy."""
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(
            f"sampling strategy must be one of {SAMPLING_STRATEGIES}, got {strategy!r}"
        )
    rng = np.random.RandomState(seed) if strategy == "random" else None
    sampled: dict[str, list[T]] = {}
    sampled_indices: dict[str, list[int]] = {}
    for sequence_name, pool in pools.items():
        _validate_pool_size(len(pool), num_frames, sequence_name=sequence_name)
        if strategy == "random":
            assert rng is not None
            indices = rng.choice(len(pool), num_frames, replace=False).tolist()
        else:
            indices = uniform_first_last_indices(len(pool), num_frames)
        sampled_indices[sequence_name] = indices
        sampled[sequence_name] = [pool[index] for index in indices]
    return sampled, sampled_indices


def _validate_pool_size(
    pool_size: int,
    num_frames: int,
    *,
    sequence_name: str | None = None,
) -> None:
    prefix = f"{sequence_name}: " if sequence_name else ""
    if num_frames < 2:
        raise ValueError(f"{prefix}num_frames must be at least 2, got {num_frames}")
    if pool_size < num_frames:
        raise ValueError(f"{prefix}only {pool_size} frames, need {num_frames}")
