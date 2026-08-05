import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "vggt_omega" / "utils" / "frame_sampling.py"
SPEC = importlib.util.spec_from_file_location("frame_sampling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
frame_sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frame_sampling)

sample_record_pools = frame_sampling.sample_record_pools
uniform_first_last_indices = frame_sampling.uniform_first_last_indices


def test_uniform_first_last_indices_preserve_endpoints() -> None:
    indices = uniform_first_last_indices(pool_size=20, num_frames=6)

    assert indices == [0, 4, 8, 11, 15, 19]
    assert indices[0] == 0
    assert indices[-1] == 19
    assert len(set(indices)) == 6


def test_uniform_first_last_indices_support_500_frame_inputs() -> None:
    indices = uniform_first_last_indices(pool_size=1200, num_frames=500)

    assert indices[0] == 0
    assert indices[-1] == 1199
    assert len(indices) == 500
    assert len(set(indices)) == 500
    assert indices == sorted(indices)


def test_uniform_sampling_selects_records_by_even_pool_indices() -> None:
    sampled, sampled_indices = sample_record_pools(
        {"seq": list("abcdefghij")},
        num_frames=4,
        seed=123,
        strategy="uniform",
    )

    assert sampled_indices["seq"] == [0, 3, 6, 9]
    assert sampled["seq"] == ["a", "d", "g", "j"]


def test_default_sampling_strategy_is_uniform() -> None:
    sampled, sampled_indices = sample_record_pools(
        {"seq": list("abcdefghij")},
        num_frames=4,
        seed=123,
    )

    assert sampled_indices["seq"] == [0, 3, 6, 9]
    assert sampled["seq"] == ["a", "d", "g", "j"]


def test_random_sampling_keeps_legacy_randomstate_order() -> None:
    sampled, sampled_indices = sample_record_pools(
        {"a": list(range(10)), "b": list(range(10, 20))},
        num_frames=4,
        seed=42,
        strategy="random",
    )
    rng = np.random.RandomState(42)
    expected_a = rng.choice(10, 4, replace=False).tolist()
    expected_b = rng.choice(10, 4, replace=False).tolist()

    assert sampled_indices == {"a": expected_a, "b": expected_b}
    assert sampled["a"] == expected_a
    assert sampled["b"] == [10 + index for index in expected_b]


def test_sampling_rejects_too_small_pool() -> None:
    with pytest.raises(ValueError, match="only 3 frames, need 4"):
        uniform_first_last_indices(pool_size=3, num_frames=4)
