import pytest
import torch

from vggt_omega.models.aggregator import slice_expand_and_flatten
from vggt_omega.utils.reference_frame import (
    parse_frame_index_spec,
    reference_first_order,
    reorder_reference_first,
    resolve_first_frame_token_indices,
    resolve_reference_frame_index,
)


def test_reference_first_order_keeps_non_reference_relative_order():
    assert reference_first_order(5, 2) == [2, 0, 1, 3, 4]


def test_reorder_reference_first_preserves_frame_set():
    frames = ["f0", "f1", "f2", "f3"]

    reordered = reorder_reference_first(frames, 2)

    assert reordered == ["f2", "f0", "f1", "f3"]
    assert sorted(reordered) == sorted(frames)


def test_reference_frame_negative_index_selects_last_frame():
    assert resolve_reference_frame_index(-1, 4) == 3
    assert reference_first_order(4, -1) == [3, 0, 1, 2]


def test_reference_frame_out_of_range_raises():
    with pytest.raises(ValueError, match="reference_frame_index"):
        resolve_reference_frame_index(4, 4)


def test_uniform_frame_index_spec():
    assert parse_frame_index_spec("uniform:4", 100) == (0, 25, 50, 75)
    assert parse_frame_index_spec("all", 4) == (0, 1, 2, 3)


def test_first_frame_token_indices_must_include_zero():
    with pytest.raises(ValueError, match="include input position 0"):
        resolve_first_frame_token_indices("1,2", 4)


def test_slice_expand_and_flatten_can_assign_multiple_first_tokens():
    token_tensor = torch.tensor([[[[10.0]], [[20.0]]]])

    tokens = slice_expand_and_flatten(
        token_tensor,
        batch_size=1,
        num_frames=5,
        first_frame_token_indices=(0, 2, 4),
    )

    assert tokens[:, 0, 0].tolist() == [10.0, 20.0, 10.0, 20.0, 10.0]
