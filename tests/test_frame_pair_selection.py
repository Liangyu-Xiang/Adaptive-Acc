import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "vggt_omega" / "utils" / "frame_pair_selection.py"
SPEC = importlib.util.spec_from_file_location("frame_pair_selection", MODULE_PATH)
assert SPEC is not None
frame_pair_selection = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = frame_pair_selection
SPEC.loader.exec_module(frame_pair_selection)

RankedFramePair = frame_pair_selection.RankedFramePair
representative_frame_pairs = frame_pair_selection.representative_frame_pairs
select_top_percent_disjoint_frame_pairs = frame_pair_selection.select_top_percent_disjoint_frame_pairs


def symmetric_similarity(num_frames: int, values: dict[tuple[int, int], float]) -> np.ndarray:
    matrix = np.zeros((num_frames, num_frames), dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    for (frame_a, frame_b), value in values.items():
        matrix[frame_a, frame_b] = value
        matrix[frame_b, frame_a] = value
    return matrix


def test_top_percent_disjoint_pairs_filter_duplicates_then_overlaps():
    similarity = symmetric_similarity(
        5,
        {
            (0, 1): 0.99,
            (1, 2): 0.98,
            (2, 3): 0.97,
            (3, 4): 0.96,
            (0, 4): 0.95,
            (0, 2): 0.30,
            (0, 3): 0.20,
            (1, 3): 0.10,
            (1, 4): 0.05,
            (2, 4): 0.01,
        },
    )

    pairs, candidate_count, top_candidate_count = select_top_percent_disjoint_frame_pairs(
        similarity,
        top_percent=50.0,
    )

    assert candidate_count == 10
    assert top_candidate_count == 5
    assert [(pair.frame_a, pair.frame_b) for pair in pairs] == [(0, 1), (2, 3)]
    assert [pair.rank for pair in pairs] == [1, 3]
    assert [round(pair.percentile, 1) for pair in pairs] == [10.0, 30.0]


def test_representative_frame_pairs_pick_requested_percentile_buckets_once():
    pairs = [
        RankedFramePair(
            frame_a=rank,
            frame_b=rank + 100,
            similarity=1.0 - rank * 0.001,
            rank=rank,
            percentile=float(rank),
        )
        for rank in range(1, 101)
    ]

    representatives = representative_frame_pairs(
        pairs,
        buckets=(1.0, 10.0, 25.0, 50.0),
        per_bucket=1,
        max_pairs=4,
    )

    assert [pair.rank for pair in representatives] == [1, 10, 25, 50]
    assert [pair.representative_bucket for pair in representatives] == [
        "top_1pct",
        "top_10pct",
        "top_25pct",
        "top_50pct",
    ]
