from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vggt_omega.models.layers.block import SelfAttentionBlock
from vggt_omega.models.progressive_attention import (
    ProgressiveAttentionConfig,
    ProgressiveLayerSpec,
    ProgressiveMaskState,
    _build_next_token_pair_mask,
    _dense_inherited_mask,
    nearest_parent_positions,
    nested_patch_indices,
    patch_sample_coordinates,
    progressive_attention_block,
    resolve_progressive_schedule,
    selected_original_token_indices,
)


def _block() -> SelfAttentionBlock:
    torch.manual_seed(4)
    return SelfAttentionBlock(
        dim=32,
        num_heads=4,
        ffn_ratio=2.0,
        qkv_bias=True,
        use_qk_norm=True,
    ).eval()


def _spec(
    scope,
    *,
    position=0,
    count=2,
    stage_index=0,
    stage_name="early",
) -> ProgressiveLayerSpec:
    return ProgressiveLayerSpec(
        layer_index=position,
        stage_index=stage_index,
        stage_name=stage_name,
        global_position=position,
        global_count=count,
        scope=scope,
    )


def test_nested_random_sampling_counts_coverage_nesting_and_seed() -> None:
    num_frames = 100
    patch_grid = (4, 4)
    patches_per_frame = 16
    scopes = (8, 16, 32, 64, "full")
    samples = [
        nested_patch_indices(
            num_frames=num_frames,
            patches_per_frame=patches_per_frame,
            equivalent_scope=scope,
            sampling_type="nested_random_balanced",
            random_seed=17,
        )
        for scope in scopes
    ]
    assert [sample.numel() for sample in samples] == [
        8 * 16,
        16 * 16,
        32 * 16,
        64 * 16,
        100 * 16,
    ]
    for sample in samples:
        assert torch.equal(sample, sample.sort().values)
        assert sample.unique().numel() == sample.numel()
        frame_counts = torch.bincount(
            sample.div(patches_per_frame, rounding_mode="floor"),
            minlength=num_frames,
        )
        assert (frame_counts > 0).all()
        assert int(frame_counts.max() - frame_counts.min()) <= 1
    for coarse, fine in zip(samples, samples[1:]):
        assert torch.isin(coarse, fine).all()
    repeated = nested_patch_indices(
        num_frames=num_frames,
        patches_per_frame=patches_per_frame,
        equivalent_scope=32,
        sampling_type="nested_random_balanced",
        random_seed=17,
    )
    different_seed = nested_patch_indices(
        num_frames=num_frames,
        patches_per_frame=patches_per_frame,
        equivalent_scope=32,
        sampling_type="nested_random_balanced",
        random_seed=18,
    )
    assert torch.equal(samples[2], repeated)
    assert not torch.equal(samples[2], different_seed)

    coordinates = patch_sample_coordinates(
        samples[0],
        patches_per_frame=patches_per_frame,
        patch_grid_size=patch_grid,
    )
    assert coordinates.shape == (8 * 16, 3)
    assert int(coordinates[:, 0].min()) == 0
    assert int(coordinates[:, 0].max()) == 99
    assert int(coordinates[:, 1:].min()) == 0
    assert int(coordinates[:, 1].max()) == 3
    assert int(coordinates[:, 2].max()) == 3


def test_nearest_parent_positions_are_monotonic_and_exact_on_anchors() -> None:
    parent = torch.tensor([0, 8, 16, 24, 31])
    child = torch.arange(32)
    positions = nearest_parent_positions(child, parent)
    assert torch.all(positions[1:] >= positions[:-1])
    assert torch.equal(positions[parent], torch.arange(parent.numel()))


def test_parent_expansion_uses_exact_token_pair_mask() -> None:
    coarse = torch.tensor(
        [[[True, False], [False, True]]],
        dtype=torch.bool,
    )
    parent_positions = torch.tensor([0, 0, 1, 1])
    expanded, stats = _dense_inherited_mask(
        ProgressiveMaskState(
            patch_indices=torch.tensor([0, 3]),
            pair_mask=coarse,
        ),
        parent_positions,
        patch_count=4,
        special_count=1,
        max_pair_elements=100,
    )
    expected_patch = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )
    assert torch.equal(expanded[0, :, :4], expected_patch)
    assert expanded[0, :, 4].all()
    assert stats["patch_attention_pairs_per_batch"] == 8


def test_empty_parent_row_falls_back_to_highest_anchor_region() -> None:
    state = ProgressiveMaskState(
        patch_indices=torch.tensor([0, 3]),
        pair_mask=torch.tensor(
            [[[False, False], [False, True]]],
            dtype=torch.bool,
        ),
        highest_score_key=torch.tensor([[1, 1]]),
    )
    parent_positions = torch.tensor([0, 0, 1, 1])
    expanded, stats = _dense_inherited_mask(
        state,
        parent_positions,
        patch_count=4,
        special_count=1,
        max_pair_elements=100,
    )
    assert expanded[0, :2, :4].tolist() == [
        [False, False, True, True],
        [False, False, True, True],
    ]
    assert stats["mask_expansion_fallback_parent_rows"] == 2
    assert stats["mask_expansion_fallback_highest_anchor_rows"] == 2
    assert stats["mask_expansion_fallback_special_only_rows"] == 0


def test_mask_selection_matches_token_pair_row_and_column_topk() -> None:
    torch.manual_seed(23)
    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    scale = 4**-0.5
    config = ProgressiveAttentionConfig(
        row_keep_ratio=0.25,
        column_keep_ratio=0.25,
        min_pairs_per_query=1,
        query_neighbor_radius=1,
        key_neighbor_radius=1,
        dilation_query=0,
        dilation_key=0,
        mask_query_chunk_size=2,
    )
    actual, highest_score_key, stats = _build_next_token_pair_mask(
        q,
        k,
        patch_indices=torch.arange(4),
        previous_state=None,
        parent_positions=None,
        config=config,
        scale=scale,
    )

    probabilities = (
        torch.matmul(q[:, :, :4], k.transpose(-2, -1)) * scale
    ).softmax(dim=-1).mean(dim=1)[..., :4]
    image = probabilities[:, None]
    score = (
        0.25 * probabilities
        + 0.25
        * F.avg_pool2d(
            image,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            count_include_pad=False,
        )[:, 0]
        + 0.25
        * F.avg_pool2d(
            image,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            count_include_pad=False,
        )[:, 0]
        + 0.25
        * F.avg_pool2d(
            image,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            count_include_pad=False,
        )[:, 0]
    )
    expected = torch.zeros_like(actual)
    expected.scatter_(-1, score.topk(1, dim=-1).indices, True)
    column = torch.zeros(1, 4, 4, dtype=torch.bool)
    column.scatter_(
        -1,
        score.transpose(-2, -1).topk(1, dim=-1).indices,
        True,
    )
    expected |= column.transpose(-2, -1)
    assert torch.equal(actual, expected)
    assert torch.equal(highest_score_key, score.argmax(dim=-1))
    assert stats["mask_selection_granularity"] == "exact_patch_token_pair"


def test_selected_layout_keeps_all_special_tokens_in_frame_order() -> None:
    patch_indices = torch.tensor([0, 3, 4, 11])
    selected = selected_original_token_indices(
        patch_indices,
        num_frames=3,
        tokens_per_frame=6,
        num_special_tokens=2,
    )
    assert selected.tolist() == [2, 5, 8, 17, 0, 1, 6, 7, 12, 13]


def test_random_sampling_reuses_within_stage_and_resamples_across_stages() -> None:
    block = _block()
    x = torch.randn(1, 60, 32)
    config = ProgressiveAttentionConfig(
        enabled=True,
        scope_schedule=(2,),
        require_stage_final_full=False,
        final_scope_mode="sampled",
        mask_enabled=False,
        sampling_random_seed=5,
    )

    def run(spec: ProgressiveLayerSpec):
        return progressive_attention_block(
            block,
            x.clone(),
            num_frames=10,
            tokens_per_frame=6,
            num_special_tokens=2,
            patch_grid_size=(2, 2),
            layer_spec=spec,
            config=config,
            previous_state=None,
            build_next_mask=False,
        )

    early_first = run(_spec(2, position=0))
    early_repeat = run(_spec(2, position=1))
    middle = run(
        _spec(
            2,
            position=0,
            stage_index=1,
            stage_name="middle",
        )
    )
    assert torch.equal(
        early_first.sample_coordinates,
        early_repeat.sample_coordinates,
    )
    assert not torch.equal(
        early_first.sample_coordinates,
        middle.sample_coordinates,
    )
    assert early_first.stats["sampling_effective_seed"] == 5
    assert middle.stats["sampling_effective_seed"] == 1_000_008


def test_stage_scope_schedule_matches_global_layer_slots() -> None:
    kinds = ["global"] * 24
    for layer in (2, 6, 9, 14, 20):
        kinds[layer] = "register"
    config = ProgressiveAttentionConfig(enabled=True)
    schedule = resolve_progressive_schedule(
        depth=24,
        inter_frame_attention_types=kinds,
        config=config,
    )
    assert sorted(schedule) == [
        0,
        1,
        3,
        4,
        5,
        7,
        8,
        10,
        11,
        12,
        13,
        15,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
    ]
    assert [schedule[layer].scope for layer in (0, 1, 3, 4, 5, 7, 8)] == [
        32,
        32,
        32,
        64,
        64,
        64,
        "full",
    ]
    assert [schedule[layer].scope for layer in (10, 11, 12, 13, 15, 16)] == [
        32,
        32,
        32,
        64,
        64,
        "full",
    ]
    assert schedule[8].is_stage_last
    assert schedule[16].is_stage_last
    assert schedule[23].is_stage_last


def test_full_sampled_scope_matches_original_block() -> None:
    block = _block()
    x = torch.randn(1, 18, 32)
    expected = block(x.clone(), None)
    config = ProgressiveAttentionConfig(
        enabled=True,
        scope_schedule=("full",),
        mask_enabled=False,
    )
    actual = progressive_attention_block(
        block,
        x.clone(),
        num_frames=3,
        tokens_per_frame=6,
        num_special_tokens=2,
        patch_grid_size=(2, 2),
        layer_spec=_spec("full", count=1),
        config=config,
        previous_state=None,
        build_next_mask=False,
    )
    torch.testing.assert_close(actual.output, expected, rtol=2e-5, atol=2e-5)
    assert actual.stats["qkv_projection_tokens"] == 18
    assert actual.output.shape == x.shape


def test_full_scope_all_one_pair_mask_matches_original_block() -> None:
    block = _block()
    x = torch.randn(1, 18, 32)
    expected = block(x.clone(), None)
    patch_indices = torch.arange(12)
    state = ProgressiveMaskState(
        patch_indices=patch_indices,
        pair_mask=torch.ones(1, 12, 12, dtype=torch.bool),
    )
    config = ProgressiveAttentionConfig(
        enabled=True,
        scope_schedule=(2, "full"),
    )
    actual = progressive_attention_block(
        block,
        x.clone(),
        num_frames=3,
        tokens_per_frame=6,
        num_special_tokens=2,
        patch_grid_size=(2, 2),
        layer_spec=_spec("full", position=1, count=2),
        config=config,
        previous_state=state,
        build_next_mask=False,
    )
    torch.testing.assert_close(actual.output, expected, rtol=2e-5, atol=2e-5)


def test_sampled_attention_scatter_preserves_unselected_attention_residual() -> None:
    block = _block()
    for parameter in block.mlp.parameters():
        parameter.data.zero_()
    x = torch.randn(1, 36, 32)
    config = ProgressiveAttentionConfig(
        enabled=True,
        scope_schedule=(2,),
        require_stage_final_full=False,
        final_scope_mode="sampled",
        mask_enabled=False,
        sampling_random_seed=11,
    )
    result = progressive_attention_block(
        block,
        x.clone(),
        num_frames=6,
        tokens_per_frame=6,
        num_special_tokens=2,
        patch_grid_size=(2, 2),
        layer_spec=_spec(2, count=1),
        config=config,
        previous_state=None,
        build_next_mask=False,
    )
    sampled_patch = nested_patch_indices(
        num_frames=6,
        patches_per_frame=4,
        equivalent_scope=2,
        sampling_type=config.sampling_type,
        random_seed=config.sampling_random_seed,
    )
    sampled_original = (
        sampled_patch.div(4, rounding_mode="floor") * 6
        + 2
        + sampled_patch.remainder(4)
    )
    all_patch = torch.cat(
        [torch.arange(frame * 6 + 2, frame * 6 + 6) for frame in range(6)]
    )
    unselected = all_patch[~torch.isin(all_patch, sampled_original)]
    torch.testing.assert_close(result.output[:, unselected], x[:, unselected])
    assert result.output.shape == x.shape
    assert result.stats["sampled_patch_tokens"] == 8
    assert result.stats["qkv_projection_tokens"] == 20


def test_mask_inheritance_has_no_empty_rows_or_nonfinite_output() -> None:
    block = _block()
    x = torch.randn(1, 60, 32)
    config = ProgressiveAttentionConfig(
        enabled=True,
        scope_schedule=(2, 4, "full"),
        row_keep_ratio=0.5,
        column_keep_ratio=0.25,
        dilation_query=0,
        dilation_key=0,
    )
    first = progressive_attention_block(
        block,
        x.clone(),
        num_frames=10,
        tokens_per_frame=6,
        num_special_tokens=2,
        patch_grid_size=(2, 2),
        layer_spec=_spec(2, position=0, count=3),
        config=config,
        previous_state=None,
        build_next_mask=True,
    )
    assert first.next_state is not None
    assert first.next_state.pair_mask.shape == (1, 8, 8)
    assert first.next_state.pair_mask.any(dim=-1).all()
    second = progressive_attention_block(
        block,
        first.output,
        num_frames=10,
        tokens_per_frame=6,
        num_special_tokens=2,
        patch_grid_size=(2, 2),
        layer_spec=_spec(4, position=1, count=3),
        config=config,
        previous_state=first.next_state,
        build_next_mask=True,
    )
    assert torch.isfinite(second.output).all()
    assert second.output.shape == x.shape
    assert second.next_state is not None
    assert second.next_state.pair_mask.shape == (1, 16, 16)
    assert second.next_state.pair_mask.any(dim=-1).all()
    assert second.stats["mask_selection_granularity"] == (
        "exact_patch_token_pair"
    )
    assert second.stats["mask_fallback_special_rows"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_inherited_mask_uses_exact_token_pair_reference() -> None:
    block = _block().cuda()
    x = torch.randn(1, 660, 32, device="cuda")
    config = ProgressiveAttentionConfig(
        enabled=True,
        scope_schedule=(2, 4, "full"),
        row_keep_ratio=0.5,
        column_keep_ratio=0.25,
        dilation_query=0,
        dilation_key=0,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = progressive_attention_block(
            block,
            x,
            num_frames=10,
            tokens_per_frame=66,
            num_special_tokens=2,
            patch_grid_size=(8, 8),
            layer_spec=_spec(2, position=0, count=3),
            config=config,
            previous_state=None,
            build_next_mask=True,
        )
        second = progressive_attention_block(
            block,
            first.output,
            num_frames=10,
            tokens_per_frame=66,
            num_special_tokens=2,
            patch_grid_size=(8, 8),
            layer_spec=_spec(4, position=1, count=3),
            config=config,
            previous_state=first.next_state,
            build_next_mask=True,
        )
    assert torch.isfinite(second.output).all()
    assert second.stats["attention_backend"] == (
        "sdpa_dense_token_pair_reference_mask"
    )
    assert second.stats["efficient_sparse_kernel"] is False


def test_reference_pair_budget_fails_instead_of_block_fallback() -> None:
    block = _block()
    x = torch.randn(1, 60, 32)
    config = ProgressiveAttentionConfig(
        enabled=True,
        scope_schedule=(2, 4, "full"),
        max_reference_pair_elements=63,
    )
    with pytest.raises(
        RuntimeError,
        match="will not silently replace token-pair routing",
    ):
        progressive_attention_block(
            block,
            x,
            num_frames=10,
            tokens_per_frame=6,
            num_special_tokens=2,
            patch_grid_size=(2, 2),
            layer_spec=_spec(2, position=0, count=3),
            config=config,
            previous_state=None,
            build_next_mask=True,
        )
