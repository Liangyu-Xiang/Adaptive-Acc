from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import vggt_omega.models.adaptive_pair_scope_attention as adaptive_module
from vggt_omega.models.adaptive_pair_scope_attention import (
    AdaptivePairScopeConfig,
    AxisScope,
    LeafPairScope,
    _batched_gather_tokens,
    adaptive_pair_scope_attention_block,
    build_axis_scopes_from_anchors,
    build_coarse_pair_scopes,
    build_patch_mask_rows,
    compile_leaf_patch_mask,
    gather_compiled_patch_mask_rows,
    materialize_full_patch_mask,
    original_patch_token_indices,
    refine_active_pair_scopes,
    select_active_pair_scopes,
    uniform_flat_anchor_indices,
    validate_axis_scope_partition,
    validate_scope_partition,
)
from vggt_omega.models.layers.block import SelfAttentionBlock
from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.progressive_attention import (
    progressive_config_from_dict,
)


def _block() -> SelfAttentionBlock:
    torch.manual_seed(41)
    return SelfAttentionBlock(
        dim=32,
        num_heads=4,
        ffn_ratio=2.0,
        qkv_bias=True,
        use_qk_norm=True,
    ).eval()


def _config(**overrides) -> AdaptivePairScopeConfig:
    values = {
        "enabled_layers": (0,),
        "coarse_num_anchors": 2,
        "coarse_stride": None,
        "coarse_keep_ratio": 0.5,
        "fine_keep_ratio": 0.5,
        "refine_factor": 2,
        "save_pair_scopes": True,
        "profile_components": False,
        "query_chunk_size": 64,
    }
    values.update(overrides)
    return AdaptivePairScopeConfig(**values)


def test_axis_scopes_are_disjoint_and_cover_complete_axis() -> None:
    anchors = uniform_flat_anchor_indices(10, num_anchors=4)
    scopes = build_axis_scopes_from_anchors(
        anchors,
        token_count=10,
    )
    assert anchors.tolist() == [1, 3, 6, 8]
    assert validate_axis_scope_partition(scopes, token_count=10)
    ownership = torch.zeros(10, dtype=torch.int64)
    for scope in scopes:
        ownership[scope.start : scope.end] += 1
        assert scope.start <= scope.anchor < scope.end
    assert torch.equal(ownership, torch.ones_like(ownership))


def test_coarse_pair_scopes_partition_complete_qk_matrix() -> None:
    anchors = uniform_flat_anchor_indices(10, num_anchors=4)
    axis_scopes = build_axis_scopes_from_anchors(
        anchors,
        token_count=10,
    )
    pair_scopes = build_coarse_pair_scopes(axis_scopes)
    assert len(pair_scopes) == 16
    assert validate_scope_partition(pair_scopes, token_count=10)
    ownership = torch.zeros(10, 10, dtype=torch.int64)
    for scope in pair_scopes:
        ownership[
            scope.q_scope.start : scope.q_scope.end,
            scope.k_scope.start : scope.k_scope.end,
        ] += 1
    assert torch.equal(ownership, torch.ones_like(ownership))


def test_only_active_parent_generates_nonoverlapping_children() -> None:
    torch.manual_seed(2)
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    axis_scopes = build_axis_scopes_from_anchors(
        torch.tensor([1, 5]),
        token_count=8,
    )
    active = torch.zeros(1, 2, 2, dtype=torch.bool)
    active[0, 0, 1] = True
    fine, leaves, probes, _ = refine_active_pair_scopes(
        q,
        k,
        coarse_axis_scopes=axis_scopes,
        coarse_refine=active,
        config=_config(),
        scale=0.5,
    )
    active_children = [
        scope for scope in fine[0] if scope.parent_id == 1
    ]
    assert len(active_children) == 4
    assert not any(scope.parent_id != 1 for scope in fine[0])
    assert probes == (4,)
    parent = (
        axis_scopes[0].start,
        axis_scopes[0].end,
        axis_scopes[1].start,
        axis_scopes[1].end,
    )
    child_area = sum(
        (scope.q_scope.end - scope.q_scope.start)
        * (scope.k_scope.end - scope.k_scope.start)
        for scope in active_children
    )
    parent_area = (parent[1] - parent[0]) * (parent[3] - parent[2])
    assert child_area == parent_area
    assert validate_scope_partition(leaves[0], token_count=8)


def test_fine_probe_count_is_sum_of_local_parent_products() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    axis_scopes = build_axis_scopes_from_anchors(
        torch.tensor([1, 5]),
        token_count=8,
    )
    active = torch.zeros(1, 2, 2, dtype=torch.bool)
    active[0, 0, 0] = True
    active[0, 1, 1] = True
    fine, _, probes, _ = refine_active_pair_scopes(
        q,
        k,
        coarse_axis_scopes=axis_scopes,
        coarse_refine=active,
        config=_config(),
        scale=0.5,
    )
    assert len(fine[0]) == 8
    assert probes == (8,)
    # Pooling all four local Q anchors and all four local K anchors would
    # incorrectly evaluate 16 pairs.  The implementation evaluates 4 + 4.
    assert probes[0] < 16


def test_active_parent_gather_does_not_replicate_full_token_axes() -> None:
    tensor = torch.arange(2 * 3 * 11 * 4).reshape(2, 3, 11, 4)
    batches = torch.tensor([0, 1, 0, 1, 1])
    indices = torch.tensor(
        [[1, 7], [2, 8], [3, 9], [4, 10], [0, 6]]
    )
    gathered = _batched_gather_tokens(
        tensor,
        batch_indices=batches,
        token_indices=indices,
    )
    assert gathered.shape == (5, 3, 2, 4)
    for row in range(batches.numel()):
        expected = tensor[
            batches[row],
            :,
            indices[row],
            :,
        ].transpose(0, 1)
        torch.testing.assert_close(
            gathered[row].transpose(0, 1),
            expected,
        )


def test_materialized_final_mask_matches_hand_constructed_leaves() -> None:
    leaves = (
        (
            LeafPairScope(0, 2, 0, 2, False, 0, 0),
            LeafPairScope(2, 4, 0, 2, False, 0, 2),
            LeafPairScope(2, 4, 2, 4, False, 0, 3),
            LeafPairScope(0, 1, 2, 3, True, 1, 1),
            LeafPairScope(0, 1, 3, 4, False, 1, 1),
            LeafPairScope(1, 2, 2, 3, False, 1, 1),
            LeafPairScope(1, 2, 3, 4, True, 1, 1),
        ),
    )
    assert validate_scope_partition(leaves[0], token_count=4)
    actual = materialize_full_patch_mask(leaves, token_count=4)
    expected = torch.zeros(1, 4, 4, dtype=torch.bool)
    expected[0, 0, 2] = True
    expected[0, 1, 3] = True
    assert torch.equal(actual, expected)


def test_compiled_patch_mask_rows_match_direct_leaf_expansion() -> None:
    leaves = (
        (
            LeafPairScope(0, 2, 0, 3, True, 1, 0),
            LeafPairScope(0, 2, 3, 6, False, 1, 0),
            LeafPairScope(2, 4, 0, 2, False, 1, 1),
            LeafPairScope(2, 4, 2, 6, True, 1, 1),
            LeafPairScope(4, 6, 0, 6, True, 0, 2),
        ),
        (
            LeafPairScope(0, 3, 0, 1, True, 1, 0),
            LeafPairScope(0, 3, 1, 6, False, 1, 0),
            LeafPairScope(3, 6, 0, 4, True, 1, 1),
            LeafPairScope(3, 6, 4, 6, False, 1, 1),
        ),
    )
    rows = torch.tensor([5, 0, 3, 2, 4, 1])
    direct = build_patch_mask_rows(rows, leaves, token_count=6)
    compiled = compile_leaf_patch_mask(leaves, token_count=6)
    gathered = gather_compiled_patch_mask_rows(rows, compiled)
    assert torch.equal(gathered, direct)


def test_selection_falls_back_to_best_key_for_every_query_scope() -> None:
    scores = torch.tensor([[[10.0, 9.0], [0.0, -1.0]]])
    selected = select_active_pair_scopes(
        scores,
        selection_mode="global_quantile",
        keep_ratio=0.25,
        min_active_key_scopes_per_query_scope=1,
    )
    assert selected.any(dim=-1).all()
    assert selected[0, 1].tolist() == [True, False]


def test_all_one_final_mask_matches_original_dense_block() -> None:
    block = _block()
    x = torch.randn(1, 12, 32)
    expected = block(x.clone(), None)
    result = adaptive_pair_scope_attention_block(
        block,
        x.clone(),
        num_frames=2,
        tokens_per_frame=6,
        num_special_tokens=2,
        patch_grid_size=(2, 2),
        layer_index=0,
        config=_config(
            coarse_keep_ratio=1.0,
            fine_keep_ratio=1.0,
        ),
    )
    torch.testing.assert_close(
        result.output,
        expected,
        rtol=2e-5,
        atol=2e-5,
    )
    assert result.stats["final_patch_pair_density"] == 1.0


def test_special_token_queries_and_keys_are_always_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _block()
    x = torch.randn(1, 6, 32)
    captured_masks: list[torch.Tensor] = []
    original_sdpa = adaptive_module.F.scaled_dot_product_attention

    def capture_sdpa(q, k, v, **kwargs):
        mask = kwargs.get("attn_mask")
        assert mask is not None
        captured_masks.append(mask.detach().cpu())
        return original_sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(
        adaptive_module.F,
        "scaled_dot_product_attention",
        capture_sdpa,
    )
    adaptive_pair_scope_attention_block(
        block,
        x,
        num_frames=1,
        tokens_per_frame=6,
        num_special_tokens=2,
        patch_grid_size=(2, 2),
        layer_index=0,
        config=_config(query_chunk_size=6),
    )
    assert len(captured_masks) == 1
    mask = captured_masks[0][0, 0]
    assert mask[:2].all()
    assert mask[2:, :2].all()


def test_every_patch_query_receives_an_attention_update() -> None:
    block = _block()
    for parameter in block.mlp.parameters():
        parameter.data.zero_()
    x = torch.randn(1, 12, 32)
    result = adaptive_pair_scope_attention_block(
        block,
        x.clone(),
        num_frames=2,
        tokens_per_frame=6,
        num_special_tokens=2,
        patch_grid_size=(2, 2),
        layer_index=0,
        config=_config(),
    )
    patch_indices = original_patch_token_indices(
        num_frames=2,
        tokens_per_frame=6,
        num_special_tokens=2,
    )
    updates = (result.output - x).abs().sum(dim=-1)
    assert (updates[:, patch_indices] > 0).all()
    assert result.stats["all_patch_queries_receive_attention_output"]
    assert result.stats["sampled_token_scatter"] is False


def test_adaptive_block_projects_qkv_exactly_once() -> None:
    block = _block()
    x = torch.randn(1, 12, 32)
    calls = 0

    def count_qkv(module, args, output):
        nonlocal calls
        calls += 1

    handle = block.attn.qkv.register_forward_hook(count_qkv)
    try:
        result = adaptive_pair_scope_attention_block(
            block,
            x,
            num_frames=2,
            tokens_per_frame=6,
            num_special_tokens=2,
            patch_grid_size=(2, 2),
            layer_index=0,
            config=_config(),
        )
    finally:
        handle.remove()
    assert calls == 1
    assert result.stats["qkv_projection_count"] == 1


def test_one_hundred_frame_flat_indices_and_scopes_are_in_range() -> None:
    frame_count = 100
    patches_per_frame = 4
    patch_count = frame_count * patches_per_frame
    anchors = uniform_flat_anchor_indices(
        patch_count,
        num_anchors=37,
    )
    scopes = build_axis_scopes_from_anchors(
        anchors,
        token_count=patch_count,
    )
    original = original_patch_token_indices(
        num_frames=frame_count,
        tokens_per_frame=patches_per_frame + 2,
        num_special_tokens=2,
    )
    assert anchors.numel() == 37
    assert int(anchors.min()) >= 0
    assert int(anchors.max()) < patch_count
    assert validate_axis_scope_partition(scopes, token_count=patch_count)
    assert original[0].item() == 2
    assert original[4].item() == 8
    assert original[-1].item() == frame_count * 6 - 1


def test_adaptive_config_and_legacy_config_are_both_supported() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "progressive_attention"
        / "adaptive_pair_scope_reference.json"
    )
    adaptive = progressive_config_from_dict(
        json.loads(config_path.read_text())
    )
    assert adaptive.algorithm == "adaptive_pair_scope"
    assert adaptive.adaptive_pair_scope_config is not None
    assert adaptive.adaptive_pair_scope_config.enabled_layers == (
        10,
        11,
        12,
        13,
        15,
        16,
    )
    payload = json.loads(config_path.read_text())
    payload["profile_components"] = False
    timing_config = progressive_config_from_dict(payload)
    assert timing_config.adaptive_pair_scope_config is not None
    assert timing_config.adaptive_pair_scope_config.profile_components is False
    legacy = progressive_config_from_dict(
        {
            "enabled": True,
            "scope_schedule": [32, 64, "full"],
            "sampling": {
                "type": "nested_random_balanced",
                "random_seed": 0,
            },
        }
    )
    assert legacy.algorithm == "legacy_token_scope"
    assert legacy.adaptive_pair_scope_config is None


def test_coarse_anchor_count_and_stride_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        uniform_flat_anchor_indices(
            10,
            num_anchors=3,
            stride=2,
        )
    with pytest.raises(ValueError, match="exactly one"):
        progressive_config_from_dict(
            {
                "enabled": True,
                "algorithm": "adaptive_pair_scope",
                "enabled_layers": [0],
                "coarse_sampling": {
                    "num_anchors": 3,
                    "stride": 2,
                },
            }
        )


def test_aggregator_dispatches_adaptive_mode_without_cross_layer_state() -> None:
    model = Aggregator(
        patch_size=16,
        embed_dim=64,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        num_register_tokens=2,
        register_attention_block_indices=[],
        cached_layer_indices=(0,),
        global_merging=False,
        merging=None,
        merge_ratio=0.0,
        progressive_attention={
            "enabled": True,
            "algorithm": "adaptive_pair_scope",
            "enabled_layers": [0],
            "coarse_sampling": {
                "type": "uniform_flat",
                "num_anchors": 2,
            },
            "routing": {
                "coarse_keep_ratio": 0.5,
                "fine_keep_ratio": 0.5,
                "refine_factor": 2,
            },
            "backend": {
                "query_chunk_size": 16,
            },
            "debug": {
                "save_pair_scopes": False,
                "profile_components": False,
            },
        },
    ).eval()
    images = torch.rand(1, 2, 3, 32, 32)
    with torch.inference_mode():
        outputs, patch_start = model(images)
    assert patch_start == 3
    assert outputs[0] is not None
    assert outputs[0].shape == (1, 2, 7, 128)
    stats = model.last_progressive_attention_stats[0]
    assert stats["algorithm"] == "adaptive_pair_scope"
    assert stats["qkv_projection_count"] == 1
    assert stats["full_patch_tokens"] == 8
    assert stats["all_patch_queries_receive_attention_output"]
    assert model._progressive_stage_states == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_adaptive_reference_block_is_finite() -> None:
    block = _block().cuda()
    x = torch.randn(1, 72, 32, device="cuda")
    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        result = adaptive_pair_scope_attention_block(
            block,
            x,
            num_frames=6,
            tokens_per_frame=12,
            num_special_tokens=3,
            patch_grid_size=(3, 3),
            layer_index=0,
            config=_config(
                coarse_num_anchors=6,
                query_chunk_size=16,
            ),
        )
    assert torch.isfinite(result.output).all()
    assert result.output.shape == x.shape
    assert result.stats["attention_backend"] == (
        "query_chunked_dense_mask_reference"
    )
    assert result.stats["efficient_sparse_kernel"] is False
