import torch

from vggt_omega.models.layers.attention import SelfAttention


def _make_attention() -> SelfAttention:
    torch.manual_seed(7)
    attention = SelfAttention(dim=32, num_heads=4, qkv_bias=True, use_qk_norm=False)
    attention.eval()
    return attention


def test_adaptive_anchor_disabled_matches_dense_path():
    attention = _make_attention()
    x = torch.randn(2, 18, 32)
    with torch.inference_mode():
        dense = attention(x)
        disabled = attention(
            x,
            patch_grid_size=(2, 2),
            num_special_tokens=2,
            use_adaptive_kv_anchor=False,
        )
    assert torch.allclose(dense, disabled, atol=1e-6, rtol=1e-6)


def test_lifting_shape_batch_two():
    attention = _make_attention()
    x = torch.randn(2, 18, 32)
    with torch.inference_mode():
        out = attention(
            x,
            patch_grid_size=(2, 2),
            num_special_tokens=2,
            use_adaptive_kv_anchor=True,
            adaptive_anchor_strategy="lifting",
            adaptive_anchor_ratio=0.25,
        )
    assert out.shape == x.shape


def test_frame_pair_gated_shape_batch_two():
    attention = _make_attention()
    x = torch.randn(2, 18, 32)
    with torch.inference_mode():
        out = attention(
            x,
            patch_grid_size=(2, 2),
            num_special_tokens=2,
            use_adaptive_kv_anchor=True,
            adaptive_anchor_strategy="frame_pair_gated",
            adaptive_anchor_ratio=0.5,
            adaptive_anchor_topm_frames=2,
        )
    assert out.shape == x.shape


def test_tiny_anchor_budget_reduces_min_per_frame():
    attention = _make_attention()
    x = torch.randn(1, 18, 32)
    with torch.inference_mode():
        out = attention(
            x,
            patch_grid_size=(2, 2),
            num_special_tokens=2,
            use_adaptive_kv_anchor=True,
            adaptive_anchor_strategy="lifting",
            adaptive_anchor_total=1,
            adaptive_anchor_min_per_frame=4,
            adaptive_anchor_debug=True,
        )
    debug = attention.last_adaptive_anchor_debug
    assert out.shape == x.shape
    assert int(debug["anchor_budget"]) == 1
    assert int(debug["anchor_counts"].sum()) == 1


def test_large_anchor_budget_uses_full_patch_kv():
    attention = _make_attention()
    x = torch.randn(1, 18, 32)
    with torch.inference_mode():
        out = attention(
            x,
            patch_grid_size=(2, 2),
            num_special_tokens=2,
            use_adaptive_kv_anchor=True,
            adaptive_anchor_strategy="lifting",
            adaptive_anchor_total=999,
            adaptive_anchor_debug=True,
        )
    debug = attention.last_adaptive_anchor_debug
    assert out.shape == x.shape
    assert int(debug["anchor_budget"]) == 12
    assert int(debug["kv_token_count"]) == 18
