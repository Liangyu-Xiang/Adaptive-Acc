from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.layers.attention import SelfAttention


def _make_attention() -> SelfAttention:
    torch.manual_seed(7)
    attention = SelfAttention(dim=32, num_heads=4, qkv_bias=True, use_qk_norm=False)
    attention.eval()
    return attention


def _run_attention(attention: SelfAttention, x: torch.Tensor, **kwargs) -> torch.Tensor:
    with torch.inference_mode():
        out = attention(
            x,
            patch_grid_size=(2, 2),
            num_special_tokens=2,
            **kwargs,
        )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    return out


def test_attention_from_qkv_supports_full_query_and_compressed_key_value_lengths():
    attention = _make_attention()
    query_input = torch.randn((1, 7, 32), generator=torch.Generator().manual_seed(1))
    key_value_input = torch.randn((1, 4, 32), generator=torch.Generator().manual_seed(2))

    q, _, _ = attention.project_qkv(query_input)
    _, k, v = attention.project_qkv(key_value_input)
    actual = attention.attention_from_qkv(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, 7, 32)

    assert actual.shape == (1, 7, 32)
    assert torch.allclose(actual, expected)


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


def test_intra_only_runs():
    attention = _make_attention()
    x = torch.randn(2, 18, 32)
    _run_attention(
        attention,
        x,
        use_adaptive_kv_anchor=True,
        adaptive_anchor_strategy="intra_only",
        adaptive_anchor_ratio=0.25,
    )


def test_register_gated_intra_runs_with_profile():
    attention = _make_attention()
    x = torch.randn(2, 18, 32)
    _run_attention(
        attention,
        x,
        use_adaptive_kv_anchor=True,
        adaptive_anchor_strategy="register_gated_intra",
        adaptive_anchor_ratio=0.25,
        adaptive_anchor_topm_frames=2,
        adaptive_anchor_profile=True,
        adaptive_anchor_debug=True,
    )
    debug = attention.last_adaptive_anchor_debug
    assert debug["strategy"] == "register_gated_intra"
    assert debug["score_mode"] == "intra"
    assert debug["topm_key_frames"] is not None
    assert "profile" in debug
    assert debug["profile"]["total_time_ms"] >= 0.0


def test_all_frame_intra_uses_cached_scores():
    attention = _make_attention()
    attention.precomputed_intra_scores = torch.rand(1, 3, 4)
    x = torch.randn(1, 18, 32)
    _run_attention(
        attention,
        x,
        use_adaptive_kv_anchor=True,
        adaptive_anchor_strategy="all_frame_intra",
        adaptive_anchor_ratio=0.25,
        adaptive_anchor_intra_source="cached_frame_qk",
        adaptive_anchor_debug=True,
    )
    debug = attention.last_adaptive_anchor_debug
    assert debug["strategy"] == "all_frame_intra"
    assert debug["cached_intra_available"] is True
    assert debug["topm_key_frames"] is None


def test_random_and_temporal_frame_gating_run():
    for strategy in ("random_frame_intra", "temporal_neighbor_intra"):
        attention = _make_attention()
        x = torch.randn(1, 18, 32)
        _run_attention(
            attention,
            x,
            use_adaptive_kv_anchor=True,
            adaptive_anchor_strategy=strategy,
            adaptive_anchor_ratio=0.5,
            adaptive_anchor_topm_frames=2,
            adaptive_anchor_debug=True,
        )
        debug = attention.last_adaptive_anchor_debug
        assert debug["strategy"] == strategy
        assert debug["topm_key_frames"] is not None


def test_oracle_frame_intra_runs_on_tiny_input():
    attention = _make_attention()
    x = torch.randn(1, 18, 32)
    _run_attention(
        attention,
        x,
        use_adaptive_kv_anchor=True,
        adaptive_anchor_strategy="oracle_frame_intra",
        adaptive_anchor_ratio=0.5,
        adaptive_anchor_topm_frames=2,
        adaptive_anchor_debug=True,
    )
    debug = attention.last_adaptive_anchor_debug
    assert debug["strategy"] == "oracle_frame_intra"
    assert debug["topm_key_frames"] is not None


def test_register_gated_intra_query_runs_batch_one():
    attention = _make_attention()
    x = torch.randn(1, 18, 32)
    _run_attention(
        attention,
        x,
        use_adaptive_kv_anchor=True,
        adaptive_anchor_strategy="register_gated_intra_query",
        adaptive_anchor_ratio=0.5,
        adaptive_anchor_topm_frames=2,
        adaptive_anchor_query_conditioned_eta=0.2,
        adaptive_anchor_debug=True,
    )
    debug = attention.last_adaptive_anchor_debug
    assert debug["strategy"] == "register_gated_intra_query"
    assert debug["topm_key_frames"] is not None


def test_quota_intra_proxy_runs():
    attention = _make_attention()
    x = torch.randn(1, 18, 32)
    _run_attention(
        attention,
        x,
        use_adaptive_kv_anchor=True,
        adaptive_anchor_strategy="quota_intra_proxy",
        adaptive_anchor_ratio=0.5,
        adaptive_anchor_score_mode="quota_union",
        adaptive_anchor_proxy_quota_ratio=0.25,
        adaptive_anchor_debug=True,
    )
    debug = attention.last_adaptive_anchor_debug
    assert debug["strategy"] == "quota_intra_proxy"
    assert "selected_by_intra" in debug
    assert "selected_by_proxy" in debug
    assert float(debug["proxy_quota_ratio"]) == 0.25


def test_tiny_anchor_budget_reduces_min_per_frame():
    attention = _make_attention()
    x = torch.randn(1, 18, 32)
    _run_attention(
        attention,
        x,
        use_adaptive_kv_anchor=True,
        adaptive_anchor_strategy="lifting",
        adaptive_anchor_total=1,
        adaptive_anchor_min_per_frame=4,
        adaptive_anchor_debug=True,
    )
    debug = attention.last_adaptive_anchor_debug
    assert int(debug["anchor_budget"]) == 1
    assert int(debug["anchor_counts"].sum()) == 1


def test_large_anchor_budget_uses_full_patch_kv():
    attention = _make_attention()
    x = torch.randn(1, 18, 32)
    _run_attention(
        attention,
        x,
        use_adaptive_kv_anchor=True,
        adaptive_anchor_strategy="lifting",
        adaptive_anchor_total=999,
        adaptive_anchor_debug=True,
    )
    debug = attention.last_adaptive_anchor_debug
    assert int(debug["anchor_budget"]) == 12
    assert int(debug["kv_token_count"]) == 18


def test_cached_intra_scores_are_used():
    attention = _make_attention()
    attention.precomputed_intra_scores = torch.rand(1, 3, 4)
    x = torch.randn(1, 18, 32)
    _run_attention(
        attention,
        x,
        use_adaptive_kv_anchor=True,
        adaptive_anchor_strategy="register_gated_intra",
        adaptive_anchor_ratio=0.25,
        adaptive_anchor_intra_source="cached_frame_qk",
        adaptive_anchor_debug=True,
    )
    debug = attention.last_adaptive_anchor_debug
    assert debug["cached_intra_available"] is True
    assert debug["intra_source_used"] == "cached_frame_qk"


def test_cached_intra_scores_fallbacks_when_missing():
    attention = _make_attention()
    attention.precomputed_intra_scores = torch.rand(1, 2, 4)
    x = torch.randn(1, 18, 32)
    _run_attention(
        attention,
        x,
        use_adaptive_kv_anchor=True,
        adaptive_anchor_strategy="register_gated_intra",
        adaptive_anchor_ratio=0.25,
        adaptive_anchor_intra_source="cached_frame_qk",
        adaptive_anchor_debug=True,
    )
    debug = attention.last_adaptive_anchor_debug
    assert debug["cached_intra_available"] is False
    assert debug["intra_source_used"] == "current_inter_qk_fallback"
    assert any("falling back" in warning for warning in debug["warnings"])


def test_debug_payload_can_be_saved():
    with tempfile.TemporaryDirectory() as tmpdir:
        aggregator = object.__new__(Aggregator)
        aggregator.adaptive_anchor_debug = True
        aggregator.adaptive_anchor_debug_dir = Path(tmpdir)
        aggregator._adaptive_anchor_debug_step = 0
        aggregator.adaptive_anchor_strategy = "register_gated_intra"
        aggregator.inter_frame_blocks = [SimpleNamespace(attn=SimpleNamespace(last_adaptive_anchor_debug={"a": 1}))]
        Aggregator._maybe_save_adaptive_anchor_debug(aggregator, 0)
        saved = sorted(Path(tmpdir).glob("*.pt"))
        assert len(saved) == 1
        payload = torch.load(saved[0], map_location="cpu")
        assert payload["a"] == 1


def run_all_tests() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    run_all_tests()
