# Adaptive Pair-Scope Attention Implementation Report

## Scope

This report describes the `adaptive_pair_scope` implementation of Progressive
Multi-Level Attention.  It is a same-layer semantic reference for
scope-driven hierarchical sampling of the original flattened patch Q-K matrix.
It is explicitly separated from the retained cross-layer
`legacy_token_scope` implementation.

The 300-frame all-sequence protocol has now completed on 7Scenes and
TUM-Dynamics, including VGGT-Omega, VGGT-Omega + FastVGGT, and this method.
The reference backend is substantially slower and loses camera accuracy; the
unfavorable results are retained in Section 15.

## 1. How initial token anchors are sampled

Let `F` be the number of frames, `P` the patch count per frame, and
`N = F × P`.  Patch tokens use their original frame-major index:

```text
patch_index = frame_index × P + row_major_patch_index
```

`uniform_flat_anchor_indices()` deterministically samples this complete
`[0, N)` axis.  Count-based sampling uses the integer centers of equal-width
strata.  Stride-based sampling uses regularly spaced representatives.  The
anchors are sorted, unique, in range, and shared by Query and Key.

Exactly one of `coarse_num_anchors` and `coarse_stride` may be configured.  If
a requested anchor count exceeds `N`, all `N` patch tokens are used once.

## 2. How each anchor's one-dimensional scope is defined

`build_axis_scopes_from_anchors()` places boundaries at integer midpoints
between adjacent anchors:

```text
boundary[0] = 0
boundary[M] = N
boundary[i] = ceil((anchor[i-1] + anchor[i]) / 2)
scope[i] = [boundary[i], boundary[i+1])
```

Every scope is non-empty, contains its representative anchor, and uses
half-open `[start, end)` semantics.

## 3. Why PairScopes exactly partition the full matrix

The one-dimensional scopes are disjoint and their union is `[0, N)`.
`build_coarse_pair_scopes()` takes their Cartesian product on Query and Key.
Consequently every original pair `(q, k)` belongs to exactly one rectangle.

`validate_scope_partition()` checks every elementary Query slab and requires
its sorted Key intervals to cover `[0, N)` without gaps or overlaps.  The same
validator is applied to each batch element's final leaf rectangles.

## 4. Whether coarse sampling is still an anchor Cartesian product

Yes.  Coarse routing gathers the same anchor set from full patch Q and K and
computes:

```text
Q[:, :, anchors, :] × K[:, :, anchors, :]^T
```

This is an `M0 × M0` Cartesian grid in the original `N × N` patch matrix.  It
is not random independent pair sampling.

## 5. Whether fine sampling occurs only inside active parents

Yes.  Rejected coarse parents immediately become `allowed=false` leaves and
do not generate children or fine Q-K probes.

Only active coarse rectangles are subdivided.  Child Query anchors lie inside
the parent Query interval, and child Key anchors lie inside the parent Key
interval.

## 6. Whether fine routing avoids a global refined-anchor product

Yes.  Fine parents with equal child shapes may be batched for GPU execution,
but each parent is treated as an independent batch item.  The operation is:

```text
for active parent:
    local_queries(parent) × local_keys(parent)
```

Anchors from different parents never interact during fine probing.
`fine_probe_pairs` records the sum of each active parent's local product.

## 7. Whether coarse, fine, and Full use one QKV projection

Yes.  `adaptive_pair_scope_attention_block()` executes:

1. `block.norm1()` on all original tokens;
2. `block.attn.qkv()` exactly once;
3. Q/K normalization once when enabled;
4. coarse and fine gathers from those full Q/K tensors;
5. final Full masked attention using the same full Q/K/V tensors.

Statistics record `qkv_projection_count=1`, and a unit test verifies the QKV
module is called once.

## 8. How the final mask is generated from leaves

For every batch element, final leaves contain either:

* one rejected coarse rectangle; or
* all selected and rejected fine children replacing an active coarse parent.

The complete leaf set remains a disjoint partition of `N × N`.
`compile_leaf_patch_mask()` compiles allowed rectangles once per layer into
exact Query-slab Key-row templates using `O(S × N)` boolean storage, where
`S` is the number of distinct Query slabs. Each Query chunk then gathers its
rows with `gather_compiled_patch_mask_rows()` instead of traversing every leaf
again.

`materialize_full_patch_mask()` exists only for small tests and diagnostics.
The ordinary reference path does not permanently store a full `N × N`
boolean mask.

## 9. Whether every patch Query executes final attention

Yes.  QKV projection and final attention cover the complete original token
sequence.  The output is written for every contiguous Query chunk in original
frame-major order.  There is no sampled-token attention update or sampled
scatter in `adaptive_pair_scope`.

## 10. How special tokens are handled

Camera and register tokens are retained in their original per-frame
frame-major positions.

The final mask controls only:

```text
patch Query → patch Key
```

The following remain dense:

```text
patch Query → camera/register Key
camera/register Query → patch Key
camera/register Query → camera/register Key
```

The first reference version validates these requirements and rejects
configurations that disable dense special-token behavior.

## 11. Whether the backend is still masked dense

Yes.  The backend identifier is:

```text
query_chunked_dense_mask_reference
```

It calls PyTorch scaled-dot-product attention with a boolean mask for every
Query chunk.  Statistics always record:

```text
efficient_sparse_kernel = false
```

Logical pair density must not be described as measured speedup or reduced
kernel work.

## 12. Work needed for a real sparse kernel

A future acceleration backend must preserve the exact leaf-rectangle
semantics while replacing:

* Python leaf-to-template compilation;
* dense Query-slab Key-row templates;
* masked dense SDPA;
* dense evaluation of logically rejected Q-K pairs.

Candidate implementations include a rectangle-aware Triton kernel, CUDA
kernel, or semantics-preserving block-sparse compilation.  Any block
regularization must be measured separately for routing overhead, compilation,
kernel latency, memory, and end-to-end runtime.

## 13. Legacy compatibility

`progressive_attention.algorithm` supports:

```text
legacy_token_scope
adaptive_pair_scope
```

Existing B1/B2 and P1--P6 files omit `algorithm` and therefore continue to
resolve to `legacy_token_scope`.  Their random nested sampling, cross-layer
state, parent mapping, and sampled scatter code remains intact.

Adaptive mode does not call `nested_patch_indices()`,
`nearest_parent_positions()`, or legacy `ProgressiveMaskState`.

## 14. Validation status

Implemented validation covers:

* complete one-dimensional scope coverage;
* complete coarse two-dimensional coverage;
* active/inactive parent refinement;
* active-parent-only fine probe counts;
* hand-constructed final masks;
* non-empty Query-row fallback;
* all-one-mask equivalence with the original dense block;
* dense special-token interactions;
* output for every patch Query;
* exactly one QKV projection;
* 100-frame frame-major indexing;
* adaptive/legacy configuration compatibility;
* Aggregator dispatch without cross-layer adaptive state;
* CUDA output finiteness;
* active-parent gathers that do not replicate the full token axis;
* compiled-mask equivalence with direct leaf expansion;
* 300-frame checkpoint-backed forward completion.

Final command results are recorded after the concluding validation run.

```text
python -m py_compile:
  adaptive_pair_scope_attention.py
  progressive_attention.py
  aggregator.py
  vggt_omega.py
  test_adaptive_pair_scope_attention.py
Result: passed

python -m pytest -q:
  tests/test_adaptive_pair_scope_attention.py
  tests/test_progressive_attention.py
Result in clean publish worktree: 31 passed in 3.59s
```

The test set includes a tiny synthetic Aggregator forward and a CUDA adaptive
reference-block smoke. A separate 300-frame `chess/seq-03` checkpoint-backed
diagnostic completed before the formal matrix. The broader dirty development
worktree, which also contains the paper evaluators, completed 43 adaptive,
progressive, camera, and depth tests in 5.31s.

## Files

Added:

* `vggt_omega/models/adaptive_pair_scope_attention.py`
* `configs/progressive_attention/adaptive_pair_scope_reference.json`
* `tests/test_adaptive_pair_scope_attention.py`
* `implementation_report.md`
* `tools/build_official_comparison.py`

Modified:

* `vggt_omega/models/progressive_attention.py`
* `vggt_omega/models/aggregator.py`
* `docs/FEATURES.md`
* `docs/PROJECT_STATE.md`

`vggt_omega/models/vggt_omega.py` required no behavioral change because its
existing progressive-attention dictionary API already passes the complete
configuration to `Aggregator`.

## 15. Official 300-frame all-sequence evaluation

Protocol: seed 0, 512 `max_size`, Pi3 depth, one untimed warm-up and three
CUDA-event repeats, one process per physical RTX 4090. The proposed run reused
the baseline `sampled_frames.json` exactly. 7Scenes contains all 18 official
test sequences; TUM-Dynamics contains all eight protocol sequences.

### 7Scenes

| Method | AUC@3 | AUC@30 | AbsRel | δ1.25 | Latency ms | Peak GiB | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VGGT-Omega | 34.11 | 87.87 | 0.0554 | 94.4786 | 34084.59 | 24.05 | 1.00× |
| VGGT-Omega + FastVGGT | 34.31 | 87.58 | 0.0557 | 94.4649 | 14951.45 | 25.32 | 2.28× |
| Proposed | 10.93 | 74.84 | 0.0608 | 94.4656 | 184607.80 | 27.03 | 0.18× |

### TUM-Dynamics

| Method | AUC@3 | AUC@30 | AbsRel | δ1.25 | Latency ms | Peak GiB | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VGGT-Omega | 40.29 | 84.50 | 0.0352 | 97.3838 | 34093.69 | 24.05 | 1.00× |
| VGGT-Omega + FastVGGT | 37.70 | 83.44 | 0.0361 | 97.3428 | 14959.50 | 25.31 | 2.28× |
| Proposed | 10.85 | 63.41 | 0.0376 | 97.3222 | 179654.30 | 27.03 | 0.19× |

Raw and rendered artifacts:

```text
outputs/adaptive-pair-scope__300frames__all-sequences__seed-0__20260725
```

The first proposed launch failed before any sequence completed because the
active-parent gather replicated full `[H,N,D]` tensors and requested about
3.6 TiB. Its logs remain visible. Simultaneous batch/token indexing fixed that
failure. A subsequent diagnostic exposed repeated per-chunk Python leaf
traversal; compiling reusable Query-slab templates preserved mask semantics
and made the formal run feasible.

## Known limitations

* Inference only.
* Reference masked-dense backend, not an acceleration claim.
* The measured proposed latency is about 5.42× dense VGGT-Omega on 7Scenes
  and 5.27× on TUM-Dynamics; there is no runtime or memory benefit.
* Camera accuracy drops substantially under the current routing settings.
* No Triton, CUDA, or block-sparse execution kernel.
* `mean_head_sampled_probability` is normalized only over sampled Key anchors,
  not over the original full Key axis.
* Coarse and fine anchors are deterministic; adaptive mode does not implement
  random sampling inside 32-frame or 64-frame groups.
* Related Git commit: `TODO: needs confirmation` (no commit created).
