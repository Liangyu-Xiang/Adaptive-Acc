# Implemented Features

## Standalone adaptive spatial representatives

Set `frame_fusion_mode: adaptive-spatial-representative` with
`merge_ratio: 0` to run the single-frame spatial scheme independently of the
adaptive temporal scheme. Frame 0 remains a one-to-one reference frame. Each
frame from index 1 onward selects a global spatial medoid, then adds the token
with the largest mean distortion reduction. The selected prefix minimizes:

```text
D_S(k) + lambda_cost * k / P
```

Only global attention uses the representative sequence. Its attention
residual is mapped back to every original patch position, after which the
original full token sequence goes through the per-token MLP. The implementation
is in `vggt_omega/models/aggregator.py` and is exposed by both paper evaluation
scripts.

## Adaptive spatiotemporal representatives

The four independent schemes from the adaptive spatiotemporal token
compression design are exposed as `h-m`, `h-r`, `u-m`, and `u-r`:

- `h-m`: adaptive temporal representatives followed by local spatial whole-group merges;
- `h-r`: adaptive temporal representatives followed by local spatial deletion and reassignment;
- `u-m`: unified local time/space whole-group merges;
- `u-r`: unified local deletion and reassignment.

All four strictly isolate frame 0 patch tokens: they remain one-to-one
representatives, are excluded from candidate edges, and cannot represent
tokens from later frames. Special tokens are always preserved. The schemes
operate only in global attention, restore the attention residual to the
original token layout, and run the full per-token MLP afterward. M modes use
dynamic whole-group cosine reconstruction losses, maintain the complete
post-merge group adjacency, choose the better parent representative, and do
not use `max_group_size`. R modes use the same absolute reconstruction-plus-
token-cost objective to select a deletion prefix, then perform one-pass
reassignment to the surviving representatives without allowing reassigned
tokens to become new representatives. Every mode uses
`D + lambda_cost * M / ((F - 1) * P)` over non-reference patch tokens,
subject to the 5% minimum active ratio. Balanced defaults are N4/N8,
temporal window 1, overlap threshold 0.5, minimum active ratio 0.05, and
reassignment candidate limit 8. The evaluation scripts expose these as
`--frame-fusion-*` options.

## Progressive Multi-Level Attention

`progressive_attention` implements the exact token-pair semantic reference for
progressively refined global attention.

### Adaptive pair-scope reference

Set `algorithm: adaptive_pair_scope` to run independent two-level routing
inside selected global layers. The implementation:

1. projects the complete original token sequence to Q/K/V exactly once;
2. selects deterministic uniform patch anchors in frame-major order;
3. forms midpoint-bounded Query and Key scopes whose Cartesian product exactly
   partitions the original patch Q-K matrix;
4. scores the coarse anchor grid and refines only active parent rectangles;
5. evaluates independent local child products without mixing anchors from
   different parents;
6. compiles selected leaf rectangles into reusable Query-slab Key-row mask
   templates;
7. applies the exact patch mask with query-chunked SDPA while keeping all
   camera/register interactions dense.

The configuration is:

```text
configs/progressive_attention/adaptive_pair_scope_reference.json
```

This is a masked-dense correctness reference, not a sparse acceleration
kernel. The 300-frame all-sequence evaluation measured 0.18× speedup on
7Scenes and 0.19× on TUM-Dynamics, with substantial camera-accuracy loss.
Full tables and failure history are recorded in `implementation_report.md`.

### Model structure

VGGT-Omega has 24 zero-based layers. In the default alternating execution
order, every layer runs frame-local attention followed by inter-frame
attention. The global inter-frame layers are:

```text
0, 1, 3, 4, 5, 7, 8,
10, 11, 12, 13, 15, 16,
17, 18, 19, 21, 22, 23
```

Register-only inter-frame layers are `2, 6, 9, 14, 20`. Per-frame token order
is camera, 16 registers, then row-major patches. Global flattening is
frame-major.

### Sampling

For equivalent scope `S`, the number of selected patch tokens is:

```text
M_S = min(S, T) * P
```

The implementation uses a seeded, random, frame-balanced ordering over all
patch tokens. Repeated equal scopes inside one stage reuse the same set.
Different stages use different derived seeds. Scope prefixes are nested, so
`I_32` is a subset of `I_64`. Tokens are sampled before normalization and QKV;
all camera/register tokens remain present.

### Exact pair routing

Attention heads are averaged. Every patch token pair receives:

```text
score =
  0.25 * self
  + 0.25 * key-axis neighborhood
  + 0.25 * query-axis neighborhood
  + 0.25 * 2-D attention-matrix neighborhood
```

The default neighborhood radius is one. Row Top-K and column Top-K are applied
to individual token pairs, followed by token-pair dilation. There is no
128-by-64 block pooling or block-level maximum in the logical routing path.

For a finer sampled set, every fine token is assigned to its nearest coarse
token along the original flattened patch index. The inherited pair mask is:

```text
B_fine[i, j] = B_coarse[parent(i), parent(j)]
```

The state also records the highest-scoring coarse key per query for the
specified empty-row fallback. Camera/register keys remain available.

### Schedules

Configurations under `configs/progressive_attention/` provide B1/B2 and
P1-P6. P6 is:

```text
32 -> 64 -> Full
final_scope_mode = inherited_sparse
```

Every stage begins without inherited state and clears its state after the
stage-final Full layer.

### Current backend and limitation

The current implementation is a correctness reference. It stores the exact
boolean token-pair mask, computes routing scores in query chunks, and applies
inherited masks through query-chunked masked SDPA. This preserves method
semantics but does not establish real sparse-kernel speedup. Pair-mask storage
is quadratic and guarded by `max_reference_pair_elements`.

An optimized backend may regularize an already-selected logical mask into
execution blocks, but execution-block pooling must not replace token-pair
scoring, Top-K selection, dilation, or parent expansion.
