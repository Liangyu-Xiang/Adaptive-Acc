# Implemented Features

## Progressive Multi-Level Attention

`progressive_attention` implements the exact token-pair semantic reference for
progressively refined global attention.

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
