# Project State

## Progressive Multi-Level Attention

Status: exact token-pair semantic reference implemented, syntax checked, unit
tested, CUDA tested, and checkpoint-backed smoke tested.

Relevant files:

* `vggt_omega/models/progressive_attention.py`
* `vggt_omega/models/aggregator.py`
* `vggt_omega/models/vggt_omega.py`
* `configs/progressive_attention/`
* `tests/test_progressive_attention.py`

Verified behavior:

* seeded random, frame-balanced and nested patch sampling;
* repeated same-scope layers reuse a stage's random set;
* stages resample and reset inherited state independently;
* token features are sampled before normalization and QKV;
* camera/register tokens are retained;
* masks are selected and dilated at exact token-pair granularity;
* parent expansion uses exact nearest coarse token-pair mapping;
* sampled outputs scatter back to unchanged frame-major positions;
* Full with an all-one pair mask matches dense attention;
* disabling progressive attention leaves the original path active.

Validation:

```text
tests/test_progressive_attention.py: 14 passed
```

The focused multi-file suite completed with 33 passes in the development
worktree. A two-frame 64-by-64 forward with the released 1B checkpoint recorded
19 progressive global layers, finite camera/register, pose, depth, and
depth-confidence outputs, and zero retained stage states after the forward.
The disabled path was bitwise identical to the original path.

Known limitations:

* exact pair-mask storage is quadratic;
* routing still computes dense sampled Q-K chunks;
* inherited execution currently uses masked dense SDPA;
* corrected P1-P6 official quality, mask-recall, and speed results are
  `not_run`;
* related commit: `TODO: needs confirmation`.
