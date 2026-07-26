# Project State

## Adaptive Pair-Scope Attention

Status: implemented, syntax checked, focused CPU/CUDA tested, and evaluated
with 300-frame all-sequence inputs on 7Scenes and TUM-Dynamics.

Relevant files:

* `vggt_omega/models/adaptive_pair_scope_attention.py`
* `vggt_omega/models/progressive_attention.py`
* `vggt_omega/models/aggregator.py`
* `configs/progressive_attention/adaptive_pair_scope_reference.json`
* `tests/test_adaptive_pair_scope_attention.py`
* `implementation_report.md`

Verified behavior:

* one normalization and QKV projection per enabled adaptive layer;
* deterministic uniform anchors on the original frame-major patch axis;
* complete midpoint-bounded coarse Q-K rectangle partition;
* active-parent-only local refinement;
* simultaneous batch/token gathers without full-token-axis replication;
* complete selected/rejected leaf partition;
* one-time `O(S × N)` Query-slab mask-template compilation;
* full Query coverage with dense camera/register interactions;
* masked-dense reference SDPA and no cross-layer adaptive state.

Validation:

```text
tests/test_adaptive_pair_scope_attention.py +
tests/test_progressive_attention.py: 31 passed
```

Official 300-frame all-sequence results are under:

```text
outputs/adaptive-pair-scope__300frames__all-sequences__seed-0__20260725
```

| Dataset | Method | AUC@3 | AUC@30 | AbsRel | δ1.25 | Latency ms | Peak GiB | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7Scenes | VGGT-Omega | 34.11 | 87.87 | 0.0554 | 94.4786 | 34084.59 | 24.05 | 1.00× |
| 7Scenes | FastVGGT | 34.31 | 87.58 | 0.0557 | 94.4649 | 14951.45 | 25.32 | 2.28× |
| 7Scenes | Proposed | 10.93 | 74.84 | 0.0608 | 94.4656 | 184607.80 | 27.03 | 0.18× |
| TUM-Dynamics | VGGT-Omega | 40.29 | 84.50 | 0.0352 | 97.3838 | 34093.69 | 24.05 | 1.00× |
| TUM-Dynamics | FastVGGT | 37.70 | 83.44 | 0.0361 | 97.3428 | 14959.50 | 25.31 | 2.28× |
| TUM-Dynamics | Proposed | 10.85 | 63.41 | 0.0376 | 97.3222 | 179654.30 | 27.03 | 0.19× |

Known limitations:

* the backend is a correctness reference and is 5.3--5.4 times slower than
  dense VGGT-Omega;
* camera accuracy drops substantially with the current routing settings;
* anchors are deterministic and do not implement random 32/64-frame groups;
* a genuine acceleration path requires a semantics-preserving sparse kernel;
* related commit: `TODO: needs confirmation`.

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
