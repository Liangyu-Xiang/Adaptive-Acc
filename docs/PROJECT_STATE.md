# Project State

## DP-Medoid Frame Fusion With Token Copy-Back

Status: implemented, syntax checked, focused unit tested, smoke tested, and
preliminarily evaluated on 300-frame two-sequence subsets of TUM-Dynamics and
7Scenes with both required baselines.

The method partitions contiguous frames using dynamic programming. Frame
representations are obtained by global average pooling all tokens within each
frame. Pairwise frame distance is `1 - cosine_similarity`. For each candidate
contiguous group, the medoid minimizes the sum of distances to group members,
and the group cost is mean distance to medoid plus `beta` times max distance to
medoid. Within each selected group, all token positions are fused into the
group medoid using non-negative cosine similarities normalized inside that
group. Subsequent VGGT-Omega layers run on the fused medoid tokens. Cached
tokens are copied back to the original 300 frame positions before camera and
dense heads so output shapes remain unchanged.

Relevant files:

* `vggt_omega/models/aggregator.py`
* `vggt_omega/models/vggt_omega.py`
* `scripts/eval_7scenes_paper.py`
* `scripts/eval_tum_dynamics_paper.py`
* `tests/test_frame_fusion_partition.py`

Entry point:

```text
python scripts/eval_tum_dynamics_paper.py \
  --data-root /path/to/TUM-Dynamics \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --output-dir outputs/<experiment>/frame_fusion_pre0 \
  --device cuda:<gpu> \
  --seed 0 \
  --num-frames 300 \
  --sequences <seq-a> <seq-b> \
  --attention-mode default \
  --merge-ratio 0 \
  --frame-fusion-mode dp-medoid \
  --frame-fusion-k 80 \
  --frame-fusion-max-group-size 5 \
  --frame-fusion-beta 1.0 \
  --frame-fusion-start-layer -1 \
  --timing-repeats 1
```

`--frame-fusion-start-layer -1` fuses before block 0. Non-negative values fuse
after that inter-frame block and copy expanded cached tokens only for later
cached layers.

Preliminary result root:

```text
outputs/frame-fusion-dp-medoid__300frames__K80_M5_beta1__2seq__20260730-090351
```

Protocol:

* original dense VGGT-Omega baseline: `attention_mode=default`,
  `merge_ratio=0`;
* FastVGGT baseline: `attention_mode=default`, `merge_ratio=0.9`, using the
  repository's protected token merge path;
* proposed variants: `frame_fusion_start_layer=-1`, `11`, and `18`;
* 300 sampled frames per sequence, seed 0, 512 `max_size`;
* TUM-Dynamics sequences: `rgbd_dataset_freiburg3_sitting_halfsphere` and
  `rgbd_dataset_freiburg3_sitting_rpy`;
* 7Scenes sequences: `chess/seq-03` and `chess/seq-05`;
* `K=80`, max group size `M=5`, `beta=1.0`;
* timing uses one warm-up and `timing_repeats=1`;
* dirty working tree; no commit created.

TUM-Dynamics two-sequence results:

| Method | AUC@3 | AUC@30 | AbsRel | delta1.25 | Latency ms | Speedup | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VGGT-Omega | 31.33 | 81.61 | 0.0366 | 97.1619 | 33068.1 | 1.00x | 24.05 |
| VGGT-Omega + FastVGGT | 28.83 | 80.51 | 0.0370 | 97.1312 | 15019.2 | 2.20x | 25.67 |
| FrameFusion pre0 | 0.04 | 3.33 | 0.3604 | 60.4852 | 6609.5 | 5.00x | 17.48 |
| FrameFusion after11 | 0.10 | 5.64 | 0.3168 | 64.0566 | 19443.6 | 1.70x | 19.89 |
| FrameFusion after18 | 0.04 | 5.12 | 0.0370 | 97.1723 | 28801.9 | 1.15x | 23.82 |

7Scenes two-sequence results:

| Method | AUC@3 | AUC@30 | AbsRel | delta1.25 | Latency ms | Speedup | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VGGT-Omega | 46.88 | 93.09 | 0.0381 | 96.1732 | 33509.8 | 1.00x | 24.05 |
| VGGT-Omega + FastVGGT | 46.84 | 92.96 | 0.0378 | 96.1738 | 14617.2 | 2.29x | 25.67 |
| FrameFusion pre0 | 0.28 | 10.83 | 0.1764 | 68.4847 | 6883.7 | 4.87x | 17.48 |
| FrameFusion after11 | 0.28 | 10.43 | 0.1679 | 70.4145 | 19613.7 | 1.71x | 19.89 |
| FrameFusion after18 | 0.15 | 10.61 | 0.0411 | 96.1490 | 27967.8 | 1.20x | 23.82 |

K200/M4 pre0 parameter test:

```text
outputs/frame-fusion-dp-medoid__300frames__K200_M4_beta1__2seq__20260731-0749
```

This run reuses the previous same-protocol VGGT-Omega and FastVGGT baseline
rows and newly evaluates only `FrameFusion pre0` with `K=200`, max group size
`M=4`, `beta=1.0`, and `frame_fusion_start_layer=-1`. Protocol remains 300
sampled frames per sequence, seed 0, 512 `max_size`, selected two sequences
per dataset, and `timing_repeats=1`.

| Dataset | Method | AUC@3 | AUC@30 | AbsRel | delta1.25 | Latency ms | Speedup | Peak GiB | Partition s | Fusion s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TUM-Dynamics | FrameFusion pre0 K200 M4 | 7.64 | 36.91 | 0.1399 | 85.69 | 18311.9 | 1.81x | 20.93 | 2.785 | 0.037 |
| 7Scenes | FrameFusion pre0 K200 M4 | 13.96 | 47.64 | 0.0963 | 85.10 | 19292.7 | 1.74x | 20.93 | 3.197 | 0.042 |

The same task also exported a first-100-frame partition comparison against the
external `frame_persistent_spatial` results. The partition comparison uses that
external run's first-300-frame inputs and clips the displayed groups to frames
0..99; it is intentionally separate from the random-sampled performance
protocol above.

```text
outputs/frame-fusion-dp-medoid__300frames__K200_M4_beta1__2seq__20260731-0749/partition_first100__frame_persistent_vs_dpK200M4
```

External first-300 contiguous validation:

```text
<external-validation-root>/validation_summary.json
<this-repository>/outputs/dpK200M4_first300_contiguous_external_sample__2seq__20260731-082318
outputs/dpK200M4_first300_contiguous__2seq__20260731-0820/7scenes
```

This validation reuses the external `frame_persistent_spatial` first-300-frame
sample manifests for TUM-Dynamics and the already-completed 7Scenes contiguous
run. TUM-Dynamics uses
`scripts/eval_tum_external_sample_frame_fusion.py`, which reads the external
`_time.json` frame names and associates pose/depth by nearest timestamp.

| Dataset | FrameFusion pre0 K200 M4 AUC@3 | AUC@30 | AbsRel | delta1.25 | FPS wall | Latency ms | Fused frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TUM-Dynamics | 39.32 | 83.93 | 0.0321 | 97.22 | 3.31 | 30490.6 | `[200, 200]` |
| 7Scenes | 37.45 | 90.14 | 0.0419 | 95.92 | 9.94 | 30188.2 | `[200, 200]` |

The same validation reran the external `frame_persistent_spatial` method on
random-order 300-frame inputs after fixing the non-Bonn random-order evaluation
path in the external implementation.
The sampled frame lists are non-monotonic for all four tested sequences. The
method's merge ratio drops from 27.26% to 9.33% on TUM-Dynamics and from
32.01% to 15.33% on 7Scenes, with a large quality drop relative to its
first-300 contiguous run. This supports the diagnosis that the external
method's frame fusion relies on local temporal continuity and is not equivalent
to fixed-count global DP fusion on randomly ordered frames.

Machine-readable summaries:

* `outputs/frame-fusion-dp-medoid__300frames__K80_M5_beta1__2seq__20260730-090351/summary.json`
* `outputs/frame-fusion-dp-medoid__300frames__K80_M5_beta1__2seq__20260730-090351/overall.csv`
* `outputs/frame-fusion-dp-medoid__300frames__K80_M5_beta1__2seq__20260730-090351/per_sequence.csv`
* `outputs/frame-fusion-dp-medoid__300frames__K200_M4_beta1__2seq__20260731-0749/summary.json`
* `outputs/frame-fusion-dp-medoid__300frames__K200_M4_beta1__2seq__20260731-0749/overall.csv`
* `outputs/frame-fusion-dp-medoid__300frames__K200_M4_beta1__2seq__20260731-0749/per_sequence.csv`

Validation:

```text
python -m pytest -q tests/test_frame_fusion_partition.py tests/test_reference_frame_order.py: 11 passed
python -m pytest -q tests/test_frame_fusion_partition.py tests/test_reference_frame_order.py tests/test_progressive_attention.py tests/test_adaptive_pair_scope_attention.py tests/test_register_mediated_anchor_attention.py: 55 passed
python -m compileall -q vggt_omega scripts/eval_7scenes_paper.py scripts/eval_tum_dynamics_paper.py tests/test_frame_fusion_partition.py: passed
python scripts/eval_7scenes_paper.py --help: passed
python scripts/eval_tum_dynamics_paper.py --help: passed
git diff --check: passed
```

Known limitations:

* camera metrics collapse for all tested copy-back variants because frames in
  the same group receive identical final camera/register tokens after fusion;
* `after18` preserves depth quality but provides only modest latency reduction,
  because most encoder layers still run on all 300 frames;
* timing uses one repeat and two sequences per dataset, so latency is a
  preliminary measurement rather than a final benchmark;
* the current implementation supports batch size 1 for frame fusion;
* this does not yet implement a camera-specific reconstruction strategy for
  non-medoid frames;
* related commit: `TODO: needs confirmation`.

## Original VGGT-Omega First-Frame Token Multiplicity Ablation

Status: implemented, syntax checked, focused unit tested, evaluator-help
checked, and preliminarily evaluated on 100-frame all-sequence quality
ablations for 7Scenes and TUM-Dynamics.

Original VGGT-Omega assigns learned position-0 camera/register tokens only to
model input position 0. The evaluator option `--first-frame-token-indices`
extends this distinction to additional input positions during inference. The
default `--first-frame-token-indices 0` preserves the original model behavior.
This experiment keeps `--reference-frame-index 0`, so the sampled frame order
is unchanged; only the camera/register token assignment changes.

Relevant files:

* `vggt_omega/utils/reference_frame.py`
* `vggt_omega/models/aggregator.py`
* `vggt_omega/models/vggt_omega.py`
* `scripts/eval_7scenes_paper.py`
* `scripts/eval_tum_dynamics_paper.py`
* `tests/test_reference_frame_order.py`

Entry point:

```text
python scripts/eval_tum_dynamics_paper.py \
  --data-root /path/to/TUM-Dynamics \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --output-dir outputs/<experiment>/tok_<count> \
  --device cuda:<gpu> \
  --seed 0 \
  --num-frames 100 \
  --reference-frame-index 0 \
  --first-frame-token-indices <spec> \
  --sampling-pool full \
  --image-resolution 512 \
  --resize-mode max_size \
  --attention-mode default \
  --merge-ratio 0 \
  --skip-timing
```

`<spec>` accepts comma-separated indices, positive inclusive ranges,
`uniform:N`, or `all`; input position 0 must be included.

Preliminary result roots:

```text
outputs/first-frame-token-multiplicity__tum-dynamics-vggt-omega__100frames__seed-0__20260729-060211
outputs/first-frame-token-multiplicity__7scenes-vggt-omega__100frames__seed-0__20260729-060211
outputs/first-frame-token-multiplicity__vggt-omega__100frames__seed-0__20260729
```

Protocol:

* original dense VGGT-Omega: `attention_mode=default`, `merge_ratio=0`;
* 100 sampled frames per sequence, seed 0, 512 `max_size`;
* `reference_frame_index=0`, so completed runs have identical input order and
  identical frame sets within each dataset;
* tested token specs: `0`, `0,50`, `uniform:4`, `uniform:8`, and `all`;
* TUM-Dynamics official eight sequences with full sampling pool;
* 7Scenes official test sequences with sequence-level sampling;
* checkpoint SHA-256
  `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`;
* timing skipped, so these are quality-screening results only;
* dirty working tree; no commit created.

TUM-Dynamics overall results:

| First-token count | Spec | AUC@3 | AUC@30 | AbsRel | delta1.25 | dAUC@3 vs count1 | dAUC@30 vs count1 | dAbsRel vs count1 | ddelta1.25 vs count1 | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0` | 39.58 | 83.47 | 0.0370 | 97.2858 | +0.00 | +0.00 | +0.0000 | +0.0000 | 11.98 |
| 2 | `0,50` | 30.18 | 78.40 | 0.0371 | 97.2864 | -9.41 | -5.06 | +0.0000 | +0.0007 | 11.98 |
| 4 | `uniform:4` | 26.76 | 78.62 | 0.0372 | 97.2848 | -12.82 | -4.85 | +0.0001 | -0.0009 | 11.98 |
| 8 | `uniform:8` | 30.03 | 77.84 | 0.0373 | 97.2769 | -9.56 | -5.62 | +0.0003 | -0.0089 | 11.98 |
| 100 | `all` | 13.53 | 58.48 | 0.0374 | 97.2494 | -26.05 | -24.98 | +0.0003 | -0.0363 | 11.98 |

7Scenes overall results:

| First-token count | Spec | AUC@3 | AUC@30 | AbsRel | delta1.25 | dAUC@3 vs count1 | dAUC@30 vs count1 | dAbsRel vs count1 | ddelta1.25 vs count1 | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0` | 30.67 | 87.15 | 0.0343 | 97.6187 | +0.00 | +0.00 | +0.0000 | +0.0000 | 11.98 |
| 2 | `0,50` | 11.97 | 76.44 | 0.0342 | 97.6264 | -18.69 | -10.71 | -0.0002 | +0.0077 | 11.98 |
| 4 | `uniform:4` | 15.10 | 78.94 | 0.0339 | 97.6374 | -15.57 | -8.20 | -0.0005 | +0.0186 | 11.98 |
| 8 | `uniform:8` | 17.59 | 81.08 | 0.0336 | 97.6537 | -13.08 | -6.07 | -0.0008 | +0.0350 | 11.98 |
| 100 | `all` | 12.60 | 74.19 | 0.0349 | 97.6605 | -18.07 | -12.96 | +0.0006 | +0.0418 | 11.98 |

Machine-readable summaries:

* `outputs/first-frame-token-multiplicity__tum-dynamics-vggt-omega__100frames__seed-0__20260729-060211/summary.json`
* `outputs/first-frame-token-multiplicity__tum-dynamics-vggt-omega__100frames__seed-0__20260729-060211/overall.csv`
* `outputs/first-frame-token-multiplicity__tum-dynamics-vggt-omega__100frames__seed-0__20260729-060211/per_sequence.csv`
* `outputs/first-frame-token-multiplicity__7scenes-vggt-omega__100frames__seed-0__20260729-060211/summary.json`
* `outputs/first-frame-token-multiplicity__7scenes-vggt-omega__100frames__seed-0__20260729-060211/overall.csv`
* `outputs/first-frame-token-multiplicity__7scenes-vggt-omega__100frames__seed-0__20260729-060211/per_sequence.csv`
* `outputs/first-frame-token-multiplicity__vggt-omega__100frames__seed-0__20260729/summary.json`
* `outputs/first-frame-token-multiplicity__vggt-omega__100frames__seed-0__20260729/overall.csv`

Validation:

```text
python -m pytest -q tests/test_reference_frame_order.py: 7 passed
python -m pytest -q tests/test_reference_frame_order.py tests/test_progressive_attention.py tests/test_adaptive_pair_scope_attention.py tests/test_register_mediated_anchor_attention.py: 51 passed
python -m compileall -q vggt_omega scripts/eval_7scenes_paper.py scripts/eval_tum_dynamics_paper.py tests/test_reference_frame_order.py: passed
python scripts/eval_7scenes_paper.py --help: passed
python scripts/eval_tum_dynamics_paper.py --help: passed
git diff --check: passed
```

Known limitations:

* timing was skipped, so no latency or speedup claim is made;
* this is an inference-time distribution shift: the released checkpoint was
  trained with exactly one position-0 camera/register token assignment;
* camera AUC drops sharply as soon as additional input positions receive the
  position-0 token, while depth metrics remain almost unchanged;
* one initial TUM-Dynamics `tok_100` attempt on GPU 6 failed before model load
  because the exclusive GPU guard detected another Python process; the failed
  log is preserved under `tok_100/`, and the completed rerun is under
  `tok_100_gpu4/`;
* this uses one sampling seed and five token-count settings; it is screening
  evidence, not a final statistical claim;
* related commit: `TODO: needs confirmation`.

## Original VGGT-Omega Reference-Frame Input-Position Ablation

Status: implemented, syntax checked, focused unit tested, evaluator-help
checked, and preliminarily evaluated on 100-frame all-sequence quality
ablations for 7Scenes and TUM-Dynamics.

Original VGGT-Omega distinguishes the first input position through learned
position-0 camera/register tokens in `slice_expand_and_flatten`; it does not
use FastVGGT token merging in this experiment. The evaluator option
`--reference-frame-index` selects one frame from the sampled 100-frame list and
moves it to model input position 0. The other 99 sampled frames keep their
relative order. Default `--reference-frame-index 0` preserves the existing
evaluator behavior.

Relevant files:

* `vggt_omega/utils/reference_frame.py`
* `vggt_omega/models/aggregator.py`
* `scripts/eval_7scenes_paper.py`
* `scripts/eval_tum_dynamics_paper.py`
* `tests/test_reference_frame_order.py`

Entry point:

```text
python scripts/eval_tum_dynamics_paper.py \
  --data-root /path/to/TUM-Dynamics \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --output-dir outputs/<experiment>/ref_<index> \
  --device cuda:<gpu> \
  --seed 0 \
  --num-frames 100 \
  --reference-frame-index <index> \
  --sampling-pool full \
  --image-resolution 512 \
  --resize-mode max_size \
  --attention-mode default \
  --merge-ratio 0 \
  --skip-timing
```

Preliminary result roots:

```text
outputs/reference-frame-ablation__tum-dynamics-vggt-omega__100frames__seed-0__20260728-125341
outputs/reference-frame-ablation__7scenes-vggt-omega__100frames__seed-0__20260728-125834
outputs/reference-frame-ablation__vggt-omega__100frames__seed-0__20260728
```

Protocol:

* original dense VGGT-Omega: `attention_mode=default`, `merge_ratio=0`;
* 100 sampled frames per sequence, seed 0, 512 `max_size`;
* TUM-Dynamics official eight sequences with full sampling pool;
* 7Scenes official test sequences with sequence-level sampling;
* completed runs have identical original sampled order and identical frame set
  within each dataset; only the sampled frame moved to input position 0 differs;
* checkpoint SHA-256
  `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`;
* timing skipped, so these are quality-screening results only;
* dirty working tree; no commit created.

TUM-Dynamics overall results:

| Reference frame | AUC@3 | AUC@30 | AbsRel | delta1.25 | dAUC@3 vs ref0 | dAUC@30 vs ref0 | dAbsRel vs ref0 | ddelta1.25 vs ref0 | Peak GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 39.58 | 83.47 | 0.0370 | 97.2858 | +0.00 | +0.00 | +0.0000 | +0.0000 | 11.98 |
| 25 | 39.35 | 83.08 | 0.0370 | 97.2859 | -0.23 | -0.39 | -0.0000 | +0.0001 | 11.98 |
| 50 | 39.87 | 83.00 | 0.0370 | 97.2873 | +0.29 | -0.46 | -0.0000 | +0.0015 | 11.98 |
| 75 | 38.98 | 82.48 | 0.0370 | 97.2865 | -0.60 | -0.98 | -0.0000 | +0.0008 | 11.98 |
| 99 | 39.84 | 83.61 | 0.0371 | 97.2823 | +0.26 | +0.14 | +0.0000 | -0.0035 | 11.98 |

7Scenes overall results:

| Reference frame | AUC@3 | AUC@30 | AbsRel | delta1.25 | dAUC@3 vs ref0 | dAUC@30 vs ref0 | dAbsRel vs ref0 | ddelta1.25 vs ref0 | Peak GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 30.67 | 87.15 | 0.0343 | 97.6187 | +0.00 | +0.00 | +0.0000 | +0.0000 | 11.98 |
| 25 | 31.30 | 86.99 | 0.0343 | 97.6031 | +0.63 | -0.15 | -0.0001 | -0.0156 | 11.98 |
| 50 | 30.92 | 87.04 | 0.0341 | 97.6311 | +0.25 | -0.11 | -0.0002 | +0.0124 | 11.98 |
| 75 | 32.26 | 87.35 | 0.0342 | 97.6300 | +1.59 | +0.20 | -0.0001 | +0.0112 | 11.98 |
| 99 | 30.78 | 87.04 | 0.0342 | 97.6044 | +0.12 | -0.10 | -0.0001 | -0.0144 | 11.98 |

Machine-readable summaries:

* `outputs/reference-frame-ablation__tum-dynamics-vggt-omega__100frames__seed-0__20260728-125341/summary.json`
* `outputs/reference-frame-ablation__tum-dynamics-vggt-omega__100frames__seed-0__20260728-125341/overall.csv`
* `outputs/reference-frame-ablation__tum-dynamics-vggt-omega__100frames__seed-0__20260728-125341/per_sequence.csv`
* `outputs/reference-frame-ablation__7scenes-vggt-omega__100frames__seed-0__20260728-125834/summary.json`
* `outputs/reference-frame-ablation__7scenes-vggt-omega__100frames__seed-0__20260728-125834/overall.csv`
* `outputs/reference-frame-ablation__7scenes-vggt-omega__100frames__seed-0__20260728-125834/per_sequence.csv`
* `outputs/reference-frame-ablation__vggt-omega__100frames__seed-0__20260728/overall.csv`

Validation:

```text
python -m pytest -q tests/test_reference_frame_order.py: 4 passed
python -m pytest -q tests/test_reference_frame_order.py tests/test_progressive_attention.py tests/test_adaptive_pair_scope_attention.py tests/test_register_mediated_anchor_attention.py: 48 passed
python -m compileall -q vggt_omega scripts/eval_7scenes_paper.py scripts/eval_tum_dynamics_paper.py tests/test_reference_frame_order.py: passed
python scripts/eval_7scenes_paper.py --help: passed
python scripts/eval_tum_dynamics_paper.py --help: passed
```

Known limitations:

* timing was skipped, so no latency or speedup claim is made;
* one initial `ref_000` attempt per dataset on GPU 0 failed after the first
  sequence because the exclusive GPU guard detected foreign processes; failed
  logs are preserved under `ref_000/`, and completed reruns are under
  `ref_000_gpu7/`;
* this uses one sampling seed and five sampled-frame positions; it is
  screening evidence, not a final statistical claim;
* 7Scenes and TUM-Dynamics disagree on which reference index is best for
  AUC@3, so downstream method design should not hard-code a universal
  nonzero reference-frame choice from this result alone;
* related commit: `TODO: needs confirmation`.

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
