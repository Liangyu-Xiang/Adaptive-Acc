# Layer-wise token evolution analysis

`tools/analyze_token_evolution.py` measures adjacent-layer hidden-state changes
without changing `VGGTOmega.forward` or enabling token merging. Forward hooks
are attached to all 24 frame/inter-frame block pairs. At register-attention
layers, the tool combines the updated camera/register tokens with the patch
tokens produced by that layer's frame-attention block, reproducing the complete
layer-boundary state used by the aggregator.

The analyzer retains only the previous and current complete token tensors on
GPU. Relative-L2 and cosine-distance scalars are transferred to CPU after every
layer. It distinguishes camera, 16 register, and patch tokens using the
aggregator's `patch_token_start` value.

## Metrics

For token `i` between complete layer outputs `l` and `l+1`:

```text
relative_l2 = ||z[l+1] - z[l]||_2 / (||z[l]||_2 + eps)
cosine_distance = 1 - cosine_similarity(z[l+1], z[l])
```

The configured joint low-update mask requires both metrics to be below their
thresholds. The summary also includes strict, conservative, moderate, and loose
threshold sensitivity, plus fractions that remain low-update for at least three
consecutive late-layer transitions.

## Example

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n omega \
  python tools/analyze_token_evolution.py \
  --device cuda:0 \
  --num-frames 3 \
  --frame-source full \
  --sequences \
    rgbd_dataset_freiburg3_sitting_static \
    rgbd_dataset_freiburg3_walking_xyz \
  --labels static dynamic \
  --output-dir outputs/token_evolution_3frame \
  --verify-output
```

Three frames are sampled uniformly from the beginning, middle, and end of each
sequence. `--verify-output` runs an unhooked reference forward and confirms that
pose, depth, and depth-confidence outputs are unchanged.

## Outputs

- `summary.json`: phase means, slopes, persistence, threshold sensitivity, and output-invariance checks
- `layer_stats.csv`: layer-by-layer distributions for all/camera/register/patch tokens
- `<sequence>/token_metrics.npz`: per-token relative-L2, cosine-distance, and low-update arrays
- `layerwise_relative_l2.png`
- `layerwise_cosine_distance.png`
- `layerwise_low_update_fraction.png`

Run the companion spatial visualizer on the saved NPZ files without another
model forward:

```bash
conda run -n omega python tools/visualize_token_evolution_spatial.py \
  outputs/token_evolution_3frame
```

It adds, per sequence:

- patch heatmaps overlaid on every sampled frame at layers 4/8/12/16/20/23;
- late-layer relative-L2, cosine-distance, and low-update-frequency maps;
- raw-order and late-stability-sorted token/layer heatmaps, separately for patch/register/camera tokens;
- lowest-20, highest-20, random-20 patch trajectories, plus every register/camera trajectory;
- per-layer patch-token boxplots and patch/register p10--p90 quantile bands;
- adjacent-layer low-update-set Jaccard curves and CSV (moderate thresholds by default);
- `patch_region_stats.csv` for a 4x4 spatial partition;
- lowest/highest late-update patch coordinates.

The Jaccard implementation uses the standard definition
`|S_l intersection S_(l+1)| / |S_l union S_(l+1)|`. Empty/empty pairs are
reported as NaN rather than being treated as perfect agreement.

Run the substage and register-attention analyzer with the same saved frame selection:

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n omega \
  python tools/analyze_token_substages.py \
  --device cuda:0 \
  --analysis-dir outputs/token_evolution_3frame
```

It adds:

- `substage_stats.csv`: pre-block to post-attention, post-attention to post-MLP,
  and whole-block distributions for frame/global/register/camera-head blocks;
- `substage_updates.png`: median substage curves by token and block type;
- `substage_token_metrics.npz`: exact per-token relative-L2/cosine arrays with
  shape `[24 blocks, 3 substages, batch, frame, token]` for frame and inter
  branches. Register-only inter blocks store patch entries as NaN because those
  patches are not processed;
- `substage_token_heatmap_*.png`: original-order and late-whole-block-sorted
  per-token heatmaps for attention, MLP, and whole-block updates;
- `substage_token_orders.json`: token permutations used by the sorted plots;
- `patch_register_attention.npz`: patch-query attention mass assigned to all
  register keys at every global inter-frame block;
- `patch_register_attention_spatial.png`: per-frame spatial attention-mass maps
  at representative global blocks and for the late-layer mean;
- `patch_register_correlation.png` and `substage_summary.json`: correlation of
  late patch updates with late register-attention mass;
- head norm rows for camera tokens and cached dense-head patch features.

Attention probabilities are recomputed from the read-only global-block input
in query chunks. They are not substituted into the forward pass, and the full
attention matrix is never retained. The reported patch/register interaction is
specifically global inter-frame attention.

## Three-frame preliminary result

Hooked and unhooked outputs are exactly equal (maximum absolute difference 0)
for both tested sequences.

Patch tokens do not monotonically stabilize. Their mean relative-L2 update is
approximately 0.31 early, 0.73 in the middle, and 0.74--0.75 late. Mean cosine
distance rises from approximately 0.05 early to 0.19 in the middle, then remains
around 0.14 late. The final transition has a particularly large update.

Register tokens change substantially less than patch tokens in the middle and
late layers: late relative-L2 is approximately 0.38 and late cosine distance is
approximately 0.037--0.039. Under the moderate threshold (relative-L2 <= 0.25,
cosine distance <= 0.02), 22.9% of static-sequence register tokens and 52.1% of
dynamic-sequence register tokens remain low-update for three consecutive late
transitions. No patch token does so. Under strict or conservative thresholds,
neither token type has persistent late low-update behavior.

The static and dynamic curves are very similar. This two-sequence, three-frame
pilot does not support a robust static/dynamic distinction. More sequences and
spatial motion masks are needed before interpreting the apparent register-token
difference.

Overall, this pilot does not support skipping full computation for a broad set
of patch tokens based only on depth. It does motivate further tests of
register-token-specific or token/layer-conditioned gating, with causal output
quality ablations before treating low update as computational redundancy.

Spatial variation is present even when the layer mean is similar. Depending on
the layer, 4x4 regional mean relative-L2 differs by about 1.16x--1.46x, and the
layer-23 patch p90 is roughly twice its p10. However, spatially low-update patch
events are transient: under the moderate threshold, many image locations are
low at one late transition, but no patch stays low for consecutive late
transitions. Region-aware analysis is therefore necessary, but this pilot does
not yet identify a spatial region that can be safely skipped across layers.

The sorted heatmaps strengthen that conclusion: patch tokens do not form a
contiguous low-update band across the middle/late layers. Moderate-threshold
patch Jaccard is zero wherever either adjacent set is non-empty in the late
range. Register low-update sets are more consistent, but not uniformly so:
mean non-empty late Jaccard is 0.381 for the static sequence and 0.410 for the
dynamic sequence.

The final spike is a real aggregator update rather than a final-normalization
artifact. For the static sequence at block 23, patch median relative-L2 is
0.650 across the complete frame block and 1.583 across the following global
inter-frame block. Inside the global block, attention and MLP median updates
are 0.689 and 0.778 respectively. The dynamic values are similar (0.650 frame,
1.469 global inter-frame). Layer 22 is much smaller (static: 0.284 frame and
0.230 global), so both block-23 attention and MLP genuinely drive the jump.

LayerNorm can strongly change vector scale while barely changing direction.
For example, static dense-head layer-23 normalization has median relative-L2
0.788 but median cosine distance 0.0011. Those norm values are reported
separately and are not included in the aggregator layer heatmaps. Prediction
branches change representation dimension/topology (camera 2048 to 9; patch
tokens to convolutional feature maps), so a token-wise relative update after
those projections is not mathematically comparable and is explicitly marked
undefined in `substage_summary.json`.

Patch-to-register interaction provides only weak evidence in this pilot. Late
patch-update versus late global register-attention correlation is Pearson
0.116/Spearman 0.211 for static and Pearson 0.108/Spearman 0.144 for dynamic.
The lowest and highest update quartiles also have nearly identical mean
register attention (static 0.1229 vs 0.1259; dynamic 0.1230 vs 0.1255). This
does not support register attention as a reliable standalone acceleration
signal, although more sequences are required for a general conclusion.

## Frame-register and global cross-frame attention flow

`tools/analyze_attention_information_flow.py` analyzes every frame block that
is immediately followed by a global inter-frame block:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 conda run -n omega \
  python tools/analyze_attention_information_flow.py \
  --device cuda:0 --analysis-dir outputs/token_evolution_3frame
```

For the frame block it recomputes register-query to patch-key attention and an
`attention * value-vector norm` proxy. For the following global block it saves
per-patch attention mass to every target frame and each query patch's top-5
target-frame patch keys. This produces `<sequence>/attention_flow/` with:

- `attention_flow_metrics.npz`: exact saved arrays for all 19 global layers;
- `frame_register_injection_L*.png`: spatial patch maps read by registers;
- `global_patch_flow_L*.png`: source-patch attention mass to each frame;
- `global_top_correspondences_L*.png`: strongest head-mean cross-frame links;
- `attention_flow_summary.csv` and `metadata.json`.

Register reading is spatially structured but diffuse. Across representative
layers, normalized spatial entropy is about 0.94--0.98, while the highest 10%
of patches receive about 23--35% of register-to-patch attention. Thus registers
do not read only a tiny patch subset. Global patch queries distribute most of
their mass across patch keys in all three frames; frame-level mass is often
close to balanced. The sharpest head-mean cross-frame top-1 links occur around
blocks 11--15. Block 23 is diffuse despite its large hidden-state update, so a
large update is not equivalent to a sharp patch correspondence operation.

`tools/analyze_register_path_coverage.py` composes the preceding frame
register-to-patch attention with global patch-to-register attention and compares
that two-hop distribution against direct cross-frame patch-to-patch attention.
It saves per-query metrics and uncovered direct links under
`<sequence>/register_path_coverage/`. With direct top-1 covered only when it is
inside register rollout top-20, aggregate coverage is 25.95% for static and
27.44% for dynamic. Blocks 12--15 have only about 3--8% coverage, identifying
their strong direct patch correspondences as the main behavior not represented
by the immediate register route. See
`outputs/token_evolution_3frame/register_path_coverage_REPORT.md` for details.

An oracle selective-merge experiment is implemented in
`tools/evaluate_coverage_guided_merge.py`. The merger accepts a per-token
eligibility mask and merges only covered bipartite source candidates. On the
three-frame pilot this is slower and degrades accuracy: all-global-layer merge
is about 0.59x baseline speed, while limiting merge to blocks 12/13/15 is
0.81--0.89x and still substantially reduces pose AUC. Coverage of an attention
route therefore does not imply that the query representation is mergeable.
See `outputs/token_evolution_3frame/coverage_guided_merge_REPORT.md`.

The direction was subsequently corrected: covered source patches are selected
from outgoing direct edges, only their K/V are merged, and all query tokens are
preserved. At blocks 12/13/15 this keeps about 2.1--2.2% source patches and
largely preserves the three-frame metrics, but remains 11--13% slower because
the small attention saving does not amortize selective matching overhead. The
raw result is `source_coverage_kv_merge_L12_L13_L15.json`.
