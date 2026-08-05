# 7 Scenes paper evaluation

`scripts/eval_7scenes_paper.py` evaluates the released VGGT-Omega 1B model
against Tables 1 and 2 of the paper. It reads the official `TestSplit.txt`
files and defaults to first/last-preserving uniform frame sampling. Pass
`--sampling-strategy random` to use the older NumPy `RandomState(42)` loader
convention. The paper's wording, "for each scene or sequence", is ambiguous; use
`--sampling-unit scene` to run the seven-physical-scene interpretation.
For long-sequence experiments, pass `--num-frames 500`; the default sampler keeps
the first and last frame in each sampling pool and selects the middle frames
with deterministic `linspace` spacing.

The expected paper values for Ours-1B are AUC@3 = 29.6, AUC@30 = 83.1,
delta1.25 = 94.6, and AbsRel = 0.058. Exact reproduction is not guaranteed
because the paper does not publish its sampled frame IDs.

The RGB-registered depth files are shared with FastVGGT. Generate them once if
needed:

```bash
cd /data/mmc_lyxiang/3D/FastVGGT
conda run -n omega python scripts/prepare_7scenes.py \
  /data/mmc_lyxiang/dataset/7scenes --workers 8
```

Validate the split and selected files without loading the model:

```bash
cd /data/mmc_lyxiang/3D/VGGT-omega
conda run -n omega python scripts/eval_7scenes_paper.py --dry-run
```

Run the complete evaluation on a free GPU:

```bash
conda run -n omega python scripts/eval_7scenes_paper.py \
  --device cuda:5 \
  --data-root /data/mmc_lyxiang/dataset/7scenes \
  --output-dir outputs/7scenes_paper
```

The paper baseline explicitly uses `--merge-ratio 0`. This prevents local
FastVGGT token-merging experiments from silently changing the benchmark. The
script writes the sampled frame list, aggregate/per-sequence metrics, and raw
pose errors under the output directory. Following standard monocular-depth
evaluation, predictions are median-scale aligned per frame and clipped to the
valid 7 Scenes depth range of 0.2--10 metres before metrics are computed. The
0.2 m lower bound is below the Kinect's usable range and also filters tiny
wrapped values that can be produced when a depth projector does not discard
the raw `65535` invalid-depth sentinel before converting back to `uint16`.
