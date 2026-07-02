# VGGT-Omega long-sequence scaling diagnostic

This diagnostic runs the unchanged VGGT-Omega model jointly on uniformly sampled
TUM-Dynamics frames. It separates RGB preprocessing, CPU-to-GPU transfer, and pure
model-forward time, and records CUDA allocated/reserved memory peaks.

## Dataset layout

`--data-root` must contain the original TUM sequence directories. Each directory
needs `rgb.txt`, `depth.txt`, `groundtruth.txt`, and their referenced `rgb/` and
`depth/` images. Sequence arguments may omit the `rgbd_dataset_` prefix.

## Commands

Profile only (default):

```bash
CUDA_VISIBLE_DEVICES=7 python tools/long_sequence_scaling.py \
  --data_root /mnt/nasdata/xly/dataset/TUM-Dynamics \
  --sequences freiburg3_walking_xyz freiburg3_walking_static \
  --model_path pretrained_ckpts/vggt_omega_1b_512.pt \
  --frame_counts 2 4 8 10 16 32 64 128 \
  --output_dir outputs/scaling_diagnostic \
  --device cuda:0 --num_repeats 3 --profile_only
```

Add `--eval` to compute Sim(3)-aligned camera ATE, adjacent-frame translation and
rotation RPE, depth AbsRel, and depth delta<1.25. Add `--save_predictions` to save
camera/depth arrays once per sequence/frame count. Use `--frame_counts ... 256` to
try 256 frames; OOM is recorded and does not abort later settings.

Generate plots after the run:

```bash
python tools/plot_scaling_results.py outputs/scaling_diagnostic/scaling_summary.csv
```

## Outputs and columns

- `scaling_results.csv/json`: one row per repeat. `status` is `success`, `oom`, or
  `error`; `error_message` preserves the failure reason. Memory is decimal GB.
- `scaling_summary.csv`: success rate plus mean/std timing, memory, and available
  metrics across successful repeats.
- `sampled_frames.json`: exact deterministic uniform selections.
- `scaling.log`: progress and failure log.
- `predictions/{sequence}/N_{count}/predictions.npz`: optional predicted depth,
  confidence, intrinsics, extrinsics, and timestamps.
- `time_vs_frames.png`, `memory_vs_frames.png`, and (with metrics)
  `metric_vs_frames.png`: plots produced by the plotting utility.

`inference_time_sec` covers only `model(images)` and is synchronized before and
after. `preprocessing_time_sec` and `input_transfer_time_sec` are reported
separately. `camera_ate` and `translation_error` are metres; `rotation_error` is
degrees; `depth_delta_1_25` is a fraction in `[0,1]`.
