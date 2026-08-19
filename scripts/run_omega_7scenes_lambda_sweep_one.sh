#!/usr/bin/env bash
set -euo pipefail

# usage: GPU {baseline|fastvggt|sparse-vggt|da-vggt|um_0pXX}
gpu=${1:?physical GPU index}
job=${2:?job name}
root=/data/mmc_syang
repo=${root}/VGGT-omega
out_root=${repo}/outputs/01/omega_7scenes_lambda_sweep_001_010_300
out=${out_root}/${job}
mkdir -p "${out}"

args=(
  --data-root "${root}/dataset/7scenes"
  --checkpoint "${root}/vggt-omega/checkpoints/vggt_omega_1b_512.pt"
  --output-dir "${out}"
  --device cuda:0
  --num-frames 300
  --sampling-unit sequence
  --sampling-strategy uniform
  --image-resolution 512
  --resize-mode max_size
  --timing-repeats 3
)

case "${job}" in
  baseline) args+=(--acceleration-method none --merge-ratio 0 --frame-fusion-mode none) ;;
  fastvggt) args+=(--acceleration-method fastvggt --merge-ratio 0.9 --frame-fusion-mode none) ;;
  sparse-vggt) args+=(--acceleration-method sparse-vggt --merge-ratio 0 --frame-fusion-mode none --sparse-attention --sparse-ratio 0.5 --sparse-pool-mode avg) ;;
  da-vggt) args+=(--acceleration-method da-vggt --merge-ratio 0 --frame-fusion-mode none --da-chunk-size 50) ;;
  um_0p*)
    lambda=${job#um_0p}
    lambda="0.${lambda}"
    args+=(--acceleration-method u-m --merge-ratio 0 --frame-fusion-mode u-m --frame-fusion-start-layer -1 --frame-fusion-recompute-layers 0,10,17 --frame-fusion-lambda-cost "${lambda}" --frame-fusion-merge-top-similarity-percent 100 --frame-fusion-min-keep-ratio 0.05 --frame-fusion-temporal-window 4 --frame-fusion-spatial-radius 2 --frame-fusion-attention-variant representative)
    ;;
  *) echo "unknown job: ${job}" >&2; exit 2 ;;
esac

cd "${repo}"
exec env VGGT_UM_TRITON=1 PYTHONUNBUFFERED=1 PYTHONPATH="${repo}" CUDA_VISIBLE_DEVICES="${gpu}" \
  "${root}/miniconda3/envs/fastvggt/bin/python" scripts/eval_7scenes_paper.py "${args[@]}"
