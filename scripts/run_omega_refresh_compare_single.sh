#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; refresh=${2:?refresh-layers}
root=/data/mmc_syang
seq=rgbd_dataset_freiburg3_walking_rpy
tag=${refresh//\//_}
out=${root}/VGGT-omega/outputs/refresh_041315_vs_01017_single_uniform300/refresh_${tag}
log=${root}/VGGT-omega/outputs/refresh_041315_vs_01017_single_uniform300/logs/refresh_${tag}_gpu${gpu}.log
mkdir -p "$(dirname "${log}")"
cd ${root}/VGGT-omega
CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/VGGT-omega \
  ${root}/miniconda3/envs/fastvggt/bin/python scripts/eval_tum_dynamics_paper.py \
  --data-root ${root}/dataset/TUM-Dynamics --sequences "${seq}" \
  --checkpoint ${root}/vggt-omega/checkpoints/vggt_omega_1b_512.pt \
  --output-dir "${out}" --device cuda:0 --num-frames 300 \
  --sampling-pool full --sampling-strategy uniform --image-resolution 512 --resize-mode max_size \
  --timing-repeats 3 --acceleration-method u-m --merge-ratio 0 --frame-fusion-mode u-m \
  --frame-fusion-recompute-layers "${refresh}" --frame-fusion-lambda-cost 0.03 \
  --frame-fusion-temporal-window 4 --frame-fusion-spatial-radius 2 \
  --frame-fusion-attention-variant representative >"${log}" 2>&1
