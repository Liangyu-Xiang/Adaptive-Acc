#!/usr/bin/env bash
set -euo pipefail
gpu=${1:-7}
root=/data/mmc_syang
out=${root}/VGGT-omega/outputs/refresh_041315_tum_full_uniform300
log=${out}/run_gpu${gpu}.log
mkdir -p "${out}"
cd ${root}/VGGT-omega
CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/VGGT-omega \
  ${root}/miniconda3/envs/fastvggt/bin/python scripts/eval_tum_dynamics_paper.py \
  --data-root ${root}/dataset/TUM-Dynamics --checkpoint ${root}/vggt-omega/checkpoints/vggt_omega_1b_512.pt \
  --output-dir "${out}" --device cuda:0 --num-frames 300 --sampling-pool full --sampling-strategy uniform \
  --image-resolution 512 --resize-mode max_size --timing-repeats 3 --acceleration-method u-m \
  --merge-ratio 0 --frame-fusion-mode u-m --frame-fusion-recompute-layers 0,4,13,15 \
  --frame-fusion-lambda-cost 0.03 --frame-fusion-temporal-window 4 --frame-fusion-spatial-radius 2 \
  --frame-fusion-attention-variant representative >"${log}" 2>&1
