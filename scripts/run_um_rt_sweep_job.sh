#!/usr/bin/env bash
set -euo pipefail
radius=$1
temporal_window=$2
gpu=$3
repo=/data/mmc_syang/VGGT-omega
tag="r${radius}_t${temporal_window}"
out=${repo}/outputs/tum300_um_rt_sweep_lambda_0p04/${tag}
log=${repo}/outputs/tum300_um_rt_sweep_lambda_0p04/logs/${tag}_gpu${gpu}.log
mkdir -p "${out}" "$(dirname "${log}")"
exec env VGGT_UM_TRITON=1 PYTHONUNBUFFERED=1 PYTHONPATH="${repo}" CUDA_VISIBLE_DEVICES="${gpu}" \
  /data/mmc_syang/miniconda3/envs/fastvggt/bin/python "${repo}/scripts/eval_tum_dynamics_paper.py" \
  --data-root /data/mmc_syang/dataset/TUM-Dynamics \
  --checkpoint /data/mmc_syang/vggt-omega/checkpoints/vggt_omega_1b_512.pt \
  --output-dir "${out}" --device cuda:0 --num-frames 300 --sampling-strategy uniform \
  --resize-mode max_size --image-resolution 512 --timing-repeats 3 \
  --require-exclusive-gpu --exclusive-gpu-index "${gpu}" --merge-ratio 0 \
  --frame-fusion-mode u-m --frame-fusion-start-layer -1 --frame-fusion-recompute-layers 0,10,17 \
  --frame-fusion-lambda-cost 0.04 --frame-fusion-merge-top-similarity-percent 100 \
  --frame-fusion-min-keep-ratio 0.05 --frame-fusion-temporal-window "${temporal_window}" \
  --frame-fusion-spatial-radius "${radius}" --frame-fusion-attention-variant representative \
  > "${log}" 2>&1
