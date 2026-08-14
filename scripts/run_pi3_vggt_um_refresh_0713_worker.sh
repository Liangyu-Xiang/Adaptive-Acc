#!/usr/bin/env bash
# One physical GPU worker: Pi3 followed by VGGT for the same lambda/dataset.
set -euo pipefail

gpu=${1:?usage: $0 GPU LAMBDA tum|7scenes}
lambda=${2:?usage: $0 GPU LAMBDA tum|7scenes}
tag=${3:?usage: $0 GPU LAMBDA tum|7scenes}

root=/data/mmc_syang
outroot=${root}/VGGT-omega/outputs/pi3_vggt_um_refresh_0_7_13_lambda_sweep_300
lambda_tag=${lambda/./p}
[[ ${tag} == tum ]] && data=tum_dynamic || data=7scenes
[[ ${tag} == tum ]] && data_root=${root}/dataset/TUM-Dynamics || data_root=${root}/dataset/7scenes
log=${outroot}/logs/worker_lambda_${lambda_tag}_${tag}_gpu${gpu}.log
mkdir -p "$(dirname "${log}")"

echo "[$(date -Is)] START Pi3 GPU=${gpu} lambda=${lambda} dataset=${tag}" >"${log}"
cd ${root}/Pi3
CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/Pi3 \
  ${root}/miniconda3/envs/flow3r/bin/python scripts/run_pi3_vggt_omega_eval.py \
  --dataset "${data}" --dataset-root "${data_root}" --pretrained checkpoints/Pi3 \
  --output-dir "${outroot}/pi3/lambda_${lambda_tag}/${tag}" --device cuda:0 \
  --max-frames-per-seq 300 --frame-sample-mode uniform --load-img-size 512 --timing-repeats 3 \
  --acceleration-method u-m --um-lambda "${lambda}" --um-spatial-radius 2 --um-temporal-window 4 \
  --um-refresh-layers 0,7,13 --overwrite >>"${log}" 2>&1

echo "[$(date -Is)] DONE Pi3; START VGGT" >>"${log}"
cd ${root}/vggt
CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/vggt \
  ${root}/miniconda3/envs/fastvggt/bin/python scripts/eval_standard_tum_7scenes.py \
  --dataset "${data}" --dataset-root "${data_root}" --checkpoint ckpts/model.pt \
  --output-dir "${outroot}/vggt/lambda_${lambda_tag}/${tag}" --device cuda:0 \
  --num-frames 300 --image-resolution 518 --timing-repeats 3 --acceleration-method u-m \
  --merge-ratio 0.9 --um-lambda "${lambda}" --um-spatial-radius 2 --um-temporal-window 4 \
  --um-refresh-layers 0,7,13 --overwrite >>"${log}" 2>&1
echo "[$(date -Is)] DONE VGGT" >>"${log}"
