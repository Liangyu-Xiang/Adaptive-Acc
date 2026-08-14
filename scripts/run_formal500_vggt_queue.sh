#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; method=${2:?none|da-vggt}
root=/data/mmc_syang
log=${root}/VGGT-omega/outputs/00/logs/formal500_vggt_${method}_gpu${gpu}.log
for dataset in tum 7scenes; do
  [[ ${dataset} == tum ]] && data=tum_dynamic || data=7scenes
  [[ ${dataset} == tum ]] && data_root=${root}/dataset/TUM-Dynamics || data_root=${root}/dataset/7scenes
  out=${root}/VGGT-omega/outputs/00/formal500_vggt/${method}/${dataset}
  mkdir -p "${out}"
  echo "[$(date -Is)] START ${method}/${dataset}" >>"${log}"
  cd ${root}/vggt
  CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=${root}/vggt ${root}/miniconda3/envs/fastvggt/bin/python scripts/eval_standard_tum_7scenes.py \
    --dataset "${data}" --dataset-root "${data_root}" --checkpoint ckpts/model.pt --output-dir "${out}" \
    --device cuda:0 --num-frames 500 --image-resolution 518 --timing-repeats 3 --acceleration-method "${method}" --da-chunk-size 50 >>"${log}" 2>&1
  echo "[$(date -Is)] DONE ${method}/${dataset}" >>"${log}"
done
