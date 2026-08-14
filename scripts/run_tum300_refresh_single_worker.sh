#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; model=${2:?pi3|vggt}; refresh=${3:?none|0,8|0,9|0,8,16|0,8,17|0,9,16|0,9,17}
root=/data/mmc_syang
sequence=rgbd_dataset_freiburg3_walking_xyz
safe=${refresh//,/_}
out=${root}/VGGT-omega/outputs/00/${model}/tum300_refresh/${safe}
mkdir -p "${out}"
if [[ ${model} == pi3 ]]; then
  cd ${root}/Pi3
  args=(--dataset tum_dynamic --dataset-root ${root}/dataset/TUM-Dynamics --pretrained checkpoints/Pi3 --output-dir "${out}" --device cuda:0 --sequences "${sequence}" --max-frames-per-seq 300 --frame-sample-mode uniform --load-img-size 512 --timing-repeats 3 --overwrite)
  if [[ ${refresh} == none ]]; then
    args+=(--acceleration-method none)
  else
    args+=(--acceleration-method u-m --um-lambda 0.04 --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers "${refresh}")
  fi
  CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/Pi3 ${root}/miniconda3/envs/flow3r/bin/python scripts/run_pi3_vggt_omega_eval.py "${args[@]}"
else
  cd ${root}/vggt
  args=(--dataset tum_dynamic --dataset-root ${root}/dataset/TUM-Dynamics --checkpoint ckpts/model.pt --output-dir "${out}" --device cuda:0 --sequences "${sequence}" --num-frames 300 --image-resolution 518 --timing-repeats 3)
  if [[ ${refresh} == none ]]; then
    args+=(--acceleration-method none)
  else
    args+=(--acceleration-method u-m --um-lambda 0.04 --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers "${refresh}")
  fi
  CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/vggt ${root}/miniconda3/envs/fastvggt/bin/python scripts/eval_standard_tum_7scenes.py "${args[@]}"
fi
