#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; model=${2:?model}; refresh=${3:?refresh}; lambda=${4:?lambda}
root=/data/mmc_syang
sequence=rgbd_dataset_freiburg3_walking_xyz
safe=${refresh//,/_}
out=${root}/VGGT-omega/outputs/00/${model}/tum300_refresh_l003_l002/${safe}
mkdir -p "${out}"
if [[ ${model} == pi3 ]]; then
  cd ${root}/Pi3
  args=(--dataset tum_dynamic --dataset-root ${root}/dataset/TUM-Dynamics --pretrained checkpoints/Pi3 --output-dir "${out}" --device cuda:0 --sequences "${sequence}" --max-frames-per-seq 300 --frame-sample-mode uniform --load-img-size 512 --timing-repeats 3 --overwrite)
  [[ ${refresh} == none ]] && args+=(--acceleration-method none) || args+=(--acceleration-method u-m --um-lambda "${lambda}" --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers "${refresh}")
  CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/Pi3 ${root}/miniconda3/envs/flow3r/bin/python scripts/run_pi3_vggt_omega_eval.py "${args[@]}"
else
  cd ${root}/vggt
  args=(--dataset tum_dynamic --dataset-root ${root}/dataset/TUM-Dynamics --checkpoint ckpts/model.pt --output-dir "${out}" --device cuda:0 --sequences "${sequence}" --num-frames 300 --image-resolution 518 --timing-repeats 3)
  [[ ${refresh} == none ]] && args+=(--acceleration-method none) || args+=(--acceleration-method u-m --um-lambda "${lambda}" --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers "${refresh}")
  CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/vggt ${root}/miniconda3/envs/fastvggt/bin/python scripts/eval_standard_tum_7scenes.py "${args[@]}"
fi
