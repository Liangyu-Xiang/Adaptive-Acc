#!/usr/bin/env bash
# Run two formal VGGT 7Scenes evaluations serially on one physical GPU.
set -uo pipefail

gpu=${1:?usage: $0 <gpu> <first-label> <first-method> <first-lambda-or-dash> <second-label> <second-method> <second-lambda-or-dash>}
label_a=${2:?}
method_a=${3:?}
lambda_a=${4:?}
label_b=${5:?}
method_b=${6:?}
lambda_b=${7:?}

root=/data/mmc_syang
repo=${root}/vggt
out=${VGGT_7SCENES_OUT:-${root}/VGGT-omega/outputs/01/vggt_7scenes_refresh0921_full300}
mkdir -p "${out}/logs"

run_one() {
  local label=$1 method=$2 lambda=$3
  local args=(
    scripts/eval_standard_tum_7scenes.py
    --dataset 7scenes
    --dataset-root "${root}/dataset/7scenes"
    --checkpoint ckpts/model.pt
    --output-dir "${out}/${label}"
    --device cuda:0
    --num-frames 300
    --image-resolution 518
    --timing-repeats 3
    --acceleration-method "${method}"
    --merge-ratio 0.9
    --sparse-vggt-sparse-ratio 0.5
    --da-chunk-size 50
    --um-refresh-layers 0,9,21
  )
  if [[ ${method} == u-m ]]; then
    args+=(--um-lambda "${lambda}" --um-spatial-radius 2 --um-temporal-window 4)
  fi
  echo "[$(date -Is)] starting ${label} on GPU ${gpu}"
  (
    cd "${repo}" || exit 1
    CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${repo} \
      ${root}/miniconda3/envs/fastvggt/bin/python "${args[@]}"
  ) >"${out}/logs/${label}.log" 2>&1
  status=$?
  echo "[$(date -Is)] ${label} exited with status ${status}"
  return 0
}

run_one "${label_a}" "${method_a}" "${lambda_a}"
run_one "${label_b}" "${method_b}" "${lambda_b}"
