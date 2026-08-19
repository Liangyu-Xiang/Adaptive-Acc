#!/usr/bin/env bash
# Recover one method's four failed overnight tasks serially on one GPU.
set -uo pipefail

gpu=${1:?usage: $0 <gpu> <baseline|fastvggt|sparse-vggt|da-vggt|u-m>}
method=${2:?}
root=/data/mmc_syang
out=${root}/VGGT-omega/outputs/01/overnight_45_7scenes_nrgbd_scannet30
runner=${root}/VGGT-omega/scripts/run_overnight_45_task.sh

run_one() {
  local model=$1
  local dataset=$2
  local id=${model}__${method}__${dataset}
  local status
  [[ -f ${out}/status/${id}.failed ]] && mv "${out}/status/${id}.failed" "${out}/status/${id}.running"
  echo "[$(date -u -Is)] GPU ${gpu}: recovering ${id}" >>"${out}/logs/recovery_gpu${gpu}.log"
  if bash "${runner}" "${gpu}" "${model}" "${method}" "${dataset}" "${out}/${id}" >"${out}/logs/${id}.recovery.log" 2>&1; then
    mv "${out}/status/${id}.running" "${out}/status/${id}.done"
  else
    status=$?
    mv "${out}/status/${id}.running" "${out}/status/${id}.failed"
    echo "[$(date -u -Is)] ${id} failed with ${status}" >>"${out}/logs/recovery_gpu${gpu}.log"
  fi
}

# VGGT Scannet30 is the largest remaining task; run it first. The three Omega
# tasks follow in descending expected duration.
run_one vggt scannet30
run_one omega scannet30
run_one omega 7scenes
run_one omega nrgbd
