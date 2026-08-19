#!/usr/bin/env bash
# Recover one named overnight task on a specified GPU.
set -euo pipefail

gpu=${1:?usage: $0 <gpu> <model> <method> <dataset>}
model=${2:?}
method=${3:?}
dataset=${4:?}
root=/data/mmc_syang
out=${root}/VGGT-omega/outputs/01/overnight_45_7scenes_nrgbd_scannet30
runner=${root}/VGGT-omega/scripts/run_overnight_45_task.sh
id=${model}__${method}__${dataset}

[[ -f ${out}/status/${id}.failed ]] && mv "${out}/status/${id}.failed" "${out}/status/${id}.running"
echo "[$(date -u -Is)] GPU ${gpu}: recovering ${id}" >>"${out}/logs/recovery_gpu${gpu}.log"
if bash "${runner}" "${gpu}" "${model}" "${method}" "${dataset}" "${out}/${id}" >"${out}/logs/${id}.recovery.log" 2>&1; then
  mv "${out}/status/${id}.running" "${out}/status/${id}.done"
else
  status=$?
  mv "${out}/status/${id}.running" "${out}/status/${id}.failed"
  echo "[$(date -u -Is)] ${id} failed with ${status}" >>"${out}/logs/recovery_gpu${gpu}.log"
  exit "${status}"
fi
