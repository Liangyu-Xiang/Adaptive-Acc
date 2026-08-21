#!/usr/bin/env bash
# Run one balanced queue from the ScanNet30 fixed-stride-3, 15-method matrix.
set -uo pipefail

gpu=${1:?usage: $0 <gpu> <group-1-to-6>}
group=${2:?}
root=/data/mmc_syang
output_root=${root}/VGGT-omega/outputs/01/scannet30_stride3_full300_15
runner=${root}/VGGT-omega/scripts/run_overnight_45_task.sh
mkdir -p "${output_root}/logs"

run_one() {
  local model=$1
  local method=$2
  local id=${model}__${method}__scannet30
  echo "[$(date -u -Is)] GPU ${gpu}: ${id}" | tee -a "${output_root}/logs/gpu${gpu}.log"
  if ! bash "${runner}" "${gpu}" "${model}" "${method}" scannet30 "${output_root}/${id}" \
      >"${output_root}/logs/${id}.log" 2>&1; then
    echo "[$(date -u -Is)] FAILED: ${id}" | tee -a "${output_root}/logs/gpu${gpu}.log"
  fi
}

case ${group} in
  1) run_one vggt baseline ;;
  2) run_one vggt fastvggt; run_one vggt u-m; run_one omega fastvggt ;;
  3) run_one vggt da-vggt; run_one pi3 baseline; run_one omega sparse-vggt ;;
  4) run_one pi3 fastvggt; run_one pi3 sparse-vggt; run_one omega da-vggt ;;
  5) run_one pi3 u-m; run_one pi3 da-vggt ;;
  6) run_one omega baseline; run_one omega u-m; run_one vggt sparse-vggt ;;
  *) echo "group must be 1 through 6" >&2; exit 2 ;;
esac
