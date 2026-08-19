#!/usr/bin/env bash
set -euo pipefail

root=/data/mmc_syang/VGGT-omega
out=${root}/outputs/01/omega_7scenes_lambda_sweep_001_010_300
mkdir -p "${out}/logs"

launch_worker() {
  local gpu=$1; shift
  local log="${out}/logs/gpu${gpu}.log"
  nohup bash -lc 'for job in "$@"; do
    echo "[$(date -Is)] start ${job}"
    bash "'$root'/scripts/run_omega_7scenes_lambda_sweep_one.sh" "'$gpu'" "${job}"
    echo "[$(date -Is)] finished ${job}"
  done' bash "$@" >"${log}" 2>&1 < /dev/null &
  echo $! >"${out}/logs/gpu${gpu}.pid"
}

launch_worker 1 baseline um_0p01 um_0p06
launch_worker 3 fastvggt um_0p02 um_0p07
launch_worker 5 sparse-vggt um_0p03 um_0p08
launch_worker 6 da-vggt um_0p04 um_0p09
launch_worker 7 um_0p05 um_0p10
