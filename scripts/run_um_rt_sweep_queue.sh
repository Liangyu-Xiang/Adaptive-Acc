#!/usr/bin/env bash
set -euo pipefail
gpu=$1
slot=$2
repo=/data/mmc_syang/VGGT-omega
job=${repo}/scripts/run_um_rt_sweep_job.sh
index=0
for radius in 1 2 3; do
  for temporal_window in 2 3 4 5 6; do
    if (( index % 2 == slot )); then
      bash "${job}" "${radius}" "${temporal_window}" "${gpu}"
    fi
    index=$((index + 1))
  done
done
