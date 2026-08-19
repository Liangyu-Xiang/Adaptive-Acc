#!/usr/bin/env bash
set -euo pipefail
root=/data/mmc_syang/VGGT-omega
gpus=(0 1 3 4 5 6 7)
refreshes=(0,4,13 0,10,16 0,4,10 0,4,16 0,5,10 0,5,16 0,5,13)
logdir=${root}/outputs/01/omega_um_refresh_sweep_7configs_tum_uniform300/launcher_logs
mkdir -p "${logdir}"
for index in "${!gpus[@]}"; do
  gpu=${gpus[$index]}; refresh=${refreshes[$index]}
  nohup setsid bash "${root}/scripts/run_omega_um_refresh_sweep_one.sh" "${gpu}" "${refresh}" < /dev/null > /dev/null 2>&1 &
  echo $! > "${logdir}/refresh_${refresh//,/_}_gpu${gpu}.pid"
done
