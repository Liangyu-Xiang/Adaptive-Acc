#!/usr/bin/env bash
set -euo pipefail
root=/data/mmc_syang/VGGT-omega
mkdir -p "${root}/outputs/refresh_041315_vs_01017_single_uniform300/logs"
nohup setsid bash "${root}/scripts/run_omega_refresh_compare_single.sh" 6 0,4,13,15 < /dev/null &
echo $! > "${root}/outputs/refresh_041315_vs_01017_single_uniform300/logs/refresh_0_4_13_15_gpu6.pid"
nohup setsid bash "${root}/scripts/run_omega_refresh_compare_single.sh" 7 0,10,17 < /dev/null &
echo $! > "${root}/outputs/refresh_041315_vs_01017_single_uniform300/logs/refresh_0_10_17_gpu7.pid"
