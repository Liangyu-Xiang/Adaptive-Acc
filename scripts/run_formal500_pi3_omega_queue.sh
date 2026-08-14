#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; shift
root=/data/mmc_syang/VGGT-omega
log=${root}/outputs/00/logs/formal500_pi3_omega_gpu${gpu}.log
mkdir -p "$(dirname "${log}")"
: >"${log}"
for task in "$@"; do
  IFS=: read -r model method dataset <<<"${task}"
  echo "[$(date -Is)] START ${model}/${method}/${dataset}" | tee -a "${log}"
  bash "${root}/scripts/run_formal500_pi3_omega_one.sh" "${gpu}" "${model}" "${method}" "${dataset}" >>"${log}" 2>&1
  echo "[$(date -Is)] DONE ${model}/${method}/${dataset}" | tee -a "${log}"
done
