#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; shift
root=/data/mmc_syang/VGGT-omega
logdir=${root}/outputs/00/logs
mkdir -p "${logdir}"
log=${logdir}/tum300_refresh_l003_l002_gpu${gpu}.log
: >"${log}"
for task in "$@"; do
  IFS=: read -r model refresh lambda <<<"${task}"
  echo "[$(date -Is)] START ${model}/${refresh}, lambda=${lambda}" | tee -a "${log}"
  bash "${root}/scripts/run_tum300_refresh_lambda_single.sh" "${gpu}" "${model}" "${refresh}" "${lambda}" >>"${log}" 2>&1
done
echo "[$(date -Is)] DONE" | tee -a "${log}"
