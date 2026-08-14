#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; shift
root=/data/mmc_syang/VGGT-omega
log=${root}/outputs/00/500frame_smoke_round2/log_gpu${gpu}.log
mkdir -p "$(dirname "${log}")"
: >"${log}"
for task in "$@"; do
  IFS=: read -r model method <<<"${task}"
  echo "[$(date -Is)] START ${model}/${method}" | tee -a "${log}"
  if bash "${root}/scripts/run_500frame_oom_smoke_one.sh" "${gpu}" "${model}" "${method}" >>"${log}" 2>&1; then
    echo "[$(date -Is)] PASS ${model}/${method}" | tee -a "${log}"
  else
    status=$?
    echo "[$(date -Is)] FAIL(${status}) ${model}/${method}; continuing next task" | tee -a "${log}"
  fi
done
