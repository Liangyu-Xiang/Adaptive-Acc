#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; first_model=${2:?model}; first_refresh=${3:?refresh}; second_model=${4:?model}; second_refresh=${5:?refresh}
root=/data/mmc_syang/VGGT-omega
logdir=${root}/outputs/00/logs
mkdir -p "${logdir}"
log=${logdir}/tum300_refresh_gpu${gpu}.log
echo "[$(date -Is)] START ${first_model}/${first_refresh}" >"${log}"
bash "${root}/scripts/run_tum300_refresh_single_worker.sh" "${gpu}" "${first_model}" "${first_refresh}" >>"${log}" 2>&1
echo "[$(date -Is)] START ${second_model}/${second_refresh}" >>"${log}"
bash "${root}/scripts/run_tum300_refresh_single_worker.sh" "${gpu}" "${second_model}" "${second_refresh}" >>"${log}" 2>&1
echo "[$(date -Is)] DONE" >>"${log}"
