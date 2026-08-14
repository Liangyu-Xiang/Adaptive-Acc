#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; model=${2:?model}; dataset=${3:?dataset}
base=/data/mmc_syang/VGGT-omega
mkdir -p "${base}/outputs/attention_probe_full/logs"
log="${base}/outputs/attention_probe_full/logs/${model}_${dataset}_gpu${gpu}.log"
bash "${base}/scripts/run_attention_probe_worker.sh" "${gpu}" "${model}" "${dataset}" 5 >"${log}" 2>&1
bash "${base}/scripts/run_attention_probe_worker.sh" "${gpu}" "${model}" "${dataset}" 10 >>"${log}" 2>&1
