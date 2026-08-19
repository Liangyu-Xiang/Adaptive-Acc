#!/usr/bin/env bash
# Wait for a tmux recovery queue to finish, then recover one task on its GPU.
set -euo pipefail

wait_session=${1:?usage: $0 <tmux-session> <gpu> <model> <method> <dataset>}
gpu=${2:?}
model=${3:?}
method=${4:?}
dataset=${5:?}
root=/data/mmc_syang
while tmux has-session -t "${wait_session}" 2>/dev/null; do
  sleep 30
done
exec bash "${root}/VGGT-omega/scripts/run_overnight_45_recovery_one.sh" "${gpu}" "${model}" "${method}" "${dataset}"
