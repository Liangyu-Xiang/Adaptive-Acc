#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 PHYSICAL_GPU" >&2
    exit 2
fi

physical_gpu=$1
repo_root=/data/mmc_syang/VGGT-omega
job_script=${repo_root}/scripts/run_cube_phase2_sweep_job.sh

case ${physical_gpu} in
    4)
        jobs=("7scenes baseline")
        ;;
    5)
        jobs=("7scenes u002" "7scenes u004")
        ;;
    6)
        jobs=("7scenes u003" "7scenes fastvggt" "tum u004")
        ;;
    7)
        jobs=("tum u002" "tum u003" "tum baseline" "tum fastvggt")
        ;;
    *)
        echo "this sweep queue is defined only for physical GPUs 4-7" >&2
        exit 2
        ;;
esac

for job in "${jobs[@]}"; do
    read -r dataset method <<< "${job}"
    bash "${job_script}" "${dataset}" "${method}" "${physical_gpu}"
done
