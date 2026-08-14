#!/usr/bin/env bash
set -euo pipefail
repo_root=/data/mmc_syang/VGGT-omega
job=${repo_root}/scripts/run_um_refresh_search_job.sh
for second in $(seq 7 22); do
    tag="two_refresh_0_${second}"
    for dataset in tum 7scenes; do
        bash "${job}" "${dataset}" "0,${second}" "${tag}" 0
    done
done
