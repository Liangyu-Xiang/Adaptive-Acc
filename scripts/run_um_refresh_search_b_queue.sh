#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "usage: $0 {1|2|3|4}" >&2
    exit 2
fi
gpu=$1
slot=$((gpu - 1))
repo_root=/data/mmc_syang/VGGT-omega
job=${repo_root}/scripts/run_um_refresh_search_job.sh
index=0
for second in $(seq 7 21); do
    for third in $(seq $((second + 1)) 22); do
        if (( index % 4 == slot )); then
            tag="three_refresh_0_${second}_${third}"
            for dataset in tum 7scenes; do
                bash "${job}" "${dataset}" "0,${second},${third}" "${tag}" "${gpu}"
            done
        fi
        index=$((index + 1))
    done
done
