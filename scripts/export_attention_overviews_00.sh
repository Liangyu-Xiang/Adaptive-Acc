#!/usr/bin/env bash
set -euo pipefail
model=${1:?pi3|vggt|omega}; dataset=${2:?tum|7scenes}; gap=${3:?5|10}
root=/data/mmc_syang/VGGT-omega
source=${root}/outputs/attention_probe_full/${dataset}/gap_${gap}/${model}
target=${root}/outputs/00/${model}/${dataset}/gap_${gap}
mkdir -p "${target}"
find "${source}" -name '*.tiff' -print0 | while IFS= read -r -d '' matrix; do
  name=$(basename "${matrix}" .tiff)
  [[ -f "${target}/${name}_log_overview.png" ]] && continue
  /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
    "${root}/scripts/render_attention_matrix_views.py" "${matrix}" --output-dir "${target}"
done
