#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?source directory}
find "${source_dir}" -name '*.tiff' -print0 | while IFS= read -r -d '' image; do
  target=${image%.tiff}_overview.png
  [[ -f ${target} ]] && continue
  # Display-only: original matrices stay untouched; auto-level makes tiny
  # attention probabilities visible after downsampling.
  convert "${image}" -resize '2048x2048>' -auto-level -depth 8 "${target}"
done
