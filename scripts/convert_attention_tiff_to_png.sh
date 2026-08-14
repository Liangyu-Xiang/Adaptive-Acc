#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?source directory}
find "${source_dir}" -name '*.tiff' -print0 | while IFS= read -r -d '' image; do
  target=${image%.tiff}.png
  [[ -f ${target} ]] || convert "${image}" -depth 16 "${target}"
done
