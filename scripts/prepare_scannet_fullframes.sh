#!/usr/bin/env bash
# Decode ScanNet .sens files while preserving SensReader's native layout.
set -euo pipefail

root=/data/mmc_syang
source_root=${root}/dataset/scannet_test/scans_test
output_root=${root}/dataset/scannet_fullframes
reader=""
frame_skip=1
python_bin=${root}/miniconda3/envs/fastvggt/bin/python
jobs=4

usage() {
  echo "Usage: $0 --reader /path/to/SensReader/python/reader.py [--source DIR] [--output DIR] [--frame-skip N] [--jobs N] [--python-bin PYTHON]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reader) reader=$2; shift 2 ;;
    --source) source_root=$2; shift 2 ;;
    --output) output_root=$2; shift 2 ;;
    --frame-skip) frame_skip=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --python-bin) python_bin=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n ${reader} && -f ${reader} ]] || { echo "--reader must point to official SensReader/python/reader.py" >&2; exit 2; }
[[ -x ${python_bin} ]] || { echo "Python does not exist or is not executable: ${python_bin}" >&2; exit 2; }
[[ -d ${source_root} ]] || { echo "Source does not exist: ${source_root}" >&2; exit 2; }
[[ ${frame_skip} =~ ^[1-9][0-9]*$ ]] || { echo "--frame-skip must be a positive integer" >&2; exit 2; }
[[ ${frame_skip} == 1 ]] || { echo "This official reader.py exports every frame and has no frame-skip option; use --frame-skip 1." >&2; exit 2; }
[[ ${jobs} =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }

mkdir -p "${output_root}/raw" "${output_root}/logs"
mapfile -t sens_files < <(find "${source_root}" -mindepth 2 -maxdepth 2 -name '*.sens' -type f | sort)
total=${#sens_files[@]}
(( total > 0 )) || { echo "No .sens files below ${source_root}" >&2; exit 1; }

export output_root reader python_bin total
decode_one() {
  local index=$1 sens=$2 scene raw_dir scene_marker log
  scene=$(basename "${sens}" .sens)
  raw_dir="${output_root}/raw/${scene}"
  scene_marker="${raw_dir}/.complete"
  log="${output_root}/logs/${scene}.log"
  if [[ -f "${scene_marker}" ]]; then
    echo "[$((index + 1))/${total}] ${scene}: already complete"
    return 0
  fi
  echo "[$((index + 1))/${total}] ${scene}: decoding"
  mkdir -p "${raw_dir}"
  if "${python_bin}" "${reader}" --filename "${sens}" --output_path "${raw_dir}" \
      --export_color_images --export_depth_images --export_poses --export_intrinsics \
      >"${log}" 2>&1; then
    touch "${scene_marker}"
    echo "[$((index + 1))/${total}] ${scene}: complete"
  else
    echo "[$((index + 1))/${total}] ${scene}: FAILED (see ${log})" >&2
    return 1
  fi
}
export -f decode_one

for i in "${!sens_files[@]}"; do
  printf '%s\t%s\n' "${i}" "${sens_files[$i]}"
done | xargs -r -P "${jobs}" -n 2 bash -c 'decode_one "$@"' _

echo "Done. Native SensReader scenes are in: ${output_root}/raw"
