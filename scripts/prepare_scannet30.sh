#!/usr/bin/env bash
# Materialize the 30 longest ScanNet test streams as an independent dataset.
# No symbolic links are created; each selected native SensReader directory is copied.
set -euo pipefail

source_root=/data/mmc_syang/dataset/scannet_fullframes/raw
output_root=/data/mmc_syang/dataset/scannet30
num_sequences=30

usage() {
  echo "Usage: $0 [--source RAW_ROOT] [--output DATASET_ROOT] [--num N]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) source_root=$2; shift 2 ;;
    --output) output_root=$2; shift 2 ;;
    --num) num_sequences=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -d ${source_root} ]] || { echo "Missing source root: ${source_root}" >&2; exit 2; }
[[ ${num_sequences} =~ ^[1-9][0-9]*$ ]] || { echo "--num must be positive" >&2; exit 2; }
command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 2; }

mkdir -p "${output_root}/raw"
mapfile -t selected < <(
  for scene_dir in "${source_root}"/scene*; do
    [[ -d ${scene_dir}/color && -f ${scene_dir}/.complete ]] || continue
    frames=$(find "${scene_dir}/color" -maxdepth 1 -type f -name '*.jpg' | wc -l)
    printf '%010d\t%s\n' "${frames}" "${scene_dir}"
  done | sort -rn | head -n "${num_sequences}"
)

(( ${#selected[@]} == num_sequences )) || {
  echo "Expected ${num_sequences} complete scenes, found ${#selected[@]}" >&2
  exit 1
}

manifest_tmp=$(mktemp)
trap 'rm -f "${manifest_tmp}"' EXIT
printf 'rank\tscene\tframes\n' >"${manifest_tmp}"

for i in "${!selected[@]}"; do
  row=${selected[$i]}
  frames=$((10#${row%%$'\t'*}))
  scene_dir=${row#*$'\t'}
  scene=$(basename "${scene_dir}")
  target_dir="${output_root}/raw/${scene}"
  printf '%d\t%s\t%d\n' "$((i + 1))" "${scene}" "${frames}" >>"${manifest_tmp}"

  if [[ -f ${target_dir}/.complete ]]; then
    echo "[$((i + 1))/${num_sequences}] ${scene}: already complete (${frames} frames)"
    continue
  fi
  echo "[$((i + 1))/${num_sequences}] ${scene}: copying ${frames} frames"
  mkdir -p "${target_dir}"
  rsync -a --partial --info=progress2 "${scene_dir}/" "${target_dir}/"
  copied=$(find "${target_dir}/color" -maxdepth 1 -type f -name '*.jpg' | wc -l)
  [[ ${copied} == ${frames} ]] || {
    echo "${scene}: copied ${copied}/${frames} RGB frames; rerun to resume" >&2
    exit 1
  }
  touch "${target_dir}/.complete"
  echo "[$((i + 1))/${num_sequences}] ${scene}: complete"
done

mv "${manifest_tmp}" "${output_root}/selected_sequences.tsv"
trap - EXIT
echo "Done: ${output_root}/raw (${num_sequences} sequences)"
