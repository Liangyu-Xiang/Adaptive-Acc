#!/usr/bin/env bash
# Calibrated overnight scheduler: reality 23:00 Beijing = 15:00 UTC on this server.
set -euo pipefail

root=/data/mmc_syang
runner=${root}/VGGT-omega/scripts/run_overnight_45_task.sh
out=${root}/VGGT-omega/outputs/01/overnight_45_7scenes_nrgbd_scannet30
target_utc=${TARGET_UTC:-'2026-08-18 15:00:00 UTC'}
mkdir -p "${out}/status" "${out}/logs"

now=$(date -u +%s); target=$(date -u -d "${target_utc}" +%s)
if (( target > now )); then
  echo "Waiting until ${target_utc} ($((${target}-${now})) seconds)."
  sleep "$((target-now))"
fi

tasks=${out}/tasks.tsv
if [[ ! -f ${tasks} ]]; then
  : >"${tasks}"
  for model in omega vggt pi3; do
    case ${model} in omega) base=20;; vggt) base=76;; pi3) base=46;; esac
    for dataset in 7scenes nrgbd scannet30; do
      case ${dataset} in 7scenes) sequences=18;; nrgbd) sequences=9;; scannet30) sequences=30;; esac
      for method in baseline fastvggt sparse-vggt da-vggt u-m; do
        case ${method} in baseline) factor=100;; fastvggt) factor=50;; sparse-vggt) factor=55;; da-vggt) factor=65;; u-m) factor=40;; esac
        estimate=$((base * sequences * factor))
        id=${model}__${method}__${dataset}
        printf '%09d\t%s\t%s\t%s\t%s\n' "${estimate}" "${id}" "${model}" "${method}" "${dataset}" >>"${tasks}"
      done
    done
  done
  sort -rn "${tasks}" -o "${tasks}"
fi

gpu_is_idle() {
  local gpu=$1 free util apps
  free=$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
  util=$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -dc '0-9')
  apps=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^No running processes found$/d;/^$/d' | wc -l)
  (( free >= 44000 && util <= 5 && apps == 0 ))
}

claim_task() {
  local result="" estimate id model method dataset
  exec 9>"${out}/task.lock"; flock -x 9
  while IFS=$'\t' read -r estimate id model method dataset; do
    [[ -f ${out}/status/${id}.done || -f ${out}/status/${id}.failed || -f ${out}/status/${id}.running ]] && continue
    touch "${out}/status/${id}.running"
    result="${id}|${model}|${method}|${dataset}"
    break
  done <"${tasks}"
  flock -u 9; exec 9>&-
  printf '%s' "${result}"
}

worker() {
  local gpu=$1 item id model method dataset code
  while true; do
    item=$(claim_task)
    [[ -n ${item} ]] || return 0
    IFS='|' read -r id model method dataset <<<"${item}"
    echo "[$(date -u -Is)] GPU ${gpu}: ${id}" >>"${out}/logs/scheduler.log"
    if bash "${runner}" "${gpu}" "${model}" "${method}" "${dataset}" "${out}/${id}" >"${out}/logs/${id}.log" 2>&1; then
      mv "${out}/status/${id}.running" "${out}/status/${id}.done"
    else
      code=$?
      mv "${out}/status/${id}.running" "${out}/status/${id}.failed"
      echo "[$(date -u -Is)] ${id} failed with ${code}" >>"${out}/logs/scheduler.log"
    fi
  done
}

all_settled() { [[ $(find "${out}/status" -name '*.done' -o -name '*.failed' | wc -l) -ge 45 ]]; }
declare -A workers=()
while ! all_settled; do
  for gpu in {0..7}; do
    if [[ -n ${workers[$gpu]:-} ]] && kill -0 "${workers[$gpu]}" 2>/dev/null; then continue; fi
    unset 'workers[$gpu]'
    gpu_is_idle "${gpu}" || continue
    worker "${gpu}" & workers[$gpu]=$!
    echo "[$(date -u -Is)] worker started on GPU ${gpu}" >>"${out}/logs/scheduler.log"
  done
  sleep 60
done
wait || true
echo "[$(date -u -Is)] all 45 tasks settled" >>"${out}/logs/scheduler.log"
