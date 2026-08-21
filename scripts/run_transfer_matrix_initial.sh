#!/usr/bin/env bash
# Initial portable runner for the 4 sampling/frame-budget experiment groups.
#
# One invocation can execute the selected group/model/method matrix serially on
# one GPU, or distribute independent items round-robin across several GPUs.
# A failed item (including CUDA OOM) is recorded and never stops its worker.
# Paths and Python executables are environment-overridable so the script can be
# copied to another server without changing its logic.
set -uo pipefail

ROOT=${ROOT:-/data/mmc_syang}
OMEGA_ROOT=${OMEGA_ROOT:-${ROOT}/VGGT-omega}
VGGT_ROOT=${VGGT_ROOT:-${ROOT}/vggt}
PI3_ROOT=${PI3_ROOT:-${ROOT}/Pi3}
OMEGA_CHECKPOINT=${OMEGA_CHECKPOINT:-${ROOT}/vggt-omega/checkpoints/vggt_omega_1b_512.pt}
VGGT_CHECKPOINT=${VGGT_CHECKPOINT:-${VGGT_ROOT}/ckpts/model.pt}
PI3_CHECKPOINT=${PI3_CHECKPOINT:-${PI3_ROOT}/checkpoints/Pi3}
OMEGA_PYTHON=${OMEGA_PYTHON:-${ROOT}/miniconda3/envs/fastvggt/bin/python}
VGGT_PYTHON=${VGGT_PYTHON:-${ROOT}/miniconda3/envs/fastvggt/bin/python}
PI3_PYTHON=${PI3_PYTHON:-${ROOT}/miniconda3/envs/flow3r/bin/python}
SEVEN_SCENES_ROOT=${SEVEN_SCENES_ROOT:-${ROOT}/dataset/7scenes}
NRGBD_ROOT=${NRGBD_ROOT:-${ROOT}/dataset/NRGBD}
SCANNET30_ROOT=${SCANNET30_ROOT:-${ROOT}/dataset/scannet30/raw}
TUM_ROOT=${TUM_ROOT:-${ROOT}/dataset/TUM-Dynamics}
OUTPUT_ROOT=${OUTPUT_ROOT:-${OMEGA_ROOT}/outputs/transfer_matrix_initial}

usage() {
  cat <<'EOF'
Usage: run_transfer_matrix_initial.sh (--gpu ID | --gpus ID[,ID...]) [options]

Options:
  --group NAME       stride3_300 | stride2_300 | stride2_500 | stride2_1000 | all
  --model NAME       omega | vggt | pi3 | all                 (default: all)
  --method NAME      baseline | fastvggt | sparse-vggt | da-vggt | u-m |
                     u-m-l003 | u-m-l0035 | all (default: all)
  --dataset NAME     tum | 7scenes | nrgbd | scannet30 | all    (default: all)
  --sequence NAME    Run only this one sequence (for smoke tests).
  --timing-repeats N CUDA Event repeats; default: 3.
  --gpu ID           Run all selected items serially on one physical GPU.
  --gpus IDS         Comma-separated physical GPU IDs. Items are distributed
                     round-robin; one serial worker is started per GPU.
  --dry-run          Print commands only.

For every dataset, sequences shorter than N are skipped; N <= L <= stride*N
uses first/last-preserving uniform sampling, and L > stride*N uses fixed stride
sampling from source frame 0. For stride2_1000, 7Scenes sequences with 1000
valid frames therefore receive all their frames (L == N).
EOF
}

gpu=
gpus_csv=
group=all
model_filter=all
method_filter=all
dataset_filter=all
sequence_filter=
timing_repeats=3
dry_run=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) gpu=${2:?}; shift 2 ;;
    --gpus) gpus_csv=${2:?}; shift 2 ;;
    --group) group=${2:?}; shift 2 ;;
    --model) model_filter=${2:?}; shift 2 ;;
    --method) method_filter=${2:?}; shift 2 ;;
    --dataset) dataset_filter=${2:?}; shift 2 ;;
    --sequence) sequence_filter=${2:?}; shift 2 ;;
    --timing-repeats) timing_repeats=${2:?}; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -n ${gpu} && -n ${gpus_csv} ]]; then
  echo "Use exactly one of --gpu or --gpus" >&2
  exit 2
fi
if [[ -z ${gpu} && -z ${gpus_csv} ]]; then
  echo "One of --gpu or --gpus is required" >&2
  exit 2
fi
if [[ -n ${gpu} && ! ${gpu} =~ ^[0-9]+$ ]]; then
  echo "Invalid --gpu ID: ${gpu}" >&2
  exit 2
fi
if [[ ! ${timing_repeats} =~ ^[1-9][0-9]*$ ]]; then
  echo "--timing-repeats must be a positive integer" >&2
  exit 2
fi

group_config() {
  case $1 in
    stride3_300) echo '300 3' ;;
    stride2_300) echo '300 2' ;;
    stride2_500) echo '500 2' ;;
    stride2_1000) echo '1000 2' ;;
    *) echo "Unknown group: $1" >&2; return 2 ;;
  esac
}

want() { [[ $1 == all || $1 == $2 ]]; }

validate_filter() {
  local label=$1 value=$2
  shift 2
  local candidate
  [[ ${value} == all ]] && return 0
  for candidate in "$@"; do
    [[ ${value} == ${candidate} ]] && return 0
  done
  echo "Invalid --${label}: ${value}" >&2
  exit 2
}

run_one() {
  local group_name=$1 model=$2 method=$3 dataset=$4 frames=$5 stride=$6
  local data_root dataset_arg output method_arg log status workdir um_lambda=
  case ${dataset} in
    tum) data_root=${TUM_ROOT}; dataset_arg=tum_dynamic ;;
    7scenes) data_root=${SEVEN_SCENES_ROOT}; dataset_arg=7scenes ;;
    nrgbd) data_root=${NRGBD_ROOT}; dataset_arg=nrgbd ;;
    scannet30) data_root=${SCANNET30_ROOT}; dataset_arg=scannet ;;
  esac
  case ${method} in
    baseline) method_arg=none ;;
    u-m) method_arg=u-m; um_lambda=0.04 ;;
    u-m-l003) method_arg=u-m; um_lambda=0.03 ;;
    u-m-l0035) method_arg=u-m; um_lambda=0.035 ;;
    *) method_arg=${method} ;;
  esac
  output=${OUTPUT_ROOT}/${group_name}/${model}__${method}__${dataset}
  log=${OUTPUT_ROOT}/${group_name}/logs/${model}__${method}__${dataset}.log
  status=${OUTPUT_ROOT}/${group_name}/status/${model}__${method}__${dataset}
  mkdir -p "$(dirname "${log}")" "$(dirname "${status}")"

  local -a cmd
  if [[ ${model} == omega ]]; then
    workdir=${OMEGA_ROOT}
    local script
    case ${dataset} in
      tum) script=scripts/eval_tum_dynamics_paper.py ;;
      7scenes) script=scripts/eval_7scenes_paper.py ;;
      nrgbd) script=scripts/eval_nrgbd_paper.py ;;
      scannet30) script=scripts/eval_scannet_paper.py ;;
    esac
    cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" VGGT_UM_TRITON=1 "PYTHONPATH=${OMEGA_ROOT}" "${OMEGA_PYTHON}" "${script}"
      --data-root "${data_root}" --checkpoint "${OMEGA_CHECKPOINT}" --output-dir "${output}" --device cuda:0
      --num-frames "${frames}" --sampling-strategy uniform
      --image-resolution 512 --resize-mode max_size --timing-repeats "${timing_repeats}" --acceleration-method "${method_arg}")
    if [[ ${dataset} == tum ]]; then
      cmd+=(--sampling-stride "${stride}")
    else
      cmd+=(--sampling-stride "${stride}" --sampling-unit sequence)
    fi
    case ${method} in
      baseline) cmd+=(--merge-ratio 0 --frame-fusion-mode none) ;;
      fastvggt) cmd+=(--merge-ratio 0.9 --frame-fusion-mode none) ;;
      sparse-vggt) cmd+=(--merge-ratio 0 --frame-fusion-mode none --sparse-attention --sparse-ratio 0.5 --sparse-pool-mode avg) ;;
      da-vggt) cmd+=(--merge-ratio 0 --frame-fusion-mode none --da-chunk-size 50) ;;
      u-m|u-m-l003|u-m-l0035) cmd+=(--merge-ratio 0 --frame-fusion-mode u-m --frame-fusion-lambda-cost "${um_lambda}" --frame-fusion-spatial-radius 2 --frame-fusion-temporal-window 4 --frame-fusion-attention-variant representative) ;;
    esac
  elif [[ ${model} == vggt ]]; then
    workdir=${VGGT_ROOT}
    cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" VGGT_UM_TRITON=1 "PYTHONPATH=${VGGT_ROOT}" "${VGGT_PYTHON}"
      scripts/eval_standard_tum_7scenes.py --dataset "${dataset_arg}" --dataset-root "${data_root}"
      --checkpoint "${VGGT_CHECKPOINT}" --output-dir "${output}" --device cuda:0 --num-frames "${frames}"
      --sampling-stride "${stride}" --image-resolution 518 --timing-repeats "${timing_repeats}" --acceleration-method "${method_arg}"
      --merge-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --da-chunk-size 50)
    [[ -z ${um_lambda} ]] || cmd+=(--um-lambda "${um_lambda}")
  else
    workdir=${PI3_ROOT}
    cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" VGGT_UM_TRITON=1 "PYTHONPATH=${PI3_ROOT}" "${PI3_PYTHON}"
      scripts/run_pi3_vggt_omega_eval.py --dataset "${dataset_arg}" --dataset-root "${data_root}"
      --pretrained "${PI3_CHECKPOINT}" --output-dir "${output}" --device cuda:0 --max-frames-per-seq "${frames}"
      --sampling-stride "${stride}" --frame-sample-mode uniform --load-img-size 512 --timing-repeats "${timing_repeats}"
      --acceleration-method "${method_arg}" --token-merging-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --da-chunk-size 50)
    [[ -z ${um_lambda} ]] || cmd+=(--um-lambda "${um_lambda}")
  fi
  [[ -z ${sequence_filter} ]] || cmd+=(--sequences "${sequence_filter}")

  printf '[%s] %s/%s/%s/%s\n' "$(date -u -Is)" "${group_name}" "${model}" "${method}" "${dataset}" | tee -a "${OUTPUT_ROOT}/launcher.log"
  if (( dry_run )); then printf '%q ' "${cmd[@]}"; printf '\n'; return 0; fi
  if (cd "${workdir}" && "${cmd[@]}") >"${log}" 2>&1; then
    : >"${status}.done"
  else
    : >"${status}.failed"
    printf '[%s] FAILED %s\n' "$(date -u -Is)" "${model}__${method}__${dataset}" | tee -a "${OUTPUT_ROOT}/launcher.log"
  fi
}

groups=(stride3_300 stride2_300 stride2_500 stride2_1000)
models=(omega vggt pi3)
methods=(baseline fastvggt sparse-vggt da-vggt u-m u-m-l003 u-m-l0035)
datasets=(tum 7scenes nrgbd scannet30)
validate_filter group "${group}" "${groups[@]}"
validate_filter model "${model_filter}" "${models[@]}"
validate_filter method "${method_filter}" "${methods[@]}"
validate_filter dataset "${dataset_filter}" "${datasets[@]}"
tasks=()
for group_name in "${groups[@]}"; do
  want "${group}" "${group_name}" || continue
  read -r frames stride <<<"$(group_config "${group_name}")"
  for model in "${models[@]}"; do
    want "${model_filter}" "${model}" || continue
    for method in "${methods[@]}"; do
      want "${method_filter}" "${method}" || continue
      for dataset in "${datasets[@]}"; do
        want "${dataset_filter}" "${dataset}" || continue
        tasks+=("${group_name}|${model}|${method}|${dataset}|${frames}|${stride}")
      done
    done
  done
done
((${#tasks[@]} > 0)) || { echo "No tasks selected" >&2; exit 2; }

run_worker() {
  local worker_gpu=$1 worker_index=$2 worker_count=$3 task_entry
  local group_name model method dataset frames stride
  gpu=${worker_gpu}
  for ((task_index=worker_index; task_index<${#tasks[@]}; task_index+=worker_count)); do
    task_entry=${tasks[task_index]}
    IFS='|' read -r group_name model method dataset frames stride <<<"${task_entry}"
    run_one "${group_name}" "${model}" "${method}" "${dataset}" "${frames}" "${stride}"
  done
}

if [[ -n ${gpus_csv} ]]; then
  IFS=',' read -r -a gpu_list <<<"${gpus_csv}"
  ((${#gpu_list[@]} > 0)) || { echo "--gpus is empty" >&2; exit 2; }
  for gpu_id in "${gpu_list[@]}"; do
    [[ ${gpu_id} =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID in --gpus: ${gpu_id}" >&2; exit 2; }
  done
  for ((worker_index=0; worker_index<${#gpu_list[@]}; worker_index++)); do
    run_worker "${gpu_list[worker_index]}" "${worker_index}" "${#gpu_list[@]}" &
  done
  # Each worker converts individual failures to .failed and returns success, so
  # wait only synchronizes workers; it does not cancel a remaining queue.
  wait
else
  run_worker "${gpu}" 0 1
fi
