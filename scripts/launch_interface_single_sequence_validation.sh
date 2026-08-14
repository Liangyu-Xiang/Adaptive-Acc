#!/usr/bin/env bash
# 24 single-sequence validation runs: 3 models x 4 interfaces x 2 datasets.
# A worker owns one physical GPU and executes its assigned jobs serially.
set -euo pipefail

gpu=${1:?usage: $0 PHYSICAL_GPU WORKER_INDEX}
worker=${2:?usage: $0 PHYSICAL_GPU WORKER_INDEX}
start_index=${3:-0}
root=/data/mmc_syang
output_root=${root}/VGGT-omega/outputs/interface_single_sequence_300
log_root=${output_root}/logs
mkdir -p "${log_root}"

jobs=(
  'omega|fastvggt|tum' 'omega|fastvggt|7scenes'
  'omega|sparse-vggt|tum' 'omega|sparse-vggt|7scenes' 'omega|da-vggt|tum' 'omega|da-vggt|7scenes'
  'omega|u-m|tum' 'omega|u-m|7scenes'
  'vggt|fastvggt|tum' 'vggt|fastvggt|7scenes'
  'vggt|sparse-vggt|tum' 'vggt|sparse-vggt|7scenes' 'vggt|da-vggt|tum' 'vggt|da-vggt|7scenes'
  'vggt|u-m|tum' 'vggt|u-m|7scenes'
  'pi3|fastvggt|tum' 'pi3|fastvggt|7scenes'
  'pi3|sparse-vggt|tum' 'pi3|sparse-vggt|7scenes' 'pi3|da-vggt|tum' 'pi3|da-vggt|7scenes'
  'pi3|u-m|tum' 'pi3|u-m|7scenes'
)

run_omega() {
  local method=$1 dataset=$2 out=$3
  local script seq extra=()
  if [[ ${dataset} == tum ]]; then
    script=scripts/eval_tum_dynamics_paper.py; seq=rgbd_dataset_freiburg3_walking_rpy
    extra=(--data-root "${root}/dataset/TUM-Dynamics" --sequences "${seq}" --sampling-pool full --sampling-strategy uniform)
  else
    script=scripts/eval_7scenes_paper.py; seq=chess/seq-03
    extra=(--data-root "${root}/dataset/7scenes" --sequences "${seq}" --sampling-unit sequence --sampling-strategy uniform)
  fi
  local accel=(--acceleration-method "${method}" --merge-ratio 0 --frame-fusion-mode none)
  case ${method} in
    fastvggt) accel=(--acceleration-method fastvggt --merge-ratio 0.9 --frame-fusion-mode none) ;;
    sparse-vggt) accel=(--acceleration-method sparse-vggt --merge-ratio 0 --frame-fusion-mode none --sparse-attention --sparse-ratio 0.5 --sparse-pool-mode avg) ;;
    da-vggt) accel=(--acceleration-method da-vggt --merge-ratio 0 --frame-fusion-mode none --da-chunk-size 50) ;;
  esac
  if [[ ${method} == u-m ]]; then
    accel=(--acceleration-method u-m --merge-ratio 0 --frame-fusion-mode u-m --frame-fusion-start-layer -1 --frame-fusion-recompute-layers 0,10,17 --frame-fusion-lambda-cost 0.04 --frame-fusion-merge-top-similarity-percent 100 --frame-fusion-min-keep-ratio 0.05 --frame-fusion-temporal-window 4 --frame-fusion-spatial-radius 2 --frame-fusion-attention-variant representative)
  fi
  (cd "${root}/VGGT-omega" && CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH="${root}/VGGT-omega" /data/mmc_syang/miniconda3/envs/fastvggt/bin/python "${script}" "${extra[@]}" --checkpoint "${root}/vggt-omega/checkpoints/vggt_omega_1b_512.pt" --output-dir "${out}" --device cuda:0 --num-frames 300 --image-resolution 512 --resize-mode max_size --timing-repeats 3 "${accel[@]}")
}

run_vggt() {
  local method=$1 dataset=$2 out=$3 seq
  [[ ${dataset} == tum ]] && seq=rgbd_dataset_freiburg3_walking_rpy || seq=chess/seq-03
  (cd "${root}/vggt" && CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH="${root}/vggt" /data/mmc_syang/miniconda3/envs/fastvggt/bin/python scripts/eval_standard_tum_7scenes.py --dataset "$([[ ${dataset} == tum ]] && echo tum_dynamic || echo 7scenes)" --dataset-root "$([[ ${dataset} == tum ]] && echo ${root}/dataset/TUM-Dynamics || echo ${root}/dataset/7scenes)" --checkpoint ckpts/model.pt --output-dir "${out}" --device cuda:0 --sequences "${seq}" --num-frames 300 --image-resolution 518 --timing-repeats 3 --acceleration-method "${method}" --merge-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --um-lambda 0.04 --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers 0,10,17 --da-chunk-size 50 --overwrite)
}

run_pi3() {
  local method=$1 dataset=$2 out=$3 seq
  [[ ${dataset} == tum ]] && seq=rgbd_dataset_freiburg3_walking_rpy || seq=chess/seq-03
  (cd "${root}/Pi3" && CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH="${root}/Pi3" /data/mmc_syang/miniconda3/envs/flow3r/bin/python scripts/run_pi3_vggt_omega_eval.py --dataset "$([[ ${dataset} == tum ]] && echo tum_dynamic || echo 7scenes)" --dataset-root "$([[ ${dataset} == tum ]] && echo ${root}/dataset/TUM-Dynamics || echo ${root}/dataset/7scenes)" --pretrained checkpoints/Pi3 --output-dir "${out}" --device cuda:0 --sequences "${seq}" --max-frames-per-seq 300 --load-img-size 512 --timing-repeats 3 --acceleration-method "${method}" --token-merging-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --um-lambda 0.04 --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers 0,10,17 --da-chunk-size 50 --overwrite)
}

for i in "${!jobs[@]}"; do
  (( i >= start_index )) || continue
  (( i % 3 == worker )) || continue
  IFS='|' read -r model method dataset <<< "${jobs[i]}"
  if [[ ${method} == da-vggt ]]; then
    echo "[$(date -Is)] SKIP gpu=${gpu} ${model}/${method}/${dataset}: incomplete adapter, excluded from comparison"
    continue
  fi
  out="${output_root}/${model}/${method}/${dataset}"
  log="${log_root}/${model}_${method}_${dataset}.log"
  echo "[$(date -Is)] START gpu=${gpu} ${model}/${method}/${dataset}" | tee -a "${log}"
  case ${model} in
    omega) run_omega "${method}" "${dataset}" "${out}" >>"${log}" 2>&1 ;;
    vggt) run_vggt "${method}" "${dataset}" "${out}" >>"${log}" 2>&1 ;;
    pi3) run_pi3 "${method}" "${dataset}" "${out}" >>"${log}" 2>&1 ;;
  esac
  echo "[$(date -Is)] DONE gpu=${gpu} ${model}/${method}/${dataset}" | tee -a "${log}"
done
