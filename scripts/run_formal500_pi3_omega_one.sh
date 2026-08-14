#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; model=${2:?pi3|omega}; method=${3:?none|fastvggt|sparse-vggt|da-vggt|u-m}; dataset=${4:?tum|7scenes}
root=/data/mmc_syang
out=${root}/VGGT-omega/outputs/00/formal500_pi3_omega/${model}/${method}/${dataset}
mkdir -p "${out}"
if [[ ${model} == omega ]]; then
  [[ ${dataset} == tum ]] && script=scripts/eval_tum_dynamics_paper.py || script=scripts/eval_7scenes_paper.py
  [[ ${dataset} == tum ]] && data_root=${root}/dataset/TUM-Dynamics || data_root=${root}/dataset/7scenes
  cd ${root}/VGGT-omega
  args=(--data-root "${data_root}" --checkpoint ${root}/vggt-omega/checkpoints/vggt_omega_1b_512.pt --output-dir "${out}" --device cuda:0 --num-frames 500 --image-resolution 512 --resize-mode max_size --sampling-strategy uniform --timing-repeats 3)
  [[ ${dataset} == tum ]] && args+=(--sampling-pool full) || args+=(--sampling-unit sequence)
  case ${method} in
    none) args+=(--acceleration-method none --merge-ratio 0 --frame-fusion-mode none) ;;
    fastvggt) args+=(--acceleration-method fastvggt --merge-ratio 0.9 --frame-fusion-mode none) ;;
    sparse-vggt) args+=(--acceleration-method sparse-vggt --merge-ratio 0 --frame-fusion-mode none --sparse-attention --sparse-ratio 0.5 --sparse-pool-mode avg) ;;
    da-vggt) args+=(--acceleration-method da-vggt --merge-ratio 0 --frame-fusion-mode none --da-chunk-size 50) ;;
    u-m) args+=(--acceleration-method u-m --merge-ratio 0 --frame-fusion-mode u-m --frame-fusion-recompute-layers 0,10,17 --frame-fusion-lambda-cost 0.03 --frame-fusion-temporal-window 4 --frame-fusion-spatial-radius 2 --frame-fusion-attention-variant representative) ;;
  esac
  CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/VGGT-omega ${root}/miniconda3/envs/fastvggt/bin/python "${script}" "${args[@]}"
else
  [[ ${dataset} == tum ]] && data=tum_dynamic || data=7scenes
  [[ ${dataset} == tum ]] && data_root=${root}/dataset/TUM-Dynamics || data_root=${root}/dataset/7scenes
  cd ${root}/Pi3
  CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/Pi3 ${root}/miniconda3/envs/flow3r/bin/python scripts/run_pi3_vggt_omega_eval.py --dataset "${data}" --dataset-root "${data_root}" --pretrained checkpoints/Pi3 --output-dir "${out}" --device cuda:0 --max-frames-per-seq 500 --frame-sample-mode uniform --load-img-size 512 --timing-repeats 3 --acceleration-method "${method}" --token-merging-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --da-chunk-size 50 --overwrite
fi
