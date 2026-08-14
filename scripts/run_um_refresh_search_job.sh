#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 {tum|7scenes} REFRESH_LAYERS TAG PHYSICAL_GPU" >&2
    exit 2
fi

dataset=$1
refresh_layers=$2
tag=$3
physical_gpu=$4
repo_root=/data/mmc_syang/VGGT-omega
python_bin=/data/mmc_syang/miniconda3/envs/fastvggt/bin/python
checkpoint=/data/mmc_syang/vggt-omega/checkpoints/vggt_omega_1b_512.pt
run_root=${repo_root}/outputs/um_refresh_layer_search_lambda_0p04_300
output_dir=${run_root}/${tag}/${dataset}
log_dir=${run_root}/logs
log_file=${log_dir}/${tag}_${dataset}_gpu${physical_gpu}.log

mkdir -p "${output_dir}" "${log_dir}"
common_args=(
    --checkpoint "${checkpoint}"
    --output-dir "${output_dir}"
    --device cuda:0
    --num-frames 300
    --sampling-strategy uniform
    --resize-mode max_size
    --image-resolution 512
    --timing-repeats 3
    --require-exclusive-gpu
    --exclusive-gpu-index "${physical_gpu}"
    --merge-ratio 0
    --frame-fusion-mode u-m
    --frame-fusion-start-layer -1
    --frame-fusion-recompute-layers "${refresh_layers}"
    --frame-fusion-lambda-cost 0.04
    --frame-fusion-merge-top-similarity-percent 100
    --frame-fusion-min-keep-ratio 0.05
    --frame-fusion-temporal-window 4
    --frame-fusion-spatial-radius 2
    --frame-fusion-attention-variant representative
)

echo "[$(date -Is)] dataset=${dataset} refresh_layers=${refresh_layers} lambda=0.04 gpu=${physical_gpu}" > "${log_file}"
if [[ ${dataset} == tum ]]; then
    exec env VGGT_UM_TRITON=1 PYTHONUNBUFFERED=1 PYTHONPATH="${repo_root}" CUDA_VISIBLE_DEVICES="${physical_gpu}" \
        "${python_bin}" "${repo_root}/scripts/eval_tum_dynamics_paper.py" \
        --data-root /data/mmc_syang/dataset/TUM-Dynamics --sampling-pool full \
        "${common_args[@]}" >> "${log_file}" 2>&1
elif [[ ${dataset} == 7scenes ]]; then
    exec env VGGT_UM_TRITON=1 PYTHONUNBUFFERED=1 PYTHONPATH="${repo_root}" CUDA_VISIBLE_DEVICES="${physical_gpu}" \
        "${python_bin}" "${repo_root}/scripts/eval_7scenes_paper.py" \
        --data-root /data/mmc_syang/dataset/7scenes --sampling-unit sequence \
        "${common_args[@]}" >> "${log_file}" 2>&1
else
    echo "unknown dataset: ${dataset}" >&2
    exit 2
fi
