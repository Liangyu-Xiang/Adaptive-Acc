#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 {tum|7scenes} {u002|u003|u004|baseline|fastvggt} PHYSICAL_GPU" >&2
    exit 2
fi

dataset=$1
method=$2
physical_gpu=$3
repo_root=/data/mmc_syang/VGGT-omega
python_bin=/data/mmc_syang/miniconda3/envs/fastvggt/bin/python
checkpoint=/data/mmc_syang/vggt-omega/checkpoints/vggt_omega_1b_512.pt
run_root=${repo_root}/outputs/cube_phase2_sweep_002_004_baselines_300
output_dir=${run_root}/${method}/${dataset}
log_dir=${run_root}/logs
log_file=${log_dir}/${method}_${dataset}_gpu${physical_gpu}.log

mkdir -p "${output_dir}" "${log_dir}"

case ${method} in
    u002)
        lambda=0.02
        merge_ratio=0
        fusion_mode=u-m
        ;;
    u003)
        lambda=0.03
        merge_ratio=0
        fusion_mode=u-m
        ;;
    u004)
        lambda=0.04
        merge_ratio=0
        fusion_mode=u-m
        ;;
    baseline)
        lambda=none
        merge_ratio=0
        fusion_mode=none
        ;;
    fastvggt)
        lambda=none
        merge_ratio=0.9
        fusion_mode=none
        ;;
    *)
        echo "unknown method: ${method}" >&2
        exit 2
        ;;
esac

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
    --merge-ratio "${merge_ratio}"
    --frame-fusion-mode "${fusion_mode}"
)

if [[ ${fusion_mode} == u-m ]]; then
    common_args+=(
        --frame-fusion-start-layer -1
        --frame-fusion-recompute-layers 0,10,17
        --frame-fusion-lambda-cost "${lambda}"
        --frame-fusion-merge-top-similarity-percent 100
        --frame-fusion-min-keep-ratio 0.05
        --frame-fusion-temporal-window 4
        --frame-fusion-spatial-radius 2
        --frame-fusion-attention-variant representative
    )
fi

echo "[$(date -Is)] dataset=${dataset} method=${method} lambda=${lambda} merge_ratio=${merge_ratio} physical_gpu=${physical_gpu}" > "${log_file}"
if [[ ${dataset} == tum ]]; then
    exec env \
        VGGT_UM_TRITON=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="${repo_root}" \
        CUDA_VISIBLE_DEVICES="${physical_gpu}" \
        "${python_bin}" "${repo_root}/scripts/eval_tum_dynamics_paper.py" \
        --data-root /data/mmc_syang/dataset/TUM-Dynamics \
        --sampling-pool full \
        "${common_args[@]}" >> "${log_file}" 2>&1
elif [[ ${dataset} == 7scenes ]]; then
    exec env \
        VGGT_UM_TRITON=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="${repo_root}" \
        CUDA_VISIBLE_DEVICES="${physical_gpu}" \
        "${python_bin}" "${repo_root}/scripts/eval_7scenes_paper.py" \
        --data-root /data/mmc_syang/dataset/7scenes \
        --sampling-unit sequence \
        "${common_args[@]}" >> "${log_file}" 2>&1
else
    echo "unknown dataset: ${dataset}" >&2
    exit 2
fi
