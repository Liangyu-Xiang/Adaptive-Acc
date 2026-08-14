#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; model=${2:?pi3|vggt}; dataset=${3:?tum|7scenes}; wait_session=${4:?old tmux session}
root=/data/mmc_syang
out=${root}/VGGT-omega/outputs/00/${model}/best_um_full300/${dataset}
log=${root}/VGGT-omega/outputs/00/logs/best_um_full300_${model}_${dataset}_gpu${gpu}.log
mkdir -p "$(dirname "${log}")"
echo "[$(date -Is)] waiting for ${wait_session}" >"${log}"
while tmux has-session -t "${wait_session}" 2>/dev/null; do sleep 30; done
echo "[$(date -Is)] START ${model}/${dataset}" >>"${log}"
if [[ ${model} == pi3 ]]; then
  [[ ${dataset} == tum ]] && data=tum_dynamic || data=7scenes
  [[ ${dataset} == tum ]] && data_root=${root}/dataset/TUM-Dynamics || data_root=${root}/dataset/7scenes
  cd ${root}/Pi3
  CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/Pi3 \
    ${root}/miniconda3/envs/flow3r/bin/python scripts/run_pi3_vggt_omega_eval.py \
    --dataset "${data}" --dataset-root "${data_root}" --pretrained checkpoints/Pi3 --output-dir "${out}" \
    --device cuda:0 --max-frames-per-seq 300 --frame-sample-mode uniform --load-img-size 512 \
    --timing-repeats 3 --acceleration-method u-m --overwrite >>"${log}" 2>&1
else
  [[ ${dataset} == tum ]] && data=tum_dynamic || data=7scenes
  [[ ${dataset} == tum ]] && data_root=${root}/dataset/TUM-Dynamics || data_root=${root}/dataset/7scenes
  cd ${root}/vggt
  CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/vggt \
    ${root}/miniconda3/envs/fastvggt/bin/python scripts/eval_standard_tum_7scenes.py \
    --dataset "${data}" --dataset-root "${data_root}" --checkpoint ckpts/model.pt --output-dir "${out}" \
    --device cuda:0 --num-frames 300 --image-resolution 518 --timing-repeats 3 --acceleration-method u-m >>"${log}" 2>&1
fi
echo "[$(date -Is)] DONE" >>"${log}"
