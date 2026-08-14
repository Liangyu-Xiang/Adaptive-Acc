#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; model=${2:?model}; dataset=${3:?tum|7scenes}; gap=${4:?gap}
root=/data/mmc_syang
out=${root}/VGGT-omega/outputs/attention_probe_full/${dataset}/gap_${gap}/${model}
mkdir -p "${out}"
if [[ ${dataset} == tum ]]; then
  seq=rgbd_dataset_freiburg3_walking_xyz; data=${root}/dataset/TUM-Dynamics
else
  seq=chess/seq-03; data=${root}/dataset/7scenes
fi
case ${model} in
pi3)
  cd ${root}/Pi3
  CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=${root}/Pi3 ${root}/miniconda3/envs/flow3r/bin/python scripts/run_pi3_vggt_omega_eval.py --dataset "$([[ ${dataset} == tum ]] && echo tum_dynamic || echo 7scenes)" --dataset-root "${data}" --pretrained checkpoints/Pi3 --output-dir "${out}/eval" --device cuda:0 --sequences "${seq}" --max-frames-per-seq 10 --timing-repeats 1 --acceleration-method none --attention-probe-output "${out}/matrices" --attention-probe-frame-gap ${gap} --overwrite ;;
vggt)
  cd ${root}/vggt
  CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=${root}/vggt ${root}/miniconda3/envs/fastvggt/bin/python scripts/eval_standard_tum_7scenes.py --dataset "$([[ ${dataset} == tum ]] && echo tum_dynamic || echo 7scenes)" --dataset-root "${data}" --checkpoint ckpts/model.pt --output-dir "${out}/eval" --device cuda:0 --sequences "${seq}" --num-frames 10 --image-resolution 518 --timing-repeats 1 --acceleration-method none --attention-probe-output "${out}/matrices" --attention-probe-frame-gap ${gap} ;;
omega)
  cd ${root}/VGGT-omega
  CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=${root}/VGGT-omega ${root}/miniconda3/envs/fastvggt/bin/python tools/analyze_attention_information_flow.py --data-root "${data}" --sequence "${seq}" --frame-gap ${gap} --analysis-dir "${out}" --checkpoint ${root}/vggt-omega/checkpoints/vggt_omega_1b_512.pt --device cuda:0 --image-resolution 512 --resize-mode max_size --export-full-attention-matrices ;;
esac
