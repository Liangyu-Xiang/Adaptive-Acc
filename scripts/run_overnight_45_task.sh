#!/usr/bin/env bash
# Execute exactly one formal 300-frame evaluation task on CUDA_VISIBLE_DEVICES.
set -euo pipefail

gpu=${1:?gpu}; model=${2:?omega|vggt|pi3}; method=${3:?method}; dataset=${4:?dataset}; out=${5:?output_dir}
root=/data/mmc_syang
mkdir -p "${out}"
common_env=(CUDA_VISIBLE_DEVICES="${gpu}" VGGT_UM_TRITON=1)
case ${method} in baseline) method=none;; esac

if [[ ${model} == omega ]]; then
  cd "${root}/VGGT-omega"
  case ${dataset} in
    7scenes) script=scripts/eval_7scenes_paper.py; data_root=${root}/dataset/7scenes;;
    nrgbd) script=scripts/eval_nrgbd_paper.py; data_root=${root}/dataset/NRGBD;;
    scannet30) script=scripts/eval_scannet_paper.py; data_root=${root}/dataset/scannet30/raw;;
  esac
  args=("${script}" --data-root "${data_root}" --checkpoint "${root}/vggt-omega/checkpoints/vggt_omega_1b_512.pt" --output-dir "${out}" --device cuda:0 --num-frames 300 --sampling-unit sequence --sampling-strategy uniform --image-resolution 512 --resize-mode max_size --timing-repeats 3 --acceleration-method "${method}")
  case ${method} in
    none) args+=(--merge-ratio 0 --frame-fusion-mode none);;
    fastvggt) args+=(--merge-ratio 0.9 --frame-fusion-mode none);;
    sparse-vggt) args+=(--merge-ratio 0 --frame-fusion-mode none --sparse-attention --sparse-ratio 0.5 --sparse-pool-mode avg);;
    da-vggt) args+=(--merge-ratio 0 --frame-fusion-mode none --da-chunk-size 50);;
    u-m) args+=(--merge-ratio 0 --frame-fusion-mode u-m --frame-fusion-temporal-window 4 --frame-fusion-spatial-radius 2 --frame-fusion-attention-variant representative);;
  esac
  env "${common_env[@]}" PYTHONPATH="${root}/VGGT-omega" "${root}/miniconda3/envs/fastvggt/bin/python" "${args[@]}"
elif [[ ${model} == vggt ]]; then
  cd "${root}/vggt"
  case ${dataset} in 7scenes) data_root=${root}/dataset/7scenes;; nrgbd) data_root=${root}/dataset/NRGBD;; scannet30) data_root=${root}/dataset/scannet30/raw;; esac
  env "${common_env[@]}" PYTHONPATH="${root}/vggt" "${root}/miniconda3/envs/fastvggt/bin/python" scripts/eval_standard_tum_7scenes.py --dataset "${dataset/scannet30/scannet}" --dataset-root "${data_root}" --checkpoint ckpts/model.pt --output-dir "${out}" --device cuda:0 --num-frames 300 --image-resolution 518 --timing-repeats 3 --acceleration-method "${method}" --merge-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --da-chunk-size 50
else
  cd "${root}/Pi3"
  case ${dataset} in 7scenes) data_root=${root}/dataset/7scenes; pi3_dataset=7scenes;; nrgbd) data_root=${root}/dataset/NRGBD; pi3_dataset=nrgbd;; scannet30) data_root=${root}/dataset/scannet30/raw; pi3_dataset=scannet;; esac
  env "${common_env[@]}" PYTHONPATH="${root}/Pi3" "${root}/miniconda3/envs/flow3r/bin/python" scripts/run_pi3_vggt_omega_eval.py --dataset "${pi3_dataset}" --dataset-root "${data_root}" --pretrained checkpoints/Pi3 --output-dir "${out}" --device cuda:0 --max-frames-per-seq 300 --frame-sample-mode uniform --load-img-size 512 --timing-repeats 3 --acceleration-method "${method}" --token-merging-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --da-chunk-size 50
fi
