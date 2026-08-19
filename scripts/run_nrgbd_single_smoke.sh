#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu}; model=${2:?omega|vggt|pi3}; method=${3:?method}
root=/data/mmc_syang; out=${root}/VGGT-omega/outputs/01/nrgbd_single_smoke/${model}/${method}
mkdir -p "${out}"
seq=complete_kitchen
if [[ ${model} == omega ]]; then
  cd ${root}/VGGT-omega
  args=(--data-root ${root}/dataset/NRGBD --checkpoint ${root}/vggt-omega/checkpoints/vggt_omega_1b_512.pt --output-dir "${out}" --device cuda:0 --sequences ${seq} --num-frames 300 --sampling-unit sequence --sampling-strategy uniform --image-resolution 512 --resize-mode max_size --timing-repeats 1)
  case ${method} in
    baseline) args+=(--acceleration-method none --merge-ratio 0 --frame-fusion-mode none);;
    fastvggt) args+=(--acceleration-method fastvggt --merge-ratio 0.9 --frame-fusion-mode none);;
    sparse-vggt) args+=(--acceleration-method sparse-vggt --merge-ratio 0 --frame-fusion-mode none --sparse-attention --sparse-ratio 0.5 --sparse-pool-mode avg);;
    da-vggt) args+=(--acceleration-method da-vggt --merge-ratio 0 --frame-fusion-mode none --da-chunk-size 50);;
    u-m) args+=(--acceleration-method u-m --merge-ratio 0 --frame-fusion-mode u-m --frame-fusion-recompute-layers 0,10,17 --frame-fusion-lambda-cost 0.03 --frame-fusion-temporal-window 4 --frame-fusion-spatial-radius 2 --frame-fusion-attention-variant representative);;
  esac
  exec env CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/VGGT-omega ${root}/miniconda3/envs/fastvggt/bin/python scripts/eval_nrgbd_paper.py "${args[@]}"
elif [[ ${model} == vggt ]]; then
  cd ${root}/vggt
  exec env CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/vggt ${root}/miniconda3/envs/fastvggt/bin/python scripts/eval_nrgbd_paper.py --dataset-root ${root}/dataset/NRGBD --checkpoint ckpts/model.pt --output-dir "${out}" --device cuda:0 --sequences ${seq} --num-frames 300 --image-resolution 518 --timing-repeats 1 --acceleration-method "${method/baseline/none}" --merge-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --da-chunk-size 50
else
  cd ${root}/Pi3
  exec env CUDA_VISIBLE_DEVICES=${gpu} VGGT_UM_TRITON=1 PYTHONPATH=${root}/Pi3 ${root}/miniconda3/envs/flow3r/bin/python scripts/eval_nrgbd_paper.py --dataset-root ${root}/dataset/NRGBD --pretrained checkpoints/Pi3 --output-dir "${out}" --device cuda:0 --sequences ${seq} --max-frames-per-seq 300 --frame-sample-mode uniform --load-img-size 512 --timing-repeats 1 --acceleration-method "${method/baseline/none}" --token-merging-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --da-chunk-size 50
fi
