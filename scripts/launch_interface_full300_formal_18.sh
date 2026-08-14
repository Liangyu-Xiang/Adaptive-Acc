#!/usr/bin/env bash
# 18 acceleration configurations x {TUM, 7Scenes}; all use 300 uniform frames
# and formal CUDA Event median timing.  Ω/Pi3 are always exhausted before VGGT.
set -euo pipefail

gpu=${1:?usage: $0 PHYSICAL_GPU WORKER_INDEX}
worker=${2:?usage: $0 PHYSICAL_GPU WORKER_INDEX}
root=/data/mmc_syang
outroot=${root}/VGGT-omega/outputs/interface_full300_formal_18
logroot=${outroot}/logs
barrier_dir=${outroot}/.phase1_omega_pi3_done
mkdir -p "$logroot"

run_omega() {
  local method=$1 lambda=$2 dataset=$3 out=$4 script accel=()
  if [[ $dataset == tum ]]; then
    script=scripts/eval_tum_dynamics_paper.py
    accel=(--data-root "$root/dataset/TUM-Dynamics" --sampling-pool full --sampling-strategy uniform)
  else
    script=scripts/eval_7scenes_paper.py
    accel=(--data-root "$root/dataset/7scenes" --sampling-unit sequence --sampling-strategy uniform)
  fi
  case $method in
    fastvggt) accel+=(--acceleration-method fastvggt --merge-ratio 0.9 --frame-fusion-mode none) ;;
    sparse-vggt) accel+=(--acceleration-method sparse-vggt --merge-ratio 0 --frame-fusion-mode none --sparse-attention --sparse-ratio 0.5 --sparse-pool-mode avg) ;;
    da-vggt) accel+=(--acceleration-method da-vggt --merge-ratio 0 --frame-fusion-mode none --da-chunk-size 50) ;;
    u-m) accel+=(--acceleration-method u-m --merge-ratio 0 --frame-fusion-mode u-m --frame-fusion-start-layer -1 --frame-fusion-recompute-layers 0,10,17 --frame-fusion-lambda-cost "$lambda" --frame-fusion-merge-top-similarity-percent 100 --frame-fusion-min-keep-ratio 0.05 --frame-fusion-temporal-window 4 --frame-fusion-spatial-radius 2 --frame-fusion-attention-variant representative) ;;
  esac
  (cd "$root/VGGT-omega" && CUDA_VISIBLE_DEVICES=$gpu VGGT_UM_TRITON=1 PYTHONPATH="$root/VGGT-omega" "$root/miniconda3/envs/fastvggt/bin/python" "$script" "${accel[@]}" --checkpoint "$root/vggt-omega/checkpoints/vggt_omega_1b_512.pt" --output-dir "$out" --device cuda:0 --num-frames 300 --image-resolution 512 --resize-mode max_size --timing-repeats 3)
}

run_pi3() {
  local method=$1 lambda=$2 dataset=$3 out=$4 data
  [[ $dataset == tum ]] && data=tum_dynamic || data=7scenes
  (cd "$root/Pi3" && CUDA_VISIBLE_DEVICES=$gpu VGGT_UM_TRITON=1 PYTHONPATH="$root/Pi3" "$root/miniconda3/envs/flow3r/bin/python" scripts/run_pi3_vggt_omega_eval.py --dataset "$data" --dataset-root "$([[ $dataset == tum ]] && echo "$root/dataset/TUM-Dynamics" || echo "$root/dataset/7scenes")" --pretrained checkpoints/Pi3 --output-dir "$out" --device cuda:0 --max-frames-per-seq 300 --frame-sample-mode uniform --load-img-size 512 --timing-repeats 3 --acceleration-method "$method" --token-merging-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --um-lambda "$lambda" --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers 0,10,17 --da-chunk-size 50 --overwrite)
}

run_vggt() {
  local method=$1 lambda=$2 dataset=$3 out=$4 data
  [[ $dataset == tum ]] && data=tum_dynamic || data=7scenes
  (cd "$root/vggt" && CUDA_VISIBLE_DEVICES=$gpu VGGT_UM_TRITON=1 PYTHONPATH="$root/vggt" "$root/miniconda3/envs/fastvggt/bin/python" scripts/eval_standard_tum_7scenes.py --dataset "$data" --dataset-root "$([[ $dataset == tum ]] && echo "$root/dataset/TUM-Dynamics" || echo "$root/dataset/7scenes")" --checkpoint ckpts/model.pt --output-dir "$out" --device cuda:0 --num-frames 300 --image-resolution 518 --timing-repeats 3 --acceleration-method "$method" --merge-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --um-lambda "$lambda" --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers 0,10,17 --da-chunk-size 50 --overwrite)
}

configs=(
  'fastvggt|0.04' 'sparse-vggt|0.04' 'da-vggt|0.04'
  'u-m|0.02' 'u-m|0.03' 'u-m|0.04'
)
task=0
phase1_arrived=0
for model in omega pi3 vggt; do
  # This is a global (not merely per-worker) barrier.  VGGT cannot start on
  # any of GPUs 5/6/7 until every worker has exhausted Ω and Pi3.
  if [[ $model == vggt && $phase1_arrived == 0 ]]; then
    mkdir -p "$barrier_dir"
    touch "$barrier_dir/worker_${worker}"
    echo "[$(date -Is)] PHASE1_DONE gpu=$gpu; waiting for all Ω/Pi3 workers" >>"$logroot/worker_gpu${gpu}.log"
    while [[ $(find "$barrier_dir" -maxdepth 1 -type f -name 'worker_*' | wc -l) -lt 3 ]]; do
      sleep 15
    done
    echo "[$(date -Is)] PHASE2_START gpu=$gpu; all Ω/Pi3 workers complete" >>"$logroot/worker_gpu${gpu}.log"
    phase1_arrived=1
  fi
  for config in "${configs[@]}"; do
    IFS='|' read -r method lambda <<< "$config"
    for dataset in tum 7scenes; do
      if (( task % 3 == worker )); then
        tag="${model}_${method}_lambda_${lambda/./p}_${dataset}"
        out="$outroot/$model/$method/lambda_${lambda/./p}/$dataset"
        log="$logroot/$tag.log"
        echo "[$(date -Is)] START gpu=$gpu $model/$method lambda=$lambda $dataset" | tee -a "$log"
        case $model in
          omega) run_omega "$method" "$lambda" "$dataset" "$out" >>"$log" 2>&1 ;;
          pi3) run_pi3 "$method" "$lambda" "$dataset" "$out" >>"$log" 2>&1 ;;
          vggt) run_vggt "$method" "$lambda" "$dataset" "$out" >>"$log" 2>&1 ;;
        esac
        echo "[$(date -Is)] DONE gpu=$gpu $model/$method lambda=$lambda $dataset" | tee -a "$log"
      fi
      ((task+=1))
    done
  done
done
