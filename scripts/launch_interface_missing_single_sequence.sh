#!/usr/bin/env bash
# Run only missing or superseded interface validation groups (300 uniform frames).
set -euo pipefail
gpu=${1:?usage: $0 PHYSICAL_GPU WORKER}
worker=${2:?usage: $0 PHYSICAL_GPU WORKER}
root=/data/mmc_syang
outroot=${root}/VGGT-omega/outputs/interface_single_sequence_300
logroot=${outroot}/logs
mkdir -p "${logroot}"

# DA rows are deliberately rerun: older rows used the retired fixed-anchor wrapper.
jobs=(
 'omega|da-vggt|tum' 'omega|da-vggt|7scenes'
 'vggt|fastvggt|tum' 'vggt|fastvggt|7scenes' 'vggt|sparse-vggt|7scenes' 'vggt|u-m|tum' 'vggt|u-m|7scenes' 'vggt|da-vggt|tum' 'vggt|da-vggt|7scenes'
 'pi3|fastvggt|7scenes' 'pi3|sparse-vggt|tum' 'pi3|u-m|7scenes' 'pi3|da-vggt|tum' 'pi3|da-vggt|7scenes'
)

omega() {
 local method=$1 dataset=$2 out=$3 script seq extra accel
 if [[ $dataset == tum ]]; then script=scripts/eval_tum_dynamics_paper.py; seq=rgbd_dataset_freiburg3_walking_rpy; extra=(--data-root "$root/dataset/TUM-Dynamics" --sequences "$seq" --sampling-pool full --sampling-strategy uniform); else script=scripts/eval_7scenes_paper.py; seq=chess/seq-03; extra=(--data-root "$root/dataset/7scenes" --sequences "$seq" --sampling-unit sequence --sampling-strategy uniform); fi
 accel=(--acceleration-method "$method" --merge-ratio 0 --frame-fusion-mode none)
 [[ $method == da-vggt ]] && accel=(--acceleration-method da-vggt --merge-ratio 0 --frame-fusion-mode none --da-chunk-size 50)
 (cd "$root/VGGT-omega" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$root/VGGT-omega" "$root/miniconda3/envs/fastvggt/bin/python" "$script" "${extra[@]}" --checkpoint "$root/vggt-omega/checkpoints/vggt_omega_1b_512.pt" --output-dir "$out" --device cuda:0 --num-frames 300 --image-resolution 512 --resize-mode max_size --timing-repeats 1 "${accel[@]}")
}
vggt() {
 local method=$1 dataset=$2 out=$3 seq data
 [[ $dataset == tum ]] && seq=rgbd_dataset_freiburg3_walking_rpy && data=tum_dynamic || { seq=chess/seq-03; data=7scenes; }
 (cd "$root/vggt" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$root/vggt" "$root/miniconda3/envs/fastvggt/bin/python" scripts/eval_standard_tum_7scenes.py --dataset "$data" --dataset-root "$([[ $dataset == tum ]] && echo "$root/dataset/TUM-Dynamics" || echo "$root/dataset/7scenes")" --checkpoint ckpts/model.pt --output-dir "$out" --device cuda:0 --sequences "$seq" --num-frames 300 --image-resolution 518 --timing-repeats 1 --acceleration-method "$method" --merge-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --um-lambda 0.04 --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers 0,10,17 --da-chunk-size 50 --overwrite)
}
pi3() {
 local method=$1 dataset=$2 out=$3 seq data
 [[ $dataset == tum ]] && seq=rgbd_dataset_freiburg3_walking_rpy && data=tum_dynamic || { seq=chess/seq-03; data=7scenes; }
 (cd "$root/Pi3" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$root/Pi3" "$root/miniconda3/envs/flow3r/bin/python" scripts/run_pi3_vggt_omega_eval.py --dataset "$data" --dataset-root "$([[ $dataset == tum ]] && echo "$root/dataset/TUM-Dynamics" || echo "$root/dataset/7scenes")" --pretrained checkpoints/Pi3 --output-dir "$out" --device cuda:0 --sequences "$seq" --max-frames-per-seq 300 --load-img-size 512 --timing-repeats 1 --acceleration-method "$method" --token-merging-ratio 0.9 --sparse-vggt-sparse-ratio 0.5 --um-lambda 0.04 --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers 0,10,17 --da-chunk-size 50 --overwrite)
}
for i in "${!jobs[@]}"; do
 (( i % 3 == worker )) || continue
 IFS='|' read -r model method dataset <<< "${jobs[i]}"
 out="$outroot/$model/$method/$dataset"; log="$logroot/missing_${model}_${method}_${dataset}.log"
 echo "[$(date -Is)] START gpu=$gpu $model/$method/$dataset" | tee -a "$log"
 case $model in omega) omega "$method" "$dataset" "$out" >>"$log" 2>&1;; vggt) vggt "$method" "$dataset" "$out" >>"$log" 2>&1;; pi3) pi3 "$method" "$dataset" "$out" >>"$log" 2>&1;; esac
 echo "[$(date -Is)] DONE gpu=$gpu $model/$method/$dataset" | tee -a "$log"
done
