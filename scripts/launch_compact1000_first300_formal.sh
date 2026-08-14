#!/usr/bin/env bash
# 15 model/interface configurations x TUM/7Scenes, six independent GPU queues.
set -euo pipefail
root=/data/mmc_syang
outroot=${root}/VGGT-omega/outputs/compact1000_first300_formal
logroot=${outroot}/logs
mkdir -p "${logroot}"
gpus=(1 3 4 5 6 7)
jobs=()
for model in omega pi3 vggt; do
  for method in none fastvggt sparse-vggt da-vggt u-m; do
    for dataset in tum 7scenes; do jobs+=("${model} ${method} ${dataset}"); done
  done
done
for slot in "${!gpus[@]}"; do
  gpu=${gpus[$slot]}
  queue=${logroot}/gpu${gpu}.queue
  : >"${queue}"
  for idx in "${!jobs[@]}"; do
    if (( idx % ${#gpus[@]} == slot )); then
      echo "${jobs[$idx]}" >>"${queue}"
    fi
  done
  nohup setsid bash -lc "while read -r model method dataset; do echo \"[\$(date -Is)] START \$model \$method \$dataset\"; bash '${root}/VGGT-omega/scripts/run_compact1000_first300_one.sh' '${gpu}' \"\$model\" \"\$method\" \"\$dataset\"; echo \"[\$(date -Is)] DONE \$model \$method \$dataset rc=\$?\"; done < '${queue}'" >"${logroot}/gpu${gpu}.log" 2>&1 < /dev/null &
  echo $! >"${logroot}/gpu${gpu}.pid"
done
