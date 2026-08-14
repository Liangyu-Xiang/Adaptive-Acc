#!/usr/bin/env bash
set -euo pipefail
# Do not interrupt the current Pi3 baseline-TUM run. Once its queue advances
# to 7Scenes, terminate the obsolete queue because that job runs on GPU1.
while true; do
  if ps -eo args | grep -F 'run_formal500_pi3_omega_one.sh 7 pi3 none 7scenes' | grep -v grep >/dev/null; then
    tmux kill-session -t formal500_retry_gpu7
    exit 0
  fi
  sleep 2
done
