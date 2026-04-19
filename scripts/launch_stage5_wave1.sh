#!/usr/bin/env bash
# Launch Stage 5 wave-1 trials (6 variants, seed 0) across 6 GPUs on
# the current node. Each trial's full command.txt pipeline (leakage
# check → build labels → patch meta → train) runs in the background
# under nohup + setsid with GPU pinning.
#
# Usage: bash scripts/launch_stage5_wave1.sh [GPU_OFFSET]
# GPU_OFFSET shifts the CUDA_VISIBLE_DEVICES assignment (default 0).
# Run this on the server after syncing the repo + experiment dirs.

set -euo pipefail

cd /data/muskliu/npmaster
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc
export PYTHONPATH="src:${PYTHONPATH:-}"

OFFSET="${1:-0}"

TRIALS=(
  "baseline_v1"
  "tight_pos_075_v1"
  "tight_pos_090_v1"
  "truncated_exclude_v1"
  "short_bgc_upweight_v1"
  "hardpos_hardneg_v2"
)

LAUNCH_LOG="experiments/stage5_no9_data_v1-launch.log"
echo "=== Stage 5 wave-1 launch at $(date -Is) ===" | tee -a "$LAUNCH_LOG"
echo "host $(hostname) gpu_offset=${OFFSET}" | tee -a "$LAUNCH_LOG"

for i in "${!TRIALS[@]}"; do
  t="${TRIALS[$i]}"
  gpu=$(( i + OFFSET ))
  exp="experiments/stage5_no9_data_v1-${t}"
  mkdir -p "${exp}/logs"
  log="${exp}/logs/run_s0.log"
  echo "launching trial=${t} gpu=${gpu} log=${log}" | tee -a "$LAUNCH_LOG"

  # nohup + setsid so the process survives ssh disconnect.
  CUDA_VISIBLE_DEVICES="${gpu}" nohup setsid bash "${exp}/command.txt" \
    > "${log}" 2>&1 &
  pid=$!
  echo "  started pid=${pid}" | tee -a "$LAUNCH_LOG"
  echo "${t},${gpu},${pid},$(date -Is)" >> \
    experiments/stage5_no9_data_v1-launch-pids.csv
  sleep 1  # stagger start to avoid simultaneous label-CSV writes.
done

echo "=== all 6 trials launched ===" | tee -a "$LAUNCH_LOG"
