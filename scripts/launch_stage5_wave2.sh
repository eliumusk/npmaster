#!/usr/bin/env bash
# Launch Stage 5 wave-2 trials (seed 1 replicas, 6 variants) on 6 GPUs.
# Run this on node8 after wave-1 is up on node7.

set -euo pipefail

cd /data/muskliu/npmaster
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc
export PYTHONPATH="src:${PYTHONPATH:-}"

OFFSET="${1:-0}"

TRIALS=(
  "baseline_v1_s1"
  "tight_pos_075_v1_s1"
  "tight_pos_090_v1_s1"
  "truncated_exclude_v1_s1"
  "short_bgc_upweight_v1_s1"
  "hardpos_hardneg_v2_s1"
)

LAUNCH_LOG="experiments/stage5_no9_data_v1-launch_wave2.log"
echo "=== Stage 5 wave-2 launch at $(date -Is) ===" | tee -a "$LAUNCH_LOG"
echo "host $(hostname) gpu_offset=${OFFSET}" | tee -a "$LAUNCH_LOG"

for i in "${!TRIALS[@]}"; do
  t="${TRIALS[$i]}"
  gpu=$(( i + OFFSET ))
  exp="experiments/stage5_no9_data_v1-${t}"
  mkdir -p "${exp}/logs"
  log="${exp}/logs/run_s1.log"
  echo "launching trial=${t} gpu=${gpu} log=${log}" | tee -a "$LAUNCH_LOG"

  CUDA_VISIBLE_DEVICES="${gpu}" nohup setsid bash "${exp}/command.txt" \
    > "${log}" 2>&1 &
  pid=$!
  echo "  started pid=${pid}" | tee -a "$LAUNCH_LOG"
  echo "${t},${gpu},${pid},$(date -Is)" >> \
    experiments/stage5_no9_data_v1-launch_wave2-pids.csv
  sleep 1
done

echo "=== all 6 wave-2 trials launched ===" | tee -a "$LAUNCH_LOG"
