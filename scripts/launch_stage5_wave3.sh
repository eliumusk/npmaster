#!/usr/bin/env bash
# Launch Stage 5 wave-3 seed=2 trials. Splits across both nodes:
#   node7 GPU 6,7: baseline_v1_s2, tight_pos_075_v1_s2
#   node8 GPU 6,7: tight_pos_090_v1_s2, hardpos_hardneg_v2_s2
#
# Run on the target node; pass --host-trials to pick its pair.
# Usage: bash scripts/launch_stage5_wave3.sh node7|node8

set -euo pipefail

cd /data/muskliu/npmaster
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc
export PYTHONPATH="src:${PYTHONPATH:-}"

TARGET="${1:-node7}"

if [[ "$TARGET" == "node7" ]]; then
  TRIALS=("baseline_v1_s2" "tight_pos_075_v1_s2")
  GPU_BASE=6
elif [[ "$TARGET" == "node8" ]]; then
  TRIALS=("tight_pos_090_v1_s2" "hardpos_hardneg_v2_s2")
  GPU_BASE=6
else
  echo "unknown target: $TARGET" >&2
  exit 1
fi

LAUNCH_LOG="experiments/stage5_no9_data_v1-launch_wave3_${TARGET}.log"
echo "=== Stage 5 wave-3 launch at $(date -Is) on ${TARGET} ===" | tee -a "$LAUNCH_LOG"

for i in "${!TRIALS[@]}"; do
  t="${TRIALS[$i]}"
  gpu=$(( i + GPU_BASE ))
  exp="experiments/stage5_no9_data_v1-${t}"
  mkdir -p "${exp}/logs"
  log="${exp}/logs/run_s2.log"
  echo "launching trial=${t} gpu=${gpu}" | tee -a "$LAUNCH_LOG"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup setsid bash "${exp}/command.txt" \
    > "${log}" 2>&1 &
  pid=$!
  echo "  pid=${pid}" | tee -a "$LAUNCH_LOG"
  echo "${t},${gpu},${pid},$(date -Is)" >> \
    "experiments/stage5_no9_data_v1-launch_wave3-pids.csv"
  sleep 1
done

echo "=== all wave-3 trials launched on ${TARGET} ===" | tee -a "$LAUNCH_LOG"
