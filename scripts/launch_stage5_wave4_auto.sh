#!/usr/bin/env bash
# Auto-launches Stage 5 wave-4 trials as GPUs free up on the current node.
# Polls `nvidia-smi` every 30s; as soon as a GPU shows <500 MiB used, it
# dispatches the next unlaunched trial to that GPU.
#
# Intended to run on each node AFTER wave-1/2/3 have been started — the
# script will sit idle while those trials use the GPUs, then pick them
# up as they finish.
#
# Usage: bash scripts/launch_stage5_wave4_auto.sh node7|node8 &
# node7 gets 4 trials, node8 gets 4 — split the 8-trial wave evenly.

set -euo pipefail

cd /data/muskliu/npmaster
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc
export PYTHONPATH="src:${PYTHONPATH:-}"

TARGET="${1:-node7}"
MAX_POLL_HOURS="${2:-6}"
POLL_INTERVAL_SEC=30
FREE_THRESHOLD_MIB=500

if [[ "$TARGET" == "node7" ]]; then
  TRIALS=(
    "tight_pos_085_v1_s0"
    "tight_pos_095_v1_s0"
    "tight_pos_099_v1_s0"
    "tight_pos_090_v1_long_s0"
  )
elif [[ "$TARGET" == "node8" ]]; then
  TRIALS=(
    "tight_pos_085_v1_s1"
    "tight_pos_095_v1_s1"
    "tight_pos_095_v1_s2"
    "tight_pos_099_v1_s1"
  )
else
  echo "unknown target: $TARGET" >&2
  exit 1
fi

LAUNCH_LOG="experiments/stage5_no9_data_v1-launch_wave4_${TARGET}.log"
PIDS_CSV="experiments/stage5_no9_data_v1-launch_wave4_${TARGET}-pids.csv"

echo "=== wave-4 auto-launcher start $(date -Is) target=${TARGET} pid=$$ ===" \
  | tee -a "$LAUNCH_LOG"
echo "queued trials: ${TRIALS[*]}" | tee -a "$LAUNCH_LOG"

deadline=$(( $(date +%s) + MAX_POLL_HOURS * 3600 ))
next_idx=0

while (( next_idx < ${#TRIALS[@]} )); do
  if (( $(date +%s) > deadline )); then
    echo "$(date -Is) deadline reached with ${#TRIALS[@]} - ${next_idx} trials unlaunched; giving up" \
      | tee -a "$LAUNCH_LOG"
    exit 2
  fi

  # Find a GPU with <FREE_THRESHOLD_MIB used memory.
  free_gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -v t="${FREE_THRESHOLD_MIB}" -F', ' '$2 < t {print $1; exit}')

  if [[ -z "${free_gpu:-}" ]]; then
    sleep "${POLL_INTERVAL_SEC}"
    continue
  fi

  t="${TRIALS[$next_idx]}"
  exp="experiments/stage5_no9_data_v1-${t}"
  if [[ ! -f "${exp}/command.txt" ]]; then
    echo "$(date -Is) skipping ${t}: no command.txt at ${exp}" \
      | tee -a "$LAUNCH_LOG"
    next_idx=$(( next_idx + 1 ))
    continue
  fi

  # Double-check the GPU still looks free (race against another launcher).
  current_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${free_gpu}")
  if (( current_used >= FREE_THRESHOLD_MIB )); then
    sleep "${POLL_INTERVAL_SEC}"
    continue
  fi

  # Infer seed from trial name suffix (_s<N>) for the log filename.
  seed_tag="$(echo "$t" | grep -oE '_s[0-9]+$' | tr -d '_' || echo s0)"
  log="${exp}/logs/run_${seed_tag}.log"
  mkdir -p "${exp}/logs"

  echo "$(date -Is) dispatching ${t} -> gpu=${free_gpu}" | tee -a "$LAUNCH_LOG"
  CUDA_VISIBLE_DEVICES="${free_gpu}" nohup setsid bash "${exp}/command.txt" \
    > "${log}" 2>&1 &
  pid=$!
  echo "${t},${free_gpu},${pid},$(date -Is)" >> "${PIDS_CSV}"
  echo "  pid=${pid} log=${log}" | tee -a "$LAUNCH_LOG"

  next_idx=$(( next_idx + 1 ))
  # Sleep a little so the new process claims its GPU memory before we poll.
  sleep 10
done

echo "=== wave-4 auto-launcher finished $(date -Is) on ${TARGET} ===" \
  | tee -a "$LAUNCH_LOG"
