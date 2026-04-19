#!/usr/bin/env bash
# Server-side end-to-end orchestrator for Stage 5.
#
#   1. Wait until (almost) all expected trials have finished training
#      (poll nvidia-smi — done when ≥7 of 8 GPUs are idle for 2 consecutive
#      polls; robust against a single stuck trial).
#   2. Run scripts/summarize_stage5_results.py.
#   3. Run scripts/stage5_pick_winners.py (top-3 variants by 3-seed mean
#      peak val_region_f1).
#   4. For each winner variant, pick the best-seed checkpoint and launch
#      final 9-genome inference + eval via launch_stage5_9genome_eval.sh.
#
# Designed to run under nohup on node7 and survive the Claude Code
# session ending. Logs under experiments/stage5_no9_data_v1-orchestrator.log.

set -euo pipefail

cd /data/muskliu/npmaster
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc

LOG="experiments/stage5_no9_data_v1-orchestrator.log"
mkdir -p "$(dirname "${LOG}")"
exec >>"${LOG}" 2>&1

echo "=== orchestrator start $(date -Is) pid=$$ on $(hostname) ==="

IDLE_POLLS_NEEDED=2
POLL_INTERVAL_SEC=120
MAX_WAIT_HOURS=8
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))

idle_count=0
while true; do
  if (( $(date +%s) > deadline )); then
    echo "$(date -Is) deadline reached while waiting for idle GPUs; proceeding anyway"
    break
  fi

  busy_gpus=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 >= 500 {c++} END {print c+0}')
  echo "$(date -Is) busy_gpus=${busy_gpus}"
  if (( busy_gpus <= 1 )); then
    idle_count=$(( idle_count + 1 ))
    if (( idle_count >= IDLE_POLLS_NEEDED )); then
      break
    fi
  else
    idle_count=0
  fi
  sleep "${POLL_INTERVAL_SEC}"
done

echo "$(date -Is) training phase appears complete; summarizing..."
python scripts/summarize_stage5_results.py

echo "$(date -Is) picking top-3 winners..."
python scripts/stage5_pick_winners.py --top 3 \
  --selection-note "Autonomous orchestrator selection; rungs 0/1/2 + tight threshold extension (0.85/0.95/0.99). Peak val_region_f1 across completed seeds."

echo "$(date -Is) parsing selection_decision.md for best-seed winner checkpoints..."
freeze="experiments/stage5_no9_data_v1-selection_decision.md"
if [[ ! -f "${freeze}" ]]; then
  echo "$(date -Is) FATAL: selection_decision.md missing, aborting"
  exit 1
fi

# Extract best-seed (highest peak val F1) per variant from the freeze file.
# The markdown table format is: | <seed> | <trial_id> | <peakF1 or —> | `<ckpt_path>` |
# Strategy: per variant section (### <variant>), scan rows, pick max F1 → emit ckpt path.
winner_ckpts=$(python - <<'PY'
from pathlib import Path
import re

freeze = Path("experiments/stage5_no9_data_v1-selection_decision.md")
text = freeze.read_text(encoding="utf-8")

sections = re.split(r"(?m)^### ", text)[1:]  # each starts with the variant name
winners = []
for block in sections:
    variant, _, body = block.partition("\n")
    variant = variant.strip()
    best_ckpt = None
    best_f1 = -1.0
    for line in body.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*`([^`]+)`\s*\|", line)
        if not m:
            continue
        _seed, _trial, f1_text, ckpt = m.groups()
        f1_text = f1_text.strip()
        if f1_text == "—":
            continue
        try:
            f1 = float(f1_text)
        except ValueError:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_ckpt = ckpt.strip()
    if best_ckpt:
        winners.append(best_ckpt)

print("\n".join(winners))
PY
)

if [[ -z "${winner_ckpts}" ]]; then
  echo "$(date -Is) FATAL: no winner checkpoints parsed from freeze file, aborting"
  exit 2
fi

echo "$(date -Is) winner checkpoints:"
echo "${winner_ckpts}"

# Convert newlines to space-separated args for the launcher.
IFS=$'\n' read -rd '' -a winner_array <<<"${winner_ckpts}" || true

echo "$(date -Is) launching 9-genome eval..."
bash scripts/launch_stage5_9genome_eval.sh "${winner_array[@]}"

echo "$(date -Is) orchestrator done — 9-genome eval running in background, " \
  "results will appear under experiments/<winner>/metrics/final_9genome_*.json"
