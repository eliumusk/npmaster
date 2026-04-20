#!/usr/bin/env bash
# Run the Stage 5b decoder sweep in parallel — one process per
# checkpoint tag, up to N at a time on CPU.

set -euo pipefail

cd /data/muskliu/npmaster
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc
export PYTHONPATH="src:${PYTHONPATH:-}"

N_PARALLEL="${1:-12}"
OUT_DIR="experiments/stage5b_per_tag"
mkdir -p "${OUT_DIR}"

mapfile -t TAGS < <(find experiments -type d -name 'val_predictions_*' \
  | sed 's|.*/val_predictions_||' | sort -u)
echo "found ${#TAGS[@]} tags; running up to ${N_PARALLEL} in parallel"

PIDS=()
for tag in "${TAGS[@]}"; do
  out_prefix="${OUT_DIR}/${tag}"
  # Skip if already done
  if [[ -f "${out_prefix}_all_cells.csv" ]]; then
    echo "skip ${tag} (already done)"
    continue
  fi

  (
    python scripts/stage5b_val_decoder_sweep.py \
      --val-pred-root experiments \
      --out-prefix "${out_prefix}" \
      --only-checkpoint-tags "${tag}" \
      > "${OUT_DIR}/${tag}.log" 2>&1
  ) &
  PIDS+=("$!")

  if (( ${#PIDS[@]} >= N_PARALLEL )); then
    wait -n  # wait for any one child to finish, then dispatch next
  fi
done

wait
echo "all sweeps done"

# Concatenate the per-tag outputs.
python - <<'PY'
import pandas as pd, glob, os
all_cells = [pd.read_csv(p) for p in sorted(glob.glob("experiments/stage5b_per_tag/*_all_cells.csv"))]
selected = [pd.read_csv(p) for p in sorted(glob.glob("experiments/stage5b_per_tag/*_selected.csv"))]
if all_cells:
    pd.concat(all_cells, ignore_index=True).to_csv(
        "experiments/stage5b_val_decoder_sweep_all_cells.csv", index=False
    )
if selected:
    pd.concat(selected, ignore_index=True).to_csv(
        "experiments/stage5b_val_decoder_sweep_selected.csv", index=False
    )
print("merged outputs in experiments/")
PY
