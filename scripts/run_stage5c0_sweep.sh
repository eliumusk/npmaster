#!/usr/bin/env bash
# Stage 5c-0: re-run the Stage 5b decoder sweep on the 24 existing val
# predictions, but with --bgc-csv pointing to bgc_annotations_core.csv
# (proto_core tight coords) instead of bgc_annotations.csv (region wide).
# No training, no inference.

set -euo pipefail

cd /data/muskliu/npmaster
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc
export PYTHONPATH="src:${PYTHONPATH:-}"

N_PARALLEL="${1:-16}"
OUT_DIR="experiments/stage5c0_per_tag"
mkdir -p "${OUT_DIR}"
BGC_CORE="/data/muskliu/npmaster/data/bgc_annotations_core.csv"

mapfile -t TAGS < <(find experiments -type d -name 'val_predictions_*' \
  | sed 's|.*/val_predictions_||' | sort -u)
echo "found ${#TAGS[@]} tags; running up to ${N_PARALLEL} in parallel against core-val GT"

PIDS=()
for tag in "${TAGS[@]}"; do
  out_prefix="${OUT_DIR}/${tag}"
  if [[ -f "${out_prefix}_all_cells.csv" ]]; then
    echo "skip ${tag} (already done)"
    continue
  fi
  (
    python scripts/stage5b_val_decoder_sweep.py \
      --val-pred-root experiments \
      --bgc-csv "${BGC_CORE}" \
      --out-prefix "${out_prefix}" \
      --only-checkpoint-tags "${tag}" \
      > "${OUT_DIR}/${tag}.log" 2>&1
  ) &
  PIDS+=("$!")
  if (( ${#PIDS[@]} >= N_PARALLEL )); then
    wait -n
  fi
done

wait
echo "all core-val sweeps done"

python - <<'PY'
import pandas as pd, glob
all_cells = [pd.read_csv(p) for p in sorted(glob.glob("experiments/stage5c0_per_tag/*_all_cells.csv"))]
selected = [pd.read_csv(p) for p in sorted(glob.glob("experiments/stage5c0_per_tag/*_selected.csv"))]
if all_cells:
    pd.concat(all_cells, ignore_index=True).to_csv(
        "experiments/stage5c0_core_decoder_sweep_all_cells.csv", index=False
    )
if selected:
    pd.concat(selected, ignore_index=True).to_csv(
        "experiments/stage5c0_core_decoder_sweep_selected.csv", index=False
    )
print("merged outputs in experiments/")
PY
