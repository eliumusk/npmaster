#!/usr/bin/env bash
# Discover every Stage 5 best_*.pt under experiments/ and dump val
# predictions for each, 16 in parallel across both nodes.
#
# Usage (from the server):
#   bash scripts/launch_stage5b_val_dumps.sh node7   # handles half
#   bash scripts/launch_stage5b_val_dumps.sh node8   # other half
#
# Hash-split by checkpoint name mod 2: node7 takes the even bucket,
# node8 the odd bucket. Simpler than a coordinator and avoids the two
# nodes clobbering the same output dir on NFS.

set -euo pipefail

cd /data/muskliu/npmaster
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc
export PYTHONPATH="src:${PYTHONPATH:-}"

TARGET="${1:-node7}"
META="/data/muskliu/npmaster/data/phase3/route_a/meta_arrays.pkl"
EMB="/data/muskliu/npmaster/data/embeddings/edge_padding/embeddings_merged.npy"
CIX="/data/muskliu/npmaster/data/phase3/route_a/contig_index.pkl"

LOG="experiments/stage5b_val_dumps_${TARGET}.log"
exec >"${LOG}" 2>&1

echo "=== stage 5b val dumps start $(date -Is) on ${TARGET} ==="

# Gather all best_*.pt checkpoints under the sweep.
mapfile -t ALL_CKPTS < <(find experiments -path 'experiments/stage5_no9_data_v1-*/artifacts/checkpoints/best_*.pt' | sort)
echo "found ${#ALL_CKPTS[@]} checkpoints"

# Split by hash of filename so each node picks half.
MINE=()
OTHER=0
for ck in "${ALL_CKPTS[@]}"; do
  # Even idx → node7, odd idx → node8.
  name="$(basename "$ck")"
  hash=$(cksum <<<"$name" | awk '{print $1}')
  bucket=$(( hash % 2 ))
  if { [[ "$TARGET" == "node7" && "$bucket" == 0 ]] || [[ "$TARGET" == "node8" && "$bucket" == 1 ]]; }; then
    MINE+=("$ck")
  fi
done
echo "this node handles ${#MINE[@]} checkpoints"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "GPU count: ${GPU_COUNT}"

# Dispatch 1 checkpoint per GPU slot, then wait between waves.
PIDS=()
SLOT=0
for ck in "${MINE[@]}"; do
  trial_dir="$(dirname "$(dirname "$(dirname "$ck")")")"
  tag="$(basename "$ck" .pt)"
  tag="${tag#best_}"
  out_dir="${trial_dir}/artifacts/val_predictions_${tag}"
  mkdir -p "${out_dir}"

  # If it's already populated with 169 genomes, skip.
  if [[ $(ls -1 "${out_dir}" 2>/dev/null | wc -l) -ge 169 ]]; then
    echo "skip ${tag} (already has $(ls -1 "${out_dir}" | wc -l) files)"
    continue
  fi

  gpu=$(( SLOT % GPU_COUNT ))
  SLOT=$(( SLOT + 1 ))

  echo "dispatch ${tag} -> gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python scripts/dump_val_predictions.py \
      --ckpt "${ck}" \
      --meta "${META}" \
      --embeddings "${EMB}" \
      --contig-index "${CIX}" \
      --out-dir "${out_dir}" \
    > "${trial_dir}/logs/val_dump_${tag}.log" 2>&1 &
  PIDS+=("$!")

  # Every GPU_COUNT dispatches, wait for the wave.
  if (( SLOT % GPU_COUNT == 0 )); then
    echo "waiting for wave..."
    wait "${PIDS[@]}" || true
    PIDS=()
  fi
done

# Final wave
if (( ${#PIDS[@]} > 0 )); then
  echo "waiting for final wave..."
  wait "${PIDS[@]}" || true
fi

echo "=== stage 5b val dumps done $(date -Is) ==="
