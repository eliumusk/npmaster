#!/usr/bin/env bash
# Per-genome parallel inference for the 3 remaining Stage 5b-4
# checkpoints. Splits (checkpoint × genome) pairs across free GPUs on
# the current node; caller invokes once per node with a GPU offset.
#
# Usage:
#   bash scripts/launch_stage5b_9genome_parallel.sh node7
#   bash scripts/launch_stage5b_9genome_parallel.sh node8
#
# Each node handles a hash-split subset so the two sides don't collide
# on NFS writes. Genome 1 (GCA_000154945.1) is already saved for each
# checkpoint by the earlier serial jobs — skipped here.

set -euo pipefail

cd /data/muskliu/npmaster

TARGET="${1:-node7}"

# Environment: inference needs evo2_bgc; we activate on spawn.
source /data/conda/ourconda_bashrc
conda activate evo2_bgc
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HUB_CACHE=/data/public_models/hub

FNA_DIR="/data/syh/benchmark/nine_genome/fna"
WINDOW=8192
STRIDE=2048

GENOMES=(
  "GCA_000156435.1"
  "GCA_000156475.1"
  "GCA_000158815.1"
  "GCA_000158875.1"
  "GCA_000158895.1"
  "GCA_000158915.1"
  "GCA_000568255.1"
  "GCA_000568915.1"
)

CHECKPOINTS=(
  "experiments/stage5_no9_data_v1-baseline_v1|baseline_v1_s0"
  "experiments/stage5_no9_data_v1-short_bgc_upweight_v1|short_bgc_upweight_v1_s0"
  "experiments/stage5_no9_data_v1-tight_pos_075_v1|tight_pos_075_v1_s0"
)

LAUNCH_LOG="experiments/stage5b_parallel_${TARGET}.log"
mkdir -p "$(dirname "${LAUNCH_LOG}")"

echo "=== stage5b parallel launch ${TARGET} $(date -Is) ===" | tee -a "${LAUNCH_LOG}"

# All free GPUs on this node.
mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 500 {print $1}')
NUM_GPUS=${#FREE_GPUS[@]}
echo "free GPUs: ${FREE_GPUS[*]} (count=${NUM_GPUS})" | tee -a "${LAUNCH_LOG}"
if (( NUM_GPUS == 0 )); then
  echo "no free GPUs, nothing to do on ${TARGET}" | tee -a "${LAUNCH_LOG}"
  exit 0
fi

# Enumerate all (checkpoint, genome) pairs.
ALL_JOBS=()
for ck_entry in "${CHECKPOINTS[@]}"; do
  for g in "${GENOMES[@]}"; do
    ALL_JOBS+=("${ck_entry}|${g}")
  done
done

# Hash-split by job index so node7 takes half, node8 takes the other half.
MINE=()
idx=0
for job in "${ALL_JOBS[@]}"; do
  if [[ "$TARGET" == "node7" && $(( idx % 2 )) == 0 ]]; then
    MINE+=("${job}")
  elif [[ "$TARGET" == "node8" && $(( idx % 2 )) == 1 ]]; then
    MINE+=("${job}")
  fi
  idx=$(( idx + 1 ))
done
echo "this node: ${#MINE[@]} jobs" | tee -a "${LAUNCH_LOG}"

# Dispatch with concurrency = NUM_GPUS, wait -n after each full wave.
PIDS=()
GPU_IDX=0
for job in "${MINE[@]}"; do
  IFS='|' read -r trial_dir tag genome <<<"${job}"
  ckpt="${trial_dir}/artifacts/checkpoints/best_${tag}.pt"
  pred_dir="${trial_dir}/artifacts/predictions_9genome_${tag}"
  mkdir -p "${pred_dir}"

  # Skip if already saved.
  if [[ -f "${pred_dir}/${genome}_predictions.csv" ]]; then
    echo "skip ${tag} ${genome} (already saved)" | tee -a "${LAUNCH_LOG}"
    continue
  fi

  gpu="${FREE_GPUS[$(( GPU_IDX % NUM_GPUS ))]}"
  GPU_IDX=$(( GPU_IDX + 1 ))

  log="${trial_dir}/logs/final_9genome_b_${tag}_${genome}.log"
  echo "dispatch ${tag} ${genome} -> gpu=${gpu}" | tee -a "${LAUNCH_LOG}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
    python scripts/inference_stride.py \
      --fna-dir "${FNA_DIR}" \
      --out-dir "${pred_dir}" \
      --ckpt "${ckpt}" \
      --src-dir src \
      --genome-ids "${genome}" \
      --window "${WINDOW}" \
      --stride "${STRIDE}" \
      --layer blocks.20.mlp.l3 \
      --evo-model evo2_7b \
      --cache-dir /data/public_models/hub \
      > "${log}" 2>&1 &
  PIDS+=("$!")

  # Limit in-flight workers to NUM_GPUS.
  if (( ${#PIDS[@]} >= NUM_GPUS )); then
    wait -n || true
    # Rebuild PIDS with only still-running entries (cheap enough).
    NEW=()
    for p in "${PIDS[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        NEW+=("$p")
      fi
    done
    PIDS=("${NEW[@]}")
  fi
done

wait
echo "=== ${TARGET} parallel done $(date -Is) ===" | tee -a "${LAUNCH_LOG}"
