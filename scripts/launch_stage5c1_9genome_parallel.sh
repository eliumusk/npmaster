#!/usr/bin/env bash
# Stage 5c-1 final 9-genome: per-(checkpoint, genome) parallel inference
# for the 2 Stage 5c-1 winners, then evaluation under their frozen
# decoders.
#
# Usage:
#   bash scripts/launch_stage5c1_9genome_parallel.sh node7
#   bash scripts/launch_stage5c1_9genome_parallel.sh node8
# Split by hash; each node takes half.

set -euo pipefail

cd /data/muskliu/npmaster

TARGET="${1:-node7}"

source /data/conda/ourconda_bashrc
conda activate evo2_bgc
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HUB_CACHE=/data/public_models/hub

FNA_DIR="/data/syh/benchmark/nine_genome/fna"
WINDOW=8192
STRIDE=2048

GENOMES=(
  "GCA_000154945.1"
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
  "experiments/stage5c1-short_core_upweight_v1|short_core_upweight_v1_s0"
  "experiments/stage5c1-baseline_core_v1|baseline_core_v1_s0"
)

LAUNCH_LOG="experiments/stage5c1_9genome_parallel_${TARGET}.log"
mkdir -p "$(dirname "${LAUNCH_LOG}")"
echo "=== stage5c1 9-genome parallel ${TARGET} $(date -Is) ===" | tee -a "${LAUNCH_LOG}"

mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 500 {print $1}')
NUM_GPUS=${#FREE_GPUS[@]}
if (( NUM_GPUS == 0 )); then
  FREE_GPUS=(0); NUM_GPUS=1
fi
echo "free GPUs: ${FREE_GPUS[*]}" | tee -a "${LAUNCH_LOG}"

# Enumerate (checkpoint, genome) pairs.
ALL_JOBS=()
for ck in "${CHECKPOINTS[@]}"; do
  for g in "${GENOMES[@]}"; do
    ALL_JOBS+=("${ck}|${g}")
  done
done
echo "total jobs: ${#ALL_JOBS[@]}" | tee -a "${LAUNCH_LOG}"

# Hash-split even/odd across nodes.
MINE=()
idx=0
for j in "${ALL_JOBS[@]}"; do
  parity=$(( idx % 2 ))
  if [[ "${TARGET}" == "node7" && "${parity}" == "0" ]]; then
    MINE+=("${j}")
  elif [[ "${TARGET}" == "node8" && "${parity}" == "1" ]]; then
    MINE+=("${j}")
  fi
  idx=$(( idx + 1 ))
done
echo "this node: ${#MINE[@]} jobs" | tee -a "${LAUNCH_LOG}"

PIDS=()
GPU_IDX=0
for job in "${MINE[@]}"; do
  IFS='|' read -r trial_dir tag genome <<<"${job}"
  ckpt="${trial_dir}/artifacts/checkpoints/best_${tag}.pt"
  pred_dir="${trial_dir}/artifacts/predictions_9genome_${tag}"
  mkdir -p "${pred_dir}"

  if [[ -f "${pred_dir}/${genome}_predictions.csv" ]]; then
    echo "skip ${tag} ${genome} (already saved)" | tee -a "${LAUNCH_LOG}"
    continue
  fi

  gpu="${FREE_GPUS[$(( GPU_IDX % NUM_GPUS ))]}"
  GPU_IDX=$(( GPU_IDX + 1 ))
  log="${trial_dir}/logs/final_9genome_${tag}_${genome}.log"
  mkdir -p "$(dirname "${log}")"

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

  if (( ${#PIDS[@]} >= NUM_GPUS )); then
    wait -n || true
    NEW=()
    for p in "${PIDS[@]}"; do
      if kill -0 "$p" 2>/dev/null; then NEW+=("$p"); fi
    done
    PIDS=("${NEW[@]}")
  fi
done

wait
echo "=== ${TARGET} done $(date -Is) ===" | tee -a "${LAUNCH_LOG}"
