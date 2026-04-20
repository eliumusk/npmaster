#!/usr/bin/env bash
# Stage 5c-1: train 3 core-label variants in parallel on 3 GPUs.
# Warm-starts from the same H_hardneg_r10 checkpoint as Stage 5a/5b.

set -euo pipefail

cd /data/muskliu/npmaster
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc
export PYTHONPATH="src:${PYTHONPATH:-}"

EMB=/data/muskliu/npmaster/data/embeddings/edge_padding/embeddings_merged.npy
PROJ=/data/muskliu/npmaster/data/phase3/route_a/projections_256.npy
CHUNK=/data/muskliu/npmaster/data/phase3/route_b/chunk_index.pkl
CIX=/data/muskliu/npmaster/data/phase3/route_a/contig_index.pkl
# bgc_csv in train.py is only used by a val plotting helper; safe to
# point at the original (unused for loss).
BGC_CSV=/data/muskliu/npmaster/data/bgc_annotations.csv
WARM=/data/muskliu/npmaster/data/stage3_results/H_hardneg_r10/best_H_hardneg_r10.pt

TRIALS=(baseline_core_v1 tight_core_v1 short_core_upweight_v1)

LAUNCH_LOG=experiments/stage5c1_train_launch.log
mkdir -p "$(dirname "${LAUNCH_LOG}")"
echo "=== stage5c1 training launch $(date -Is) ===" | tee -a "${LAUNCH_LOG}"

PIDS=()
for i in "${!TRIALS[@]}"; do
  trial="${TRIALS[$i]}"
  gpu="${i}"
  exp_dir="experiments/stage5c1-${trial}"
  mkdir -p "${exp_dir}"/artifacts/checkpoints "${exp_dir}"/logs "${exp_dir}"/metrics "${exp_dir}"/inputs
  meta="/data/muskliu/npmaster/data/stage5/meta_arrays_${trial}.pkl"
  tag="${trial}_s0"
  log="${exp_dir}/logs/run.log"

  echo "launching ${trial} on gpu=${gpu}" | tee -a "${LAUNCH_LOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup setsid python src/train.py \
    --embeddings "${EMB}" \
    --projections "${PROJ}" \
    --chunk-index "${CHUNK}" \
    --meta-arrays "${meta}" \
    --contig-index "${CIX}" \
    --bgc-csv "${BGC_CSV}" \
    --ckpt "${WARM}" \
    --out-dir "${exp_dir}/artifacts/checkpoints" \
    --seed 0 \
    --tag "${tag}" \
    --batch-size 64 \
    --epochs 60 \
    --hard-neg-weight 4.0 \
    > "${log}" 2>&1 &
  pid=$!
  echo "  pid=${pid} log=${log}" | tee -a "${LAUNCH_LOG}"
  PIDS+=("${pid}")
done

echo "=== all 3 trials dispatched; pids=${PIDS[*]} ===" | tee -a "${LAUNCH_LOG}"
