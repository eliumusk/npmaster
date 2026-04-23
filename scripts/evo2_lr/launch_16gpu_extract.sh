#!/usr/bin/env bash
# Launch 8 parallel Evo2 extract processes on one node (GPUs 0-7), each bound to
# a distinct shard. Run this on node7 with SHARD_START=8 SHARD_END=15 and on
# node8 with SHARD_START=0 SHARD_END=7 to cover all 16 shards.
#
# Every process writes its own manifest_{shard}.json (thanks to
# extract_batch_9genome.py auto-naming when --genome-ids picks exactly one).
# Embeddings + windows.csv land in the shared OUT_DIR.
#
# Required env vars:
#   SHARD_START, SHARD_END      (inclusive, 0..15)
#   WINDOW, STRIDE              geometry
#   OUT_DIR                     output embedding dir (shared across all shards)
#   TAG                         used for log filename prefix (e.g. v41_short_w4096_s1024)
# Optional:
#   SHARD_DIR     (default: /data/muskliu/npmaster/experiments/OER004256_compare/v41_shards_gt10kb_16)
#   SHARD_PREFIX  (default: OER004256_v41_shard)
#   BATCH_SIZE    (default: 4)
#   MIN_CONTIG    (default: 10000)
#   LAYER         (default: blocks.20.mlp.l3)
#   LOG_DIR       (default: /data/muskliu/npmaster/experiments/OER004256_compare/logs)
#
# Example usage (on node7, for shards 8-15 at w=4096/s=1024):
#   SHARD_START=8 SHARD_END=15 WINDOW=4096 STRIDE=1024 \
#   OUT_DIR=/data/muskliu/npmaster/experiments/OER004256_compare/v41_short_w4096_s1024_embeddings \
#   TAG=v41_short_w4096_s1024 BATCH_SIZE=8 \
#   bash scripts/evo2_lr/launch_16gpu_extract.sh

set -euo pipefail

: "${SHARD_START:?need SHARD_START}"
: "${SHARD_END:?need SHARD_END}"
: "${WINDOW:?need WINDOW}"
: "${STRIDE:?need STRIDE}"
: "${OUT_DIR:?need OUT_DIR}"
: "${TAG:?need TAG}"

SHARD_DIR="${SHARD_DIR:-/data/muskliu/npmaster/experiments/OER004256_compare/v41_shards_gt10kb_16}"
SHARD_PREFIX="${SHARD_PREFIX:-OER004256_v41_shard}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MIN_CONTIG="${MIN_CONTIG:-10000}"
LAYER="${LAYER:-blocks.20.mlp.l3}"
LOG_DIR="${LOG_DIR:-/data/muskliu/npmaster/experiments/OER004256_compare/logs}"
REPO="${REPO:-/data/muskliu/npmaster}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Activate evo2 env (prefer the ourconda_bashrc if present)
if [[ -f /data/conda/ourconda_bashrc ]]; then
    source /data/conda/ourconda_bashrc
elif [[ -f /data/conda/miniconda/etc/profile.d/conda.sh ]]; then
    source /data/conda/miniconda/etc/profile.d/conda.sh
fi
conda activate evo2_bgc

cd "$REPO"

NODE_HOST="$(hostname)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LABEL="${TAG}_${STAMP}_${NODE_HOST}"
PID_CSV="${LOG_DIR}/${LABEL}.pids.csv"
echo "node,shard,gpu,pid,log" > "$PID_CSV"

N_LAUNCHED=0
for i in $(seq "$SHARD_START" "$SHARD_END"); do
    GPU=$(( i - SHARD_START ))
    SHARD_ID=$(printf "%02d" "$i")
    SHARD_NAME="${SHARD_PREFIX}${SHARD_ID}"
    LOG="${LOG_DIR}/${LABEL}_${SHARD_NAME}_gpu${GPU}.log"

    CUDA_VISIBLE_DEVICES="$GPU" nohup python scripts/evo2_lr/extract_batch_9genome.py \
        --fna-dir "$SHARD_DIR" \
        --out-dir "$OUT_DIR" \
        --window "$WINDOW" --stride "$STRIDE" \
        --batch-size "$BATCH_SIZE" \
        --min-contig-len "$MIN_CONTIG" \
        --genome-ids "$SHARD_NAME" \
        --layer "$LAYER" \
        --cache-dir /data/public_models/hub \
        > "$LOG" 2>&1 &
    PID=$!
    echo "${NODE_HOST},${SHARD_NAME},${GPU},${PID},${LOG}" >> "$PID_CSV"
    echo "launched shard=${SHARD_NAME} gpu=${GPU} pid=${PID}"
    N_LAUNCHED=$((N_LAUNCHED + 1))
    # small stagger so all 8 don't hit HF cache simultaneously
    sleep 2
done

# detach so the shell can exit without killing the jobs
disown -a 2>/dev/null || true

echo
echo "launched ${N_LAUNCHED} shards on ${NODE_HOST}"
echo "pid csv: ${PID_CSV}"
echo "monitor with: tail -f ${LOG_DIR}/${LABEL}_*.log"
