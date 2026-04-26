#!/usr/bin/env bash
# End-to-end per_token 9g evaluation after training completes.
# Usage: per_token_pipeline.sh <run_dir>
#   <run_dir>/best.pt  must exist
# Writes:
#   <run_dir>/9g_probs/        (per-stem probs.npz + coords.csv)
#   <run_dir>/9g_regions_t{0.3,0.5,0.7}.csv
#   <run_dir>/9g_readout_t{0.3,0.5,0.7}.json
set -euo pipefail

RUN_DIR="${1:?usage: $0 <run_dir>}"
CKPT="$RUN_DIR/best.pt"
FEATS_DIR="/data/muskliu/npmaster/experiments/step2_per_token/9g_w8192_s2048"
GT_JSON="/data/syh/benchmark/nine_genome/ground_truth.json"
REPO_ROOT="/data/muskliu/npmaster"
PROBS_DIR="$RUN_DIR/9g_probs"

cd "$REPO_ROOT"
source /data/conda/miniconda/bin/activate evo2_bgc
export CUDA_VISIBLE_DEVICES=1

echo "=== inference ==="
python -u scripts/inference/per_token/infer_shards.py \
  --features-dir "$FEATS_DIR" \
  --ckpt "$CKPT" \
  --out-dir "$PROBS_DIR" \
  --batch 64

for th in 0.3 0.5 0.7; do
  echo "=== decode threshold=$th ==="
  python -u scripts/decode/per_token_v1.py \
    --probs-dir "$PROBS_DIR" \
    --out-regions-csv "$RUN_DIR/9g_regions_t${th}.csv" \
    --threshold "$th" --min-len-bp 2000 --max-gap-bp 500

  echo "=== evaluate threshold=$th ==="
  python -u scripts/evaluate/per_token_readout.py \
    --regions-csv "$RUN_DIR/9g_regions_t${th}.csv" \
    --method-name "per_token_v1_seed0_t${th}" \
    --gt-json "$GT_JSON" \
    --out-json "$RUN_DIR/9g_readout_t${th}.json" \
    --n-boot 500
done
echo "=== DONE ==="
