#!/bin/bash
# P0: Reproduce Phase 3 Route B 3-seed training from scratch.
#
# Usage: ./reproduce_mainline.sh [seed]
# Default runs s0 only. Pass "all" to run all 3 seeds sequentially.
#
# Expected: each seed ~35 min on 1 A800 (4096-dim E2E).

set -e
SEED=${1:-0}
GPU=${GPU:-0}

cd /data/muskliu/np_master/code/np-master/bgc-gcd

run_seed() {
  local s=$1
  local g=$2
  echo "=== Route B seed=$s on GPU $g ==="
  CUDA_VISIBLE_DEVICES=$g python -u train_context_e2e.py \
    --embeddings /data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy \
    --projections /data/muskliu/np_master/data/phase3/route_a/projections_256.npy \
    --chunk-index /data/muskliu/np_master/data/phase3/route_b/chunk_index.pkl \
    --meta-arrays /data/muskliu/np_master/data/phase3/route_a/meta_arrays.pkl \
    --contig-index /data/muskliu/np_master/data/phase3/route_a/contig_index.pkl \
    --bgc-csv /data/muskliu/np_master/data/bgc_annotations.csv \
    --ckpt /data/muskliu/np_master/data/stage3_results/H_hardneg_r10/best_H_hardneg_r10.pt \
    --out-dir /data/muskliu/np_master/data/phase3/route_b \
    --seed $s --epochs 100 --batch-size 64 \
    --lr-proj 1e-4 --lr-context 1e-3 --lr-classifier 1e-3
}

if [ "$SEED" = "all" ]; then
  for s in 0 1 2; do
    run_seed $s $GPU
  done
else
  run_seed $SEED $GPU
fi
