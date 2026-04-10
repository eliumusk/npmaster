#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc
cd /data/muskliu/np_master/code/np-master/bgc-gcd
CUDA_VISIBLE_DEVICES=1 python -u train_context_e2e.py \
  --embeddings /data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy \
  --projections /data/muskliu/np_master/data/phase3/route_a/projections_256.npy \
  --chunk-index /data/muskliu/np_master/data/phase3/route_b/chunk_index.pkl \
  --meta-arrays /data/muskliu/np_master/data/phase3/route_a/meta_arrays.pkl \
  --contig-index /data/muskliu/np_master/data/phase3/route_a/contig_index.pkl \
  --bgc-csv /data/muskliu/np_master/data/bgc_annotations.csv \
  --ckpt /data/muskliu/np_master/data/stage3_results/H_hardneg_r10/best_H_hardneg_r10.pt \
  --out-dir /data/muskliu/np_master/data/phase3/route_b \
  --seed 2 --epochs 100 --batch-size 64 \
  --lr-proj 1e-4 --lr-context 1e-3 --lr-classifier 1e-3
