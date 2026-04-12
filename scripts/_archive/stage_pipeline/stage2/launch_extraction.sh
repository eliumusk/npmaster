#!/bin/bash
# Launch Evo2 embedding extraction across 8 GPUs on THIS node.
# Run this script on EACH node with different GPU_OFFSET:
#   Node7: GPU_OFFSET=0  N_GPUS=16  (gpu_id 0-7)
#   Node8: GPU_OFFSET=8  N_GPUS=16  (gpu_id 8-15)
set -u

ROOT=/data/muskliu/np_master
SCRIPT=$ROOT/scripts/stage2/extract_embeddings.py
OUT=$ROOT/data/embeddings/per_genome
GPU_OFFSET=${GPU_OFFSET:-0}
N_GPUS=${N_GPUS:-16}
N_LOCAL=${N_LOCAL:-8}

source /data/conda/ourconda_bashrc
conda activate evo2_bgc
export HF_HUB_CACHE=/data/public_models/hub

mkdir -p "$OUT"

echo "[$(date)] Starting extraction: GPU_OFFSET=$GPU_OFFSET N_GPUS=$N_GPUS N_LOCAL=$N_LOCAL"

for local_id in $(seq 0 $((N_LOCAL - 1))); do
    gpu_id=$((GPU_OFFSET + local_id))
    log="$OUT/gpu${gpu_id}_full.log"
    echo "[$(date)] Launching gpu_id=$gpu_id on CUDA:$local_id → $log"
    CUDA_VISIBLE_DEVICES=$local_id nohup python "$SCRIPT" \
        --gpu_id "$gpu_id" \
        --n_gpus "$N_GPUS" \
        --genomes_dir "$ROOT/data/genomes" \
        --metadata_csv "$ROOT/data/windows/windows_metadata.csv" \
        --output_dir "$OUT" \
        > "$log" 2>&1 &
done

echo "[$(date)] All $N_LOCAL processes launched. PIDs: $(jobs -p | tr '\n' ' ')"
wait
echo "[$(date)] All GPUs finished."
