#!/bin/bash
# Launch per-token Evo2 extraction for 9-genome inference arm.
# 9 genomes → 9 GPUs: node7 GPUs 0..7 take one each, node8 GPU 0 takes the leftover.
# Window/stride = 8192/2048 (same as v4.1 primary arm).
#
# Usage: bash launch_9g.sh
# Requires: HF_HUB_CACHE=/data/public_models/hub, evo2_bgc conda env.
set -euo pipefail

REPO=/data/muskliu/npmaster
OUT=$REPO/experiments/step2_per_token/9g_w8192_s2048
LOG=$REPO/experiments/step2_per_token/logs
FNA=/data/syh/benchmark/nine_genome/fna

ssh Voc_gpu_node7_a800 "mkdir -p $OUT $LOG"

# Load-balance: sort genomes by bp size ascending; biggest ones go to idle GPUs last.
# (every GPU only gets one genome anyway so order mostly doesn't matter.)
GENOMES=(
  GCA_000568915.1   # 6.78 Mb, 14 contigs (most fragmented — run first)
  GCA_000158815.1   # 6.82 Mb,  4 contigs
  GCA_000156475.1   # 6.82 Mb,  2 contigs
  GCA_000154945.1   # 8.13 Mb,  1 contig
  GCA_000158895.1   # 8.46 Mb,  4 contigs
  GCA_000156435.1   # 8.51 Mb,  3 contigs
  GCA_000158875.1   # 9.18 Mb,  4 contigs
  GCA_000158915.1   # 11.03 Mb, 2 contigs
  GCA_000568255.1   # 11.65 Mb, 1 contig (biggest)
)

launch_node7() {
  local gpu=$1 gid=$2
  ssh Voc_gpu_node7_a800 "nohup bash -lc '
    source /data/conda/miniconda/bin/activate evo2_bgc && \
    export HF_HUB_CACHE=/data/public_models/hub && \
    export HF_HUB_OFFLINE=1 && \
    export CUDA_VISIBLE_DEVICES=$gpu && \
    cd $REPO && \
    python -u src/per_token/extract/production_extract.py \
      --source genome \
      --fasta $FNA/${gid}.fna \
      --out-dir $OUT \
      --stem $gid \
      --window 8192 --stride 2048 \
      --downsample 8 --d-reduced 128 \
      --flush-every 1000 \
      --log-file $LOG/node7_gpu${gpu}_${gid}.log
  ' > $LOG/node7_gpu${gpu}_${gid}.nohup 2>&1 &"
}

launch_node8() {
  local gpu=$1 gid=$2
  ssh Voc_gpu_node8_a800 "nohup bash -lc '
    source /data/conda/miniconda/bin/activate evo2_bgc && \
    export HF_HUB_CACHE=/data/public_models/hub && \
    export HF_HUB_OFFLINE=1 && \
    export CUDA_VISIBLE_DEVICES=$gpu && \
    cd $REPO && \
    python -u src/per_token/extract/production_extract.py \
      --source genome \
      --fasta $FNA/${gid}.fna \
      --out-dir $OUT \
      --stem $gid \
      --window 8192 --stride 2048 \
      --downsample 8 --d-reduced 128 \
      --flush-every 1000 \
      --log-file $LOG/node8_gpu${gpu}_${gid}.log
  ' > $LOG/node8_gpu${gpu}_${gid}.nohup 2>&1 &"
}

# node7: GPUs 0..7 get first 8 genomes
for i in 0 1 2 3 4 5 6 7; do
  launch_node7 $i ${GENOMES[$i]}
  sleep 1
done
# node8: GPU 0 gets the 9th (biggest) genome
launch_node8 0 ${GENOMES[8]}

echo "9g launch dispatched; see $LOG for per-GPU progress."
