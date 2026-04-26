#!/bin/bash
# Launch per-token Evo2 extraction for the training pool (1.064M windows).
# 16 GPUs total: node7 GPUs 0..7, node8 GPUs 0..7. Each GPU takes one md5-mod-16
# shard of window_ids from windows_metadata_edgepad_filtered.csv. Source = sliced
# (read {genome}.fna.gz, slice [window_start:window_end], N-pad tail).
#
# Window/stride = 16384/8192; same proj (D=128, K=8, seed=0xE2E2, fp16, layers
# 21..31 truncated) as the 9g arm, only the window length differs.
#
# Usage
# -----
#   1) Immediate kick-off (node8 GPUs 1..7 are idle right now while the 9g arm
#      finishes on the other 9 GPUs):
#
#        bash launch_train.sh start-free
#
#   2) After the 9g arm finishes on a given GPU, auto-launch the training shard
#      on that GPU. This runs a watcher loop per node that polls the 9g manifest
#      files and launches the queued training worker as soon as the 9g worker on
#      the same GPU completes:
#
#        bash launch_train.sh start-watchers
#
#   3) Or, in one shot:  bash launch_train.sh all
#
# Shard assignment (md5(window_id) mod 16):
#   shards 00..07 -> node7 GPU 0..7
#   shards 08..15 -> node8 GPU 0..7
#
# Shard files are pre-built at:
#   $REPO/experiments/step2_per_token/train_w16384_s8192/shards/shard_XX.txt
# and contain one window_id per line.
#
# CRITICAL: --source sliced requires that sliced output be byte-equivalent to the
# --source fasta output on the edge-padded subset. Validation is performed by
# validate_sliced_vs_fasta/ before this script is safe to launch.
set -euo pipefail

REPO=/data/muskliu/npmaster
OUT=$REPO/experiments/step2_per_token/train_w16384_s8192
LOG=$REPO/experiments/step2_per_token/logs
SHARDS=$OUT/shards
META=$REPO/data/windows/edge_padding/windows_metadata_edgepad_filtered.csv
GENOMES=$REPO/data/genomes
NINE_G_OUT=$REPO/experiments/step2_per_token/9g_w8192_s2048

# GPU -> (node, local_gpu_idx, shard_idx, 9g_stem_on_that_gpu)
# 9g stems are the 9 genome IDs — we need to know which stem is running on each
# GPU so we can wait on its manifest.json. Keep this map in sync with launch_9g.sh.
NINE_G_NODE7_GPU0=GCA_000568915.1
NINE_G_NODE7_GPU1=GCA_000158815.1
NINE_G_NODE7_GPU2=GCA_000156475.1
NINE_G_NODE7_GPU3=GCA_000154945.1
NINE_G_NODE7_GPU4=GCA_000158895.1
NINE_G_NODE7_GPU5=GCA_000156435.1
NINE_G_NODE7_GPU6=GCA_000158875.1
NINE_G_NODE7_GPU7=GCA_000158915.1
NINE_G_NODE8_GPU0=GCA_000568255.1

launch_one() {
  local host=$1 gpu=$2 shard=$3
  local stem=$(printf 'train_%s_gpu%d' "$(echo $host | sed 's/.*node//')" "$gpu")
  local sshhost=""
  case $host in
    node7) sshhost=Voc_gpu_node7_a800 ;;
    node8) sshhost=Voc_gpu_node8_a800 ;;
  esac
  ssh $sshhost "mkdir -p $OUT $LOG"
  ssh $sshhost "nohup bash -lc '
    source /data/conda/miniconda/bin/activate evo2_bgc && \
    export HF_HUB_CACHE=/data/public_models/hub && \
    export HF_HUB_OFFLINE=1 && \
    export CUDA_VISIBLE_DEVICES=$gpu && \
    cd $REPO && \
    python -u src/per_token/extract/production_extract.py \
      --source sliced \
      --genome-dir $GENOMES \
      --metadata-csv $META \
      --window-ids-subset $SHARDS/shard_$(printf %02d $shard).txt \
      --out-dir $OUT \
      --stem $stem \
      --window 16384 --stride 16384 \
      --downsample 8 --d-reduced 128 \
      --flush-every 1000 \
      --log-file $LOG/${stem}.log
  ' > $LOG/${stem}.nohup 2>&1 &"
  echo "  launched $host GPU $gpu -> shard $shard (stem=$stem)"
}

wait_and_launch() {
  # Poll for a 9g manifest.json to appear on $host (meaning the 9g worker on
  # $gpu finished), then launch the training worker for $shard on the same GPU.
  local host=$1 gpu=$2 shard=$3 nine_g_stem=$4
  local sshhost=""
  case $host in
    node7) sshhost=Voc_gpu_node7_a800 ;;
    node8) sshhost=Voc_gpu_node8_a800 ;;
  esac
  (
    while true; do
      if ssh $sshhost "test -f $NINE_G_OUT/${nine_g_stem}.manifest.json" 2>/dev/null; then
        echo "[watcher] 9g done on $host GPU $gpu (stem=$nine_g_stem); launching training shard $shard"
        launch_one $host $gpu $shard
        break
      fi
      sleep 60
    done
  ) &
  echo "  watcher spawned for $host GPU $gpu (awaiting 9g $nine_g_stem)"
}

cmd=${1:-help}

case $cmd in
  start-free)
    # Launch training on node8 GPUs 1..7 right now. Shards 09..15.
    for gpu in 1 2 3 4 5 6 7; do
      shard=$((gpu + 8))
      launch_one node8 $gpu $shard
      sleep 2
    done
    ;;
  start-watchers)
    # Spawn background watchers that wait for each 9g worker to finish, then
    # launch the training worker for that GPU.
    # node7 GPUs 0..7 -> shards 00..07
    for gpu in 0 1 2 3 4 5 6 7; do
      var=NINE_G_NODE7_GPU${gpu}
      wait_and_launch node7 $gpu $gpu ${!var}
    done
    # node8 GPU 0 -> shard 08
    wait_and_launch node8 0 8 $NINE_G_NODE8_GPU0
    echo "all watchers spawned; they will background-launch as 9g workers finish."
    echo "watcher PIDs belong to THIS shell — keep this shell alive (use nohup or tmux)"
    ;;
  all)
    $0 start-free
    $0 start-watchers
    ;;
  __launch_one)
    # internal: bash launch_train.sh __launch_one <node> <gpu> <shard>
    launch_one "$2" "$3" "$4"
    ;;
  help|*)
    cat <<EOF
Usage: $0 {start-free|start-watchers|all}

  start-free       launch training workers on the 7 idle node8 GPUs (1..7)
  start-watchers   spawn watchers that launch training on each GPU once its
                   9g worker produces a manifest.json (i.e. finishes)
  all              both of the above
EOF
    ;;
esac
