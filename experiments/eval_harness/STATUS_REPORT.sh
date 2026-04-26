#!/usr/bin/env bash
# Quick status report for the overnight Step 2 per-token run.
# Run from anywhere: `bash experiments/eval_harness/STATUS_REPORT.sh`

set -euo pipefail

echo "=== Step 2 per-token extraction status (as of $(date)) ==="
echo

NODES=("Voc_gpu_node7_a800" "Voc_gpu_node8_a800")
OUT_ROOT="/data/muskliu/npmaster/experiments/step2_per_token"

for NODE in "${NODES[@]}"; do
    echo "--- $NODE ---"
    ssh "$NODE" bash <<EOF 2>/dev/null
echo "[hostname] \$(hostname)"
echo "[gpu]"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -8
echo "[disk /data/muskliu]"
df -h /data/muskliu | tail -1
echo "[shard count 9g]"
ls $OUT_ROOT/9g_w8192_s2048/ 2>/dev/null | wc -l
echo "[shard count train]"
ls $OUT_ROOT/train_w16384_s8192/ 2>/dev/null | wc -l
echo "[tail of most recent log]"
tail -n 3 \$(ls -t $OUT_ROOT/logs/*.log 2>/dev/null | head -1) 2>/dev/null || echo "  (no logs yet)"
EOF
    echo
done

echo "=== Local artifacts ==="
ls -la "$(dirname "$0")/"
echo
echo "=== NIGHT_LOG last 30 lines ==="
tail -n 30 "$(dirname "$0")/NIGHT_LOG.md"
