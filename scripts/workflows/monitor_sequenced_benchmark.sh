#!/usr/bin/env bash
set -u
REPO=/data/muskliu/npmaster
BENCH=/data/syh/bench_34
FASTA_DIR=$BENCH/fasta
AS_ROOT=$BENCH/antismash_out
NP_ROOT=$BENCH/npmaster_out
OUT=$BENCH/comparison_safe
LOG=$BENCH/logs/monitor_safe_benchmark.log
PY=/data/conda/miniconda/bin/python
mkdir -p "$OUT" "$BENCH/logs"
expected=$(find "$FASTA_DIR" -maxdepth 1 -type f -name "*.fa" | wc -l)
start=$(date +%s)
timeout=$((20*3600))
echo "[$(date +%F_%T)] monitor start expected=$expected" >> "$LOG"
while true; do
  as_done=0
  np_done=0
  while IFS= read -r fa; do
    sample=$(basename "$fa" .fa)
    [[ -f "$AS_ROOT/$sample/$sample.json" ]] && as_done=$((as_done+1))
    [[ -f "$NP_ROOT/$sample/regions.csv" && -f "$NP_ROOT/$sample/cds_features.json" ]] && np_done=$((np_done+1))
  done < <(find "$FASTA_DIR" -maxdepth 1 -type f -name "*.fa" | sort)
  echo "[$(date +%F_%T)] status antismash=$as_done/$expected npmaster=$np_done/$expected" >> "$LOG"

  if [[ "$np_done" -gt 0 ]]; then
    (cd "$REPO" && "$PY" scripts/annotate/safe_bgc_annotation.py --npmaster-root "$NP_ROOT" --out-root "$NP_ROOT") >> "$LOG" 2>&1 || true
  fi
  if [[ "$np_done" -gt 0 && "$as_done" -gt 0 ]]; then
    (cd "$REPO" && "$PY" scripts/evaluate/compare_antismash_npmaster_safe.py --antismash-root "$AS_ROOT" --npmaster-root "$NP_ROOT" --out-dir "$OUT") >> "$LOG" 2>&1 || true
  fi

  if [[ "$as_done" -eq "$expected" && "$np_done" -eq "$expected" ]]; then
    echo "FINAL_DONE $(date +%F_%T) antismash=$as_done npmaster=$np_done" | tee "$OUT/FINAL_STATUS.txt" >> "$LOG"
    exit 0
  fi
  now=$(date +%s)
  if (( now - start > timeout )); then
    echo "TIMEOUT_PARTIAL $(date +%F_%T) antismash=$as_done/$expected npmaster=$np_done/$expected" | tee "$OUT/FINAL_STATUS.txt" >> "$LOG"
    exit 2
  fi
  sleep 300
done
