#!/bin/bash
# 扩充Streptomyces antiSMASH标注: 从25个扩展到~105个
# 用法: bash scripts/run_antismash_strep_expand.sh [并行数, 默认4] [最大数量, 默认80]

set -e
PARALLEL=${1:-4}
MAX_NEW=${2:-80}
BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$BASE/data"
RESULT="$DATA/antismash_result"
LOGS="$RESULT/_logs"
mkdir -p "$RESULT" "$LOGS"

genus="streptomyces"
TASKS=()
count=0

for fna in "$DATA/ref_genomes_batch/$genus/ncbi_dataset/data/GCF_"*/*.fna; do
  gcf=$(basename "$(dirname "$fna")")
  fname=$(basename "$fna")

  # 跳过已有结果的 (zip或目录)
  if [ -f "$RESULT/${gcf}.zip" ]; then
    continue
  fi
  if [ -d "$RESULT/$gcf" ] && ls "$RESULT/$gcf"/*.region*.gbk >/dev/null 2>&1; then
    continue
  fi

  # 清理之前失败的空目录
  rm -rf "$RESULT/$gcf"
  TASKS+=("$genus|$gcf|$fname")
  count=$((count + 1))
  if [ $count -ge $MAX_NEW ]; then
    break
  fi
done

echo "========================================="
echo " Streptomyces扩充: ${#TASKS[@]}个新基因组, 并行=$PARALLEL"
echo "========================================="

running=0
total=${#TASKS[@]}
idx=0

for info in "${TASKS[@]}"; do
  IFS='|' read -r genus gcf fname <<< "$info"
  idx=$((idx+1))

  (
    echo "[START] ($idx/$total) $gcf"
    docker run --rm \
      -v "$DATA":/input \
      -v "$RESULT/$gcf":/output \
      antismash/standalone:latest \
      "ref_genomes_batch/$genus/ncbi_dataset/data/$gcf/$fname" \
      --output-dir /output \
      --minimal \
      --genefinding-tool prodigal \
      >"$LOGS/$gcf.log" 2>&1

    if [ $? -eq 0 ]; then
      echo "[DONE]  ($idx/$total) $gcf"
    else
      echo "[FAIL]  ($idx/$total) $gcf"
    fi
  ) &

  running=$((running+1))
  if [ $running -ge $PARALLEL ]; then
    wait -n 2>/dev/null || wait
    running=$((running-1))
  fi
done

wait

echo ""
echo "========================================="
# 统计
success=0
total_strep=0
for fna in "$DATA/ref_genomes_batch/$genus/ncbi_dataset/data/GCF_"*/*.fna; do
  gcf=$(basename "$(dirname "$fna")")
  total_strep=$((total_strep+1))
  if [ -f "$RESULT/${gcf}.zip" ] || ([ -d "$RESULT/$gcf" ] && ls "$RESULT/$gcf"/*.region*.gbk >/dev/null 2>&1); then
    success=$((success+1))
  fi
done
echo " Streptomyces antiSMASH总完成数: $success / $total_strep"
echo "========================================="
