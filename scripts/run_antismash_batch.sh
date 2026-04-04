#!/bin/bash
# 批量用Docker跑antiSMASH
# 用法: bash scripts/run_antismash_batch.sh [并行数, 默认2]

set -e
PARALLEL=${1:-2}
BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$BASE/data"
RESULT="$DATA/antismash_result"
LOGS="$RESULT/_logs"
mkdir -p "$RESULT" "$LOGS"

# 收集所有需要跑的基因组
TASKS=()
for genus in micromonospora amycolatopsis pseudomonas bacillus; do
  for fna in "$DATA/ref_genomes_batch/$genus/ncbi_dataset/data/GCF_"*/*.fna; do
    gcf=$(basename "$(dirname "$fna")")
    fname=$(basename "$fna")
    # 跳过已完成的（有 .region*.gbk 文件说明跑过了）
    if [ -d "$RESULT/$gcf" ] && ls "$RESULT/$gcf"/*.region*.gbk >/dev/null 2>&1; then
      echo "[SKIP] $gcf (已有结果)"
      continue
    fi
    # 清理之前失败的空目录
    rm -rf "$RESULT/$gcf"
    TASKS+=("$genus|$gcf|$fname")
  done
done

echo "========================================="
echo " 待处理: ${#TASKS[@]} 个基因组, 并行数: $PARALLEL"
echo "========================================="

running=0
total=${#TASKS[@]}
idx=0

for info in "${TASKS[@]}"; do
  IFS='|' read -r genus gcf fname <<< "$info"
  idx=$((idx+1))

  (
    echo "[START] ($idx/$total) $genus / $gcf"
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
      echo "[DONE]  ($idx/$total) $genus / $gcf"
    else
      echo "[FAIL]  ($idx/$total) $genus / $gcf (see $LOGS/$gcf.log)"
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
echo " 全部完成! 结果在: $RESULT/"
echo "========================================="

success=0
fail=0
for genus in micromonospora amycolatopsis pseudomonas bacillus; do
  genus_ok=0
  genus_fail=0
  for fna in "$DATA/ref_genomes_batch/$genus/ncbi_dataset/data/GCF_"*/*.fna; do
    gcf=$(basename "$(dirname "$fna")")
    if [ -d "$RESULT/$gcf" ] && ls "$RESULT/$gcf"/*.region*.gbk >/dev/null 2>&1; then
      genus_ok=$((genus_ok+1))
    else
      genus_fail=$((genus_fail+1))
    fi
  done
  echo " $genus: $genus_ok 成功, $genus_fail 失败"
  success=$((success+genus_ok))
  fail=$((fail+genus_fail))
done
echo " 总计: $success 成功, $fail 失败"
