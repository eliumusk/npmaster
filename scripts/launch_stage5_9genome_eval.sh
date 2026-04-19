#!/usr/bin/env bash
# Runs the final 9-genome inference + eval on a list of Stage 5 winner
# checkpoints. Intended to be invoked AFTER winners are frozen via
# scripts/stage5_pick_winners.py.
#
# Usage:
#   bash scripts/launch_stage5_9genome_eval.sh <winner_ckpt1> <winner_ckpt2> ...
#
# Each checkpoint path is the absolute on-server path from
# selection_decision.md. Per checkpoint, we use one idle GPU: Evo2
# inference on all 9 genomes (evo2_bgc env), then evaluate_nine_genome
# (bgc env) under the frozen val_frozen_fused operating point.
#
# Results land alongside the checkpoint, at
#   <trial_dir>/artifacts/predictions_9genome/
#   <trial_dir>/metrics/final_9genome.json

set -euo pipefail

cd /data/muskliu/npmaster

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <winner_ckpt> [<winner_ckpt> ...]" >&2
  exit 1
fi

BENCH_BASE="/data/syh/benchmark/nine_genome"
FNA_DIR="${BENCH_BASE}/fna"
GENOME_IDS="GCA_000154945.1,GCA_000156435.1,GCA_000156475.1,GCA_000158815.1,GCA_000158875.1,GCA_000158895.1,GCA_000158915.1,GCA_000568255.1,GCA_000568915.1"
OPERATING_POINT="val_frozen_fused"
WINDOW=8192
STRIDE=2048

LAUNCH_LOG="experiments/stage5_no9_data_v1-final9genome_launch.log"
echo "=== Stage 5 final 9-genome eval $(date -Is) ===" | tee -a "$LAUNCH_LOG"

# Find free GPUs up front so each checkpoint gets pinned to a distinct GPU.
FREE_GPUS=( $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 500 {print $1}') )
if (( ${#FREE_GPUS[@]} < $# )); then
  echo "only ${#FREE_GPUS[@]} free GPUs but $# checkpoints — will queue sequentially" \
    | tee -a "$LAUNCH_LOG"
fi

i=0
for ckpt in "$@"; do
  # Trial dir = checkpoint's grandparent (experiments/<trial>/artifacts/checkpoints/best_...)
  trial_dir="$(dirname "$(dirname "$(dirname "$ckpt")")")"
  # Use the Stage 5 tag (variant_sN) from the checkpoint filename.
  tag="$(basename "$ckpt" .pt)"; tag="${tag#best_}"
  pred_dir="${trial_dir}/artifacts/predictions_9genome_${tag}"
  summary_json="${trial_dir}/metrics/final_9genome_${tag}.json"
  mkdir -p "${pred_dir}" "$(dirname "${summary_json}")"

  # Assign GPU cyclically from the available pool.
  if (( ${#FREE_GPUS[@]} )); then
    gpu="${FREE_GPUS[$(( i % ${#FREE_GPUS[@]} ))]}"
  else
    gpu="0"
  fi
  log="${trial_dir}/logs/final_9genome_${tag}.log"
  mkdir -p "$(dirname "${log}")"

  echo "dispatch ckpt=${ckpt} gpu=${gpu} pred_dir=${pred_dir}" \
    | tee -a "$LAUNCH_LOG"

  # Run in background; each job activates evo2_bgc for inference, then
  # bgc for evaluation.
  (
    source /data/conda/ourconda_bashrc
    conda activate evo2_bgc
    export PYTHONPATH="src:${PYTHONPATH:-}"
    export HF_HUB_CACHE=/data/public_models/hub
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/inference_stride.py \
      --fna-dir "${FNA_DIR}" \
      --out-dir "${pred_dir}" \
      --ckpt "${ckpt}" \
      --src-dir src \
      --genome-ids "${GENOME_IDS}" \
      --window "${WINDOW}" \
      --stride "${STRIDE}" \
      --layer blocks.20.mlp.l3 \
      --evo-model evo2_7b \
      --cache-dir /data/public_models/hub

    conda deactivate
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate bgc
    python scripts/evaluate_nine_genome.py \
      --base "${BENCH_BASE}" \
      --results-dir "${pred_dir}" \
      --genome-ids "${GENOME_IDS}" \
      --operating-point "${OPERATING_POINT}" \
      --out "${summary_json}"
  ) >"${log}" 2>&1 &

  pid=$!
  echo "  pid=${pid} log=${log}" | tee -a "$LAUNCH_LOG"
  echo "${ckpt},${gpu},${pid},$(date -Is)" \
    >> experiments/stage5_no9_data_v1-final9genome-pids.csv
  i=$(( i + 1 ))
done

echo "=== all ${i} checkpoints dispatched ===" | tee -a "$LAUNCH_LOG"
echo "wait with: watch -n 10 tail -n 2 experiments/stage5_no9_data_v1-*/logs/final_9genome_*.log" \
  | tee -a "$LAUNCH_LOG"
