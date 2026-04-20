#!/usr/bin/env bash
# Stage 5b-4: run 9-genome inference + eval on the 4 frozen candidates
# from experiments/stage5_no9_data_v1-selection_decision_b.md.
#
# Three of the four need fresh inference (baseline_v1_s0,
# short_bgc_upweight_v1_s0, tight_pos_075_v1_s0); the fourth
# (tight_pos_099_v1_s0) reuses the Stage 5a prediction dir but applies
# a new decoder. All four are evaluated under their frozen decoder
# params, not val_frozen_fused.

set -euo pipefail

cd /data/muskliu/npmaster

BENCH_BASE="/data/syh/benchmark/nine_genome"
FNA_DIR="${BENCH_BASE}/fna"
GENOME_IDS="GCA_000154945.1,GCA_000156435.1,GCA_000156475.1,GCA_000158815.1,GCA_000158875.1,GCA_000158895.1,GCA_000158915.1,GCA_000568255.1,GCA_000568915.1"
WINDOW=8192
STRIDE=2048

LAUNCH_LOG="experiments/stage5b_final9genome_launch.log"
mkdir -p "$(dirname "${LAUNCH_LOG}")"
echo "=== Stage 5b final 9-genome at $(date -Is) ===" | tee -a "${LAUNCH_LOG}"

# Each entry: <trial_dir>|<tag>|<start>|<ext>|<ms>|<needs_inference>
CANDIDATES=(
  "experiments/stage5_no9_data_v1-baseline_v1|baseline_v1_s0|0.99|0.85|1|1"
  "experiments/stage5_no9_data_v1-short_bgc_upweight_v1|short_bgc_upweight_v1_s0|0.99|0.95|1|1"
  "experiments/stage5_no9_data_v1-tight_pos_075_v1|tight_pos_075_v1_s0|0.90|0.85|2|1"
  "experiments/stage5_no9_data_v1-tight_pos_099_v1_s0|tight_pos_099_v1_s0|0.90|0.85|2|0"
)

# Pick one free GPU per job.
FREE_GPUS=( $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 500 {print $1}') )
echo "free GPUs on $(hostname): ${FREE_GPUS[*]:-none}" | tee -a "${LAUNCH_LOG}"

PIDS=()
i=0
for entry in "${CANDIDATES[@]}"; do
  IFS='|' read -r trial_dir tag start ext ms needs_inf <<<"${entry}"
  ckpt="${trial_dir}/artifacts/checkpoints/best_${tag}.pt"
  pred_dir="${trial_dir}/artifacts/predictions_9genome_${tag}"
  # For the reused Stage 5a predictions the pred_dir lives at
  # predictions_9genome_<tag>; confirm before skipping inference.
  summary_json="${trial_dir}/metrics/final_9genome_b_${tag}.json"
  log="${trial_dir}/logs/final_9genome_b_${tag}.log"
  mkdir -p "${pred_dir}" "$(dirname "${summary_json}")" "$(dirname "${log}")"

  gpu="${FREE_GPUS[$(( i % ${#FREE_GPUS[@]:-1} ))]:-0}"
  i=$(( i + 1 ))
  echo "=> ${tag}  gpu=${gpu}  needs_inf=${needs_inf}  decoder=start_${start}_ext_${ext}_ms_${ms}" | tee -a "${LAUNCH_LOG}"

  (
    if [[ "${needs_inf}" == "1" ]]; then
      # inference needs evo2_bgc
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
    else
      echo "[skip inference] reusing existing ${pred_dir}"
    fi

    # Eval under Stage 5b-selected decoder (CLI flags override any
    # named operating point).
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate bgc
    python scripts/evaluate_nine_genome.py \
      --base "${BENCH_BASE}" \
      --results-dir "${pred_dir}" \
      --genome-ids "${GENOME_IDS}" \
      --start-threshold "${start}" \
      --extend-threshold "${ext}" \
      --min-support-windows "${ms}" \
      --max-gap 0 \
      --out "${summary_json}"
  ) >"${log}" 2>&1 &

  pid=$!
  PIDS+=("${pid}")
  echo "  pid=${pid} log=${log}" | tee -a "${LAUNCH_LOG}"
done

echo "=== ${#PIDS[@]} jobs dispatched; wait with: tail -F experiments/stage5_no9_data_v1-*/logs/final_9genome_b_*.log ===" | tee -a "${LAUNCH_LOG}"
