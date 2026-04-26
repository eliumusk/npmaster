#!/usr/bin/env bash
# Launch per-token Evo2 extraction for 16 OER metagenome FASTA shards.
#
# Split across two GPU nodes:
#   --node node8  =>  shards 00..07  on GPUs 0..7
#   --node node7  =>  shards 08..15  on GPUs 0..7
#
# After BOTH nodes complete extraction, run inference + decode on node8:
#   extract_evo2_per_token_oer.sh --stage infer-decode
#
# Stages:
#   extract        (default) launch 8 nohup workers, one per GPU
#   infer-decode   run per-token inference then decode at 3 thresholds
#
# Usage:
#   # On node8 (run via ssh or directly):
#   bash scripts/extract/extract_evo2_per_token_oer.sh --node node8
#   # On node7:
#   bash scripts/extract/extract_evo2_per_token_oer.sh --node node7
#   # After both finish:
#   bash scripts/extract/extract_evo2_per_token_oer.sh --stage infer-decode
set -euo pipefail

# ── paths (server layout) ─────────────────────────────────────────────
REPO=/data/muskliu/npmaster
SHARD_DIR=${SHARD_DIR:-$REPO/data/OER004256/shards_gt10kb_16}
SHARD_PREFIX=${SHARD_PREFIX:-OER004256_shard}
OUT_DIR=$REPO/data/evo2_per_token_unet/oer_w8192_s2048
LOG_DIR=$OUT_DIR/logs

# ── defaults ──────────────────────────────────────────────────────────
NODE=""
STAGE="extract"
CKPT="$REPO/experiments/evo2_per_token_unet/round2/E10_h64_lr1e3_b32/best.pt"
TAG="e10"

# ── arg parse ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node)   NODE="$2"; shift 2 ;;
    --stage)  STAGE="$2"; shift 2 ;;
    --ckpt)   CKPT="$2"; shift 2 ;;
    --tag)    TAG="$2"; shift 2 ;;
    *)        echo "unknown arg: $1"; exit 1 ;;
  esac
done

PROBS_DIR=$OUT_DIR/${TAG}_probs

# ── stage: extract ────────────────────────────────────────────────────
if [[ "$STAGE" == "extract" ]]; then
  if [[ -z "$NODE" ]]; then
    echo "ERROR: --node required for extract stage (node7 or node8)"
    exit 1
  fi

  case "$NODE" in
    node8) SHARD_START=0;  SHARD_END=7  ;;
    node7) SHARD_START=8;  SHARD_END=15 ;;
    *)     echo "ERROR: --node must be node7 or node8"; exit 1 ;;
  esac

  mkdir -p "$OUT_DIR" "$LOG_DIR"

  echo "=== per-token extract: $NODE shards $(printf '%02d' $SHARD_START)..$(printf '%02d' $SHARD_END) ==="
  echo "    out_dir: $OUT_DIR"
  echo "    log_dir: $LOG_DIR"

  PIDS=()
  for i in $(seq "$SHARD_START" "$SHARD_END"); do
    GPU=$(( i - SHARD_START ))
    SHARD_ID=$(printf "%02d" "$i")
    SHARD_NAME="${SHARD_PREFIX}${SHARD_ID}"
    SHARD_FNA="${SHARD_DIR}/${SHARD_NAME}.fna"
    LOG_FILE="${LOG_DIR}/${NODE}_gpu${GPU}_${SHARD_NAME}.log"

    echo "  launching shard=${SHARD_NAME} gpu=${GPU} ..."

    CUDA_VISIBLE_DEVICES="$GPU" nohup python -u \
      "$REPO/scripts/extract/extract_evo2_per_token_features.py" \
        --source genome \
        --fasta "$SHARD_FNA" \
        --out-dir "$OUT_DIR" \
        --stem "$SHARD_NAME" \
        --window 8192 --stride 2048 \
        --downsample 8 --d-reduced 128 \
        --flush-every 1000 \
        --log-file "$LOG_FILE" \
      > "${LOG_DIR}/${NODE}_gpu${GPU}_${SHARD_NAME}.nohup" 2>&1 &

    PIDS+=($!)
    # stagger slightly so 8 GPUs don't all hit the Evo2 model load at once
    sleep 3
  done

  echo ""
  echo "=== waiting for ${#PIDS[@]} workers (PIDs: ${PIDS[*]}) ==="
  echo "    monitor: tail -f ${LOG_DIR}/${NODE}_gpu*_*.log"

  FAILED=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      echo "ERROR: worker pid=$pid exited non-zero"
      FAILED=$((FAILED + 1))
    fi
  done

  if [[ "$FAILED" -gt 0 ]]; then
    echo "=== FAILED: $FAILED / ${#PIDS[@]} workers had errors ==="
    exit 1
  fi

  echo "=== $NODE extraction complete: all ${#PIDS[@]} shards done ==="
  echo "    outputs in $OUT_DIR"
  echo ""
  echo "After BOTH nodes finish, run:"
  echo "  bash scripts/extract/extract_evo2_per_token_oer.sh --stage infer-decode"
  exit 0
fi

# ── stage: infer-decode ───────────────────────────────────────────────
if [[ "$STAGE" == "infer-decode" ]]; then
  echo "=== stage: infer-decode ==="

  # sanity check: at least 16 manifest files should exist
  N_MANIFESTS=$(find "$OUT_DIR" -name "*.manifest.json" | wc -l)
  echo "  found $N_MANIFESTS manifest files in $OUT_DIR"
  if [[ "$N_MANIFESTS" -lt 16 ]]; then
    echo "WARNING: expected 16 manifests (shards 00..15), found $N_MANIFESTS"
    echo "  make sure both node7 and node8 extraction completed before running infer-decode"
    read -r -p "  continue anyway? [y/N] " reply
    if [[ "$reply" != "y" && "$reply" != "Y" ]]; then
      echo "aborted."
      exit 1
    fi
  fi

  # ── inference ──
  echo ""
  echo "--- per-token inference (tag=$TAG) ---"
  echo "  features: $OUT_DIR"
  echo "  ckpt:     $CKPT"
  echo "  out:      $PROBS_DIR"

  python -u "$REPO/scripts/inference/evo2_per_token/infer_shards.py" \
    --features-dir "$OUT_DIR" \
    --ckpt "$CKPT" \
    --out-dir "$PROBS_DIR" \
    --batch 64

  echo "  inference done."

  # ── decode at multiple thresholds ──
  echo ""
  echo "--- decoding at thresholds 0.3 / 0.4 / 0.5 ---"
  for THR in 0.3 0.4 0.5; do
    THR_TAG=$(echo "$THR" | tr -d '.')
    OUT_CSV="$OUT_DIR/${TAG}_regions_thr${THR_TAG}.csv"
    echo "  threshold=$THR -> $OUT_CSV"

    python -u "$REPO/scripts/decode/evo2_per_token_regions.py" \
      --probs-dir "$PROBS_DIR" \
      --out-regions-csv "$OUT_CSV" \
      --threshold "$THR" \
      --min-len-bp 2000 \
      --max-gap-bp 500
  done

  echo ""
  echo "=== infer-decode complete ==="
  echo "  probs:   $PROBS_DIR"
  echo "  regions: $OUT_DIR/${TAG}_regions_thr*.csv"
  exit 0
fi

echo "ERROR: unknown --stage '$STAGE' (must be extract or infer-decode)"
exit 1
