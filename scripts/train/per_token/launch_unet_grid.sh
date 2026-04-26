#!/usr/bin/env bash
# Round 1: 2×2 grid — architecture (pertoken / unet) × loss (bce / dice_bce)
# Sequential on one GPU to avoid RAM duplication (4× 237 GiB = OOM).
#
# Usage:  GPU=1 bash scripts/train/per_token/launch_unet_grid.sh
set -euo pipefail

REPO=/data/muskliu/npmaster
GPU=${GPU:-1}

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_FEAT_DIR=${TRAIN_FEAT_DIR:-${REPO}/experiments/step2_per_token/train_w16384_s8192}
LABELS_CSV=${REPO}/data/windows/edge_padding/windows_metadata_edgepad_filtered.csv
BGC_ANNOT=${REPO}/data/bgc_annotations.csv

# 9g inference features (already extracted, w=8192/s=2048).
NINEG_FEAT_DIR=${REPO}/experiments/step2_per_token/9g_w8192_s2048

# ── Output root ────────────────────────────────────────────────────────────
OUT=${REPO}/experiments/step2_per_token/unet_grid
mkdir -p "${OUT}"

# ── Grid configs ───────────────────────────────────────────────────────────
# A = baseline (pertoken + bce)
# B = unet + bce
# C = pertoken + dice_bce
# D = unet + dice_bce

declare -A MODELS=( [A]=pertoken [B]=unet [C]=pertoken [D]=unet )
declare -A LOSSES=( [A]=bce      [B]=bce  [C]=dice_bce [D]=dice_bce )

echo "=== Round 1 grid: 4 configs SEQUENTIAL on GPU ${GPU} ==="
echo "TRAIN_FEAT_DIR=${TRAIN_FEAT_DIR}"
echo ""

for CFG in A B C D; do
    M=${MODELS[$CFG]}
    L=${LOSSES[$CFG]}
    RUN_DIR=${OUT}/${CFG}_${M}_${L}
    mkdir -p "${RUN_DIR}"

    echo ""
    echo "============================================"
    echo "[${CFG}] model=${M} loss=${L} gpu=${GPU}"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=${GPU} python -u ${REPO}/scripts/train/per_token/train_shards.py \
        --features-dir "${TRAIN_FEAT_DIR}" \
        --labels-csv   "${LABELS_CSV}" \
        --bgc-annot    "${BGC_ANNOT}" \
        --out-dir      "${RUN_DIR}" \
        --model        "${M}" \
        --loss         "${L}" \
        --d-reduced    128 \
        --hidden       64 \
        --batch        16 \
        --epochs       15 \
        --patience     5 \
        --pos-weight   3.0 \
        --lr           1e-3 \
        --seed         0 \
        --max-wall-minutes 45 \
        2>&1 | tee "${RUN_DIR}/train.log"

    echo "[${CFG}] training done."
done

echo ""
echo "=== All training done ==="

# ── 9g inference + decode + eval for each config ───────────────────────────
GT_JSON=/data/syh/benchmark/nine_genome/ground_truth.json

for CFG in A B C D; do
    M=${MODELS[$CFG]}
    L=${LOSSES[$CFG]}
    RUN_DIR=${OUT}/${CFG}_${M}_${L}
    CKPT=${RUN_DIR}/best.pt
    PROBS_DIR=${RUN_DIR}/9g_probs
    REGIONS_DIR=${RUN_DIR}/9g_regions
    READOUT_DIR=${RUN_DIR}/9g_readouts
    mkdir -p "${PROBS_DIR}" "${REGIONS_DIR}" "${READOUT_DIR}"

    if [ ! -f "${CKPT}" ]; then
        echo "[${CFG}] SKIP — no best.pt found"
        continue
    fi

    echo "[${CFG}] inference on 9g..."
    CUDA_VISIBLE_DEVICES=0 python -u ${REPO}/scripts/inference/per_token/infer_shards.py \
        --features-dir "${NINEG_FEAT_DIR}" \
        --ckpt         "${CKPT}" \
        --out-dir      "${PROBS_DIR}" \
        --batch        32

    for THR in 0.3 0.4 0.5; do
        echo "[${CFG}] decode thr=${THR}..."
        python ${REPO}/scripts/decode/per_token_v1.py \
            --probs-dir       "${PROBS_DIR}" \
            --out-regions-csv "${REGIONS_DIR}/regions_thr${THR}.csv" \
            --threshold       "${THR}" \
            --min-len-bp      2000 \
            --max-gap-bp      500

        echo "[${CFG}] eval thr=${THR}..."
        python ${REPO}/scripts/evaluate/per_token_readout.py \
            --regions-csv "${REGIONS_DIR}/regions_thr${THR}.csv" \
            --method-name "${CFG}_${M}_${L}_thr${THR}" \
            --gt-json     "${GT_JSON}" \
            --out-json    "${READOUT_DIR}/readout_thr${THR}.json"
    done
    echo "[${CFG}] done."
done

echo ""
echo "=== Round 1 complete.  Results: ==="
echo "  cat ${OUT}/*/9g_readouts/*.json | python -m json.tool"
echo ""
echo "Compare dedup_F1, bench_P, bench_R, max_preds_per_GT across A/B/C/D."
echo "Gate: dedup_F1 > 0.30 → proceed to Round 2."
