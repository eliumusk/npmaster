#!/usr/bin/env bash
# Round 2: consolidate D (unet+dice_bce) — the Round 1 winner.
# Two parallel jobs (each ~237 GiB RAM), then sequential follow-ups.
#
# Usage:  bash scripts/train/per_token/launch_round2.sh
set -euo pipefail

REPO=/data/muskliu/npmaster

TRAIN_FEAT_DIR=${REPO}/experiments/step2_per_token/train_w16384_s8192
LABELS_CSV=${REPO}/data/windows/edge_padding/windows_metadata_edgepad_filtered.csv
BGC_ANNOT=${REPO}/data/bgc_annotations.csv
NINEG_FEAT_DIR=${REPO}/experiments/step2_per_token/9g_w8192_s2048
GT_JSON=/data/syh/benchmark/nine_genome/ground_truth.json

OUT=${REPO}/experiments/step2_per_token/round2
mkdir -p "${OUT}"

COMMON_ARGS="--features-dir ${TRAIN_FEAT_DIR} \
  --labels-csv ${LABELS_CSV} \
  --bgc-annot ${BGC_ANNOT} \
  --model unet \
  --loss dice_bce \
  --d-reduced 128 \
  --batch 16 \
  --seed 0"

# ── Phase 1: two parallel jobs ───────────────────────────────────────────
echo "=== Round 2 Phase 1: D-long + D-crop in parallel ==="

# E = D-long: same as D but 30 epochs, patience=10
E_DIR=${OUT}/E_unet_dice_long
mkdir -p "${E_DIR}"
echo "[E] unet+dice, 30 epochs, patience=10, hidden=64, lr=1e-3 → ${E_DIR}"
CUDA_VISIBLE_DEVICES=0 nohup python -u ${REPO}/scripts/train/per_token/train_shards.py \
    ${COMMON_ARGS} \
    --hidden 64 \
    --lr 1e-3 \
    --epochs 30 \
    --patience 10 \
    --max-wall-minutes 120 \
    --out-dir "${E_DIR}" \
    > "${E_DIR}/train.log" 2>&1 &
PID_E=$!
echo "  PID=${PID_E}"

# F = D-crop: same as D but with random crop augmentation
F_DIR=${OUT}/F_unet_dice_crop
mkdir -p "${F_DIR}"
echo "[F] unet+dice, 30 epochs, patience=10, crop 256-2048 → ${F_DIR}"
CUDA_VISIBLE_DEVICES=1 nohup python -u ${REPO}/scripts/train/per_token/train_shards.py \
    ${COMMON_ARGS} \
    --hidden 64 \
    --lr 1e-3 \
    --epochs 30 \
    --patience 10 \
    --crop-min 256 \
    --crop-max 2048 \
    --max-wall-minutes 120 \
    --out-dir "${F_DIR}" \
    > "${F_DIR}/train.log" 2>&1 &
PID_F=$!
echo "  PID=${PID_F}"

echo ""
echo "Waiting for E and F..."
wait ${PID_E} ${PID_F}
echo "=== Phase 1 training done ==="

# ── Phase 2: two more parallel jobs ──────────────────────────────────────
echo ""
echo "=== Round 2 Phase 2: D-h128 + D-lr5e4 ==="

# G = D-h128: bigger model capacity
G_DIR=${OUT}/G_unet_dice_h128
mkdir -p "${G_DIR}"
echo "[G] unet+dice, hidden=128, 30 epochs → ${G_DIR}"
CUDA_VISIBLE_DEVICES=0 nohup python -u ${REPO}/scripts/train/per_token/train_shards.py \
    ${COMMON_ARGS} \
    --hidden 128 \
    --lr 1e-3 \
    --epochs 30 \
    --patience 10 \
    --max-wall-minutes 120 \
    --out-dir "${G_DIR}" \
    > "${G_DIR}/train.log" 2>&1 &
PID_G=$!
echo "  PID=${PID_G}"

# H = D-lr5e4: lower LR for more stable long training
H_DIR=${OUT}/H_unet_dice_lr5e4
mkdir -p "${H_DIR}"
echo "[H] unet+dice, lr=5e-4, 30 epochs → ${H_DIR}"
CUDA_VISIBLE_DEVICES=1 nohup python -u ${REPO}/scripts/train/per_token/train_shards.py \
    ${COMMON_ARGS} \
    --hidden 64 \
    --lr 5e-4 \
    --epochs 30 \
    --patience 10 \
    --max-wall-minutes 120 \
    --out-dir "${H_DIR}" \
    > "${H_DIR}/train.log" 2>&1 &
PID_H=$!
echo "  PID=${PID_H}"

echo ""
echo "Waiting for G and H..."
wait ${PID_G} ${PID_H}
echo "=== Phase 2 training done ==="

# ── 9g eval for all Round 2 configs ──────────────────────────────────────
echo ""
echo "=== 9g inference + decode + eval ==="

for CFG_DIR in ${E_DIR} ${F_DIR} ${G_DIR} ${H_DIR}; do
    NAME=$(basename ${CFG_DIR})
    CKPT=${CFG_DIR}/best.pt
    PROBS_DIR=${CFG_DIR}/9g_probs
    REGIONS_DIR=${CFG_DIR}/9g_regions
    READOUT_DIR=${CFG_DIR}/9g_readouts
    mkdir -p "${PROBS_DIR}" "${REGIONS_DIR}" "${READOUT_DIR}"

    if [ ! -f "${CKPT}" ]; then
        echo "[${NAME}] SKIP — no best.pt"
        continue
    fi

    echo "[${NAME}] inference..."
    CUDA_VISIBLE_DEVICES=0 python -u ${REPO}/scripts/inference/per_token/infer_shards.py \
        --features-dir "${NINEG_FEAT_DIR}" \
        --ckpt "${CKPT}" \
        --out-dir "${PROBS_DIR}" \
        --batch 32

    for THR in 0.3 0.4 0.5; do
        python ${REPO}/scripts/decode/per_token_v1.py \
            --probs-dir "${PROBS_DIR}" \
            --out-regions-csv "${REGIONS_DIR}/regions_thr${THR}.csv" \
            --threshold "${THR}" \
            --min-len-bp 2000 \
            --max-gap-bp 500

        python ${REPO}/scripts/evaluate/per_token_readout.py \
            --regions-csv "${REGIONS_DIR}/regions_thr${THR}.csv" \
            --method-name "${NAME}_thr${THR}" \
            --gt-json "${GT_JSON}" \
            --out-json "${READOUT_DIR}/readout_thr${THR}.json"
    done

    # Also run decoder sweep with gap-fill variants
    bash ${REPO}/scripts/train/per_token/decoder_sweep.sh \
        "${PROBS_DIR}" "${REGIONS_DIR}" "${READOUT_DIR}"

    echo "[${NAME}] done."
done

echo ""
echo "=== Round 2 complete ==="
