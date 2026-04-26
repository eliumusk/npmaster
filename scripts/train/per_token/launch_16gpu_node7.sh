#!/usr/bin/env bash
# Node7 half: configs E09-E16. Same structure as launch_16gpu.sh.
set -euo pipefail

REPO=/data/muskliu/npmaster
PREPARED=/dev/shm/per_token_data
OUT=${REPO}/experiments/step2_per_token/round2
GT_JSON=/data/syh/benchmark/nine_genome/ground_truth.json
NINEG_FEAT_DIR=${REPO}/experiments/step2_per_token/9g_w8192_s2048

mkdir -p "${OUT}"

COMMON="--prepared-dir ${PREPARED} --model unet --loss dice_bce --d-reduced 128 --seed 0 --epochs 30 --patience 10 --max-wall-minutes 150 --subsample-train 80000 --subsample-val 20000"

run_config() {
    local NAME=$1 GPU=$2 EXTRA=$3
    local DIR=${OUT}/${NAME}
    mkdir -p "${DIR}"
    echo "[${NAME}] GPU=${GPU}"
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u ${REPO}/scripts/train/per_token/train_shards.py \
        ${COMMON} ${EXTRA} --out-dir "${DIR}" \
        > "${DIR}/train.log" 2>&1 &
    echo "  PID=$!"
}

echo "=== Wave 1: 4 configs on GPUs 0-3 ==="
run_config "E09_h256_lr5e4"        0  "--hidden 256 --lr 5e-4 --batch 16"
run_config "E10_h64_lr1e3_b32"     1  "--hidden 64  --lr 1e-3 --batch 32"
run_config "E11_h64_lr1e3_b64"     2  "--hidden 64  --lr 1e-3 --batch 64"
run_config "E12_h128_lr5e4_crop"   3  "--hidden 128 --lr 5e-4 --batch 16 --crop-min 256 --crop-max 2048"

echo "Waiting for wave 1..."
wait
echo "=== Wave 1 done ==="

echo ""
echo "=== Wave 2: 4 configs on GPUs 0-3 ==="
run_config "E13_h64_lr1e3_crop512" 0  "--hidden 64  --lr 1e-3 --batch 16 --crop-min 512 --crop-max 2048"
run_config "E14_h64_lr2e3"         1  "--hidden 64  --lr 2e-3 --batch 16"
run_config "E15_h128_lr3e4"        2  "--hidden 128 --lr 3e-4 --batch 16"
run_config "E16_h64_lr1e3_seed42"  3  "--hidden 64  --lr 1e-3 --batch 16 --seed 42"

echo "Waiting for wave 2..."
wait
echo "=== Wave 2 done ==="

echo ""
echo "=== 9g inference + eval ==="
for CFG_DIR in ${OUT}/E{09,10,11,12,13,14,15,16}*; do
    [ -d "${CFG_DIR}" ] || continue
    NAME=$(basename ${CFG_DIR})
    CKPT=${CFG_DIR}/best.pt
    [ -f "${CKPT}" ] || { echo "[${NAME}] SKIP — no best.pt"; continue; }
    [ -f "${CFG_DIR}/9g_readouts/readout_thr0.5.json" ] && { echo "[${NAME}] already evaluated"; continue; }

    PROBS=${CFG_DIR}/9g_probs
    REGIONS=${CFG_DIR}/9g_regions
    READOUT=${CFG_DIR}/9g_readouts
    mkdir -p "${PROBS}" "${REGIONS}" "${READOUT}"

    echo "[${NAME}] inference..."
    CUDA_VISIBLE_DEVICES=0 python -u ${REPO}/scripts/inference/per_token/infer_shards.py \
        --features-dir "${NINEG_FEAT_DIR}" --ckpt "${CKPT}" --out-dir "${PROBS}" --batch 64

    for THR in 0.3 0.4 0.5 0.55; do
        python ${REPO}/scripts/decode/per_token_v1.py \
            --probs-dir "${PROBS}" \
            --out-regions-csv "${REGIONS}/regions_thr${THR}.csv" \
            --threshold "${THR}" --min-len-bp 2000 --max-gap-bp 500 2>/dev/null

        python ${REPO}/scripts/evaluate/per_token_readout.py \
            --regions-csv "${REGIONS}/regions_thr${THR}.csv" \
            --method-name "${NAME}_thr${THR}" \
            --gt-json "${GT_JSON}" \
            --out-json "${READOUT}/readout_thr${THR}.json" 2>/dev/null
    done
    echo "[${NAME}] done."
done

echo ""
echo "=== Node7 complete ==="
