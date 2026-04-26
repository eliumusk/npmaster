#!/usr/bin/env bash
# 16-config grid: 4 parallel per node × 2 waves × 2 nodes = 16 configs.
# Data pre-saved to /dev/shm; each process copies to local RAM (~300 GiB).
# 4 × 300 GiB = 1.2 TiB out of 1.8 TiB — fits comfortably.
#
# Usage:  bash scripts/train/per_token/launch_16gpu.sh
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

# ── Wave 1: 4 configs ────────────────────────────────────────────────────
echo "=== Wave 1: 4 configs on GPUs 0-3 ==="
run_config "E01_h64_lr1e3"       0  "--hidden 64  --lr 1e-3 --batch 16"
run_config "E02_h64_lr5e4"       1  "--hidden 64  --lr 5e-4 --batch 16"
run_config "E04_h64_lr1e3_crop"  2  "--hidden 64  --lr 1e-3 --batch 16 --crop-min 256 --crop-max 2048"
run_config "E06_h128_lr1e3"      3  "--hidden 128 --lr 1e-3 --batch 16"

echo "Waiting for wave 1..."
wait
echo "=== Wave 1 done ==="

# ── Wave 2: 4 more configs ───────────────────────────────────────────────
echo ""
echo "=== Wave 2: 4 configs on GPUs 0-3 ==="
run_config "E03_h64_lr3e4"       0  "--hidden 64  --lr 3e-4 --batch 16"
run_config "E05_h64_lr5e4_crop"  1  "--hidden 64  --lr 5e-4 --batch 16 --crop-min 256 --crop-max 2048"
run_config "E07_h128_lr5e4"      2  "--hidden 128 --lr 5e-4 --batch 16"
run_config "E08_h128_lr1e3_crop" 3  "--hidden 128 --lr 1e-3 --batch 16 --crop-min 256 --crop-max 2048"

echo "Waiting for wave 2..."
wait
echo "=== Wave 2 done ==="

# ── 9g eval ──────────────────────────────────────────────────────────────
echo ""
echo "=== 9g inference + eval ==="
for CFG_DIR in ${OUT}/E*; do
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

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=== Results summary (thr=0.5) ==="
python3 -c "
import json, glob, os
rows = []
for f in sorted(glob.glob('${OUT}/E*/9g_readouts/readout_thr0.5.json')):
    d = json.load(open(f))
    b = d['nine_genome']['summary']['bench']
    name = os.path.basename(os.path.dirname(os.path.dirname(f)))
    rows.append((name, b['dedup_f1'], b['precision'], b['recall'], b['n_pred'], b['max_preds_per_gt']))
print(f'{\"config\":<30s} {\"dedup_F1\":>8s} {\"bench_P\":>8s} {\"bench_R\":>8s} {\"n_pred\":>6s} {\"max/gt\":>6s}')
print('-' * 70)
for r in sorted(rows, key=lambda x: -x[1]):
    print(f'{r[0]:<30s} {r[1]:8.4f} {r[2]:8.3f} {r[3]:8.3f} {r[4]:6d} {r[5]:6d}')
print()
print('v4.1-hysteresis reference:     0.4878    0.598    0.459    246')
"
