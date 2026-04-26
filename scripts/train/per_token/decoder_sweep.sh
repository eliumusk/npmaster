#!/usr/bin/env bash
set -euo pipefail

REPO=/data/muskliu/npmaster
PROBS=${1:-${REPO}/experiments/step2_per_token/unet_grid/D_unet_dice_bce/9g_probs}
REGIONS_BASE=${2:-${REPO}/experiments/step2_per_token/unet_grid/D_unet_dice_bce/9g_regions}
READOUT_BASE=${3:-${REPO}/experiments/step2_per_token/unet_grid/D_unet_dice_bce/9g_readouts}
GT=/data/syh/benchmark/nine_genome/ground_truth.json

echo "=== Decoder sweep on $(basename $(dirname $PROBS)) ==="
echo ""
printf "%-8s %-6s  %-8s %-8s %-8s %-6s %-10s\n" "thr" "gap" "dedup_F1" "bench_P" "bench_R" "n_pred" "max_per_gt"
echo "--------------------------------------------------------------"

for THR in 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6; do
    for GAP in 500 1000 2000 4000; do
        TAG="thr${THR}_gap${GAP}"
        RCSV="${REGIONS_BASE}/regions_${TAG}.csv"
        RJSON="${READOUT_BASE}/readout_${TAG}.json"

        python ${REPO}/scripts/decode/per_token_v1.py \
            --probs-dir "$PROBS" \
            --out-regions-csv "$RCSV" \
            --threshold "$THR" \
            --min-len-bp 2000 \
            --max-gap-bp "$GAP" 2>/dev/null

        python ${REPO}/scripts/evaluate/per_token_readout.py \
            --regions-csv "$RCSV" \
            --method-name "D_${TAG}" \
            --gt-json "$GT" \
            --out-json "$RJSON" 2>/dev/null

        python -c "
import json
d = json.load(open('${RJSON}'))
b = d['nine_genome']['summary']['bench']
print(f'${THR}     ${GAP}   {b[\"dedup_f1\"]:.4f}   {b[\"precision\"]:.3f}    {b[\"recall\"]:.3f}    {b[\"n_pred\"]:<6d} {b[\"max_preds_per_gt\"]}')
"
    done
done
