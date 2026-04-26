#!/usr/bin/env bash
# End-to-end multi-layer Evo2 probe diagnostic.
#
# Runs once data lands from scripts/extract/multi_layer.py.
# 1. Train 4 LRs (L13, L20, L27, concat) with per-layer standardization.
# 2. Apply each LR to 9g multi-layer features.
# 3. Score each preds dir under v4.1's published OP (start=0.95, extend=0.80,
#    ms=3, max_gap=0) plus the honest_readout sensitivity band.
# 4. Print the comparison table.
#
# Decision rule:
#   if dedup_F1[concat] - dedup_F1[L20] >= 0.02   → multi-layer wins, scale up.
#   if abs(diff) < 0.01                           → tied, commit to LoRA path.
#   else                                          → marginal, run full 1.064M
#                                                   to confirm.
set -euo pipefail

REPO=${REPO:-/data/muskliu/npmaster}
PROBE=${PROBE:-${REPO}/experiments/multilayer_probe}
SHARD_DIR=${SHARD_DIR:-${PROBE}/train_subset_200k}
NINEG_FEAT=${NINEG_FEAT:-${PROBE}/9g}
MASTER_CSV=${MASTER_CSV:-${REPO}/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv}
GT_JSON=${GT_JSON:-/data/syh/benchmark/nine_genome/ground_truth.json}
OUT=${OUT:-${PROBE}/readout}

mkdir -p "${OUT}"
cd "${REPO}"

echo "=== 1/3 train LRs ==="
python scripts/train/lr_multilayer.py \
    --shard-dir "${SHARD_DIR}" \
    --master-csv "${MASTER_CSV}" \
    --out-dir "${OUT}/lrs"

echo "=== 2/3 infer on 9g ==="
PREDS_ARGS=""
for combo in L13 L20 L27 concat; do
    OUT_DIR="${OUT}/9g_preds_${combo}"
    python scripts/inference/apply_lr_multilayer.py \
        --lr-npz "${OUT}/lrs/lr_${combo}.npz" \
        --feat-dir "${NINEG_FEAT}" \
        --out-dir "${OUT_DIR}" \
        --variant-tag "${combo}"
    if [ -n "${PREDS_ARGS}" ]; then PREDS_ARGS+=","; fi
    PREDS_ARGS+="${combo}:${OUT_DIR}"
done

echo "=== 3/3 honest_readout (v4.1 OP + sensitivity band) ==="
python scripts/evaluate/honest_readout.py \
    --preds-dirs "${PREDS_ARGS}" \
    --gt-json "${GT_JSON}" \
    --out-json "${OUT}/multilayer_probe_readout.json" \
    --out-md   "${OUT}/multilayer_probe_readout.md" \
    --bootstrap 1000

echo
echo "=== diagnostic summary ==="
python - <<PY
import json
from pathlib import Path
p = Path("${OUT}/multilayer_probe_readout.json")
r = json.loads(p.read_text())
print(f"{'combo':<8s}  {'dedup_F1 (CI95)':>22s}  {'bench_P':>8s}  {'bench_R':>8s}  {'n_pred':>6s}")
for combo in ("L13","L20","L27","concat"):
    if combo not in r: continue
    e = r[combo]["alt"]   # alt = v4.1 published OP (s=0.95 e=0.80 ms=3)
    pp = e["pooled"]; ci = e["bootstrap_ci_95"]["dedup_f1"]
    print(f"{combo:<8s}  {pp['dedup_f1']:.4f} ({ci['ci_lo']:.3f}..{ci['ci_hi']:.3f})  "
          f"{pp['bench_precision']:>8.4f}  {pp['bench_recall']:>8.4f}  {pp['n_pred']:>6d}")
print()
def f(c): return r[c]["alt"]["pooled"]["dedup_f1"]
diff = f("concat") - f("L20")
print(f"Δ(concat − L20) = {diff:+.4f}")
if diff >= 0.02:
    print(">>> VERDICT: multi-layer WINS — scale up to full 1.064M extraction.")
elif abs(diff) < 0.01:
    print(">>> VERDICT: TIED — frozen Evo2 layers saturate. Commit to LoRA.")
else:
    print(">>> VERDICT: marginal. Run full 1.064M to confirm before committing.")
PY
