#!/bin/bash
# =================================================================
# Phase 1 + Phase 2 Parallel Execution Launcher
# =================================================================
# Run on GPU server (node7/node8)
#
# Phase 1: Short-BGC coverage (P1-T1 audit → P1-T2 edge padding → retrain → eval)
# Phase 2: OOD robustness (P2-T1 FP decomposition, independent of Phase 1)
#
# Usage:
#   bash run_phase1_phase2.sh [--skip-extraction]
# =================================================================

set -euo pipefail

BASE="/data/muskliu/np_master"
DATA="$BASE/data"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPTS_DIR/../.." && pwd)"

# Canonical inputs
META="$DATA/windows/windows_metadata_cap08.csv"
BGC_CSV="$DATA/bgc_annotations.csv"
GENOMES_DIR="$DATA/genomes"
EMB="$DATA/embeddings/embeddings.npy"
NLL="$DATA/embeddings/nll_features.npy"
CKPT="$DATA/stage3_results/F_cap08/best_run_s0.pt"
EVAL_JSON="$DATA/stage4_results/ws3_cap08_eval/score_swap_results.json"

# Outputs
P1_AUDIT_OUT="$DATA/stage4_results/p1_t1_audit"
P1_EDGE_OUT="$DATA/windows/edge_padding"
P1_EMB_OUT="$DATA/embeddings/edge_padding"
P1_TRAIN_OUT="$DATA/stage3_results/G_edgepad"
P1_EVAL_OUT="$DATA/stage4_results/p1_t3_edgepad_eval"
P2_OUT="$DATA/stage4_results/p2_t1_ood_decomposition"

SKIP_EXTRACTION=false
if [[ "${1:-}" == "--skip-extraction" ]]; then
    SKIP_EXTRACTION=true
fi

echo "=============================================="
echo "Phase 1 + Phase 2 Execution"
echo "=============================================="
echo "Base: $BASE"
echo "Skip extraction: $SKIP_EXTRACTION"
echo ""

# ─── Phase 2 Step 1: OOD FP decomposition (can run immediately, no GPU needed) ───
echo ""
echo ">>> P2-T1: OOD False-Positive Decomposition"
echo "=============================================="
python "$SCRIPTS_DIR/p2_t1_ood_fp_decomposition.py" \
    --eval-json "$EVAL_JSON" \
    --bgc-csv "$BGC_CSV" \
    --meta "$META" \
    --out "$P2_OUT"
echo ">>> P2-T1 DONE: $P2_OUT/ood_fp_decomposition.json"

# ─── Phase 1 Step 1: Short-BGC audit ───
echo ""
echo ">>> P1-T1: Short-BGC Support Audit"
echo "=============================================="
python "$SCRIPTS_DIR/p1_t1_shortbgc_audit.py" \
    --meta "$META" \
    --bgc-csv "$BGC_CSV" \
    --genomes-dir "$GENOMES_DIR" \
    --out "$P1_AUDIT_OUT"
echo ">>> P1-T1 DONE: $P1_AUDIT_OUT/shortbgc_audit.json"

# ─── Phase 1 Step 2: Edge-padding window generation ───
echo ""
echo ">>> P1-T2: Edge-Padding Window Generation"
echo "=============================================="
python "$SCRIPTS_DIR/p1_t2_edge_padding.py" \
    --meta "$META" \
    --bgc-csv "$BGC_CSV" \
    --genomes-dir "$GENOMES_DIR" \
    --out "$P1_EDGE_OUT" \
    --min-contig 1000
echo ">>> P1-T2 DONE: $P1_EDGE_OUT/"

# ─── Phase 1 Step 2b: Extract Evo2 embeddings for padded windows ───
if [ "$SKIP_EXTRACTION" = false ]; then
    echo ""
    echo ">>> P1-T2b: Evo2 Embedding Extraction for Padded Windows"
    echo "=============================================="
    python "$SCRIPTS_DIR/p1_t2_extract_padded_embeddings.py" \
        --fasta "$P1_EDGE_OUT/padded_sequences.fasta" \
        --existing-emb "$EMB" \
        --existing-nll "$NLL" \
        --out "$P1_EMB_OUT" \
        --layer blocks.20.mlp.l3
    echo ">>> P1-T2b DONE: $P1_EMB_OUT/"

    MERGED_EMB="$P1_EMB_OUT/embeddings_merged.npy"
    EDGEPAD_META="$P1_EDGE_OUT/windows_metadata_edgepad.csv"
else
    echo ""
    echo ">>> SKIPPING P1-T2b (--skip-extraction)"
    echo "    Assuming merged embeddings already exist at $P1_EMB_OUT/embeddings_merged.npy"
    MERGED_EMB="$P1_EMB_OUT/embeddings_merged.npy"
    EDGEPAD_META="$P1_EDGE_OUT/windows_metadata_edgepad.csv"
fi

# ─── Phase 1 Step 3: Retrain with edge-padded data ───
echo ""
echo ">>> P1-T3a: Retrain with Edge-Padded Data"
echo "=============================================="

# Create a config for edge-padded training
EDGEPAD_CONFIG="$PROJECT_DIR/bgc-gcd/configs/p1_edgepad.yaml"
cat > "$EDGEPAD_CONFIG" << 'YAMLEOF'
# P1 edge-padding config: same as ws3_cap08 but with merged embeddings
data:
  embeddings_path: PLACEHOLDER_EMB
  labels_path: PLACEHOLDER_META
  input_dim: 4096
  batch_size: 256

model:
  proj_hidden: 2048
  proj_out: 256
  dropout: 0.3
  dann_hidden: 128

training:
  n_epochs: 100
  lr: 0.1
  momentum: 0.9
  weight_decay: 0.0001
  alpha: 1.0
  beta: 0.35
  gamma: 0.1
  tau: 0.07
  dann_warmup: 20
  augmentation: true
  noise_std: 0.05
  dropout_rate: 0.1
  pos_ratio: 0.15
  checkpoint_metric: bgc_binary_auroc

output:
  dir: PLACEHOLDER_OUT
YAMLEOF

# Substitute paths
sed -i "s|PLACEHOLDER_EMB|$MERGED_EMB|g" "$EDGEPAD_CONFIG"
sed -i "s|PLACEHOLDER_META|$EDGEPAD_META|g" "$EDGEPAD_CONFIG"
sed -i "s|PLACEHOLDER_OUT|$P1_TRAIN_OUT|g" "$EDGEPAD_CONFIG"

if [ -f "$MERGED_EMB" ] && [ -f "$EDGEPAD_META" ]; then
    cd "$PROJECT_DIR/bgc-gcd"
    python train.py --config "$EDGEPAD_CONFIG" --seed 0 --tag "G_edgepad"
    echo ">>> P1-T3a DONE: $P1_TRAIN_OUT/"

    # ─── Phase 1 Step 3b: Evaluate with score_swap ───
    echo ""
    echo ">>> P1-T3b: Evaluate Edge-Padded Model"
    echo "=============================================="
    EDGEPAD_CKPT="$P1_TRAIN_OUT/best_run_s0.pt"
    if [ -f "$EDGEPAD_CKPT" ]; then
        python "$SCRIPTS_DIR/ws1_score_swap.py" \
            --ckpt "$EDGEPAD_CKPT" \
            --emb "$MERGED_EMB" \
            --meta "$EDGEPAD_META" \
            --bgc-csv "$BGC_CSV" \
            --out "$P1_EVAL_OUT" \
            --fullwindow
        echo ">>> P1-T3b DONE: $P1_EVAL_OUT/score_swap_results.json"
    else
        echo ">>> WARNING: Checkpoint not found at $EDGEPAD_CKPT"
        echo "    Training may have failed. Check $P1_TRAIN_OUT/ for logs."
    fi
else
    echo ">>> SKIPPING P1-T3: Missing merged embeddings or metadata."
    echo "    Run with Evo2 extraction first (remove --skip-extraction)."
fi

# ─── Summary ───
echo ""
echo "=============================================="
echo "EXECUTION COMPLETE"
echo "=============================================="
echo ""
echo "Phase 1 outputs:"
echo "  P1-T1 audit:     $P1_AUDIT_OUT/shortbgc_audit.json"
echo "  P1-T2 windows:   $P1_EDGE_OUT/"
echo "  P1-T2b emb:      $P1_EMB_OUT/"
echo "  P1-T3 train:     $P1_TRAIN_OUT/"
echo "  P1-T3 eval:      $P1_EVAL_OUT/score_swap_results.json"
echo ""
echo "Phase 2 outputs:"
echo "  P2-T1 decompose: $P2_OUT/ood_fp_decomposition.json"
echo ""
echo "Next steps:"
echo "  1. Review P1-T1 audit and P2-T1 decomposition results"
echo "  2. If P1-T3 eval shows improvement, compare with mainline"
echo "  3. Based on P2-T1, decide P2-T2 (hard negatives) strategy"
