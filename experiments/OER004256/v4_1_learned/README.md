# OER004256 / v4_1_learned — production pipeline output

End-to-end OER004256 results with **v4.1 + learned CNN decoder** (not the frozen hysteresis OP of `v4_1/`). The learned decoder replaces the `min_support_windows=3` rule that structurally rejected short metagenomic BGCs. See [`docs/learned-decoder.md`](../../../docs/learned-decoder.md) for motivation, training, and OP selection.

## Frozen operating point

| parameter | value |
|---|---|
| primary arm | w=8192 / s=2048 |
| short arm | w=4096 / s=1024 |
| primary threshold | 0.95 |
| short threshold | 0.98 |
| short length cap | 10,000 bp |
| union-dedup IoU | 0.3 |
| decoder config | c00: CNN-3L, hidden=32, antismash labels, BCE loss |
| decoder params | ~50 K |

## Headline numbers

| metric | hysteresis (`v4_1`) | **learned (`v4_1_learned`)** | antiSMASH |
|---|---:|---:|---:|
| total regions | 254 | **1,230** | 1,976 |
| primary / short split | 141 / 113 | 531 / 699 | — |
| regions covered by antiSMASH | 239 | **801** | — |
| antiSMASH-missed "novel" candidates | 15 | **429** | — |
| **AS-recall** (fraction of antiSMASH hit) | 12% | **37%** | 100% |
| **9g dedup F1 vs GT** | 0.541 | 0.461 | 0.625* |
| 15 original novel candidates preserved | — | **15/15** ✓ | — |
| v4.1 ∩ defrag_v2 type agreement | 87% | 80% | — |

\* antiSMASH dedup F1 on 9g uses its own output; not directly comparable since antiSMASH is the source of ~half of the training labels.

## Files

| File | Content |
|---|---|
| `regions.csv` | 1,230 regions — (genome, contig, start, end, max_score, source_scale) |
| `regions_fully_typed.csv` | Same + v4.1 Stage-2 per-type LR predictions + defrag_v2 CNN per-type predictions. 14 cols total. |
| `novel_triage/novel_triage_full.csv` | HMM + MIBiG three-layer verdict for the 429 novel (antiSMASH-missed) candidates. Use `PASS_for_wetlab=True` filter to shortlist ≈ 261 candidates. Top-86 are STRONG × (NOVEL or LIKELY_NOVEL). |

## Reproduce

Requires:
- Cached Evo2 w=8192 primary + w=4096 short per-window predictions on OER (from stride inference)
- Decoder checkpoint (produced by `scripts/train/decoder.py` config_id=0)

```bash
# (1) Decode regions (multi-scale, learned decoder)
python scripts/decode/v4_1_learned.py \
    --primary-preds <OER primary preds dir> \
    --short-preds   <OER short preds dir> \
    --decoder-ckpt  <path to c00 decoder.pt> \
    --out-dir       experiments/OER004256/v4_1_learned/

# (2) Add v4.1 Stage-2 per-type classification
python scripts/classify/v4_1_regions.py \
    --regions-csv  experiments/OER004256/v4_1_learned/all_regions.csv \
    --emb-dir      <cached w=8192 embedding dir> \
    --type-lr-dir  experiments/evo2_first_principles/stage2/type_lr/ \
    --out-csv      experiments/OER004256/v4_1_learned/regions_v41_typed.csv

# (3) Add defrag_v2 per-type classification
python scripts/classify/defrag_v2_regions.py \
    --regions-csv  experiments/OER004256/v4_1_learned/regions_v41_typed.csv \
    --preds-dir    <defrag_v2 OER preds dir> \
    --out-csv      experiments/OER004256/v4_1_learned/regions_fully_typed.csv

# (4) Triage novel candidates (HMM + MIBiG)
python scripts/diagnose/oer_novel_candidates.py \
    --regions-csv   experiments/OER004256/v4_1_learned/regions_fully_typed.csv \
    --antismash-dir /data/muskliu/oer_bgc_scan/results/antismash \
    --out-dir       experiments/OER004256/v4_1_learned/novel_triage/
python scripts/classify/hmm_biosynth_scan.py     # inputs in script constants
python scripts/classify/mibig_novelty.py
```
