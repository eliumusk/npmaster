# Stage 5b Diagnostic Report

_raw F1 high but fragmented; internal-val proxy inverts the ranking_

**Date:** 2026-04-20
**Selection protocol:** no 9-genome data inspected during selection.
Primary metric convention (per 2026-04-20): `dedup_f1`,
`bench_precision`, `bench_recall`. Other metrics are secondary.

---

## 0. TL;DR

- **Stage 5a's "matched the 9-genome-tuned upper bound" claim is retracted.** It
  was an artifact of the asymmetric `bench_f1` counting multiple fragmented
  predictions per GT. Honest `dedup_f1` on Stage 5a was 0.365 — **below**
  BGC-Prophet (0.391), **far below** the 9-genome-tuned `defrag_v2` (0.439).
- **Stage 5b's decoder gate (`dedup_precision / bench_precision ≥ 0.75`)
  successfully removed fragmentation.** For the same `tight_pos_099` checkpoint,
  switching from `val_frozen_fused` to the Stage 5b-selected `(start=0.90,
  ms=2, cap=None)` decoder drops `bench_f1` slightly (0.439 → 0.408) but
  lifts honest `dedup_f1` (0.365 → **0.398**), clearing BGC-Prophet by a hair
  and leaving a ~0.04 gap to `defrag_v2`.
- **But Stage 5b's internal val (shared antiSMASH-region GT) selects the
  wrong model.** Baseline wins internal val (`dedup_f1` 0.733) but collapses
  on 9-genome (`dedup_f1` 0.256). tight_pos_099 is the **worst** on internal
  val (0.587) and the **best** on 9-genome (0.398). The two rankings are
  **inverted**.
- **Training-side interventions move 10-50 kb precision, not <10 kb recall.**
  Every variant, including `short_bgc_upweight_v1`, gets ~0.30 recall on
  <5 kb GT. The window-size×overlap-threshold interaction makes short BGCs
  unrepresentable in training (see §4 for mechanism).

This is a **methodology-failure diagnosis**, not a performance claim.
Neither Stage 5a (self-label F1) nor Stage 5b (shared antiSMASH-region val)
is a valid selection proxy for 9-genome tight-core GT. **Stage 5c (antiSMASH
`cand_cluster` / `proto_core` coords as val GT) is therefore not an
optimisation — it's a necessary correction.**

---

## 1. 9-genome final results

All four candidates use their Stage 5b-selected (gate-passing) decoder.
No 9-genome data was used to pick any parameter. Reference rows are from
prior runs, included for comparison.

| candidate | source | decoder (start / ext / ms / cap) | **dedup_f1** | **bench_P** | **bench_R** | n_pred | max_preds/GT | dup_TP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `defrag_v2_ms10_s099` (ref) | 9-genome-tuned | 0.99 / 0.95 / 10 / — | **0.439** | 0.471 | 0.410 | 140 | 1 | 0 |
| `tight_pos_099_v1_s0` | Stage 5b diagnostic | 0.90 / 0.85 / 2 / — | **0.398** | 0.322 | 0.557 | 342 | 2 | 4 |
| BGC-Prophet (ref) | external baseline | — | **0.391** | 0.453 | 0.348 | 170 | 1 | 0 |
| `tight_pos_075_v1_s0` | Stage 5b top tight | 0.90 / 0.85 / 2 / — | **0.314** | 0.224 | 0.554 | 415 | 2 | 2 |
| `baseline_v1_s0` | **Stage 5b winner** | 0.99 / 0.85 / 1 / — | **0.256** | 0.165 | 0.577 | 437 | 1 | 0 |
| `short_bgc_upweight_v1_s0` | Stage 5b runner-up | 0.99 / 0.95 / 1 / — | **0.254** | 0.169 | 0.587 | 445 | 2 | 3 |

Notes:

- Every Stage 5b candidate has `max_preds_per_gt ≤ 2` and `duplicate_TP ≤ 4`
  — the fragmentation gate held on 9-genome, not just on val.
- tight_pos_099 → tight_pos_075 → baseline shows a **monotonic** relationship
  between training-label tightness and 9-genome `dedup_f1` (0.398 → 0.314
  → 0.256). The tighter the label, the better the generalisation.
- short_bgc_upweight_v1_s0 is indistinguishable from baseline on 9-genome
  (0.254 vs 0.256). The upweighting did not produce short-BGC generalisation.

---

## 2. Internal-val vs 9-genome inversion

Stage 5b's *selected* winner is baseline_v1; the *correct* winner for the
9-genome target is tight_pos_099. These are nearly opposite in the two
rankings:

| variant | internal val dedup_f1 (antiSMASH regions) | 9-genome dedup_f1 | gap |
|---|---:|---:|---:|
| baseline_v1 | **0.733** (rank 1) | **0.256** (rank 4) | **-0.477** |
| short_bgc_upweight_v1 | 0.731 (rank 2) | 0.254 (rank 5) | -0.477 |
| tight_pos_075_v1 | 0.716 (rank 3) | 0.314 (rank 3) | -0.402 |
| tight_pos_099_v1 | **0.587** (rank 9) | **0.398** (rank 2) | **-0.189** |

The internal-val metric is **monotonically wrong** for the 9-genome
target. A Spearman-style reading: the tight_pos variants that drop
furthest on internal val are the ones that gain most (relatively) on
9-genome.

**Why:** antiSMASH regions are wide (the core biosynthetic genes +
~10-20 kb flanking extension). 9-genome GT is human-curated tight core.
A model trained to predict tight cores scores *worse* against wide val
GT (predictions don't reach the wide GT boundaries) but *better* against
tight 9-genome GT. Stage 5b's val GT sign is opposite to the final
target.

---

## 3. Length-bucket metrics on 9-genome

GT distribution: 48 `<5kb`, 89 `5-10kb`, 85 `10-20kb`, 61 `20-50kb`,
22 `>=50kb`.

### Recall by bucket (fraction of GT covered)

| variant | <5kb | 5-10kb | 10-20kb | 20-50kb | >=50kb |
|---|---:|---:|---:|---:|---:|
| baseline_v1 | 0.333 | 0.539 | 0.518 | 0.770 | 0.955 |
| short_bgc_upweight_v1 | 0.333 | 0.551 | 0.506 | 0.770 | 0.955 |
| tight_pos_075_v1 | 0.292 | 0.551 | 0.494 | 0.770 | 0.955 |
| tight_pos_099_v1 | 0.312 | 0.506 | 0.482 | 0.770 | 0.955 |

Recall is nearly identical across variants. **Tight training does not
improve short-BGC recall**, and `short_bgc_upweight_v1` is no better
than baseline either. The variance is noise.

### Precision by bucket (fraction of preds that hit GT)

| variant | <5kb | 5-10kb | 10-20kb | 20-50kb | >=50kb |
|---|---:|---:|---:|---:|---:|
| baseline_v1 | 0.000 | 0.000 | **0.000** | **0.184** | 0.667 |
| short_bgc_upweight_v1 | 0.000 | 0.000 | 0.033 | 0.185 | 0.674 |
| tight_pos_075_v1 | 0.000 | 0.000 | 0.038 | 0.340 | 0.750 |
| tight_pos_099_v1 | 0.000 | 0.000 | **0.126** | **0.469** | 0.730 |

**The tight-training precision gain is concentrated in the 20-50 kb
bucket** (baseline 0.184 → tight_099 0.469, **+0.285**). The 10-20 kb
bucket also improves (0.000 → 0.126, +0.126). `>=50kb` barely moves.

So the dedup F1 improvement comes almost entirely from **sharpening
boundaries of medium-length predictions**, not from catching any new
BGCs.

---

## 4. Why short-BGC recall doesn't move

Mechanism (verified against `bgc_annotations.csv` and `stage4_results/
p2_t2_hard_neg/windows_metadata_hardneg.csv`):

- Training window size = 8192 bp, stride = 2048 bp.
- Only 511 / 19 762 training BGCs (2.6 %) are `<5kb`.
- A 3 kb BGC sits inside at most one or two 8192 bp windows with
  `overlap_ratio ≈ 0.37` (3 kb / 8 kb).
- Baseline labels: `overlap ≥ 0.5` → positive → a 3 kb BGC window is
  **ambiguous** (masked), not positive. Baseline never learns to fire
  on a `<5kb` BGC cleanly.
- `tight_pos_075`: threshold 0.75 → even fewer short windows qualify
  as positive — short BGCs are *more* masked, not less.
- `tight_pos_099`: threshold 0.99 → *no* `<5kb` BGC window can contain
  a sub-region with ≥99 % overlap of the 8 kb window, so every such
  window is masked.
- `short_bgc_upweight_v1`: upweights short-BGC positives by 4×. But
  the set of short-BGC *positives* at overlap≥0.5 is nearly empty
  (most short BGCs fall into ambiguous), so the 4× is 4× nothing.

The window/stride design + overlap rule structurally excludes short
BGCs from the positive label set. No label-reweighting variant can fix
this.

**Implication for Stage 5 direction:** short-BGC recall is a
training-data + multi-scale problem. Two structurally-different fixes
are needed:

1. **Shorter windows** — retrain with `window=2048 / stride=512` so a
   3 kb BGC can have `overlap_ratio ≥ 0.9`. Requires rebuilding chunks
   and re-extracting Evo2 embeddings at the finer stride.
2. **Tight-core source coords** — use antiSMASH's `cand_cluster` /
   `proto_core` coords (already on disk under
   `/data/muskliu/npmaster/data/antismash_results/*/`) to construct
   a training label where short BGCs are fully inside 8192 bp windows
   because the *region* is tighter, not because the window is smaller.

Stage 5c does (2). Stage 5d would do (1).

---

## 5. What Stage 5b's three runs proved

Two parallel failure modes are now documented. Neither is a
performance story; both are methodology evidence for Stage 5c.

| run | model picked by | decoder picked by | 9-genome dedup_f1 |
|---|---|---|---:|
| Stage 5a | self-label F1 (circular) | fixed `val_frozen_fused` | 0.365 (fragmented) |
| Stage 5b (official winner) | shared antiSMASH-region val | gate-passing sweep | 0.256 (wrong model) |
| Stage 5b (diagnostic) | self-evident tight variant | gate-passing sweep | **0.398** (still below `defrag_v2`) |

Each selection protocol has a distinct pathology:

- **Stage 5a failure** — right model, wrong decoder. Fragmentation
  inflates bench F1 because the asymmetric metric double-counts
  TPs. `dedup_f1` exposed this.
- **Stage 5b failure** — right decoder class, wrong model. Internal val
  GT is antiSMASH-wide so it rewards models that over-extend their
  predictions. These are the models that fail on 9-genome tight GT.

The diagnostic cell (tight_pos_099 + Stage 5b decoder) shows what a
*legitimate* Stage 5 result could look like if the selection path were
sound: **dedup_f1 = 0.398, clean (frag_ratio 0.964), no 9-genome
tuning**. It beats BGC-Prophet (0.391) by a small margin, and loses to
the 9-genome-tuned `defrag_v2` upper bound by 0.041 points. This is not
reported as a Stage 5 claim because we could not have selected it
without already knowing the answer.

---

## 6. Next steps

Priority order:

1. **Stage 5c (necessary)** — parse
   `/data/muskliu/npmaster/data/antismash_results/*/{cand_cluster,proto_core}`
   coordinates, rebuild `bgc_annotations_core.csv`, regenerate
   `windows_metadata_core.csv`, re-apply the same Stage 5 variant
   ladder, and redo Stage 5b with the new tight-core val GT. This
   fixes the proxy inversion **without needing any new data**.

2. **Stage 5d (for <10 kb recall)** — retrain a short-window head
   (`window=2048`, `stride=512`). Requires fresh Evo2 extraction at
   the finer stride. Larger compute cost but the only path to the
   48 `<5kb` GT regions.

3. **MIBiG injection (only if 5c + 5d aren't enough)** — add ~2 500
   human-curated MIBiG BGC sequences as independent tight-core
   positives. Highest cost of the three.

Do 5c first. If 5c's tight-core val ranks `tight_pos_099` or
`tight_pos_075` at the top (matching 9-genome), the selection protocol
is repaired and the 0.398 dedup_f1 becomes a legitimate Stage 5 result.
If 5c still doesn't rank tight variants on top, the problem is not
just the GT domain and we escalate to 5d.

---

## 7. Artefacts

- Per-checkpoint val predictions: `experiments/stage5_no9_data_v1-<tag>/artifacts/val_predictions_<tag>/`
- Per-checkpoint 9-genome predictions:
  `experiments/stage5_no9_data_v1-<variant>/artifacts/predictions_9genome_<tag>/`
- Sweep outputs:
  `experiments/stage5b_val_decoder_sweep_all_cells.csv` (all 24 ×
  231 cells) and `..._selected.csv` (24 winners)
- Freeze: `experiments/stage5_no9_data_v1-selection_decision_b.md`
- Final 9-genome metrics:
  `experiments/stage5_no9_data_v1-<variant>/metrics/final_9genome_b_<tag>.json`
- Fragmentation figures (Stage 5a diagnosis):
  `figures/stage5_fragmentation/`
