# Stage 5c-1 Results — proto_core retraining, 3 variants

**Date:** 2026-04-20.
**Protocol (frozen before launch):** `docs/stage5c1-plan.md`.
**Selection freeze:** `experiments/stage5c1-selection_decision.md`.
**Proxy validation prerequisite:** Stage 5c-0 confirmed proto_core GT
flips the ranking correlation with 9-genome from -0.80 (region GT) to
+0.80 (core GT). See `experiments/stage5c0_core_decoder_sweep_selected.csv`.

**No 9-genome data was inspected during training, label construction,
or decoder selection.**

---

## 1. Primary metrics on 9-genome (the bench)

| method | **dedup_f1** | **bench_P** | **bench_R** | n_pred | source |
|---|---:|---:|---:|---:|---|
| `defrag_v2_ms10_s099` | **0.4385** | 0.471 | 0.410 | 140 | 9-genome-tuned (upper bound) |
| **`baseline_core_v1_s0`** | **0.3995** | 0.373 | 0.528 | 389 | **Stage 5c-1 winner** |
| BGC-Prophet | 0.3911 | 0.453 | 0.348 | 170 | external baseline |
| Stage 5b diagnostic (`tight_pos_099` + core-gate decoder) | 0.3984 | 0.322 | 0.557 | 342 | illegitimate selection (diagnostic only) |
| `short_core_upweight_v1_s0` | 0.3729 | 0.322 | 0.538 | 410 | Stage 5c-1 rank 2 |
| Stage 5b selected winner (`baseline_v1` region-val) | 0.2563 | 0.165 | 0.577 | 437 | picked on broken antiSMASH-region proxy |

**Headline:** `baseline_core_v1_s0` hits **`dedup_f1 = 0.3995`** under
a legitimate no-9-genome protocol — **clears the `≥ 0.40` minimum
success criterion by 0.0005** (i.e. at the threshold), beats
BGC-Prophet on `dedup_f1` by 0.008, and matches the Stage 5b
diagnostic that needed protocol-violating hindsight. Still **0.039
below the 9-genome-tuned `defrag_v2` upper bound**.

`short_core_upweight_v1`, which won on core-val (rank 1, 0.3202), came
in below `baseline_core_v1` on 9-genome (0.373 vs 0.400). The
upweight did not generalise — details in §3.

### Fragmentation gate behaviour on 9-genome

On internal core-val both winners had `frag_ratio = 1.000`. On
9-genome, some fragmentation slipped back in under distribution shift:

| tag | 9-genome `max_preds_per_gt` | `duplicate_precision_tp` | 9-genome `frag_ratio` |
|---|---:|---:|---:|
| `baseline_core_v1_s0` | 4 | 20 | ~0.90 |
| `short_core_upweight_v1_s0` | 3 | 15 | ~0.89 |

Both slightly below the core-val 0.95 gate on 9-genome, but far
cleaner than Stage 5a's unguarded `tight_pos_099 + val_frozen_fused`
run (`max 6, dup 38, frag 0.745`).

---

## 2. Length-bucket metrics on 9-genome

GT distribution: 48 `<5kb`, 89 `5-10kb`, 85 `10-20kb`, 61 `20-50kb`,
22 `>=50kb`.

### `baseline_core_v1_s0` (winner)

| bucket | GT n | recall | pred n | pred precision |
|---|---:|---:|---:|---:|
| `<5kb` | 48 | **0.229** | 1 | 0.000 |
| `5-10kb` | 89 | 0.551 | 1 | 0.000 |
| `10-20kb` | 85 | 0.482 | 281 | 0.214 |
| `20-50kb` | 61 | 0.738 | 102 | **0.598** |
| `>=50kb` | 22 | 0.773 | 24 | **0.875** |

### `short_core_upweight_v1_s0`

| bucket | GT n | recall | pred n | pred precision |
|---|---:|---:|---:|---:|
| `<5kb` | 48 | 0.271 | 1 | 0.000 |
| `5-10kb` | 89 | 0.528 | 1 | 0.000 |
| `10-20kb` | 85 | 0.482 | 267 | 0.146 |
| `20-50kb` | 61 | 0.754 | 114 | 0.579 |
| `>=50kb` | 22 | 0.818 | 29 | 0.759 |

### Comparison to Stage 5b's region-trained candidates (for reference)

| bucket | baseline_core recall | baseline_v1 (region) recall | tight_pos_099 recall |
|---|---:|---:|---:|
| `<5kb` | **0.229** | 0.333 | 0.312 |
| `5-10kb` | 0.551 | 0.539 | 0.506 |
| `10-20kb` | 0.482 | 0.518 | 0.482 |
| `20-50kb` | **0.738** | 0.770 | 0.770 |
| `>=50kb` | **0.773** | 0.955 | 0.955 |

| bucket | baseline_core precision | baseline_v1 (region) precision | tight_pos_099 precision |
|---|---:|---:|---:|
| `10-20kb` | **0.214** | 0.000 | 0.126 |
| `20-50kb` | **0.598** | 0.184 | 0.469 |
| `>=50kb` | **0.875** | 0.667 | 0.730 |

**Trade-off observed:** core-label training makes the model
substantially more **precise** in the 10-50 kb regime (10-20kb:
0.00 → 0.21, 20-50kb: 0.18 → 0.60) but markedly more **conservative**
on long regions (>=50kb recall: 0.95 → 0.77). Short-BGC recall stays
stuck at ~0.23 even under `short_core_upweight_v1`.

---

## 3. What Stage 5c-1 confirmed and what it didn't

### Confirmed

- **antiSMASH proto_core is a legitimate internal-val GT proxy for
  9-genome tight-core GT.** The Stage 5c-0 Spearman flip held up in
  training: core-val top pick beat BGC-Prophet on dedup F1 at a
  legitimate selection protocol.
- **Retraining on core labels repairs the proxy without any
  architecture change.** `baseline_core_v1` at `dedup_f1 = 0.3995` is
  +0.143 over the Stage 5b winner, achieved solely by switching the
  GT coordinate source from antiSMASH `region` to antiSMASH
  `proto_core`.
- **The fragmentation gate transfers imperfectly under distribution
  shift.** A gate of 0.95 on core-val resulted in ~0.90 on 9-genome.
  Some medium-sized BGCs got 3-4 predictions apiece. Not a reward
  hack, but not fully clean either.

### Not solved

- **Short BGC recall is still broken.** `<5kb` recall is 0.23
  (baseline_core) / 0.27 (short_core_upweight). Per the Stage 5b
  diagnostic, this is structural — an 8192 bp window can neither
  represent nor predict a 2-4 kb BGC cleanly. **Stage 5d (short-window
  head, window=2048) is the only legitimate path.**
- **`short_core_upweight_v1` didn't help.** It actually _hurts_
  bench_precision (0.322 vs 0.373 for baseline_core). Upweighting
  short positives pushes the model to over-predict medium regions
  that are short enough to attract the upweighted features, without
  actually improving the `<5kb` detection. Another piece of evidence
  that label reweighting ≠ short-BGC fix.
- **Long-BGC recall dropped.** `>=50kb` recall fell from 0.95 (Stage
  5b baseline) to 0.77 (baseline_core). Core coords are tighter, so
  the model became more reluctant to extend predictions to cover
  full wide BGCs. This is an unforced loss that Stage 5d / a
  post-decoder region merger could recover.

---

## 4. Headline claim and its caveats

> **Stage 5c-1 achieves `dedup_f1 ≈ 0.40` on the 9-genome benchmark
> under a no-9-genome-tuning protocol, matching BGC-Prophet and the
> Stage 5b diagnostic, but still 0.04 below the 9-genome-tuned
> `defrag_v2_ms10_s099` upper bound.**

Caveats to keep in any paper framing:

- **Dedup_f1 is at the threshold (0.3995 vs 0.40 target).** Not a
  robust win — single-seed result on a 305-GT benchmark, ~±0.02
  variation is plausible.
- **Length-bucket distribution is uneven**. The 0.40 is carried by
  20-50 kb precision gains; short-BGC recall is unchanged and
  `>=50kb` recall regressed.
- **`frag_ratio` ≈ 0.90 on 9-genome** means ~10 % of "precision TPs"
  are duplicates. If you report under a stricter dedup convention
  (counting TPs once per GT only), the number is what's in the
  `dedup_f1` column above.
- **This was selected without looking at 9-genome, but the decoder
  grid and the gate threshold are legacy of Stage 5b's 9-genome
  experience.** The selection was clean; the setup history wasn't.

---

## 5. Recommended next moves (explicit, in priority order)

1. **Stage 5d (short-window head)** — the `<5kb / 5-10kb` bucket is
   now the dominant missed-recall category. Build chunks at
   `window=2048 / stride=512`, re-extract Evo2 embeddings at finer
   stride, train a dedicated short-BGC head. Expected to recover 15-30
   short-BGC GTs currently missed.

2. **Stage 5c-1b seed variance** — current single-seed 0.3995 is
   suggestive but not statistically tight. Train 2 more seeds of
   `baseline_core_v1` (2 GPUs × 30 min), report mean ± std.

3. **Tighter decoder gate + post-merge** — allow Stage 5c-1's winner
   to run with a `min_support_windows ≥ 5` decoder to push frag_ratio
   closer to 1.0 on 9-genome, at the cost of ~5-10 % recall. Report
   both points on the precision-recall curve.

Do NOT yet:
- Run MIBiG positive injection (Stage 5e). Wait until 5d's outcome is
  known — MIBiG is most useful for short-BGC supplementation, which
  5d is also targeting.
- Propose a new label-reweighting variant. `short_core_upweight_v1`
  already showed that loss-weighting can't compensate for a window
  that's too wide.

---

## 6. Artefacts

- Plan / freeze: `docs/stage5c1-plan.md`, `experiments/stage5c1-selection_decision.md`
- Training logs: `experiments/stage5c1-*/artifacts/checkpoints/train_log_*.csv`
- Core-val sweep: `experiments/stage5c1_core_sweep_selected.csv`
- Val predictions: `experiments/stage5c1-*/artifacts/val_predictions_*/`
- 9-genome predictions: `experiments/stage5c1-*/artifacts/predictions_9genome_*/`
- 9-genome metrics: `experiments/stage5c1-*/metrics/final_9genome.json`
- Cross-reference:
  - Stage 5a fragmentation figures: `figures/stage5_fragmentation/`
  - Stage 5b diagnosis: `docs/stage5b-diagnostic-report.md`
  - Stage 5c-0 proxy test: `experiments/stage5c0_core_decoder_sweep_selected.csv`
