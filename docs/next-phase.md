# Next Phase: Training-Side Precision

## Problem Statement

After `val_frozen_fused` and `tuned_v2`, the 9-genome `bench` F1 is ~0.26 formal / ~0.34 diagnostic. BGC-Prophet sits at F1 ≈ 0.39 on the same metric. The gap is almost entirely on the **precision** side (we lead on recall on all but BGC-Prophet, and we're not far behind there).

Postprocess sweep is done (v1→v2 gain was only ΔF1 ≈ 0.004). The window-level score study shows:

- Internal val negatives are well separated (AUROC 0.98+).
- 9-genome negatives have a saturated tail — ~12% of "negative" windows score ≥ 0.95.
- Of 90 high-score 9-genome negatives, 89 are also called by at least one external tool (antiSMASH / DeepBGC / BGC-Prophet).

That last bullet is the headline: most of our precision loss on this benchmark is not model-random-error, it's **benchmark-GT narrowness**. The model is finding BGC-like regions the benchmark GT doesn't list.

## 2026-04-19 Fragmentation Audit

The Stage 3 hard-negative result and the Stage 3b strict-decoder sweep exposed a second issue: raw `bench` precision can be inflated by splitting one GT BGC into several short predictions.

Same dev split (`GCA_000154945.1`, `GCA_000156475.1`, `GCA_000158875.1`, `GCA_000158915.1`, `GCA_000568915.1`):

| Candidate | R | raw P | raw F1 | n_pred | raw TPp | uniqGT TPp | dupTP | max/GT | dedup P | dedup F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BGC-Prophet dev | 0.3353 | 0.4878 | 0.3974 | 82 | 40 | 39 | 1 | 2 | 0.4756 | 0.3933 |
| Stage2 baseline | 0.5471 | 0.2703 | 0.3618 | 259 | 70 | 45 | 25 | 6 | 0.1737 | 0.2637 |
| Stage2b best | 0.5353 | 0.2834 | 0.3706 | 247 | 70 | 45 | 25 | 6 | 0.1822 | 0.2718 |
| Stage3 default | 0.5412 | 0.2824 | 0.3711 | 255 | 72 | 46 | 26 | 6 | 0.1804 | 0.2706 |
| Stage3 raw-P top | 0.4353 | 0.4390 | 0.4372 | 451 | 198 | 63 | 135 | 19 | 0.1397 | 0.2115 |
| Stage3 raw-F1 top | 0.5059 | 0.4027 | 0.4485 | 293 | 118 | 51 | 67 | 11 | 0.1741 | 0.2590 |

Reading:

- BGC-Prophet's precision is stable after dedup (`0.4878 -> 0.4756`), so its precision is not mostly fragmentation.
- Stage3 default only adds one unique GT precision hit over Stage2b and does not improve `dedup_f1`.
- Strict Stage3b decoder points raise raw precision/F1 but lower `dedup_f1`; those are diagnostic upper bounds, not deployable operating points.

Decision: do not promote Stage3/Stage3b based on raw precision alone. Any next sweep must rank by `dedup_f1` / fragmentation diagnostics in addition to raw `bench`.

## 2026-04-19 De-fragmentation Sweep (Stage 3b v2)

Follow-up to the audit. Infrastructure + reproducible sweep + pre-registered
lockbox candidates. **Status: dev-only; not yet validated on lockbox.**

### Infrastructure changes

1. `scripts/run_precision_constrained_decoder_sweep.py` now emits `dedup_f1`, `dedup_precision`, `unique_gt_precision_tp`, `duplicate_precision_tp`, `max_preds_per_gt`, `median_pred_len`, `p75_pred_len`, and `max_gap` into `summary.csv`. Primary ranking is by `dedup_f1` (was `bench f1`). CSV sort now honors `ranking.min_recall` / `ranking.max_n_pred` gates: eligible trials appear first (dedup_f1-best), ineligible ones follow with a `False` in `meets_recall_floor` / `meets_n_pred_cap`. `resolve_params` now prefers `decoder_grid` values over `decoder_fixed` (previously `max_gap` grid values were silently dropped).
2. `scripts/evaluate_nine_genome.py` reports a top-level `pred_length_stats` block (min / mean / median / p25 / p75 / max over all predicted region lengths).
3. `configs/sweeps/stage3b_defrag_v1.yaml` is the first-pass config (only scans `min_support_windows` ∈ {1, 2}) and is **superseded** by `stage3b_defrag_v2.yaml` (scans `min_support_windows` ∈ {5..11} and `start_threshold` ∈ {0.99, 0.995, 0.998, 0.999}). The v2 artifact at `experiments/stage3b_defrag_v2/` is the reproducible source for the numbers in this section.

### Window geometry (easy to get wrong)

Stage 3 remote prediction CSVs use `window=8192, stride=2048` — not stride 8192. A single window below `extend_threshold` closes a region (see `scripts/postprocess_v2.py::decode_regions_hysteresis`), so the knob that controls within-BGC splits is `min_support_windows`, not `max_gap`. At stride=2048, `min_support_windows=K` forces a region to span at least `(K-1)*2048 + 8192` bp:

| ms | min span (bp) |
|---:|---:|
| 6 | 18,432 |
| 8 | 22,528 |
| 10 | 26,624 |
| 12 | 30,720 |

These sizes sit in the middle of the typical 20–100 kb BGC range, so raising ms is a biological-size prior, not a length-cap hack.

### Fine-grained min-support sweep

On Stage 3 hardneg predictions, dev split, `fused / start=0.99 / extend=0.95 / max_gap=0 / length_cap=None`. Source: `experiments/stage3b_defrag_v2/summary.csv` for the start=0.99 slice.

| ms | R | bench P | bench F1 | loose F1 | strict F1 | dedup P | dedup F1 | n_pred | uniqGT | median len |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.553 | 0.237 | 0.331 | 0.530 | 0.236 | 0.237 | 0.331 | 169 | 40 | 24,576 |
| 6 | 0.553 | 0.248 | 0.343 | 0.543 | 0.242 | 0.248 | 0.343 | 161 | 40 | 24,576 |
| 7 | 0.506 | 0.305 | 0.380 | 0.553 | 0.262 | 0.305 | 0.380 | 128 | 39 | 26,624 |
| 8 | 0.476 | 0.370 | 0.417 | **0.577** | 0.274 | 0.370 | 0.417 | 100 | 37 | 30,720 |
| 9 | 0.424 | 0.419 | 0.421 | 0.540 | 0.281 | 0.419 | 0.421 | 86 | 36 | 33,792 |
| 10 | 0.388 | 0.473 | 0.426 | 0.519 | 0.287 | 0.473 | **0.426** | 74 | 35 | 36,864 |
| 11 | 0.329 | 0.576 | 0.419 | 0.469 | 0.297 | 0.576 | 0.419 | 59 | 34 | 43,008 |

BGC-Prophet dev for head-to-head: bench R/P/F1 = 0.335 / **0.488** / **0.397**; dedup_P / dedup_F1 = 0.476 / **0.393**; loose F1 **0.438**; strict F1 **0.302**.

### Reading, carefully

- `dedup_P == bench_P` at ms ≥ 5 (max/GT=1 per genome). The ms knob is not a fragmentation hack in the length_cap sense: each emitted region is long (pred_len ≥ 26.6 kb at ms=10), so the short-core trick where tiny predictions automatically satisfy `overlap / pred_len ≥ 0.5` is not available. The bench precision TP condition is `overlap ≥ pred_len / 2 = 13.3 kb` at ms=10, which in turn requires `gt_len ≥ 13.3 kb` (not `gt_len ≥ 26.6 kb`) because `overlap ≤ min(pred_len, gt_len)`. So ms=10 is a biological-size prior on predictions, not on GTs.
- **At ms=10 / start=0.99, NP-Master wins dev bench F1 (0.426 vs 0.397), loose F1 (0.519 vs 0.438), dedup_F1 (0.426 vs 0.393)**, but its bench_P (0.473) is slightly below BGC-Prophet's (0.488). So "beats BGC-Prophet" is accurate on F1-style metrics but not on raw precision at this operating point.
- **At ms=10 / start=0.995 (and start=0.998; identical output above 0.995 because the score tail is empty in between) NP-Master has R=0.371, bench_P=0.500, bench_F1=0.426, dedup_F1=0.426, n_pred=70.** This point simultaneously exceeds BGC-Prophet on recall (0.371 vs 0.335), bench_P (0.500 vs 0.488), and dedup_P (0.500 vs 0.476). It is the cleanest head-to-head candidate.
- At ms=8, loose F1 peaks at **0.577** (BGC-Prophet 0.438) — the best "real coverage" operating point, at the cost of some bench precision.
- BGC-Prophet still wins strict / iou F1 (0.302 vs 0.287) because NP-Master's predictions (median 36.9 kb at ms=10) overshoot GT boundaries (BGC-Prophet median 21.4 kb).
- Per-genome at ms=10 / start=0.99, NP-Master beats BGC-Prophet on bench F1 on 4/5 dev genomes. `GCA_000154945.1` collapses to R=0.029 at every decoder setting — this is a model-domain miss, not a decoder issue, and survives all decoder settings including length_cap variants.

### Pre-registered lockbox candidates

Do **not** promote these to `configs/operating_points.yaml` yet. They were tuned on dev and need the 4-genome lockbox set to validate. To prevent lockbox-tuning, the two candidates below are frozen now; **no further parameter search is allowed on lockbox**.

| candidate | score | start | extend | max_gap | length_cap | ms | dev R | dev bench_P | dev bench_F1 | dev dedup_F1 | rationale |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `defrag_v2_ms10_s099` | fused | 0.99 | 0.95 | 0 | null | 10 | 0.388 | 0.473 | 0.426 | 0.426 | peak dev bench/dedup_F1; higher recall than BGC-Prophet |
| `defrag_v2_ms10_s0995` | fused | 0.995 | 0.95 | 0 | null | 10 | 0.371 | 0.500 | 0.426 | 0.426 | highest dev bench_P ≥ BGC-Prophet's 0.488; tied F1 |

If lockbox dedup_F1 drops more than ~0.03 on both candidates, that is a tuning-on-dev overfit signal and `defrag_v2_ms8_*` (loose-F1 candidate) should be tried as a separate pre-registered third point before calling the decoder direction done.

### Next actions

1. Re-run Stage 3 inference for the 4 lockbox genomes (`GCA_000156435.1`, `GCA_000158815.1`, `GCA_000158895.1`, `GCA_000568255.1`) on the server, same checkpoint `best_route_b_s0`, same `window=8192 / stride=2048`.
2. Apply both pre-registered candidates to lockbox via `evaluate_nine_genome.py`. Report head-to-head with BGC-Prophet lockbox.
3. Only after lockbox passes, promote the winning candidate to `configs/operating_points.yaml` under a new name (e.g. `defrag_v2_ms10_s099`). Keep `val_frozen_fused` as the generalization-safe reference.
4. Keep Route 1 (hard-negative mining) and Route 2 (data grid) going in parallel — the `GCA_000154945.1` collapse is not a decoder-fixable problem.

## Route 1: Hard-Negative Mining (active — `route-b-hardneg-v1.md`)

Mine top-K high-scoring windows from training negatives, weight their loss, retrain Route B. Minimal intervention:

- mining: `scripts/mine_hard_negatives.py` (formerly `mine_route_b_hardneg.py`)
- training: `src/train.py` + `src/data/chunk_dataset.py` accept the mined flag
- decision rule for 3-seed promotion: see `route-b-hardneg-v1.md::Decision Rule`

## Route 2: Data-Side Grid (after Route 1)

Only start if Route 1 shows movement. Knobs to sweep:

- **Window length** — current 16384 bp. Try 8192 / 32768.
- **Stride** — current 8192. Try 4096 / full-overlap.
- **Negative sampling ratio** — current train mix. Try reducing negatives or re-weighting by phylum.
- **Edge padding** — keep `edge_padding_filtered` windows; test phase-aware padding variants.

Evaluation protocol frozen (see `evaluation-protocol.md`). No decoder re-tuning — if a data change doesn't move `bench F1` under the frozen operating point, it's not real signal.

## Route 3: Benchmark GT Re-examination

Low priority but cheap. Take the 89 / 90 supported-by-other-tools high-score negatives from the score-family diagnosis. Two candidate outcomes:

- They're genuinely BGC-like → reclassify → we're underestimating our own precision.
- They're not → harvest as hard-negatives for Route 1.

Either way this feeds back into Route 1.

## What We Are NOT Doing Yet

Skip until Routes 1–2 are played out:

- CNN depth / dilation changes
- Chunk length changes
- Different Evo2 layer / different backbone
- Loss function redesign (focal, contrastive, etc.)

These are model-architecture changes. The data evidence says the bottleneck isn't there yet.
