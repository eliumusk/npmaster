# Overnight Session Log — 2026-04-24

## Plan going in

Three steps with a short overnight budget:
1. Step 0 — unified eval harness + baseline table
2. Step 1 — zero-shot Evo2 log-likelihood as scorer
3. Step 2 — per-token dense prediction (extraction + training if time)

User constraints: 9g `dedup_F1` allowed to drop slightly but not much; OER must not lag antiSMASH too much (region count + AS-overlap). Use subagents heavily to save context.

## Environment

- node7 + node8: 8× A800-80G each, **distinct machines** (hostnames `qjdn8`, `9drh5`), 16 GPUs total.
- Evo2 env: **`/data/conda/miniconda/envs/evo2_bgc`** (torch 2.7 + flash-attn 2.8). The `bgc` env at `/root/miniconda3/envs/bgc` is missing flash-attn.
- Non-Evo2 work (LR training, eval) can stay in `bgc`.

## Step 0 — DONE (subagent)

Output: `experiments/eval_harness/baseline_table.{json,md}` + `README.md`.

Fills v4.1-hysteresis fully from `v4_1_final_readout.json`. v4.1-learned has only `dedup_F1=0.461` from the ledger; CIs and per-length buckets are TBD (rerun launched).

Gates codified for future candidate architectures:
- 9g `dedup_F1 ≥ 0.458` (mainline minus 0.03 tolerance)
- OER `n_regions ≥ 1400`
- OER AS-overlap ≥ 0.50 (beat v4.1-learned's 0.37)
- At least 6 of 7 types represented

## Step 1 — RESOLVED AS NULL (no rerun)

Audit of pre-existing `experiments/evo2_first_principles/9g_eval_ll_*` found **8 likelihood scoring variants already tested** (`zscore`, `zscore_pos`, `mean_neg`, `mean_pos`, `q50_zscore`, `q50_zscore_pos`, `std_neg`, `composite`) on 9g only:
- Best: `zscore_pos` → 9g dev `dedup_F1 = 0.1296`, vs v4.1 mainline 0.4878 — **3.8× worse**.
- `mean_neg`/`mean_pos` collapsed to F1 ≈ 0 (thresholding/sign artefact).
- No OER run. No strict_F1. No CI.
- **Scoring-function script is not in the repo or anywhere on server** — only the 5-col raw LL feature arrays survive at `9g_ll_w16384_s8192/*_ll.npy` (shape `(n_win, 5)`), geometry w=16384/s=8192 which doesn't match mainline.

Decision: do **not** re-run. Pure Evo2 log-likelihood as a standalone BGC scorer is decisively worse than LR-on-embeddings. Documented in `baseline_table.md`. Moving compute budget to Step 2.

Open loose end (not pursued tonight): LR trained on the 5-col raw LL features, or (LL ∪ embedding) fused features. Prior suggests marginal at best given the 2026-04-22 null results on score-level ensembles.

## Step 2 — IN PROGRESS

### Local code (written, compiled clean)

- `src/per_token/labels.py` — dense per-nucleotide label construction from antiSMASH coords.
- `src/per_token/model.py` — `PerTokenHead` (3× dilated Conv1D, 49K params @ D_in=128, hidden=64).
- `src/per_token/decode.py` — per-token probs → regions via per-nt track averaging + threshold + min_len + gap-fill.
- `scripts/train/per_token/train.py` — training entrypoint.
- `scripts/inference/per_token/infer.py` — apply trained head to per-token features.

### Server work (subagent running)

Designing `test_extraction.py` under `/data/muskliu/npmaster/experiments/step2_per_token/`. First sanity run on an 80-bp sample at `D_reduced=128` (fp16, random projection seed 58082) produced a `(80, 128)` tensor at ~4.5 K tokens/sec/GPU. Waiting for throughput + storage estimates on real 8192-bp windows before launching full extraction.

### Key architectural choices (locked in)

- **Random Johnson-Lindenstrauss projection** 4096 → 128-d inside the forward pass (no extra training; preserves pairwise distances). `seed=58082`.
- **fp16 per-token storage.** Single window: 8192 × 128 × 2 B = 2.0 MB per window. Full training pool 1.064 M windows ≈ 2.1 TB. This is the main budget worry.
  - Fallback if storage is unworkable: token downsampling (e.g. pool every 4 tokens → 512 MB / 4 = 530 GB). Label construction already supports this trivially.
- **Layer 20 truncation** kept (drops blocks 21–31 from Evo2 forward). Previously 34% speedup.
- **Per-nt labels** painted from `data/bgc_annotations.csv` into each training window.

### Step 2 design report — RECEIVED

Verified facts (1 window, A800, GPU 0):
- Evo2 byte-level: 1 token = 1 bp
- **Training FASTA records are w=16384 bp, not w=8192 as spec'd.** 1.064M windows. This is intentional (training extraction pre-existed at w=16384); inference uses w=8192/s=2048.
- Forward bf16 (bs=1, blocks 21–31 truncated): 1.30 s per 16k-token window → **~12,500 tok/s/GPU**.
- Projection `(h @ R)` and K-pool add negligible cost.

Storage at D=128 fp16:
- K=1: 4.75 TB — blown
- **K=8: 594 GB** — fits in /data/muskliu (1.7 TB free) — recommended and adopted.
- K=16: 297 GB — fallback.

GPU-hours: ~413 single-GPU → **~16–26 h on 16 GPUs** at bs=1–2.

### Decisions committed (user asleep)

- K=8 (1024 latent tokens per 8k window, 8 bp effective resolution; well below antiSMASH's ~1 kb granularity).
- N-padding kept in stream, `clean_len` recorded per window; downstream head masks.
- fp16 everywhere.
- Projection seed `0xE2E2`.
- Scope: **9g first** (finish in ~4 h), **training pool second** (won't finish overnight). OER deferred to tomorrow.

## What's still pending

- **Extraction launcher subagent running** (a4fcd13652d1f5185) — merges `extract_batch.py` batching with `test_extraction.py` projection/downsample, dispatches across 16 GPUs, validates 100-window dry run before full launch, reports back when first shards land.
- Per-token head training — depends on extraction shards.
- Full 9g + OER evaluation — depends on trained head.

## MATERIAL FINDING — v4.1-learned 9g drop is 5× bigger than documented

TBD-fill subagent computed **v4.1-learned 9g dedup_F1 = 0.3323 (CI95 0.224–0.437)**, n_pred=363, bench_P=0.262, bench_R=0.528, strict_F1=0.246. This is under the **canonical** `benchmark/nine_genome.eval_regions` metric — the same metric used to publish v4.1-hysteresis=0.488.

The "0.461" in `REGRESSION_LEDGER #16` and `docs/learned-decoder.md` comes from `full_pipeline_eval.py`'s greedy-pairing metric, a **different** F1 definition. Same predictions, different scorer. Internally consistent within learned-decoder.md's table (hysteresis=0.541, learned=0.461, Δ=-0.08) but NOT comparable to the published v4.1-hysteresis number.

**Apples-to-apples correction**:

| method | 9g dedup_F1 (bench mode) | Δ vs hysteresis |
|---|---:|---:|
| v4.1-hysteresis | 0.488 | — |
| v4.1-learned | **0.332** | **-0.156** |

This changes the strategic picture. The learned-decoder story "trade a tiny 0.03 F1 for 3× OER recall" is not true under the canonical metric. The real trade is **-0.156 on 9g for +3× OER AS-recall**. Whether it's worth it depends on how much 9g F1 the OER-preferring user is willing to give up — not a given.

Implications for Step 2:
- Per-token head's 9g pass/fail bar (0.458) is stricter than the current metagenomic mainline.
- Beating v4.1-learned on 9g is now easy (0.332 target) — the real goal is getting **close to hysteresis (0.488) while lifting OER AS-recall above 0.50**.

Action items (added to the morning follow-ups):
- Correct `REGRESSION_LEDGER.md #16` with metric footnote.
- Correct `docs/learned-decoder.md` results table with canonical metric + note the divergence.
- Consider deprecating `full_pipeline_eval.py`'s custom `dedup_metrics` — two metrics with the same name is the root cause of this confusion.

## Extraction launched (update 2026-04-24 ~01:25)

**9g**: live on 9 GPUs (node7 0–7, node8 0). Measured throughput **1.8 win/s/GPU** — 2.3× faster than design estimate. Shards flush every 1000 windows. ETA ~60 min total; completion ~02:20. Output: `/data/muskliu/npmaster/experiments/step2_per_token/9g_w8192_s2048/`.

**Training pool**: LAUNCHED.

- Validation: 10/10 edge-padded windows byte-exact between `--source fasta` and `--source sliced` (`max_abs = 0.0`, `np.allclose` True). Validation artifacts: `experiments/step2_per_token/validate_sliced_vs_fasta/VALIDATION_PASS`. Sliced path = single source-of-truth for all 1.064M windows.
- Sharding: `md5(window_id) mod 16` → 16 shards of ~66.5 K windows each.
- **Immediate workers** (launched ~01:40): node8 GPU 1–7 running shards 09–15. Measured throughput **0.87 win/s/GPU** at w=16384 (matches design).
- **Deferred workers**: 9 shards (00–08) queued via watcher daemon `/tmp/npmaster_watchers/watcher.sh` (PID 11522 on node7) that polls each 9g genome's manifest and auto-starts the matching training shard the moment a 9g worker frees its GPU. node7 GPU 0–7 → shards 00–07, node8 GPU 0 → shard 08.
- RUN_LOG auto-snapshot every 10 min via `tee`-style appender (PID 11695 on node7).
- First training shard lands ~01:59.

Projected progress by morning (09:00):
- 9g extraction: 100% done ~02:20.
- Training extraction: ~34% done (~360K of 1.064M windows). Enough for a first per-token head training run.

## Morning action plan (for when user wakes)

1. Run `bash experiments/eval_harness/STATUS_REPORT.sh` to see live GPU + shard state.
2. Check `/data/muskliu/npmaster/experiments/step2_per_token/RUN_LOG.md` (server-side) for the 10-min throughput snapshots.
3. If training extraction ≥ 200K shards done: spin up one training run on 1 GPU using `scripts/train/per_token/train.py` with `--subsample-train 200000` on the completed shards. Short smoke (3 epochs) first, then full 10 epochs. Should take <1h.
4. If 9g extraction done: run `scripts/inference/per_token/infer.py` with the best ckpt, then `scripts/evaluate/per_token_readout.py` to get the 9g row for the unified table. Compare against the gates in `baseline_table.md`.
5. Once OER question comes up: launch OER extraction at `w=8192/s=2048` on `/data/muskliu/oer_bgc_scan/filtered/*.fa` (the ≥10 kb filtered metagenomic set that antiSMASH also saw) using the same `production_extract.py`.

If anything is misbehaving, subagent IDs for the night: `a11fefee3e7c0b2a6` (design), `a4fcd13652d1f5185` (initial launch), `aaa37ec632c4484c9` (validation+training launch). They can be resumed via SendMessage if needed.

## Realistic morning outcome

- 9g extraction: **DONE** (~4h).
- Training extraction: **30–50% done** (24h total).
- Per-token head: can train on partial training data once first 100K shards ready; evaluation on 9g immediately after.
- OER inference: next day (needs separate extraction pass).
