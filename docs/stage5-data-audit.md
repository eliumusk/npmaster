# Stage 5 Data Audit (2026-04-20)

Pre-implementation audit of training labels, benchmark provenance, and
architecture — the findings here supersede assumptions in
`docs/stage5-no-nine-genome-tuning.md` where they conflict. Each section
ends with concrete decisions that affect Stage 5 task design.

---

## 1. BGC annotation source

`/data/muskliu/npmaster/data/bgc_annotations.csv` (19 762 rows, 1 965 genomes):

| column | meaning |
|---|---|
| `bgc_id` | synthesized as `{genome_id}_BGC_{N:03d}` |
| `genome_id`, `contig_id`, `start_bp`, `end_bp`, `bgc_length` | coordinates |
| `bgc_types`, `bgc_product` | free-text antiSMASH types / product names |
| `is_truncated` | 31.7 % of regions touch a contig edge |
| `phylum` | GTDB phylum |

### Provenance evidence

- `bgc_id` pattern is `GCA_XXX_BGC_001` (assembly-local counter), **not**
  `BGC0000021` (MIBiG style).
- Zero rows start with `BGC` (MIBiG prefix) → **no MIBiG clusters are in
  this table**.
- Zero rows for any of the 9-genome IDs → training pool is cleanly
  disjoint from the benchmark (embargo is already implicit in the data
  itself).
- Raw antiSMASH outputs live under
  `/data/muskliu/npmaster/data/antismash_results/{GCA_*}/`.

### Length distribution of the annotation table

```
count  19762   mean  29 918 bp    std  22 217 bp
min     1 004   25 %  18 692 bp   50 %  22 214 bp   75 %  41 041 bp
 <5 kb:    511 (2.6 %)        <10 kb: 1 223 (6.2 %)        <20 kb: 5 163 (26 %)
```

**Decision → the training pool is pure antiSMASH. The Stage 5 plan's
`mibig_boost_v1` variant cannot be built from current data.** Any
"MIBiG-tight-core" intervention requires first constructing MIBiG
positives from scratch (see §5).

---

## 2. Training window labels — the real positive rule

`/data/muskliu/npmaster/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv`
(1 064 568 rows):

```
split       train 626 449   val 95 251   test_id 180 566   test_ood 162 302
label       negative 933 236  positive 68 347  ambiguous 62 985
hard_neg    0 → 1 056 264     1 → 8 304   (mined in Stage 3)
```

### Per-label `max_bgc_overlap_ratio`

| label | min | 1 % | 5 % | 10 % | 25 % | 50 % | 75 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `positive` | 0.061 | 0.397 | 0.541 | 0.595 | 0.746 | 1.000 | 1.000 |
| `ambiguous` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.252 |
| `negative` | 0.000 | — | — | — | — | 0.000 | 0.000 |

Reverse-engineered rule (overlap with any antiSMASH region):

- **positive** iff overlap ≥ ~0.5 (mostly, with a small tail down to 0.06
  likely from rounding or a soft cutoff)
- **ambiguous** iff 0 < overlap < 0.5, or overlap = 0 but
  `nearest_bgc_distance < 10 kb`
- **negative** iff overlap = 0 AND distance to nearest BGC ≥ 10 kb

### Why this produces wide predictions

A typical antiSMASH region (~25 kb) is covered by ~8–10 fully-positive
windows (w = 8 192, s = 2 048). At the edges, any window with ≥ 50 %
overlap is also positive — that is, a window whose centre sits ~4 kb
outside the antiSMASH region still counts as positive. So the model is
literally being trained to fire **~4 kb past every antiSMASH edge**,
which itself is already a wider annotation than 9-genome's hand-curated
tight cores.

**Short-BGC blind spot**: only 511 BGCs in the annotation table are
< 5 kb, and they are antiSMASH-style (not tight cores). The plan's
`short_bgc_boost_v1` would reweight these, but the labels themselves
are still wide — so boosting loss cannot manufacture tight boundaries.

### Internal validation scale

`split == val` → 169 distinct genomes, 95 251 windows. Statistical power
for a decoder-param selection is adequate (CI on `dedup_f1` roughly
±0.01–0.02 at n ≈ 8 k positives + 70 k negatives).

**Decision** → we can and must **rebuild labels with a stricter positive
threshold** on the existing CSV. No new data required.

---

## 3. 9-genome benchmark provenance & prior decoder tuning (soft-leak audit)

### Benchmark

- **Shared external benchmark** used by BGC-Prophet and DeepBGC papers.
- Ground truth at `/data/syh/benchmark/nine_genome/ground_truth.json`,
  **human-curated**, tight-core boundary style.
- Length distribution (from Stage 5 plan): 48/305 < 5 kb, 137/305 < 10 kb.
- This is the **only human-curated BGC benchmark in the project**;
  `phase4/mibig_eval/mibig_matches.json` is only a species-level
  cross-match between MIBiG and our test genomes (405 matches of 2 502
  clusters), no BGC coordinates inside our training genomes. It is not
  a usable alternative.

### Prior 9-genome decoder tuning — audit verdict

Commit `c181a9b` + `experiments/nine_genome/stage3b_defrag_v2/summary.csv`:

- **168 decoder configurations** evaluated on the 9-genome `dev` split
  (5 genomes).
- Grid = `start_threshold (4) × min_support_windows (7) × extend (2) × length_cap (3)`.
- Selection criterion: peak 9-genome dev `bench_F1`.
- `defrag_v2_ms10_s099` picked as the dev-peak, then **passed** the
  4-genome lockbox (F1 0.453).

**Classification** → MODERATE-TO-SEVERE soft leak. The lockbox was
honoured, but the `ms = 10 / start = 0.99` choice itself is a
9-genome-dev-optimised point, not an internal-val selection. Stage 5's
demand for a no-9-genome-tuned candidate is a strictly stronger
protocol than anything done so far.

**Decision** → we must report Stage 5 candidates under a selection
protocol that never inspects 9-genome dev or lockbox. `defrag_v2_ms10_s099`
remains a useful 9-genome-informed upper bound only.

---

## 4. Architecture — short-window head is cheap

From `scripts/inference/stride.py:116`, `src/model/context_full_model.py`:

- Evo2 (frozen, `blocks.20.mlp.l3`, 4 096-dim) is called **once per
  window**, then **mean-pooled** over tokens to a single 4 096-d vector.
  Window / stride are CLI args, not architectural.
- Head: `Projection (4096 → 2048 → 256) → Dilated CNN (dil = [1,2,4],
  receptive field 15 windows ≈ 120 kb at w = 8192) → GatedResidual
  (α ≈ 0.018 at init) → Classifier (256 → 8)`.
- Layered LRs: `lr_proj = 1e-4`, `lr_context = lr_classifier = 1e-3`.

### Short-window feasibility

Because Evo2 is mean-pooled per window, running inference at
`window = 2048` or `4096` is not an architecture change — it is the
same model fed narrower windows. Retraining on a new chunk set (via
`scripts/train/build_chunks.py`) is required, but Evo2 extraction and the
downstream CNN are unchanged. **Cost is proportional only to total
windows, not a new embedding pass.** This matches the user's intuition.

**Decision** → short-window head is viable as a Stage 5 track. Gate it
behind seeing whether cheap label-tightening (§5) already closes the
< 5 kb recall gap.

---

## 5. Revised Stage 5 intervention ladder

Replaces the §2.2 variant table in the original Stage 5 plan. Each rung
is more expensive than the last — **do not start rung N + 1 until
rung N has been run and evaluated internally.**

### Rung 0 — control

| id | intervention | new data? |
|---|---|---|
| `baseline_v1` | retrain from Stage 3 hard-neg warm-start with current labels | none |

Needed as the honest no-tuning-on-9-genome baseline (Stage 5A).

### Rung 1 — label tightening (cheap, no new data)

All three are pure relabels of `windows_metadata_hardneg.csv`; they
change only the `label` column + a `stage5_weight` column:

| id | positive rule | ambiguous rule | negative rule | what it tests |
|---|---|---|---|---|
| `tight_pos_075_v1` | overlap ≥ 0.75 | 0.1 ≤ overlap < 0.75 | overlap = 0 ∧ dist ≥ 10 kb | shrinks effective "fire zone" by ~25 % |
| `tight_pos_090_v1` | overlap ≥ 0.90 | 0.1 ≤ overlap < 0.90 | same | aggressive tightening |
| `trim_bgc_5kb_v1` | overlap ≥ 0.5 **against BGC shrunk by 5 kb each side** | likewise | same | simulates "core only" without needing core annotations |

### Rung 2 — label rebalancing (cheap, no new data)

| id | intervention |
|---|---|
| `short_bgc_upweight_v1` | upweight (×4) training loss for positives whose source BGC is < 10 kb |
| `truncated_exclude_v1` | drop `overlaps_truncated_bgc == 1` positives (14 321 windows) from training — their boundaries are unreliable by construction |
| `hardpos_hardneg_v2` | combine upweighted short + drop truncated + current hard-neg set |

### Rung 3 — multi-scale training (medium cost)

| id | intervention |
|---|---|
| `short_window_head_v1` | retrain with `window = 4096, stride = 1024`; needs new `build_chunks` pass and Evo2 re-extraction at finer stride |

### Rung 4 — MIBiG positives (highest cost)

| id | intervention |
|---|---|
| `mibig_positives_v1` | scrape MIBiG 4.0 FASTA + metadata, add MIBiG clusters as tight-core positives either as standalone training examples or mapped onto training genomes via homology |

Left as a Stage 5D follow-up — unblocked only if rungs 1–3 fail to close
the gap.

---

## 6. Internal selection substrate

Because no curated internal val exists, Stage 5 selection must use
the antiSMASH-labelled `val` split (169 genomes, 95 k windows) — but
with an important caveat: **if we train on tight labels, we should
also evaluate on tight labels on `val`, so the selection metric rewards
tight-boundary behaviour**.

Selection policy for rung-1 and rung-2 candidates:

1. Primary metric: `dedup_f1` on `val` **under the same label
   threshold used for training** (tight-vs-tight comparison).
2. Tie-breaker: `dedup_f1` on `test_ood` under the same threshold
   (stronger generalisation signal).
3. Additional audit metric (not used for selection): 9-genome
   `bench_f1` at the frozen `val_frozen_fused` operating point for the
   `baseline_v1` trial **only**, to anchor the leaderboard. Do not
   evaluate any other rung on 9-genome before its internal selection
   freezes.

`val_audit` is the untouched `test_id` split — used only once per
candidate family, after freezing, to check for val-split overfit.

---

## 7. Summary of implementation impact

- `configs/sweeps/stage5_no9_data_v1.yaml` needs to replace its variants
  with the rung ladder above.
- `scripts/build_stage5_labels.py` must handle: relabel by new overlap
  threshold, optionally shrink source BGCs by a configurable margin,
  emit `stage5_label` / `stage5_weight` / `stage5_variant` columns.
- `scripts/evaluate_length_bucket_metrics.py` stays as planned.
- `tests/test_stage5_label_builder.py` fixtures should use the real
  column set (`label`, `max_bgc_overlap_ratio`, `nearest_bgc_distance`,
  `overlaps_truncated_bgc`, `bgc_length`-joined) rather than a toy
  schema.
- Stage 5-0 proxy-validity check: score all 168 historical
  `stage3b_defrag_v2` decoder configs on the internal `val` split (by
  re-running inference over val predictions with varying decoder
  params, not by re-training). If `defrag_v2_ms10_s099` ranks in the
  top-3 by `val_dedup_f1` → selection proxy is viable. Otherwise Stage 5
  must target an alternative internal metric or invest in rung 4 (MIBiG)
  before the sweep.
