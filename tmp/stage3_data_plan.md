# Stage 3 Data Construction Sweep — Proposal v0

Date: 2026-04-18
Status: DRAFT for review (do not launch yet)
Spec reference: `docs/superpowers/specs/2026-04-18-nine-genome-sweep-design.md` §6.3, §7 Stage 3

## TL;DR

- Most of Stage 3's parameter axes already have working code wired in (hardneg mining, hardneg loss weighting, chunk overlap stride). Several do not exist yet (phylum-aware weighting, edge-padding-policy switch, ambiguous-window inclusion, dataset-mixing).
- The cheapest, highest-leverage axis is **hardneg weight + top-k**: requires only a re-mine + retrain, no rebuild of `chunk_index.pkl`. Start there.
- The most expensive axes (chunk construction, edge-padding) require rebuilding `meta_arrays.pkl` / `chunk_index.pkl` and possibly re-extracting embeddings. Hold those for a Stage 3b after Stage 3a narrows.
- Realistic budget: **8–12 retrains/week** on 16 A800 (single-seed dev runs). A full 6-axis joint grid is impossible; sequential narrow is the only honest approach.
- Biggest blocker right now: we have no recorded wall-clock for a single Route B retrain. The plan below assumes 6–10 h/run based on epoch budget alone — needs to be confirmed with one timed pilot before sweeping.

---

## 1. What's Already in the Repo

Cited file:line references — checked, not assumed.

### Hardneg pipeline (live)

- Mining script: `scripts/mine_hard_negatives.py:171-249` — CLI takes `--score-name {max_type, 1_minus_isneg, fused}`, `--top-k`, `--min-score`. Output: patched `meta_arrays.pkl` with new int32 column `hard_negative` plus `hardneg_candidates.csv` and `summary.json`.
- Selection logic: `scripts/mine_hard_negatives.py:27-41` — restricts to `(split == "train") & (label == "negative") & (scores >= min_score)`, then top-k by descending score.
- Dataset wiring: `src/data/chunk_dataset.py:39-42` reads `meta["hard_negative"]` (defaults to zeros if absent) and surfaces it as a per-position bool through `__getitem__`.
- Loss-side weighting: `src/train.py:30-47` — `apply_hard_negative_weights` multiplies flagged positions by `hard_neg_weight` before averaging. Default weight `1.0` (no-op). CLI flag `--hard-neg-weight` exists at `src/train.py:357`.
- Logging: `src/train.py:247` writes `hard_neg_frac` per epoch to `train_log_<tag>.csv`.

Concrete Stage 3 implication: the hardneg axes (definition / threshold / top-k / weighting) require **no new code** beyond a wrapper script that loops mine→train→eval and writes to the canonical experiment layout.

### Chunk construction (live but inflexible)

- Builder: `scripts/build_chunks.py:21-112` — `L_chunk` is configurable via CLI; stride is hardcoded `L_chunk // 2`, margin is hardcoded `L_chunk // 4`. Loss region is fixed center.
- Ambiguous-window policy: `scripts/build_chunks.py:55-58, 96-99` — `label == "ambiguous"` is silently masked out of `loss_mask`. There is **no flag** to include them.
- The only knob currently exposed is `--L-chunk`. Sweeping overlap or margin requires editing `build_chunks.py`.

### What does NOT exist yet (gaps for spec §6.3)

| Spec §6.3 axis | Code present? | Gap |
|---|---|---|
| hard-negative definition | partial | only score-family choice; no "hard from a specific tool" or "hard from cross-genome" definitions |
| hard-negative mining threshold and top-k | yes | CLI flags exist |
| hard-negative weighting | yes | scalar `--hard-neg-weight`; no per-class or annealed schedule |
| positive/negative sampling ratio | **no** | dataset is full-epoch traversal (`src/data/chunk_dataset.py:65-66`); no resampler / no WeightedRandomSampler in `src/train.py:71-74` |
| phylum-aware weighting or balancing | **no** | `phylum_to_code` is built (`chunk_dataset.py:44-46`) but never used in loss or sampler |
| ambiguous-window inclusion policy | **no** | hardcoded exclusion in `build_chunks.py` |
| chunk construction settings | partial | `L_chunk` exposed, others hardcoded |
| chunk overlap logic | **no** | stride and margin are hardcoded `L_chunk // 2` and `L_chunk // 4` |
| edge-padding policy for training data | **no** | embeddings already use `edge_padding/embeddings_merged.npy` (`configs/mainline.yaml:18`); the policy was baked in at embedding-extraction time and would require re-extraction to change |
| dataset mixing policy across sources | **no** | single-source pipeline; no source-tagging in `meta_arrays.pkl` schema seen |

### Existing operating point context

- Mainline checkpoint: `H_hardneg_r10_s{0,1,2}` already includes a **previous round** of hardneg work (warm-start in `configs/mainline.yaml:50`). Stage 3 hardneg trials are stacking on top of an already-mined baseline. This matters for hypothesis framing: we are not asking "does hardneg help vs none", we are asking "does fresh re-mining + heavier weighting help vs the frozen H_hardneg_r10".
- Current 9-genome dev metric (formal `val_frozen_fused`): R 0.5934, P 0.1649, F1 0.2580 (`docs/operating-points.md:17`). Target: BGC-Prophet R 0.3475 / P 0.4529 / F1 0.3933 (spec §1).
- Diagnosis (`docs/next-phase.md:10-14`): 89/90 sampled high-score "negatives" on 9-genome are also called by other tools — the precision gap is partly **benchmark-GT narrowness**, not pure model error. Hardneg mining will not fix label-incompleteness; it can only help if there really is a tail of trainable false-positives.

---

## 2. Per-Axis Proposal

For each spec §6.3 axis, one row gives values to try, retrain/rebuild requirement, rough cost, and the hypothesis under test. **All cost estimates assume 6–10 h/training-run on 1×A800; this needs a pilot to confirm (see §5).**

### Axis A: Hardneg score family + threshold

| Field | Value |
|---|---|
| Values | `score=fused @ min=0.95` (current), `score=fused @ min=0.99`, `score=max_type @ min=0.95`, `score=1_minus_isneg @ min=0.90` |
| Action | Re-run `mine_hard_negatives.py` per setting → patches `meta_arrays.pkl` only |
| Retrain? | yes, but mining is ~10 min; training is the cost |
| Cost | 4 mines × ~10 min + 4 retrains × ~8 h ≈ 32 GPU-hours |
| Hypothesis | Different score families surface different false-positive families; fused already balances type-confidence and is-negative-prob, but a stricter `min_score` may give cleaner gradients |

### Axis B: Hardneg top-k

| Field | Value |
|---|---|
| Values | `top_k = 1000`, `5000` (current), `15000`, `50000` |
| Action | Re-mine only |
| Retrain? | yes |
| Cost | 4 retrains × ~8 h ≈ 32 GPU-hours |
| Hypothesis | Current 5000 may be either too few (under-coverage) or too many (diluted by easy-negative rank-tail). The slope of dev F1 vs k tells us where the useful tail ends |

Risk flag: if the underlying hardneg pool is small (e.g. only ~12% of train negatives ever score ≥0.95 according to `docs/next-phase.md:11`), then `top_k = 50000` may simply pull in mostly-easy negatives and look identical to `top_k = 5000`. Verify pool size from current `summary.json` before committing this sub-sweep.

### Axis C: Hardneg loss weight

| Field | Value |
|---|---|
| Values | `hard_neg_weight = 1.0` (off, baseline), `2.0`, `4.0` (route-b-hardneg-v1 recommended), `8.0` |
| Action | training-only, no mining or rebuild |
| Retrain? | yes |
| Cost | 4 retrains × ~8 h ≈ 32 GPU-hours |
| Hypothesis | Spec §6.3 lists weighting as separate from definition. `route-b-hardneg-v1.md:78` already proposes weight=4 with a single-seed test. We have not run a multi-weight scan with the current mining output. This is the cheapest of A/B/C because mining is reused |

### Axis D: Positive/negative sampling ratio

| Field | Value |
|---|---|
| Values | full-epoch (current), `1:5`, `1:2`, `1:1` weighted-resampler |
| Action | requires new code: `WeightedRandomSampler` wired into `DataLoader` at `src/train.py:71` and a per-chunk weight derived from `n_positive` |
| Retrain? | yes |
| Cost | 1–2 days dev time + 4 retrains × ~8 h ≈ 32 GPU-hours |
| Hypothesis | Current chunk traversal is whatever the data distribution gives you. If train is heavily negative-dominated, the model has limited per-update positive signal. Resampling could change effective gradient direction without changing the loss itself |

Caveat: we have a hardneg-weighting mechanism already, which partially overlaps with this. Doing both at once is confounded. Probably wait until A/B/C are settled.

### Axis E: Phylum-aware weighting

| Field | Value |
|---|---|
| Values | uniform (current), inverse-phylum-frequency, capped inverse-phylum-frequency (cap=5x) |
| Action | requires new code: weight per chunk by phylum of center window or majority phylum |
| Retrain? | yes |
| Cost | 1 day dev + 3 retrains × ~8 h ≈ 24 GPU-hours |
| Hypothesis | If 9-genome under-represents phyla that are well-represented in training, current model is over-fit to majority-phylum signal patterns and underperforms on minority phyla in 9-genome dev |

Caveat: we don't have evidence yet that 9-genome dev failures correlate with phylum. Worth a quick diagnostic on existing predictions BEFORE running any phylum-weighted retrain.

### Axis F: Chunk construction + overlap + edge handling

| Field | Value |
|---|---|
| Values | `L_chunk=32, stride=16` (current), `L_chunk=32, stride=8` (4x overlap), `L_chunk=64, stride=32`, `L_chunk=32, stride=16, include_ambiguous=True` |
| Action | edit `build_chunks.py` to expose `--stride`, `--margin`, `--include-ambiguous`; rebuild `chunk_index.pkl` per setting |
| Retrain? | yes; **and** new `chunk_index.pkl` (~min) plus possibly new `meta_arrays.pkl` for ambiguous-included variant |
| Cost | 1 day dev + 4 chunk rebuilds + 4 retrains × ~8 h ≈ 32–40 GPU-hours |
| Hypothesis | More overlap means smoother boundary supervision; ambiguous-as-positive may close the precision gap if many "ambiguous" windows are real BGC fringes |

Edge-padding for training data is **not in this axis** — embeddings were already extracted under one edge-padding policy. Switching policy would require re-extracting embeddings (Evo2 7B inference on the entire training corpus = much more expensive than retraining, on the order of days per policy). Recommend: explicitly punt this sub-axis to Stage 5 or treat as out-of-scope for graduation timeline.

### Summary table (axes ordered by cost-effectiveness, lowest dev-effort first)

| Axis | Code ready? | GPU-hr estimate | Confound risk | Recommended order |
|---|---|---:|---|---:|
| C — hardneg weight | yes | 32 | low | 1 |
| B — hardneg top-k | yes | 32 | medium (pool size) | 2 |
| A — hardneg score family | yes | 32 | low | 3 |
| F — chunk overlap | no (1 day dev) | 32–40 | medium | 4 |
| D — pos/neg sampling | no (1–2 day dev) | 32 | high vs C | 5 |
| E — phylum weighting | no (1 day dev) | 24 | needs diagnostic first | 6 |

---

## 3. Proposed Sub-Sweep Order

A strict joint grid (4×4×4×4×3×3 = 2304 trials) is obviously impossible. Sequential narrow.

### Stage 3a — hardneg parameter narrow (1–1.5 weeks)

1. **Pilot (1 trial)**: re-run mining with current settings + retrain with `hard_neg_weight=4.0`. Time it. Confirm GPU-hours/run estimate and that the manifest+experiment-layout machinery works end-to-end for a training run, not just inference.
2. **Axis C scan (4 trials)**: weight ∈ {1, 2, 4, 8}, mining frozen at current.
3. Pick best weight w*; **Axis B scan (3 new trials)** at w*: top_k ∈ {1000, 15000, 50000} (5000 is already in step 1).
4. Pick best (w*, k*); **Axis A scan (3 new trials)**: score family or stricter min_score variants.

End of 3a: ~11 trials, ~88 GPU-hours, one promoted (mining_config*, w*) pair.

### Stage 3b — chunk construction narrow (1 week, only if 3a moved dev metric)

5. **Axis F scan (3 trials)**: rebuild chunk_index for two new (L_chunk, stride) settings + ambiguous-included; retrain with promoted Stage-3a hardneg config.

### Stage 3c — confound check (1 week)

6. **Axis E** if dev failure cases show phylum bias. Otherwise skip.
7. **Axis D** only if Stage-3a hardneg weighting did NOT move dev — then sampling-ratio is a different lever on the same problem.

### Stage 3 exit gate

Promote one or two configs to Stage 4 (training-hyperparam sweep) only if:
- dev recall ≥ current 0.5934 (no recall regression vs frozen mainline), AND
- dev precision improves by ≥0.02 absolute (i.e. > 0.18) under `val_frozen_fused`.

If neither lever moves the metric, the next move is to re-examine whether the precision gap really is data-side at all (cf. `docs/next-phase.md:14` — benchmark-GT narrowness hypothesis).

---

## 4. Blockers Before Stage 3a Trial 1

Concrete list. Each item is "yes build / push back".

| # | Blocker | Action needed | Estimate |
|---|---|---|---|
| 1 | **No timed Route B retrain on record** — we don't actually know GPU-hours/run | Run one pilot training (single seed, current config, no hardneg weight change) and time it. Use the experiment-layout machinery for real | 1 trial, 6–10 h wall |
| 2 | **No Stage 3 sweep runner** | New script `scripts/run_stage3_data_sweep.py` analogous to stage1/stage2 runners. Must (a) call `mine_hard_negatives.py` per trial when needed, (b) call `src/train.py` with chosen flags, (c) call `evaluate_nine_genome.py` on dev split, (d) write canonical experiment layout. Reuse `src/experiments/manifest.py`, `layout.py` | 1 day dev |
| 3 | **No Stage 3 sweep config schema** | New `configs/sweeps/stage3_data_v1.yaml` listing trial-as-row with `(score_name, min_score, top_k, hard_neg_weight)` plus reference to base mainline checkpoint | 0.5 day dev |
| 4 | **Mining inputs verified on server** | Confirm `embeddings_path`, `meta_arrays_path`, `contig_index_path` from `configs/mainline.yaml` are present on node7/node8 and readable by `mine_hard_negatives.py`. Spec mandates immutable inputs (§13) | 30 min check |
| 5 | **Dev/lockbox split awareness during training-eval loop** | `src/train.py` evaluates on internal `val`, not on 9-genome dev. We need a separate post-train `evaluate_nine_genome.py --genome-ids <dev-only>` invocation per trial. Already supported in stage1/2 runners; just wire it into stage3 runner | included in #2 |
| 6 | **Hardneg pool size diagnostic** | One-off: print current count of train-negative windows scoring ≥0.95. If < 20k, then top_k=50000 is meaningless. Influences Axis B values | 30 min |
| 7 | **Decide whether to use seed=0 only or 3-seed for promoted configs** | Spec §7 Stage 6 says multi-seed only at promotion. Stage 3 should be seed=0 only to save compute. State this explicitly in the sweep config | decision only |

Out of scope (push back if requested): re-extracting embeddings under a new edge-padding policy. Cost on the order of days per policy and not justified by current evidence.

---

## 5. Compute Budget — 16× A800 for 1 Week

Assumptions (please correct if you have actual numbers):

- 1 Route B retrain ≈ **8 h** wall on 1 A800. This is a guess from "100 epochs" + medium dataset. **Confirm with pilot before trusting any cost number below.**
- 16 GPUs across node7+node8.
- Sustainable utilization ≈ 80% (sync time, eval time, occasional restart).

Then per week: 16 GPUs × 168 h × 0.8 = ~2150 GPU-hours/week.

At 8 h/run: ≈ **270 trials/week theoretical max**. But the bottleneck is not GPU — it is dev review cycle (the user has to look at each trial's dev metric and decide). Realistic throughput: **8–12 retrains/week reviewed**, with the rest of GPU time used for stage1/stage2 followups, baselines, ablation re-runs.

Per Stage 3 phase (using the §3 plan):

| Phase | Trials | GPU-hr if 8h/run | Wall on 16 GPU 80% util |
|---|---:|---:|---:|
| Stage 3a pilot | 1 | 8 | <1 day |
| Stage 3a Axis C | 4 | 32 | <1 day |
| Stage 3a Axis B | 3 | 24 | <1 day |
| Stage 3a Axis A | 3 | 24 | <1 day |
| Stage 3b Axis F | 3 | 24 | <1 day |
| Stage 3c (E or D) | 3 | 24 | <1 day |
| **Total Stage 3** | **17** | **136** | **~3–5 calendar days at full attention** |

Calendar reality: dev-metric review, debugging mining failures, and re-running flaky trials will likely double this. Plan for **~2 weeks** of Stage 3 from kickoff to promotion decision.

If the per-run is closer to 16 h (slow data loader, eval overhead), all the above doubles. **This is the single biggest unknown in the plan.**

---

## 6. Honest Risks

Things that can waste compute or produce misleading conclusions.

### R1 — Hardneg pool may be saturated already

`H_hardneg_r10` warm-start (`configs/mainline.yaml:50`) was itself trained with hardneg. The new mining is on a model that already learned to avoid the easiest false-positives. The remaining high-score negatives may largely be the "GT-narrowness" tail (`docs/next-phase.md:14`), which the model **should** be calling positive — penalizing them would hurt recall, not help precision. **Mitigation**: before Stage 3a, sample 20 of the freshly-mined hard negatives and manually classify them as (a) clearly non-BGC, (b) BGC-like-but-not-in-GT. If (b) dominates, hardneg weighting will hurt; consider Route 3 (benchmark GT re-examination) instead.

### R2 — Top-k vs threshold collinearity

Top-k=1000 and top-k=200 may resolve to nearly the same set if the score distribution drops sharply above 0.95. Picking sweep values without knowing the score histogram = blind. **Mitigation**: compute the histogram first (Blocker #6).

### R3 — Chunk overlap regularization may dwarf data changes

If we sweep chunk overlap and hardneg jointly, we cannot attribute gains. **Mitigation**: never sweep F simultaneously with A/B/C. Stage 3a freezes chunk construction; Stage 3b freezes hardneg.

### R4 — Single-seed dev metric is noisy

A 0.02 dev F1 swing on a 5-genome dev set is easily within seed noise. We have no inter-seed CI on the 9-genome dev metric yet. **Mitigation**: any "promoted" Stage 3 config must be re-run with at least one additional seed before being declared a winner. Budget +8 h/promoted config.

### R5 — Eval contract drift

Spec §3 mandates the `bench` scorer with asymmetric 50% overlap. We use `val_frozen_fused` as the formal operating point. If anyone "tunes" thresholds during Stage 3 to make a config look better, that violates §5. **Mitigation**: lock the `--operating-point val_frozen_fused` flag in the sweep runner; do not let trials override it.

### R6 — Dataset rebuild cache invalidation

If we change `meta_arrays.pkl` for hardneg trials, but reuse the same `chunk_index.pkl`, that's fine (chunk_index doesn't store labels, only indices). But if we change chunk construction, we MUST also confirm chunk indices still point at the right embedding rows (i.e. not also re-extract embeddings unintentionally). **Mitigation**: each Stage 3 trial's `manifest.json` must record SHA256 of every input pkl and embedding file. Cheap, prevents silent corruption.

### R7 — node7/node8 environment skew

Existing `configs/servers/execution_hosts.json` only lists node7. node8 hasn't been wired in. If half the GPU budget is on a node we haven't tested with `mine_hard_negatives.py` and `src/train.py`, we may discover broken envs mid-sweep. **Mitigation**: add node8 host entry and run a smoke train on node8 during the pilot.

---

## 7. Decision Points for the User

Please respond on each before any Stage 3 trial launches.

1. **Per-run wall-clock**: do you have a recent log showing one Route B training run's elapsed time? If yes, share — replaces the §5 guess. If no, do you approve burning one trial as a pure pilot?
2. **Pool-size diagnostic**: OK to spend 30 min on the histogram of train-negative scores from the current mainline before committing top-k values?
3. **Manual hardneg sanity-check (R1)**: 20-window manual review of fresh hardneg picks. OK to ask you to do this, or punt?
4. **Edge-padding axis**: confirm we're punting it (re-extract is too expensive)?
5. **node8 wiring**: should the Stage 3 runner be node7-only for first week, or do we wire node8 first?
6. **Resampler (Axis D)**: build the code now in parallel with Stage 3a, or wait to see if Axis C moves the metric?

---

## 8. Out-of-Scope for Stage 3

Per spec §7, do NOT sweep in Stage 3:

- Training hyperparameters (lr, batch size, warmup) — that's Stage 4.
- Architecture (CNN depth, dilation) — that's Stage 5.
- Decoder thresholds — already closed in Stage 1, frozen at `val_frozen_fused`.
- Inference tiling — Stage 2 in progress; Stage 3 must lock whatever Stage 2 promotes.

If a Stage 3 trial appears to want one of these to "make it work", that is evidence the Stage 3 hypothesis itself is wrong — flag and stop, do not silently widen scope.
