# Phase 1: Short-BGC Coverage Plan

**Date:** 2026-04-09  
**Status:** Ready for execution  
**Primary owner:** Claude Code / execution workers  
**Reviewer of record:** central review against this document

---

## 1. Purpose

This phase addresses the current **coverage defect** for very short BGCs and short contig-edge BGCs.

In ML terms, this is not primarily a model-capacity problem. It is an **input-support and supervision-coverage problem**:

- some short BGCs do not produce any usable windows,
- some short BGCs produce only ambiguous or weakly supervised windows,
- therefore the model cannot learn or recover them reliably regardless of downstream threshold tuning.

The goal of this phase is to determine how much known-BGC recall can still be recovered by fixing short-BGC coverage **without changing the model architecture**.

---

## 2. Phase Hypothesis

### Primary hypothesis

Current mainline performance is still materially limited by **short-BGC support gaps** in the data pipeline.

If we increase valid support for short BGCs, then:

- short-length bucket recall should improve,
- overall known-BGC recall should improve or hold,
- precision should not collapse,
- this would indicate we are still in a **supervision-limited** regime rather than a **representation-limited** regime.

### Null hypothesis

Short-BGC coverage fixes do not materially improve short-length recall or overall recall, implying the remaining bottleneck is likely not data support alone.

---

## 3. Mainline Baseline

All Phase 1 experiments must compare against the same fixed mainline baseline:

- model family: current frozen Evo2 + current classifier
- detection score: `score_maxtype`
- evaluation mode: full-window
- training prior: prior-aligned (`pos_ratio=0.15`)
- label rule baseline: `F_cap08`

No experiment in this phase may change:

- backbone
- pooling architecture
- region aggregation logic
- detection score contract
- threshold objective

This phase is about **coverage**, not representation or decoding.

---

## 4. ML Best-Practice Rules

To keep this phase scientifically meaningful, execution must follow these rules:

1. **Single-axis changes first**
   Only one coverage intervention should be tested at a time before combinations.

2. **No test-set tuning**
   Hyperparameters may be selected on validation logic or predeclared policy only.
   `test_id` and `test_ood` are for reporting, not tuning.

3. **Short-bucket metrics are primary metrics**
   This phase is not judged only by overall recall.

4. **Promotion requires non-regression on safety**
   A short-BGC fix that improves recall but destroys precision is not a success.

5. **Prefer low-cost interventions before geometry resets**
   Edge padding and coverage fixes should be tried before shrinking window size and invalidating Stage 2 caches.

6. **Use at least one confirmatory rerun before declaring a new mainline**
   If a candidate fix wins, it must be rechecked for stability before promotion.

---

## 5. Scope

This phase covers two subproblems:

1. **No-window problem**
   Short contigs and tail regions that currently generate no window at all.

2. **Weak-supervision problem**
   Windows that exist but are not labeled positive under current rules.

This phase does **not** cover:

- sequence-model upgrades
- LoRA
- multimodal gene/domain models
- OOD-specific robustness methods

Those belong to later phases.

---

## 6. Inputs

Execution workers should assume the following inputs are canonical:

- current mainline config: [ws3_cap08.yaml](/Users/musk/Documents/code/np-master/bgc-gcd/configs/ws3_cap08.yaml)
- current redesign memo: [2026-04-08-bgc-system-redesign.md](/Users/musk/Documents/code/np-master/docs/2026-04-08-bgc-system-redesign.md)
- current execution roadmap: [2026-04-09-bgc-execution-roadmap.md](/Users/musk/Documents/code/np-master/docs/2026-04-09-bgc-execution-roadmap.md)
- current label-generation code: [05_windows_and_labels.py](/Users/musk/Documents/code/np-master/scripts/stage1/05_windows_and_labels.py)
- current WS3 findings: [ws3_capture_sweep_results.md](/Users/musk/Documents/code/np-master/docs/execution/ws3_capture_sweep_results.md)

Server-side canonical artifacts to reuse:

- mainline eval: `/data/muskliu/np_master/data/stage4_results/ws3_cap08_eval/score_swap_results.json`
- mainline labels: `/data/muskliu/np_master/data/windows/windows_metadata_cap08.csv`
- error analysis: `/data/muskliu/np_master/data/stage4_results/error_analysis/error_analysis.json`

---

## 7. Required Outputs

This phase must produce the following outputs.

### Output A: Coverage diagnosis report

A short document that quantifies:

- how many `<5kb` BGCs have zero windows,
- how many `5-10kb` BGCs have zero positive windows,
- how many short BGCs are lost because of contig-tail dropout vs label rule.

### Output B: Intervention-specific result docs

Each intervention must have its own result document containing:

- exact intervention
- exact changed files
- exact artifact paths
- before/after short-bucket metrics
- before/after overall safety metrics
- pass / fail / conditional-pass verdict

### Output C: Updated canonical recommendation

At the end of the phase, there must be one clear recommendation:

- keep current mainline unchanged
- or promote one short-BGC coverage fix as the new provisional mainline

No ambiguous “several options seem interesting” closeout.

---

## 8. Primary Metrics

### Must-report metrics

For each intervention, always report:

- `<5kb` BGC recall
- `5-10kb` BGC recall
- truncated-BGC recall
- overall `test_id` recall
- overall `test_id` prediction-level precision
- overall `test_ood` recall
- overall `test_ood` prediction-level precision
- number of predicted regions

### Secondary metrics

- per-type recall for RiPP / Terpene / Other
- count of short BGCs with zero windows
- count of short BGCs with zero positive windows

### Discovery guardrail

If an intervention is a strong candidate for mainline promotion, re-run discovery-side benchmark at least for:

- signature precision@20
- signature precision@50

This avoids improving known recovery at the cost of degrading discovery quality.

---

## 9. Success Criteria

An intervention is a **Phase 1 success candidate** if it satisfies all of:

1. **Coverage improvement**
   Short-bucket support increases measurably.

2. **Known-recovery gain**
   At least one of the key short buckets improves materially.

3. **No major ID safety regression**
   `test_id` prediction-level precision must not drop by more than `0.03` absolute from the current mainline unless recall gain is unusually large and explicitly approved.

4. **No catastrophic OOD regression**
   `test_ood` prediction-level precision must not collapse relative to mainline.

5. **Contract clarity**
   The intervention is reproducible and can be described as a new label/geometry contract.

---

## 10. Execution Order

The recommended order is:

### Step 1: Coverage diagnosis

Measure exactly where short-BGC support is missing.

### Step 2: Edge-padding MVP

Test whether adding padded tail windows fixes the “no-window” subset.

This is the most important next intervention because it addresses samples that currently have zero support.

### Step 3: Evaluate edge-padding alone

Do not combine with other label changes yet.

### Step 4: If needed, test one additional low-cost label adjustment

Only if edge padding does not recover enough short-BGC support.

### Step 5: Confirmatory rerun on the best intervention

Before promotion, do one stability recheck.

### Step 6: Decide whether geometry reset is justified

Only if low-cost fixes fail should the phase consider:

- smaller stride
- smaller window size

These are expensive and should not be the first move.

---

## 11. Recommended Task Packets

The following task packets are authorized for this phase.

### `P1-T1` Short-BGC support audit

Goal:

- quantify zero-window and zero-positive support defects by length bucket

Deliverables:

- `docs/execution/p1_t1_shortbgc_support_audit.md`
- canonical JSON artifact with counts by length bucket

### `P1-T2` Edge-padding MVP spec + implementation

Goal:

- allow short contig tails to generate padded windows

Deliverables:

- implementation in Stage 1 / Stage 2 path as needed
- `docs/execution/p1_t2_edge_padding_results.md`

### `P1-T3` Edge-padding benchmark

Goal:

- evaluate whether padded support improves short-bucket recall without breaking safety

Deliverables:

- result doc
- canonical eval artifact

### `P1-T4` Phase 1 promotion memo

Goal:

- summarize whether any coverage intervention should become new provisional mainline

Deliverables:

- `docs/execution/p1_phase_summary.md`

---

## 12. Explicit Non-Goals

This phase must not:

- introduce sequence-model upgrades
- add LoRA
- change the backbone
- change detection score away from `score_maxtype`
- silently retune using test performance
- merge multiple interventions before single-axis evidence exists

---

## 13. Final Decision Question

At the end of Phase 1, the project should be able to answer:

> How much of the remaining known-BGC gap is still caused by short-BGC support defects, and does fixing those defects still materially improve the current architecture?

That is the only question this phase exists to answer.
