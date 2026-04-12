# Phase 2: OOD Robustness Plan

**Date:** 2026-04-09  
**Status:** Ready for execution after Phase 1 baseline freeze  
**Primary owner:** Claude Code / execution workers  
**Reviewer of record:** central review against this document

---

## 1. Purpose

This phase addresses the current **OOD robustness defect**.

In ML terms, the problem is:

- the current mainline improves ID performance,
- but some interventions increase false positives under distribution shift,
- therefore the project still lacks a robust boundary between true biosynthetic signal and OOD genomic background.

The goal of this phase is to improve **OOD precision and robustness** without giving back most of the gains on:

- known-BGC recall
- discovery-side candidate quality

---

## 2. Phase Hypothesis

### Primary hypothesis

The remaining OOD weakness is primarily a **distribution-shift robustness problem**, not yet evidence that the current model family is fundamentally wrong.

More specifically, OOD degradation may come from one or more of:

- insufficient hard negatives,
- positive-label expansion that is too permissive,
- model selection that is too ID-centric,
- phylum-specific background mismatch.

If these are corrected, then OOD precision should improve materially while retaining most ID gains.

### Null hypothesis

After reasonable robustness fixes, OOD precision remains poor, suggesting a deeper representation bottleneck.

---

## 3. Mainline Baseline

All Phase 2 work must compare against the same fixed mainline candidate:

- model family: current frozen Evo2 + current classifier
- detection score: `score_maxtype`
- evaluation mode: full-window
- prior-aligned training
- current best label contract from Phase 1

If Phase 1 promotes a new coverage fix, that promoted configuration becomes the new Phase 2 baseline.

No Phase 2 work may change:

- backbone
- pooling architecture
- region aggregation logic
- discovery benchmark definition

This phase is about **robustness under shift**, not representation redesign.

---

## 4. ML Best-Practice Rules

1. **Decompose before fixing**
   No robustness intervention should be promoted before OOD error analysis identifies where false positives concentrate.

2. **Keep ID and OOD reported together**
   Robustness fixes must not be judged on OOD alone.

3. **Treat OOD precision as a first-class metric**
   This phase is not allowed to hide OOD degradation behind better ID recall.

4. **Use one intervention axis at a time**
   Hard-negative changes, label-purity changes, and selection-rule changes must not be mixed in the first pass.

5. **Do not use test splits for tuning policy in an unconstrained way**
   OOD results are for evaluation. Selection policies must be predeclared or validation-based.

6. **Require non-inferiority on discovery**
   A robustness fix that only reduces candidate volume but destroys top-K discovery quality is not acceptable.

---

## 5. Scope

This phase covers:

- OOD false-positive analysis
- hard-negative strategy
- positive-label purity adjustments if needed
- OOD-aware selection constraints

This phase does **not** cover:

- new sequence architectures
- LoRA
- multimodal models
- large geometry resets

Those only become justified if this phase fails.

---

## 6. Inputs

Execution workers should assume the following are canonical inputs:

- current redesign memo: [2026-04-08-bgc-system-redesign.md](/Users/musk/Documents/code/np-master/docs/2026-04-08-bgc-system-redesign.md)
- execution roadmap: [2026-04-09-bgc-execution-roadmap.md](/Users/musk/Documents/code/np-master/docs/2026-04-09-bgc-execution-roadmap.md)
- current discovery benchmark: [ws_discovery_benchmark.md](/Users/musk/Documents/code/np-master/docs/execution/ws_discovery_benchmark.md)
- current sweep evidence: [ws3_capture_sweep_results.md](/Users/musk/Documents/code/np-master/docs/execution/ws3_capture_sweep_results.md)

Server-side baseline artifacts:

- mainline eval artifact
- current discovery benchmark artifact
- error analysis artifact

---

## 7. Required Outputs

This phase must produce:

### Output A: OOD error decomposition

A report that clearly identifies:

- where OOD false positives cluster by phylum,
- where they cluster by predicted type,
- whether they are near known BGCs or truly far,
- whether they carry curated BGC signature domains or look like hard negatives.

### Output B: Robustness intervention results

Each intervention needs its own result doc with:

- exact intervention
- exact changed files
- before/after ID metrics
- before/after OOD metrics
- before/after discovery metrics
- pass / fail / conditional-pass verdict

### Output C: Phase conclusion

One clear answer to:

- can the current architecture be made OOD-robust enough by training/data changes,
- or are we approaching a representation bottleneck?

---

## 8. Primary Metrics

### Must-report safety metrics

For every intervention:

- `test_id` prediction-level precision
- `test_id` recall
- `test_ood` prediction-level precision
- `test_ood` recall
- predicted region count on `test_id`
- predicted region count on `test_ood`

### Must-report discovery metrics

For any intervention that is a promotion candidate:

- signature precision@20
- signature precision@50
- signature diversity@50

### Required decomposition metrics

- OOD false positives by phylum
- OOD false positives by predicted type
- OOD unmatched candidate composition
- fraction of OOD false positives with curated signature domains

---

## 9. Success Criteria

An intervention is a **Phase 2 success candidate** if it satisfies all of:

1. **OOD precision improvement**
   It materially improves OOD precision relative to the current baseline.

2. **ID non-regression**
   It does not erase the current ID gains.

3. **Discovery non-inferiority**
   Top-K discovery quality does not materially degrade.

4. **Interpretability**
   The intervention’s effect is explainable in terms of the identified OOD error modes.

5. **No architecture change**
   The gain is achieved within the current model family.

---

## 10. Execution Order

### Step 1: OOD false-positive audit

Before changing training, identify the dominant OOD failure modes.

This is the most important first step. Without it, robustness work becomes guesswork.

### Step 2: Hard-negative strategy

If OOD FPs cluster in specific background regimes, test a hard-negative intervention first.

This is the preferred first fix because it directly targets the likely source of shift-related specificity failure.

### Step 3: Positive-label purity check

If OOD degradation appears tightly linked to permissive positive expansion, test whether a more selective label rule preserves most ID gains while restoring OOD precision.

### Step 4: Selection-constraint experiment

If training changes help but remain unstable, test whether checkpoint or operating-point selection should include an OOD guardrail.

### Step 5: Phase conclusion

Only after the above should the project decide whether OOD failure is still primarily a data/training problem or has become a representation problem.

---

## 11. Recommended Task Packets

### `P2-T1` OOD false-positive decomposition

Goal:

- map OOD false positives by phylum, type, and distance regime

Deliverables:

- `docs/execution/p2_t1_ood_fp_decomposition.md`
- canonical JSON artifact with grouped OOD FP statistics

### `P2-T2` Hard-negative intervention

Goal:

- test whether better negative coverage in OOD-like background improves specificity

Deliverables:

- implementation
- `docs/execution/p2_t2_hard_negative_results.md`

### `P2-T3` Label-purity robustness check

Goal:

- determine whether current positive expansion is too permissive under shift

Deliverables:

- one focused ablation
- `docs/execution/p2_t3_label_purity_results.md`

### `P2-T4` OOD-aware selection constraint

Goal:

- test whether model selection should enforce an OOD guardrail

Deliverables:

- selection policy note
- `docs/execution/p2_t4_selection_guardrail_results.md`

### `P2-T5` Phase 2 summary memo

Goal:

- decide whether the project is still robustness-limited or has become representation-limited

Deliverables:

- `docs/execution/p2_phase_summary.md`

---

## 12. Explicit Non-Goals

This phase must not:

- introduce a new sequence model
- add LoRA
- change the backbone
- change the detection score contract
- combine multiple robustness interventions before isolated evidence exists

---

## 13. Final Decision Question

At the end of Phase 2, the project should be able to answer:

> After coverage defects are addressed, can the current architecture remain competitive under OOD distribution shift, or are we now truly limited by representation?

That is the only question this phase exists to answer.
