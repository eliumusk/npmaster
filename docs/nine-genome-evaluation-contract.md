# 9-Genome Evaluation Contract

**Date:** 2026-04-18
**Status:** active working contract
**Scope:** NP-Master / Evo2 Route B on the classical 9-genome benchmark

---

## 1. Goal

This document fixes the evaluation protocol for the 9-genome benchmark so that:

- we stop mixing score diagnosis, postprocessing diagnosis, and final reporting
- we can test whether poor 9-genome precision is mainly a postprocessing problem
- we can compare NP-Master against DeepBGC / BGC-Prophet / antiSMASH on a stable protocol

This contract is for the **9-genome benchmark only**. It does not replace the main internal antiSMASH-proxy benchmark.

---

## 2. Current Situation

The 9-genome benchmark exists independently on the server:

- benchmark root: `/data/syh/benchmark/nine_genome`
- ground truth: `/data/syh/benchmark/nine_genome/ground_truth.json`
- region-level evaluator: `/data/syh/benchmark/nine_genome/evaluate.py`
- gene-level evaluator: `/data/syh/benchmark/nine_genome/gene_level_eval.py`
- existing region-level summary: `/data/syh/benchmark/nine_genome/evaluation/results.json`
- existing gene-level summary: `/data/syh/benchmark/nine_genome/evaluation/gene_level_results.json`

Observed current benchmark numbers:

### Region-level (`evaluation/results.json`)

| Tool | Recall | Precision | F1 | Pred regions |
|---|---:|---:|---:|---:|
| antiSMASH | 0.557 | 0.512 | 0.534 | 283 |
| DeepBGC | 0.725 | 0.170 | 0.275 | 1132 |
| BGC-Prophet | 0.351 | 0.606 | 0.444 | 170 |
| **NP-Master** | **0.636** | **0.634** | **0.635** | 175 |

### Gene-level (`evaluation/gene_level_results.json`)

| Tool | Precision | Recall | F1 |
|---|---:|---:|---:|
| antiSMASH | 0.261 | 0.649 | 0.373 |
| DeepBGC | 0.171 | 0.830 | 0.284 |
| BGC-Prophet | 0.492 | 0.500 | 0.496 |
| **NP-Master** | **0.150** | **0.711** | **0.247** |

Interpretation:

- NP-Master already looks strong at **region-level**
- NP-Master still looks weak at **gene-level**
- therefore, score quality and region calling must be evaluated separately

---

## 3. Main Risk

The current 9-genome evaluator uses a deployment choice that is too arbitrary to be treated as final:

- `run_npmaster.py` writes window scores
- `evaluate.py` converts them to regions with:
  - fixed `threshold=0.5`
  - merge across `1` full-window gap (`16384 bp`)

This operating point is not tied to validation data and is likely recall-biased.

Therefore:

- any bad 9-genome precision under this default must be treated as **diagnostic**
- it cannot be used as final evidence against the model

---

## 4. Fixed Evaluation Layers

The benchmark is split into four sets. Each set answers one question only.

### Set A: Window-Level Diagnostic

**Purpose:** test whether the score itself separates BGC windows from non-BGC windows.

**Inputs:**

- per-window probabilities from NP-Master
- 9-genome ground truth

**Scores to compare:**

- `max_type = max(probs[:7])`
- `one_minus_isneg = 1 - probs[7]`
- `fused = 0.5 * max_type + 0.5 * one_minus_isneg`

**Metrics:**

- AUROC
- AUPRC
- best window-level F1
- precision / recall at best window-level F1

**Use:** diagnosis only, never final reporting.

### Set B: Region-Level Postprocessing Sweep

**Purpose:** test whether the 9-genome result is dominated by postprocessing.

**Parameters to sweep:**

- score family
- threshold
- `max_gap`
- `length_cap`

**Matching metrics to report in parallel:**

- `loose`: overlap / min(pred_len, gt_len) >= 0.5
- `strict`: overlap / max(pred_len, gt_len) >= 0.5
- `iou`: IoU >= 0.5

**Use:** diagnosis only, never final reporting.

### Set C: Internal-Val Operating Point Selection

**Purpose:** choose the default operating point without tuning on 9-genome test data.

**Selection split:**

- internal `val` from the main NP-Master data

**Parameters to select:**

- score family
- threshold
- `max_gap`
- `min_length_bp`
- `length_cap`

**Primary selection objective:**

- strict region-level F1

**Secondary outputs:**

- loose region-level F1
- IoU F1
- prediction-level precision
- number of predicted regions

### Set D: 9-Genome Final Frozen Evaluation

**Purpose:** report the final 9-genome result using the operating point selected in Set C.

**Frozen items:**

- checkpoint
- score family
- threshold
- `max_gap`
- `min_length_bp`
- `length_cap`

**Primary reported metrics:**

- region-level recall
- region-level prediction-level precision
- region-level F1

**Secondary reported metrics:**

- gene-level precision / recall / F1
- number of predicted regions
- per-genome results

---

## 5. Official Metric Definitions

### 5.1 Region-Level Recall

For each ground-truth BGC, determine whether at least one predicted region matches it under the chosen overlap rule.

### 5.2 Prediction-Level Precision

For each predicted region, determine whether it matches any ground-truth BGC under the chosen overlap rule.

This is the primary precision metric.

### 5.3 Gene-Level Metrics

Gene-level precision / recall / F1 are retained as supporting evidence only.

Reason:

- papers like BGC-Prophet emphasize gene-level detection
- but our deployment object is a **region caller**, not a per-gene classifier

### 5.4 Primary Matching Rule for Final Reporting

Final reporting should prioritize:

- **strict region-level F1** as the selection objective
- **prediction-level precision** as the precision number shown in the main table

Reason:

- `loose` can over-credit oversized regions
- `strict` penalizes over-expanded calls, which is exactly the failure mode we suspect

---

## 6. Official Execution Order

Always run in this order:

1. Set A: window-level diagnosis
2. Set B: postprocessing sweep on 9-genome
3. Set C: select frozen operating point on internal `val`
4. Set D: run final 9-genome evaluation with the frozen operating point

Do **not** tune thresholds directly on the final 9-genome report set.

---

## 7. Immediate Working Hypothesis

The leading hypothesis is:

1. NP-Master score ranking is probably acceptable
2. 9-genome degradation is mainly caused by postprocessing
3. the current default operating point is too permissive because it uses:
   - fixed `threshold=0.5`
   - one-window merge gap
   - no principled selection against strict region precision

This hypothesis is considered supported only if Set B shows that precision changes materially under operating-point changes while score-level AUROC/AUPRC remain stable.

---

## 8. Minimum Validation Matrix

These are the first runs to perform.

### Run 1: Window-Level Diagnostic on 9-genome

Use:

- `scripts/phase4/nine_genome_sweep_s1.py`

Output needed:

- AUROC / best window-F1 for `max_type`, `one_minus_isneg`, `fused`

### Run 2: Region-Level Postprocessing Sweep on 9-genome

Use:

- `scripts/phase4/nine_genome_sweep_s1.py`

Output needed:

- top configs by strict F1
- top configs by IoU F1
- baseline comparison against current default-like operating point

### Run 3: Internal-Val Postprocessing Sweep

Use:

- `scripts/phase4/postprocess_sweep.py`

Output needed:

- best strict-F1 config on internal `val`
- best loose-F1 config on internal `val`
- region count and median length at each best point

### Run 4: Frozen Final 9-genome Evaluation

Use:

- server benchmark pipeline in `/data/syh/benchmark/nine_genome`
- but replace the current hardcoded default operating point with the frozen config from Run 3

---

## 9. What Not To Do Yet

Before the four runs above are complete, do not spend time on:

- changing CNN depth / dilation
- changing chunk length
- changing sampling ratio
- changing data construction
- changing Evo2 extraction setup

Those are model-layer interventions. The current leading problem is in the decision layer.

---

## 10. Success Criteria For The Current Debug Phase

This debug phase is successful if we can answer all three questions clearly:

1. Is the 9-genome problem mostly score quality or postprocessing?
2. Which operating point gives the best strict region-level tradeoff?
3. After freezing that point on internal `val`, what is the true 9-genome final result?
