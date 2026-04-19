# Stage 5 No-9-Genome-Tuning Implementation Plan

Date: 2026-04-20 01:00:00

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a NP-Master training/data pipeline that reaches strong 9-genome performance without using any 9-genome sequence, label, score, boundary, or metric for training, threshold selection, decoder tuning, model selection, or early stopping.

**Architecture:** Treat the full 9-genome benchmark as a final external test set only. All model, label, threshold, decoder, and hyperparameter decisions must be made on internal data derived from antiSMASH/MIBiG/approved non-9-genome sources. The current `defrag_v2_ms10_s099` result remains a useful 9-genome-tuned upper bound and sanity target, but it is not the Stage 5 selection protocol. Short BGCs are handled as a separate length-bucket track because the current high-precision defragmented decoder is intentionally biased toward longer regions.

**Tech Stack:** Python, PyTorch, existing `src/train.py`, Evo2 embedding inference via `scripts/inference_stride.py`, existing region evaluation via `scripts/evaluate_nine_genome.py`, YAML configs, unittest tests, server execution through generated experiment manifests.

---

## 0. Executive Summary

We now have a strong 9-genome-tuned operating point:

| Candidate | all9 R | all9 P | all9 F1 | all9 dedup F1 | n_pred |
|---|---:|---:|---:|---:|---:|
| BGC-Prophet | 0.3475 | 0.4529 | 0.3933 | 0.3911 | 170 |
| `defrag_v2_ms10_s099` | 0.4098 | 0.4714 | 0.4385 | 0.4385 | 140 |

This proves NP-Master can reach the desired operating regime, but it used 9-genome dev to select decoder parameters. Stage 5 asks a harder and more scientifically defensible question:

> Can training data, label construction, and model training make NP-Master perform well on 9-genome without using 9-genome for any selection?

There is one important caveat: the 9-genome GT contains many short regions. As of the current benchmark file (`tmp/stage2b_predictions/benchmark/ground_truth.json`), 48/305 GT regions are shorter than 5 kb and 137/305 are shorter than 10 kb. The promoted `defrag_v2_ms10_s099` operating point uses `window=8192`, `stride=2048`, and `min_support_windows=10`, so its minimum supported span is about `8192 + 9 * 2048 = 26624 bp`. That operating point is suitable for a high-precision main detector, but it is structurally unable to make tight `<5 kb` boundary calls. Stage 5 must therefore report length-bucket metrics and should include a separate small-BGC track.

The answer must be measured by a locked final 9-genome report. A Stage 5 candidate is invalid if any parameter was chosen because it improved 9-genome.

---

## 1. Non-Negotiable Protocol

### 1.1 Embargoed 9-Genome IDs

The following IDs are fully embargoed for Stage 5 selection:

```text
GCA_000154945.1
GCA_000156435.1
GCA_000156475.1
GCA_000158815.1
GCA_000158875.1
GCA_000158895.1
GCA_000158915.1
GCA_000568255.1
GCA_000568915.1
```

Forbidden uses:

- Training examples.
- Validation examples.
- Hard-negative mining examples.
- Hard-positive mining examples.
- Threshold fitting.
- Decoder grid selection.
- Hyperparameter selection.
- Early stopping.
- Deciding which trial to continue or kill.
- Manual “one more run” decisions based on 9-genome metrics.

Allowed uses:

- One final external report per pre-registered Stage 5 candidate family.
- Re-running the exact same frozen candidate only to reproduce a failed command or fix an infrastructure error.
- Plotting after final evaluation for interpretation, not for selecting a new parameter.

### 1.2 Candidate Promotion Rule

A Stage 5 candidate can be promoted only if all selection was completed before 9-genome evaluation.

Promotion evidence must include:

- A resolved config file under `configs/sweeps/` or `configs/operating_points.yaml`.
- An internal validation summary used for selection.
- A leakage check report showing no embargoed genome IDs in training/validation/mining inputs.
- A final 9-genome report generated after candidate freeze.

### 1.3 Metrics

Primary external metric:

- `bench_f1` on all 9 genomes.

Required companion metrics:

- `bench_recall`
- `bench_precision`
- `dedup_precision`
- `dedup_f1`
- `duplicate_precision_tp`
- `max_preds_per_gt`
- `strict_f1`
- `iou_f1`
- `n_pred`
- per-genome `bench` breakdown

Required length-bucket metrics:

- GT-bucket recall for `<5kb`, `5-10kb`, `10-20kb`, `20-50kb`, and `>=50kb`.
- Pred-bucket precision for `<5kb`, `5-10kb`, `10-20kb`, `20-50kb`, and `>=50kb`.
- `n_gt`, `n_pred`, `recall_tp`, and `precision_tp` per bucket.
- `small_bgc_f1`: harmonic mean of `<5kb` GT-bucket recall and `<5kb` pred-bucket precision.
- A clear flag stating whether a candidate is a main operating point, a small-BGC secondary operating point, or an ensemble of both.

Success tiers:

| Tier | Requirement |
|---|---|
| Minimum success | all9 `bench_f1 >= 0.3933` and `dedup_f1 >= 0.3911`, matching/exceeding BGC-Prophet without 9-genome tuning |
| Strong success | all9 `bench_f1 >= 0.4200` and `bench_precision >= 0.4400` |
| Stretch success | all9 `bench_f1 >= 0.4385`, matching/exceeding the current 9-genome-tuned upper bound |
| Small-BGC success | `<5kb` GT-bucket recall improves over the main operating point without reducing all9 `bench_f1` by more than 0.02, or is reported as a separate secondary detector rather than merged into the main score |

Failure modes:

- Raw precision improves but `dedup_f1` falls: fragmentation, not progress.
- `bench_f1` improves only by making `n_pred` collapse below 50: likely over-strict decoding.
- `<5kb` recall is claimed from long predictions only: not valid evidence of tight small-BGC detection.
- Internal validation improves but 9-genome collapses: selection distribution mismatch.
- `GCA_000154945.1` remains near zero recall: domain miss still unresolved.

---

## 2. Data Strategy

Stage 5 should not assume antiSMASH + MIBiG is enough. It should test whether better labels and broader weak supervision reduce the need for 9-genome decoder tuning.

### 2.1 Source Tiers

| Source | Role | Label strength | Notes |
|---|---|---|---|
| MIBiG | high-confidence BGC cores | strong positive | small but biologically reliable |
| antiSMASH regions | large-scale weak supervision | medium positive | boundary style is wider than 9-genome tight-core GT |
| antiSMASH flanks | ambiguous boundary zone | masked or low weight | do not treat as strong positive or strong negative |
| short MIBiG/curated cores | short-BGC positives | strong positive | needed for `<5kb` and `5-10kb` recall |
| antiSMASH-negative windows with no biosynthetic domains | clean background | strong negative | safer than “outside any antiSMASH region” |
| antiSMASH-negative but multi-tool-supported regions | possible missed BGCs | unlabeled or medium positive | should not be hard negative |
| internal high-score false positives | hard negatives | weighted negative after review/filtering | must exclude 9-genome |
| internal low-score GT-overlapping regions | hard positives | weighted positive | must exclude 9-genome |

### 2.2 Label Variants To Test

Stage 5 should compare label construction variants, not only CNN hyperparameters.

| Variant | Description | Expected effect |
|---|---|---|
| `baseline_antismash` | current labels as-is | reference |
| `core_only_v1` | shrink antiSMASH positives by trimming region edges; mask flanks | tighter boundaries, higher precision |
| `core_flank_soft_v1` | core positive weight 1.0, flank positive weight 0.25, outside negative weight 1.0 | less boundary overfit |
| `domain_clean_negative_v1` | outside regions with biosynthetic domains are masked, not negative | fewer false negatives in training |
| `mibig_boost_v1` | MIBiG/core windows receive higher positive weight | improve real BGC core recall |
| `short_bgc_boost_v1` | short curated/MIBiG positives receive higher sampling and loss weight | improve `<5kb` and `5-10kb` recall |
| `hardpos_hardneg_v1` | add internal hard positives and hard negatives from non-9-genome validation | improve both recall and precision |

### 2.3 Internal Split Rules

Splits must be group-based:

- Split by genome/assembly, not by window.
- Windows from the same contig/genome must not appear in both train and val.
- MIBiG clusters with near-identical accessions should stay in the same split where possible.
- 9-genome IDs must not appear in any split file except explicit final-test references.

Recommended internal split:

| Split | Purpose |
|---|---|
| train | model fitting |
| val_select | threshold, decoder, operating point, and hyperparameter selection |
| val_audit | optional confirmation that selected params are not overfit to `val_select` |

---

## 3. Required Artifacts

### Files To Create

- `docs/stage5-no-nine-genome-tuning.md`: this plan.
- `configs/splits/nine_genome_embargo_v1.json`: explicit embargo list.
- `configs/sweeps/stage5_no9_data_v1.yaml`: data/label construction sweep.
- `configs/sweeps/stage5_no9_decoder_v1.yaml`: internal-val-only decoder sweep for Stage 5 candidates.
- `scripts/check_no_nine_genome_leakage.py`: fail-fast checker for configs, manifests, labels, and generated commands.
- `scripts/build_stage5_labels.py`: constructs label variants from existing annotations.
- `scripts/run_stage5_data_sweep.py`: plans train/inference/eval jobs for label variants.
- `scripts/evaluate_length_bucket_metrics.py`: reports GT-length and predicted-length bucket metrics.
- `configs/sweeps/stage5_small_bgc_v1.yaml`: internal-val-only sweep for a secondary small-BGC operating point.
- `scripts/summarize_stage5_results.py`: joins internal-val and final 9-genome results into one report.
- `tests/test_no_nine_genome_leakage.py`: tests embargo checker.
- `tests/test_stage5_label_builder.py`: tests label construction behavior.
- `tests/test_length_bucket_metrics.py`: tests short-BGC bucket accounting.
- `tests/test_stage5_sweep_planner.py`: tests generated commands do not use 9-genome for selection.

### Files To Modify

- `docs/evaluation-protocol.md`: add a Stage 5 section stating 9-genome is final-test only.
- `docs/next-phase.md`: point the next research route to Stage 5.
- `configs/operating_points.yaml`: keep `defrag_v2_ms10_s099`, but describe it as a lockbox-validated 9-genome-tuned reference/upper bound. Do not use it for Stage 5 selection.

---

## 4. Experiment Design

### 4.1 Stage 5A: Establish No-9 Baseline

Purpose:

- Measure current model under an operating point selected without 9-genome.

Procedure:

1. Use existing internal validation data only.
2. Select score family, thresholds, and decoder params on internal validation.
3. Freeze one no-9 operating point.
4. Run final 9-genome once.

Expected output:

- `experiments/stage5_no9_baseline_v1/summary.csv`
- `experiments/stage5_no9_baseline_v1/final_9genome.json`
- A table comparing:
  - BGC-Prophet
  - `val_frozen_fused`
  - `defrag_v2_ms10_s099` as 9-genome-tuned upper bound
  - Stage 5A no-9 baseline

### 4.2 Stage 5B: Label Construction Sweep

Purpose:

- Test whether antiSMASH-derived labels can be reshaped to reduce boundary mismatch.

Initial grid:

| Trial | Label variant | Decoder selection source |
|---|---|---|
| `baseline_antismash` | current label file | internal val |
| `core_only_v1` | trim region edges, mask flank | internal val |
| `core_flank_soft_v1` | weighted core/flank labels | internal val |
| `domain_clean_negative_v1` | mask biosynthetic-domain negatives | internal val |
| `mibig_boost_v1` | higher positive weight for MIBiG/core | internal val |
| `short_bgc_boost_v1` | higher sampling/loss weight for short curated positives | internal val |
| `hardpos_hardneg_v1` | internal hard example weighting | internal val |

Selection rule:

- Choose by internal `val_select` `dedup_f1`, with `bench_precision` and `strict_f1` as tie-breakers.
- Do not inspect 9-genome during selection.

### 4.3 Stage 5C: Training Hyperparameter Sweep

Purpose:

- Tune model training after label strategy is fixed.

Allowed knobs:

- `lr_proj`
- `lr_context`
- `lr_classifier`
- `weight_decay`
- `warmup_epochs`
- `proj_dropout`
- `cnn_dropout`
- hard-positive weight
- hard-negative weight

Disallowed at this stage:

- Changing Evo2 layer.
- Changing model architecture.
- Changing 9-genome decoder after seeing 9-genome.

### 4.4 Stage 5D: Boundary-Aware Model Extension

Only start after Stage 5B/C show that labels help but strict/IoU remains weak.

Candidate auxiliary heads:

- `inside_bgc`
- `near_left_boundary`
- `near_right_boundary`
- distance-to-boundary bin

Selection must remain internal-val-only.

### 4.5 Stage 5E: Small-BGC Track

Purpose:

- Recover short BGCs that the main `min_support_windows=10` defragmented operating point cannot represent tightly.
- Keep the main detector honest: do not lower all-region precision by merging a short-region detector blindly into the main output.

Important constraint:

- The current main operating point has a minimum practical span of about 26.6 kb. It may overlap `<5kb` GT regions, but that is not the same as detecting short BGC boundaries.

Recommended design:

| Component | Main detector | Small-BGC detector |
|---|---|---|
| Goal | high precision on medium/large BGCs | recover `<5kb` and `5-10kb` BGCs |
| Window | 8192 bp | 2048, 4096, or 8192 bp |
| Stride | 2048 bp | 512, 1024, or 2048 bp |
| `min_support_windows` | 8-10 | 1-3 |
| `length_cap` | none or 60-100 kb | 5, 10, or 15 kb |
| Selection split | internal val only | internal val only |
| Final use | primary operating point | secondary report or gated ensemble |

Initial small-BGC decoder grid:

```yaml
sweep_id: stage5_small_bgc_v1

selection_policy:
  selection_split: internal_val
  primary_metric: small_bgc_f1
  small_bgc_bucket: "<5kb"
  tie_breakers:
    - small_bgc_recall
    - bench_precision
    - dedup_f1
  forbidden_selection_sources:
    - nine_genome

decoder_grid:
  window:
    - 2048
    - 4096
    - 8192
  stride:
    - 512
    - 1024
    - 2048
  score_family:
    - fused
    - max_type
  start_threshold:
    - 0.98
    - 0.99
    - 0.995
  extend_threshold:
    - 0.90
    - 0.95
    - 0.98
  min_support_windows:
    - 1
    - 2
    - 3
  max_gap:
    - 0
  length_cap:
    - 5000
    - 10000
    - 15000
```

Training/data knobs to test before changing architecture:

- Oversample short positive windows from MIBiG/curated cores during training.
- Increase positive loss weight for short curated BGCs, for example `short_positive_weight in {2, 4, 8}`.
- Mask uncertain flanks around short positives instead of labelling them hard negative.
- Keep clean negative mining, but do not mine hard negatives from biosynthetic-domain-rich short windows without review.
- Only try focal loss or a short-BGC auxiliary head after label/sampling sweeps fail.

Promotion rule:

- If the small-BGC detector improves `<5kb` recall while preserving all9 `bench_f1` within 0.02 of the main operating point, it can be evaluated as a gated ensemble candidate.
- If it improves `<5kb` recall but hurts main precision/F1, keep it as a separate secondary detector and report it separately. Do not claim it replaces the main operating point.

---

## 5. Implementation Tasks

### Task 1: Add Embargo Config And Leakage Checker

**Files:**

- Create: `configs/splits/nine_genome_embargo_v1.json`
- Create: `scripts/check_no_nine_genome_leakage.py`
- Create: `tests/test_no_nine_genome_leakage.py`

- [ ] **Step 1: Create the embargo JSON**

Create `configs/splits/nine_genome_embargo_v1.json`:

```json
{
  "version": "nine_genome_embargo_v1",
  "reason": "Stage 5 treats all nine genomes as final external test only.",
  "embargoed_genome_ids": [
    "GCA_000154945.1",
    "GCA_000156435.1",
    "GCA_000156475.1",
    "GCA_000158815.1",
    "GCA_000158875.1",
    "GCA_000158895.1",
    "GCA_000158915.1",
    "GCA_000568255.1",
    "GCA_000568915.1"
  ]
}
```

- [ ] **Step 2: Write failing tests for path/text scanning**

Create `tests/test_no_nine_genome_leakage.py` with tests that assert:

```python
from pathlib import Path
import tempfile
import unittest

from scripts.check_no_nine_genome_leakage import scan_paths_for_embargo_ids


class NoNineGenomeLeakageTests(unittest.TestCase):
    def test_detects_embargo_id_in_text_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("genome: GCA_000158915.1\n", encoding="utf-8")
            hits = scan_paths_for_embargo_ids(
                [path],
                {"GCA_000158915.1", "GCA_000154945.1"},
            )
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["genome_id"], "GCA_000158915.1")

    def test_ignores_clean_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("genome: GCA_999999999.1\n", encoding="utf-8")
            hits = scan_paths_for_embargo_ids([path], {"GCA_000158915.1"})
            self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the failing test**

Run:

```bash
python -m unittest tests.test_no_nine_genome_leakage
```

Expected:

```text
ImportError or ModuleNotFoundError for scripts.check_no_nine_genome_leakage
```

- [ ] **Step 4: Implement the checker**

Implement `scripts/check_no_nine_genome_leakage.py` with:

- `load_embargo_ids(path: Path) -> set[str]`
- `scan_paths_for_embargo_ids(paths: list[Path], embargo_ids: set[str]) -> list[dict]`
- CLI args:
  - `--embargo configs/splits/nine_genome_embargo_v1.json`
  - `--paths <one or more files/dirs>`
  - `--json-out <optional path>`

Behavior:

- Recursively scan text-like files under provided paths.
- Skip binary files by catching `UnicodeDecodeError`.
- Return non-zero if any embargo ID is found.
- Print every hit as `path:line:genome_id`.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m unittest tests.test_no_nine_genome_leakage
python -m unittest discover -s tests
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit**

```bash
git add configs/splits/nine_genome_embargo_v1.json scripts/check_no_nine_genome_leakage.py tests/test_no_nine_genome_leakage.py
git commit -m "stage5: add nine-genome embargo checker"
```

### Task 2: Add Stage 5 Label Builder

**Files:**

- Create: `scripts/build_stage5_labels.py`
- Create: `tests/test_stage5_label_builder.py`
- Create: `configs/sweeps/stage5_no9_data_v1.yaml`

- [ ] **Step 1: Write label behavior tests**

Create tests that use a tiny synthetic annotation table:

```python
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_stage5_labels import assign_region_zone, build_label_rows


class Stage5LabelBuilderTests(unittest.TestCase):
    def test_assign_region_zone_core_flank_outside(self):
        self.assertEqual(assign_region_zone(100, 200, 100, 200, flank_bp=25), "core")
        self.assertEqual(assign_region_zone(80, 110, 100, 200, flank_bp=25), "flank")
        self.assertEqual(assign_region_zone(0, 50, 100, 200, flank_bp=25), "outside")

    def test_core_only_masks_flank(self):
        windows = pd.DataFrame(
            [
                {"contig": "c1", "start": 100, "end": 200, "label": "positive"},
                {"contig": "c1", "start": 80, "end": 110, "label": "positive"},
                {"contig": "c1", "start": 0, "end": 50, "label": "negative"},
            ]
        )
        regions = pd.DataFrame([{"contig": "c1", "start": 100, "end": 200}])
        out = build_label_rows(windows, regions, variant="core_only_v1", flank_bp=25)
        self.assertEqual(list(out["stage5_label"]), ["positive", "masked", "negative"])
        self.assertEqual(list(out["stage5_weight"]), [1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement label builder**

`scripts/build_stage5_labels.py` must:

- Read the existing window label CSV.
- Read BGC region annotations.
- Produce output CSV with original columns plus:
  - `stage5_label`
  - `stage5_weight`
  - `stage5_variant`
  - `stage5_zone`
- Support variants:
  - `baseline_antismash`
  - `core_only_v1`
  - `core_flank_soft_v1`
  - `domain_clean_negative_v1`
  - `mibig_boost_v1`
  - `short_bgc_boost_v1`
  - `hardpos_hardneg_v1`

Minimum variant semantics:

| Variant | Required behavior |
|---|---|
| `baseline_antismash` | preserve original positive/negative labels, weight 1.0 |
| `core_only_v1` | core positive weight 1.0, flank masked weight 0.0 |
| `core_flank_soft_v1` | core positive weight 1.0, flank positive weight 0.25 |
| `domain_clean_negative_v1` | windows with biosynthetic domain markers outside antiSMASH regions become masked |
| `mibig_boost_v1` | MIBiG-associated positives weight 2.0 |
| `short_bgc_boost_v1` | positives from BGC regions shorter than `short_max_bp` receive `short_positive_weight` |
| `hardpos_hardneg_v1` | consume a hard-example CSV and assign hard positive/negative weights |

If a required source column is absent for a variant, the script must fail with a clear error instead of silently falling back.

- [ ] **Step 3: Add sweep config**

Create `configs/sweeps/stage5_no9_data_v1.yaml`:

```yaml
sweep_id: stage5_no9_data_v1

defaults:
  input_labels: /data/muskliu/npmaster/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv
  bgc_csv: /data/muskliu/npmaster/data/bgc_annotations.csv
  output_root: /data/muskliu/npmaster/data/stage5/no9_labels
  flank_bp: 10000
  embargo: configs/splits/nine_genome_embargo_v1.json

variants:
  baseline_antismash:
    variant: baseline_antismash
  core_only_v1:
    variant: core_only_v1
  core_flank_soft_v1:
    variant: core_flank_soft_v1
  domain_clean_negative_v1:
    variant: domain_clean_negative_v1
  mibig_boost_v1:
    variant: mibig_boost_v1
  short_bgc_boost_v1:
    variant: short_bgc_boost_v1
    short_positive_weight: 4.0
    short_max_bp: 10000
  hardpos_hardneg_v1:
    variant: hardpos_hardneg_v1
    hard_examples_csv: /data/muskliu/npmaster/data/stage5/hard_examples/internal_hard_examples.csv
```

- [ ] **Step 4: Run tests**

```bash
python -m unittest tests.test_stage5_label_builder
python -m unittest discover -s tests
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_stage5_labels.py tests/test_stage5_label_builder.py configs/sweeps/stage5_no9_data_v1.yaml
git commit -m "stage5: add no-nine label variants"
```

### Task 3: Add Stage 5 Experiment Planner

**Files:**

- Create: `scripts/run_stage5_data_sweep.py`
- Create: `tests/test_stage5_sweep_planner.py`
- Create: `configs/sweeps/stage5_no9_decoder_v1.yaml`

- [ ] **Step 1: Write planner tests**

Test requirements:

- Unknown trial keys fail.
- Generated commands include `scripts/check_no_nine_genome_leakage.py`.
- Generated train commands use label variant paths, not 9-genome paths.
- Generated final 9-genome command is marked `final_report_only` in manifest.

- [ ] **Step 2: Implement planner**

`scripts/run_stage5_data_sweep.py` should mirror `scripts/run_stage4_training_sweep.py`:

- Read `configs/sweeps/stage5_no9_data_v1.yaml`.
- For each variant:
  - Generate label build command.
  - Generate leakage check command.
  - Generate train command.
  - Generate internal-val inference/eval command.
  - Generate final 9-genome inference/eval command only when `--include-final-9genome` is explicitly passed.
- Write:
  - `experiments/stage5_no9_data_v1-<variant>/command.txt`
  - `config.resolved.yaml`
  - `manifest.json`
  - `env.txt`

Manifest must include:

```json
{
  "stage": "stage5_no9_data_v1",
  "nine_genome_policy": "final_report_only",
  "uses_nine_genome_for_selection": false
}
```

- [ ] **Step 3: Add decoder config**

Create `configs/sweeps/stage5_no9_decoder_v1.yaml` for internal-val-only decoder search:

```yaml
sweep_id: stage5_no9_decoder_v1

selection_policy:
  selection_split: internal_val
  primary_metric: dedup_f1
  tie_breakers:
    - bench_f1
    - bench_precision
    - strict_f1
  forbidden_selection_sources:
    - nine_genome

decoder_grid:
  score_family:
    - fused
    - max_type
  start_threshold:
    - 0.95
    - 0.97
    - 0.99
  extend_threshold:
    - 0.90
    - 0.95
  max_gap:
    - 0
  min_support_windows:
    - 1
    - 3
    - 5
    - 8
    - 10
  length_cap:
    - null
    - 60000
    - 100000
```

- [ ] **Step 4: Run tests**

```bash
python -m unittest tests.test_stage5_sweep_planner
python -m unittest discover -s tests
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_stage5_data_sweep.py tests/test_stage5_sweep_planner.py configs/sweeps/stage5_no9_decoder_v1.yaml
git commit -m "stage5: plan no-nine data sweep"
```

### Task 4: Run Stage 5A No-9 Baseline

**Files:**

- Produce: `experiments/stage5_no9_baseline_v1/`
- Produce: `experiments/stage5_no9_baseline_v1/summary.csv`
- Produce: `experiments/stage5_no9_baseline_v1/final_9genome.json`

- [ ] **Step 1: Plan baseline jobs**

Run locally:

```bash
python scripts/run_stage5_data_sweep.py \
  --config configs/sweeps/stage5_no9_data_v1.yaml \
  --variants baseline_antismash \
  --out-root experiments \
  --include-final-9genome
```

- [ ] **Step 2: Inspect generated command**

Open:

```bash
sed -n '1,240p' experiments/stage5_no9_data_v1-baseline_antismash/command.txt
```

Confirm:

- The command includes leakage checking before training.
- The command does not use any 9-genome path before final report.
- Final 9-genome evaluation is after all selection output is frozen.

- [ ] **Step 3: Execute on server**

Use the generated command on the assigned GPU node. Do not edit generated commands manually except for server-specific environment activation paths.

- [ ] **Step 4: Pull results and summarize**

Run:

```bash
python scripts/summarize_stage5_results.py \
  --experiments-root experiments \
  --sweep-id stage5_no9_data_v1 \
  --out experiments/stage5_no9_data_v1/summary.csv \
  --out-md experiments/stage5_no9_data_v1/summary.md
```

- [ ] **Step 5: Commit metadata only**

Do not commit checkpoints, per-window prediction CSVs, or large JSON files.

```bash
git add experiments/stage5_no9_data_v1/summary.csv experiments/stage5_no9_data_v1/summary.md
git commit -m "stage5: record no-nine baseline"
```

### Task 5: Run Stage 5B Label Sweep

**Files:**

- Produce: `experiments/stage5_no9_data_v1/summary.csv`
- Produce: `docs/stage5-no-nine-genome-tuning-results.md`

- [ ] **Step 1: Plan all label variants**

```bash
python scripts/run_stage5_data_sweep.py \
  --config configs/sweeps/stage5_no9_data_v1.yaml \
  --out-root experiments \
  --include-final-9genome
```

- [ ] **Step 2: Run leakage check across planned commands**

```bash
python scripts/check_no_nine_genome_leakage.py \
  --paths experiments/stage5_no9_data_v1-* \
  --embargo configs/splits/nine_genome_embargo_v1.json
```

Expected:

- Hits are allowed only in command sections explicitly labelled final 9-genome report.
- If the checker cannot distinguish final-report sections yet, add a `--allow-final-report` mode before running full jobs.

- [ ] **Step 3: Execute variants**

Recommended scheduling:

- Use one GPU per trial where possible.
- Run `baseline_antismash`, `core_only_v1`, and `core_flank_soft_v1` first.
- Only run `domain_clean_negative_v1`, `mibig_boost_v1`, `short_bgc_boost_v1`, and `hardpos_hardneg_v1` after label builder outputs pass row-count and class-balance sanity checks.

- [ ] **Step 4: Select winner using internal validation only**

Create `experiments/stage5_no9_data_v1/selection_decision.md` with:

- Internal-val table.
- Selected trial.
- Frozen decoder params.
- Statement that 9-genome was not used for selection.

- [ ] **Step 5: Run final 9-genome report for selected winner**

Only after writing `selection_decision.md`, run final 9-genome evaluation.

- [ ] **Step 6: Commit results summary**

```bash
git add experiments/stage5_no9_data_v1/summary.csv experiments/stage5_no9_data_v1/selection_decision.md docs/stage5-no-nine-genome-tuning-results.md
git commit -m "stage5: evaluate no-nine label sweep"
```

### Task 6: Decide Whether To Start Boundary-Aware Modeling

**Files:**

- Modify: `docs/stage5-no-nine-genome-tuning-results.md`
- Optional later create: `docs/stage6-boundary-aware-model.md`

- [ ] **Step 1: Check boundary metrics**

Start boundary-aware modeling only if:

- Stage 5B improves `bench_f1` but strict/IoU remains below BGC-Prophet or below `0.30`.
- Per-genome plots show long region overshoot rather than missed detection.

- [ ] **Step 2: Write a separate Stage 6 plan**

Do not mix boundary-head architecture changes into Stage 5B. Stage 5B is data/label-focused.

### Task 7: Add Length-Bucket Metrics And Small-BGC Sweep

**Files:**

- Create: `scripts/evaluate_length_bucket_metrics.py`
- Create: `tests/test_length_bucket_metrics.py`
- Create: `configs/sweeps/stage5_small_bgc_v1.yaml`
- Modify: `scripts/summarize_stage5_results.py`

- [ ] **Step 1: Write failing tests for bucket assignment**

Create `tests/test_length_bucket_metrics.py`:

```python
import unittest

from scripts.evaluate_length_bucket_metrics import assign_length_bucket, compute_bucket_metrics


class LengthBucketMetricsTests(unittest.TestCase):
    def test_assigns_gt_length_buckets(self):
        cases = [
            (4999, "<5kb"),
            (5000, "5-10kb"),
            (9999, "5-10kb"),
            (10000, "10-20kb"),
            (19999, "10-20kb"),
            (20000, "20-50kb"),
            (49999, "20-50kb"),
            (50000, ">=50kb"),
        ]
        for length, expected in cases:
            with self.subTest(length=length):
                self.assertEqual(assign_length_bucket(length), expected)

    def test_computes_recall_and_precision_by_bucket(self):
        gt = [("contig1", 100, 4100), ("contig1", 10000, 22000)]
        pred = [("contig1", 50, 4050), ("contig1", 10000, 18000)]
        out = compute_bucket_metrics(gt, pred)
        self.assertEqual(out["<5kb"]["n_gt"], 1)
        self.assertEqual(out["<5kb"]["recall_tp"], 1)
        self.assertEqual(out["<5kb"]["n_pred"], 1)
        self.assertEqual(out["<5kb"]["precision_tp"], 1)
        self.assertEqual(out["10-20kb"]["n_gt"], 1)
        self.assertEqual(out["5-10kb"]["n_pred"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing test**

```bash
python -m unittest tests.test_length_bucket_metrics
```

Expected:

```text
ImportError or ModuleNotFoundError for scripts.evaluate_length_bucket_metrics
```

- [ ] **Step 3: Implement the bucket metric script**

Create `scripts/evaluate_length_bucket_metrics.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, TypeAlias


BUCKETS = ("<5kb", "5-10kb", "10-20kb", "20-50kb", ">=50kb")
Region: TypeAlias = tuple[str, int, int]


def assign_length_bucket(length_bp: int) -> str:
    if length_bp < 5000:
        return "<5kb"
    if length_bp < 10000:
        return "5-10kb"
    if length_bp < 20000:
        return "10-20kb"
    if length_bp < 50000:
        return "20-50kb"
    return ">=50kb"


def _normalize_region(region: dict | list | tuple) -> Region:
    if isinstance(region, dict):
        return (str(region["contig"]), int(region["start"]), int(region["end"]))
    if isinstance(region, (list, tuple)) and len(region) == 3:
        return (str(region[0]), int(region[1]), int(region[2]))
    raise ValueError(f"Unsupported region format: {region!r}")


def _length(region: Region) -> int:
    return int(region[2]) - int(region[1])


def _overlap(a: Region, b: Region) -> int:
    if a[0] != b[0]:
        return 0
    return max(0, min(a[2], b[2]) - max(a[1], b[1]))


def _init_counts() -> dict[str, dict[str, int | float]]:
    return {
        bucket: {
            "n_gt": 0,
            "n_pred": 0,
            "recall_tp": 0,
            "precision_tp": 0,
            "recall": 0.0,
            "precision": 0.0,
        }
        for bucket in BUCKETS
    }


def compute_bucket_metrics(gt_regions: Iterable[dict | list | tuple], pred_regions: Iterable[dict | list | tuple]) -> dict:
    gt = [_normalize_region(region) for region in gt_regions]
    pred = [_normalize_region(region) for region in pred_regions]
    counts = _init_counts()

    for region in gt:
        bucket = assign_length_bucket(_length(region))
        counts[bucket]["n_gt"] += 1
        gt_len = max(1, _length(region))
        hit = any(_overlap(region, p) / gt_len >= 0.5 for p in pred)
        if hit:
            counts[bucket]["recall_tp"] += 1

    for region in pred:
        bucket = assign_length_bucket(_length(region))
        counts[bucket]["n_pred"] += 1
        pred_len = max(1, _length(region))
        hit = any(_overlap(region, g) / pred_len >= 0.5 for g in gt)
        if hit:
            counts[bucket]["precision_tp"] += 1

    for bucket, row in counts.items():
        row["recall"] = row["recall_tp"] / row["n_gt"] if row["n_gt"] else 0.0
        row["precision"] = row["precision_tp"] / row["n_pred"] if row["n_pred"] else 0.0

    return counts


def _load_regions(path: Path) -> list[dict | list | tuple]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "regions" in data:
        return data["regions"]
    if "ground_truth" in data:
        regions = []
        for genome_regions in data["ground_truth"].values():
            regions.extend(genome_regions)
        return regions
    if "predictions" in data:
        regions = []
        for genome_regions in data["predictions"].values():
            regions.extend(genome_regions)
        return regions
    raise ValueError(
        f"{path} must contain a list, 'regions', 'ground_truth', or 'predictions'"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    metrics = compute_bucket_metrics(_load_regions(args.gt), _load_regions(args.pred))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python -m unittest tests.test_length_bucket_metrics
python -m unittest discover -s tests
```

Expected:

```text
OK
```

- [ ] **Step 5: Add the small-BGC sweep config**

Create `configs/sweeps/stage5_small_bgc_v1.yaml` using the grid from Section 4.5. The config must include:

```yaml
selection_policy:
  selection_split: internal_val
  primary_metric: small_bgc_f1
  small_bgc_bucket: "<5kb"
  forbidden_selection_sources:
    - nine_genome
```

- [ ] **Step 6: Add summary integration**

Modify `scripts/summarize_stage5_results.py` so every Stage 5 report includes:

- `length_bucket_metrics.json`
- `<5kb_recall`
- `5-10kb_recall`
- `<5kb_pred_precision`
- `small_bgc_detector_role`, one of `none`, `secondary`, or `gated_ensemble`

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate_length_bucket_metrics.py tests/test_length_bucket_metrics.py configs/sweeps/stage5_small_bgc_v1.yaml scripts/summarize_stage5_results.py
git commit -m "stage5: add small-bgc bucket evaluation plan"
```

---

## 6. Reporting Template

Every Stage 5 result report must include:

```markdown
# Stage 5 No-9 Result: <candidate>

## Selection Inputs

- Train source:
- Internal validation source:
- 9-genome use before freeze: none
- Leakage checker command:
- Leakage checker result:

## Internal Selection Result

| trial | internal R | internal P | internal F1 | internal dedup F1 | selected |
|---|---:|---:|---:|---:|---|

## Final 9-Genome Result

| tool/candidate | R | P | F1 | dedup P | dedup F1 | strict F1 | iou F1 | n_pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BGC-Prophet | 0.3475 | 0.4529 | 0.3933 | 0.4471 | 0.3911 | 0.2863 | 0.2779 | 170 |
| 9-genome-tuned upper bound `defrag_v2_ms10_s099` | 0.4098 | 0.4714 | 0.4385 | 0.4714 | 0.4385 | 0.2921 | 0.2921 | 140 |
| Stage 5 candidate |  |  |  |  |  |  |  |  |

## Length-Bucket Result

| bucket | n_gt | GT recall | n_pred | pred precision |
|---|---:|---:|---:|---:|
| `<5kb` |  |  |  |  |
| `5-10kb` |  |  |  |  |
| `10-20kb` |  |  |  |  |
| `20-50kb` |  |  |  |  |
| `>=50kb` |  |  |  |  |

Small-BGC detector role: none / secondary / gated_ensemble

## Interpretation

- Did it beat BGC-Prophet without 9-genome tuning:
- Did it approach the 9-genome-tuned upper bound:
- Did it improve `GCA_000154945.1`:
- Did it improve `<5kb` and `5-10kb` recall:
- Did it reduce boundary overshoot:
- Is the result likely due to better model/data or decoder artifacts:
```

---

## 7. Guardrails For Claude Code

Claude Code should follow these rules:

- Do not run a new 9-genome decoder sweep.
- Do not inspect 9-genome per-genome results while deciding internal winners.
- Do not change Stage 5 candidate params after final 9-genome report.
- Do not commit large files from `tmp/`, checkpoints, prediction CSVs, or server output directories.
- Do not treat `/data/syh/OER0042/OER004256` as a labelled benchmark; it is a qualitative external-real-data pilot only.
- Do not claim `<5kb` BGC boundary performance from the main `min_support_windows=10` operating point; it is too coarse by design.
- Use `rg` to search for embargo IDs in generated configs before launching jobs.
- Keep code local and use server nodes only for execution.
- Use multiple GPUs for independent trials, but keep trial manifests deterministic.

---

## 8. Immediate Next Command Sequence

Run these locally first:

```bash
python -m unittest discover -s tests
python scripts/check_no_nine_genome_leakage.py \
  --paths configs src scripts docs \
  --embargo configs/splits/nine_genome_embargo_v1.json
```

Then generate Stage 5A baseline plans:

```bash
python scripts/run_stage5_data_sweep.py \
  --config configs/sweeps/stage5_no9_data_v1.yaml \
  --variants baseline_antismash \
  --out-root experiments \
  --include-final-9genome
```

Only after reviewing the generated manifest should server execution begin.
