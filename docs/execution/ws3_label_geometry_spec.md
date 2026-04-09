# WS3: Label and Geometry Redesign Specification

**Date:** 2026-04-09  
**Task ID:** D_ws3_shortbgc_label_geometry_spec  
**Dependencies:** WS0-T2 (metric contract), WS1-T2 (threshold contract), WS2-T2 (prior-aligned training results)  
**Write scope:** `scripts/stage1/05_windows_and_labels.py`, new geometry sweep scripts, label config

---

## 1. Current System: Why Short BGCs Are Unlearnable

### 1.1 The Mathematical Blind Spot

A window of size `W` with positive-label threshold `t` (minimum overlap ratio) can only label a BGC of length `L` as positive if:

```
L >= W * t
```

Under the current system:

- `W = 16,384 bp`
- `t = 0.5`
- Minimum labelable BGC length: `16,384 * 0.5 = 8,192 bp`

**Proof that short BGCs cannot produce positive windows:**

Consider a BGC of length `L < W * t`. Even in the best case, where the BGC is entirely contained within the window, the maximum overlap ratio is:

```
max_overlap_ratio = L / W
```

For this to satisfy the positive threshold:

```
L / W >= t
L >= W * t = 8,192 bp
```

Therefore, any BGC shorter than 8,192 bp **cannot** produce a positive window under any sliding-window alignment. This is not a model failure or a threshold problem. It is a structural impossibility in the label generation code.

For a BGC of length `L` where `L < W`, the maximum possible overlap ratio is `L / W`. Examples:

| BGC length | Max overlap ratio | Positive? (t=0.5) | Positive? (t=0.3) |
|------------|------------------|--------------------|---------------------|
| 3,000 bp   | 0.183            | No                 | No                  |
| 5,000 bp   | 0.305            | No                 | Barely              |
| 8,000 bp   | 0.488            | No                 | Yes                 |
| 10,000 bp  | 0.610            | Yes                | Yes                 |
| 12,000 bp  | 0.732            | Yes                | Yes                 |

### 1.2 Empirical Confirmation from Error Analysis

Server-side error analysis confirms this is not theoretical:

- **<5 kb BGCs:** 123/123 had **zero** positive windows (100% structurally unlearnable)
- **5-10 kb BGCs:** 135/144 had **zero** positive windows (93.8% structurally unlearnable)
- Combined: 258/267 short BGCs are invisible to training (96.6%)

This accounts for approximately 20% of all missed BGCs in the test_id evaluation.

### 1.3 The Ambiguous Window Trap

Even for BGCs that can produce positive windows, the boundary regions are problematic:

- Windows with 50-60% overlap: **62% miss rate**
- Windows with 60-80% overlap: miss rate drops
- Windows with 80-100% overlap: **28% miss rate**

The current ambiguous window policy excludes from training all windows with:
- Overlap > 0 but overlap ratio < 0.5 (labeled ambiguous)
- Distance to nearest BGC < 10 kb but no overlap (labeled ambiguous)

At deployment, these windows must be predicted on. The model has never seen anything like them during training, so it has no learned behavior for the most difficult regime.

### 1.4 Contig Tail Dropout

Current code (line 116 of `05_windows_and_labels.py`):

```python
while pos + W <= clen:
```

This drops any contig tail shorter than `W = 16,384 bp`. A BGC near the contig end can lose its only viable windows entirely. This disproportionately affects truncated BGCs that are already at contig boundaries.

---

## 2. Proposed Label Rule Changes

### 2.1 Option A: Length-Adaptive Overlap Threshold

**Description:** Lower the positive-label overlap threshold for windows overlapping short BGCs. If a BGC has length `L < W`, the positive threshold becomes `min(t, L / W - epsilon)` so that at least the best-aligned window can be labeled positive.

Concretely, for a BGC of length `L`:
- If `L >= W * t`: use current threshold `t = 0.5` (no change)
- If `L < W * t`: set positive threshold to `max(0.2, L / W * 0.8)` so the BGC's best-overlap window qualifies

Implementation in `05_windows_and_labels.py`: when computing `max_or` for a window, also check whether any overlapping BGC is short (`L < W * t`). If so, apply the relaxed threshold for that BGC.

**Expected impact:**
- Recovers positive supervision for ~258 currently invisible BGCs
- Short BGC recall should improve from ~0% to a nonzero baseline
- Enables the model to learn any signal present in short-BGC windows

**Risk:**
- Dilutes positive label quality: windows with only 20-30% BGC content are noisier supervision
- May increase false positive rate if the model learns to call low-overlap regions as BGC
- Partial mitigation: use soft labels (Option C) instead of hard positive

**Compute cost:** Negligible (label generation only)

### 2.2 Option B: BGC-Centered Window Injection

**Description:** For each BGC shorter than `W * t`, add one or more BGC-centered windows (where the BGC is placed at the center of the window) as explicit positive training samples, in addition to the regular sliding-window grid.

For a BGC at position `[start, end]` with length `L`:
- Center window: `[center - W/2, center + W/2]` where `center = (start + end) / 2`
- If the centered window falls outside the contig, clamp to contig boundaries and pad with N's

These injected windows are marked with `source = "bgc_centered"` in metadata for tracking.

**Expected impact:**
- Guarantees at least one window per short BGC with maximum possible overlap
- Preserves the original sliding-window grid (no disruption to negative/ambiguous labels)
- Easier to control: injected windows can be weighted separately in the loss

**Risk:**
- Introduces non-uniform window placement, which breaks the assumption that windows are i.i.d. from a contig scan
- At deployment, the model still uses a regular grid, so it must generalize from centered placement to arbitrary alignment
- The number of injected windows is small (~260), which may be insufficient for learning without oversampling
- Partial mitigation: combine with stride reduction (Section 3.2) so regular grid also captures more short-BGC windows

**Compute cost:** Negligible for label generation. Minor increase in training set size (~260 windows out of ~500k+).

### 2.3 Option C: Soft Labels for Partial Overlap

**Description:** Replace the hard positive/ambiguous/negative trichotomy with a continuous label. Each window gets a label weight proportional to its overlap ratio with the highest-overlapping BGC.

Proposed label mapping:

```
overlap_ratio >= 0.5  -->  label_weight = 1.0  (current positive)
0.2 <= overlap < 0.5  -->  label_weight = overlap_ratio  (currently ambiguous, now soft positive)
0.0 < overlap < 0.2   -->  label_weight = overlap_ratio * 0.5  (down-weighted soft positive)
overlap == 0, dist >= 10kb  -->  label_weight = 0.0, is_negative = 1.0  (current negative)
overlap == 0, dist < 10kb   -->  label_weight = 0.0, excluded or is_negative = dist/10kb
```

Training loss modification: use `label_weight` as per-sample weight in the BCE loss, or use it directly as a soft target instead of `{0, 1}`.

**Expected impact:**
- Removes the hard ambiguous exclusion zone
- The model sees the full spectrum of BGC proximity during training
- Boundary windows get reduced but nonzero gradient signal
- Short BGCs with overlap ratio 0.2-0.5 become soft positives instead of invisible

**Risk:**
- Soft labels require loss function changes beyond `05_windows_and_labels.py` (must touch `train.py`, `dataset.py`)
- Calibration becomes harder: the model must learn a continuous mapping, not a binary one
- Risk of regression on currently well-classified interior windows if soft labels blur the decision boundary
- Partial mitigation: only apply soft labels to windows that are currently ambiguous; keep hard labels for clear positives and negatives

**Compute cost:** Negligible for label generation. Training may be slower to converge due to noisier gradients.

### 2.4 Option D: Dual-Threshold Type Assignment

**Description:** The current code uses `overlap_type = 0.3` to assign BGC types to positive windows. This means a window can be labeled positive (overlap >= 0.5) but get type labels from a different, lower-overlap BGC (overlap >= 0.3). This is a reasonable design, but if the positive threshold is lowered for short BGCs, the type threshold should track accordingly.

Proposed rule: `overlap_type = max(0.15, overlap_pos * 0.6)` so it stays proportional.

**Expected impact:** Prevents type label noise when positive threshold is lowered.

**Risk:** Minor. Only affects type assignment, not detection.

---

## 3. Proposed Geometry Changes

### 3.1 Smaller Window Size (8,192 bp)

**Description:** Reduce window size from 16,384 to 8,192 bp. This halves the minimum labelable BGC length from 8,192 bp to 4,096 bp.

| BGC length | Max overlap (W=16k) | Max overlap (W=8k) |
|------------|---------------------|---------------------|
| 3,000 bp   | 0.183               | 0.366               |
| 5,000 bp   | 0.305               | 0.610               |
| 8,000 bp   | 0.488               | 0.976               |

**Expected impact:**
- 5-10 kb BGCs become fully labelable at t=0.5
- 3-5 kb BGCs become labelable at t=0.3-0.4
- Doubles the number of windows per genome (at same stride ratio)
- All window-level metrics change because the unit of analysis changes

**Risk:**
- **Critical:** Requires re-running Stage 2 embedding extraction for all genomes. This is the most expensive step in the pipeline (~50 GPU-hours on 8 GPUs). The entire existing embedding cache becomes invalid.
- Evo2 context length at 8,192 tokens may produce weaker representations than at 16,384. Evo2 was designed for long-range context; halving the window may lose relevant genomic context for large BGCs.
- Large BGCs (>50 kb) that span many windows become even more fragmented, potentially hurting their recall.
- Window-level metrics are not directly comparable across window sizes. All baselines must be re-established.

**Compute cost:** Very high. Full re-extraction of Stage 2 embeddings.

### 3.2 Smaller Stride (4,096 bp)

**Description:** Reduce stride from 8,192 to 4,096 bp while keeping window size at 16,384 bp. This gives 75% overlap between adjacent windows instead of 50%.

**Expected impact:**
- Doubles the number of windows, giving more diverse alignments of windows relative to BGC boundaries
- For a short BGC of length `L`, the number of windows with nonzero overlap increases: a BGC gets `ceil((L + W) / S) - 1` overlapping windows
- More windows near BGC boundaries means more gradient signal in the boundary regime
- At deployment, denser scanning produces smoother score profiles for region calling

**Risk:**
- Doubles the training set size. Training time and memory increase proportionally.
- Highly correlated adjacent windows (75% shared sequence) may reduce effective sample diversity
- The overlap ratio distribution shifts: more windows with moderate overlap, which may increase ambiguous-zone pressure
- Does **not** by itself solve the minimum-labelable-length problem (still need `L >= W * t`)

**Compute cost:** Moderate. No re-extraction needed (windows are subsequences of existing embeddings if stored at contig level; otherwise extraction doubles). Training cost doubles.

**Note on implementation:** Current Stage 2 stores one embedding per window. Changing stride requires either (a) re-running extraction with new stride, or (b) switching to contig-level embedding storage with runtime windowing. Option (b) is better long-term but requires architectural change.

### 3.3 Multi-Scale Windows

**Description:** Use two window sizes simultaneously: 16,384 bp (for large BGCs and global context) and 8,192 bp (for short BGCs). Each BGC is labeled using whichever scale gives the best coverage.

**Expected impact:**
- Preserves full Evo2 context for large BGCs while gaining short-BGC coverage
- Enables scale-aware region calling at Stage 4

**Risk:**
- Requires dual Stage 2 extraction (very expensive)
- Training must handle heterogeneous input dimensions or separate forward passes
- Significantly increases system complexity for uncertain gain over simpler alternatives
- Model architecture may need modification to handle multi-scale inputs

**Compute cost:** Very high. Full dual extraction plus training complexity.

**Recommendation:** Defer. Try label rule changes (Section 2) first. If short-BGC recall remains insufficient after label changes, reconsider.

### 3.4 Edge Padding at Contig Ends

**Description:** When the remaining contig tail is shorter than `W`, pad with N characters to form a complete window instead of dropping the tail.

Implementation:

```python
# Current: drops short tails
while pos + W <= clen:
    ...

# Proposed: pad last window
while pos < clen:
    actual_end = min(pos + W, clen)
    # If actual_end - pos < W, this window is N-padded
    ...
```

The embedding extraction must also handle N-padded sequences (Evo2 tokenizer should already handle N's).

**Expected impact:**
- Recovers contig-terminal BGCs that currently lose their last windows
- Particularly benefits truncated BGCs at contig boundaries
- Small increase in total window count (one extra window per contig at most)

**Risk:**
- N-padded windows produce different embeddings than full-sequence windows; the model must handle this distribution shift
- The pad ratio varies per window, introducing input heterogeneity
- If pad ratio is large (e.g., only 2 kb of real sequence), the embedding is mostly noise
- Partial mitigation: only pad if the real-sequence portion is >= `W/2` (8,192 bp)

**Compute cost:** Negligible.

---

## 4. Ambiguous Window Handling Strategy

### 4.1 Current Problem

The current system creates three exclusive categories:

| Category  | Training | Stage 4 Evaluation | Deployment |
|-----------|----------|-------------------|------------|
| Positive  | Included | Included          | Predicted  |
| Negative  | Included | Included          | Predicted  |
| Ambiguous | Excluded | Excluded          | **Must predict** |

This means at deployment, the model must handle a window distribution it has never trained on and never been evaluated on. The ambiguous zone is exactly the BGC-boundary regime where errors matter most.

### 4.2 Proposed Strategy: Graduated Inclusion

**Phase 1 (immediate, no retraining):**

Include ambiguous windows in Stage 4 evaluation. Predict on them using the existing model and report metrics with and without ambiguous windows side by side. This gives a true full-window deployment approximation.

Implementation: modify `run_stage4.py` to include ambiguous windows in genome-level calling. Add a `--include-ambiguous` flag. Report both reduced-proxy and full-window metrics.

**Phase 2 (with retraining, after label rule changes):**

If soft labels (Option C in Section 2.3) are adopted, ambiguous windows become soft positives or soft negatives and are included in training naturally. The ambiguous category ceases to exist.

If soft labels are not adopted, include ambiguous windows as negatives with reduced loss weight:

```
ambiguous window near BGC (overlap > 0, overlap_ratio < 0.5):
    loss_weight = 0.3
    target = 0 (negative) if overlap_ratio < 0.2
    target = excluded if 0.2 <= overlap_ratio < 0.5

ambiguous window far from BGC (no overlap, dist < 10kb):
    loss_weight = 0.5
    target = 0 (negative)
```

**Phase 3 (deployment):**

At deployment, all windows are predicted. The region caller must handle the full score distribution including boundary windows. This is not a label question but an aggregation question (see WS5).

### 4.3 Expected Impact

- Phase 1: No training impact, but gives honest evaluation numbers. May reveal that current recall is lower than reduced-proxy suggests.
- Phase 2: Model learns behavior for the boundary regime. Expected to reduce the 62% miss rate on 50-60% overlap windows.
- Phase 3: Region calling on full windows should produce more stable boundaries.

### 4.4 Risk

- Including ambiguous windows as negatives could teach the model to suppress scores near BGC boundaries, making region calling harder
- Soft labels avoid this but add training complexity
- Phase 1 evaluation numbers may look worse than current numbers, which is correct but requires careful communication

---

## 5. Minimum Viable Experiment

### 5.1 Core Hypothesis

The primary hypothesis is:

> Short BGCs are missed because the label generation code structurally cannot produce positive windows for them, not because the model cannot learn short-BGC signal.

### 5.2 Smallest Test

The smallest change that tests this hypothesis is **Option A (length-adaptive overlap threshold)** combined with **Phase 1 ambiguous inclusion in evaluation**.

**Concretely:**

1. Modify `05_windows_and_labels.py` to apply a relaxed positive threshold for windows overlapping short BGCs:
   - For each overlapping BGC with length `L < W * t`, set effective threshold to `max(0.2, L / W * 0.8)`
   - Keep all other logic unchanged
2. Re-generate `windows_metadata.csv` with the new label rules
3. Count how many previously-invisible short BGCs now have at least one positive window
4. Retrain the model with identical hyperparameters on the new labels
5. Evaluate on test_id with the canonical threshold contract from WS1-T2
6. Report short-BGC recall (by length bucket) before and after

**What this tests:**
- If short-BGC recall improves materially (>= +0.10 absolute on the <10 kb bucket), the hypothesis is confirmed and the structural blind spot was real
- If short-BGC recall does not improve, the issue is deeper (the model cannot learn short-BGC signal from partial-overlap windows, or the embedding is insufficient)

**What this does NOT test:**
- Whether soft labels are better than hard labels
- Whether geometry changes help
- Whether ambiguous window training helps overall recall

### 5.3 Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Short-BGC (<10 kb) recall | ~0.03 | >= 0.15 |
| Overall test_id recall | 0.668 (post-WS1/WS2) | >= 0.68 (no regression) |
| Overall test_id precision | baseline | no material regression |
| Number of short BGCs with >= 1 positive window | 9/267 | >= 200/267 |

### 5.4 Failure Mode

If the MVE fails (short-BGC recall does not improve despite positive windows being generated), the next step is:
- Check whether the newly-positive windows actually have detectable BGC signal by examining their embedding distances to known-positive windows
- If embedding-level signal is absent, this becomes a representation problem (escalate to WS7)
- If embedding-level signal is present but the classifier misses, check calibration and loss weighting

---

## 6. Implementation Task Breakdown

### Task 1: Label Code Audit (0.5 day)

**Goal:** Fully understand the current label assignment path and enumerate every edge case.

**Inputs:** `scripts/stage1/05_windows_and_labels.py`

**Outputs:**
- Annotated list of all label decision paths
- Count of windows in each category from the current run
- Distribution of BGC lengths in the training set

**Verification:** Window counts match the existing `windows_metadata.csv`

### Task 2: Short-BGC Positive Window Recoverability Analysis (0.5 day)

**Goal:** Simulate the label change without re-running Stage 2. Count how many short BGCs would gain positive windows under different threshold relaxation schemes.

**Inputs:** Current `windows_metadata.csv`, `bgc_annotations.csv`

**Outputs:**
- Table: for each threshold variant (t=0.5, 0.4, 0.3, 0.2, length-adaptive), number of BGCs with >= 1 positive window by length bucket
- Recommended threshold variant

**Verification:** Numbers are reproducible from the CSV files alone

### Task 3: Implement Length-Adaptive Threshold in Label Code (0.5 day)

**Goal:** Modify `05_windows_and_labels.py` to support length-adaptive positive threshold.

**Implementation detail:** Add a `--short-bgc-adaptive` flag. When enabled:
- For each window-BGC overlap, compute the BGC length
- If BGC length < `W * overlap_pos`, use `effective_threshold = max(0.2, bgc_length / W * 0.8)`
- Mark these windows with `label_source = "short_bgc_adaptive"` in metadata for tracking

**Outputs:** Modified `05_windows_and_labels.py` with backward compatibility (default off)

**Verification:** Running with `--short-bgc-adaptive` produces more positive windows than without, and the difference is concentrated in the <10 kb BGC bucket

### Task 4: Re-Generate Labels (0.5 day)

**Goal:** Produce a new `windows_metadata.csv` with the adaptive threshold.

**Inputs:** Existing genome FASTA files, BGC annotations

**Outputs:** New `windows_metadata.csv` with label statistics summary

**Verification:**
- Total window count unchanged (geometry is unchanged)
- Positive count increases by roughly the number of short-BGC windows recovered
- Negative count unchanged
- Ambiguous count decreases (some ambiguous windows promoted to positive)

### Task 5: Retrain with New Labels (1 day)

**Goal:** Train one model with identical hyperparameters on the new labels.

**Inputs:** New `windows_metadata.csv`, existing Stage 2 embeddings

**Outputs:** One checkpoint

**Verification:** Training converges comparably to the baseline run

### Task 6: Short-BGC Recall Evaluation (0.5 day)

**Goal:** Evaluate the new model specifically on short-BGC recall by length bucket.

**Inputs:** New checkpoint, test_id split, BGC annotations with length

**Outputs:**
- Recall by length bucket: <5 kb, 5-10 kb, 10-20 kb, 20-50 kb, >50 kb
- Overall test_id recall and precision
- Comparison table: old labels vs new labels

**Verification:** Uses canonical threshold contract from WS1-T2

### Task 7: Ambiguous Window Evaluation Patch (0.5 day)

**Goal:** Extend Stage 4 evaluation to include ambiguous windows.

**Inputs:** `scripts/stage4/run_stage4.py`

**Outputs:** Modified `run_stage4.py` with `--include-ambiguous` flag. Reports both reduced-proxy and full-window metrics side by side.

**Verification:** Reduced-proxy metrics match existing numbers when flag is off

### Task 8: Edge Padding Implementation (0.5 day)

**Goal:** Implement optional contig-tail padding in the label code.

**Implementation:** Add `--pad-tail` flag. When enabled:
- If remaining tail >= `W/2`, create one padded window
- Mark padded windows with `is_padded = 1` in metadata
- Report how many additional windows are created

**Outputs:** Modified `05_windows_and_labels.py`

**Verification:** Padded window count is bounded by number of contigs. No change to existing windows.

### Task 9: Decision Report (0.5 day)

**Goal:** Synthesize MVE results and recommend next steps.

**Inputs:** Outputs from Tasks 6 and 7

**Outputs:** One short decision report answering:
- Did the length-adaptive threshold improve short-BGC recall?
- Did overall recall/precision regress?
- Should we proceed to soft labels (Option C)?
- Should we proceed to geometry changes (Section 3)?
- Is there evidence of representation limitation for short BGCs?

**Verification:** Conclusions are justified by the reported numbers

### Total Estimated Timeline: 4-5 days

**Execution order:**
```
Task 1 ──> Task 2 ──> Task 3 ──> Task 4 ──> Task 5 ──> Task 6 ──> Task 9
                                                    \
                                          Task 7 ───> Task 9
                                          Task 8 (parallel, low priority)
```

---

## 7. Expected Outcomes

### 7.1 Length-Adaptive Threshold (Option A, the MVE)

| Metric | Before | Expected After | Basis for Estimate |
|--------|--------|----------------|-------------------|
| Short BGCs (<10 kb) with >= 1 positive window | 9/267 (3.4%) | ~220/267 (82%) | All BGCs with L >= 0.2 * W = 3,277 bp gain windows |
| Short BGC recall (test_id) | ~0.03 | 0.10-0.25 | Having positive windows is necessary but not sufficient; model must actually learn the signal |
| Overall test_id recall | 0.668 | 0.68-0.72 | Short BGCs are ~8% of test set; recovering 20% of them adds ~0.016 absolute |
| Overall test_id precision | baseline | -0.02 to +0.01 | Minor risk of increased FP from noisier positive labels |

### 7.2 Soft Labels (Option C, if MVE succeeds)

| Metric | Before | Expected After | Basis for Estimate |
|--------|--------|----------------|-------------------|
| 50-60% overlap window miss rate | 62% | 40-50% | Model trains on this regime instead of ignoring it |
| Overall test_id recall | post-MVE | +0.02 to +0.05 | Boundary windows contribute to region recovery |
| Overall test_id precision | post-MVE | -0.03 to +0.01 | Boundary learning could improve or hurt precision |

### 7.3 Stride Reduction (if pursued later)

| Metric | Before | Expected After | Basis for Estimate |
|--------|--------|----------------|-------------------|
| Windows per genome | N | 2N | Direct consequence of halved stride |
| Training time | T | ~2T | Linear scaling with dataset size |
| Score profile smoothness | baseline | improved | Denser sampling produces smoother contig-level score curves |
| Region boundary precision | baseline | +minor | More windows near boundaries means more evidence for boundary placement |

### 7.4 Edge Padding (if pursued)

| Metric | Before | Expected After | Basis for Estimate |
|--------|--------|----------------|-------------------|
| Truncated BGC recall | baseline | +0.02 to +0.05 | Recovers terminal BGCs that lose their last window |
| Added windows | 0 | ~1 per contig (bounded) | Only long-enough tails are padded |

---

## 8. Risks and Mitigations

### 8.1 Noisy Positive Labels Degrade Overall Performance

**Risk:** Lowering the positive threshold introduces windows with only 20-30% BGC content as positives. These windows are 70-80% non-BGC DNA. If the model treats them as equivalent to high-overlap positives, it may learn to call non-BGC regions as positive.

**Severity:** Medium

**Mitigation:**
- Use loss weighting proportional to overlap ratio (e.g., weight = overlap_ratio for newly-positive windows)
- Track `label_source` so the training pipeline can apply differential weighting
- Monitor per-overlap-bucket precision in evaluation, not just overall precision
- If precision drops > 0.05, revert to BGC-centered injection (Option B) which is more targeted

### 8.2 Short-BGC Signal May Not Exist in Frozen Evo2 Embeddings

**Risk:** Evo2 mean-pooled embeddings of 16 kb windows may not preserve enough local structure to distinguish a 3 kb BGC from background DNA. The blind spot may be a representation problem, not just a label problem.

**Severity:** Medium-high. This would mean the MVE fails even after label changes.

**Mitigation:**
- Task 2 (recoverability analysis) should include an embedding-space check: measure cosine distance between newly-positive short-BGC windows and known-positive long-BGC windows
- If short-BGC embeddings are indistinguishable from negatives, this is a representation ceiling, and the correct response is to escalate to WS7 (representation upgrade) rather than continuing label experiments
- This is explicitly the "failure mode" branch described in Section 5.4

### 8.3 Retraining Cost Is Wasted If Label Change Is Wrong

**Risk:** Full retraining takes significant GPU time. If the label change does not help, the compute is wasted.

**Severity:** Low. A single retraining run is ~4-8 GPU-hours, which is affordable.

**Mitigation:**
- Task 2 provides a cheap pre-check (no GPU needed) to estimate whether the label change can possibly help
- Only proceed to Task 5 if Task 2 shows meaningful recoverability improvement

### 8.4 Ambiguous Window Inclusion Reveals Worse-Than-Expected Baseline

**Risk:** Including ambiguous windows in evaluation (Task 7) may show that current recall is significantly lower than the reduced-proxy suggests. This is correct but could be discouraging.

**Severity:** Low (this is information, not a problem)

**Mitigation:**
- Frame full-window evaluation as the honest baseline, not as regression
- Report both reduced-proxy and full-window side by side per the WS4-T1 spec
- This is exactly what the execution roadmap requires for promotion-grade evaluation

### 8.5 Soft Labels Break Existing Training Pipeline

**Risk:** The current training pipeline (`bgc-gcd/train.py`, `bgc-gcd/data/dataset.py`) assumes binary labels. Soft labels require changes to the loss function, data loading, and possibly the sampler.

**Severity:** Medium

**Mitigation:**
- Soft labels (Option C) are Phase 2, not the MVE. The MVE uses hard labels with a relaxed threshold, which requires no training pipeline changes.
- When implementing soft labels, make changes backward-compatible: soft labels degrade to hard labels when all weights are 0 or 1
- Test soft label training on a small subset before full run

### 8.6 Geometry Changes Invalidate Existing Embeddings

**Risk:** Any window size change requires re-running Stage 2, which is the most expensive step (~50 GPU-hours).

**Severity:** High for window size changes. Zero for stride changes (if contig-level embeddings are stored) or moderate (if per-window embeddings are stored).

**Mitigation:**
- The MVE explicitly avoids geometry changes
- Geometry changes (Section 3) are deferred until label-only results are available
- If geometry changes become necessary, batch them with any other pipeline changes that require re-extraction
- Consider switching to contig-level embedding storage to decouple extraction from windowing

### 8.7 Short-BGC Recovery Hurts Novel Candidate Quality

**Risk:** Recovering short BGCs may increase the number of short predicted regions that overlap with known antiSMASH annotations but do not contribute to novel discovery. In the worst case, the model becomes better at recovering known short BGCs while losing novel-candidate ranking quality.

**Severity:** Low

**Mitigation:**
- Per the discovery-first rule (execution roadmap Section 3.3), every evaluation must report both safety and discovery metrics
- If novel ranking degrades materially at fixed budget, the label change must be tuned, not blindly shipped
