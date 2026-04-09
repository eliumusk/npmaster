# Current Mainline Summary

**Date:** 2026-04-09  
**Purpose:** concise project-state summary before architecture redesign discussion

---

## 1. Official Mainline

The current official mainline is:

- **Config:** `H_hardneg_r10`
- **Backbone:** frozen Evo2 embeddings
- **Classifier family:** current window classifier, no architecture change
- **Detection score:** `score_maxtype = max(sigmoid(type_logits))`
- **Evaluation mode:** full-window
- **Training changes accumulated so far:**
  - score contract fixed
  - threshold contract fixed
  - prior-aligned sampling (`pos_ratio=0.15`)
  - `cap=0.8` label rule
  - edge padding
  - hard-negative upsampling (`hard_neg_ratio=0.10`)

Canonical references:

- [p2_phase_summary.md](phase2-ood-robustness.md)
- [stability_confirmation.md](stability-3seed.md)
- [train.py](/Users/musk/Documents/code/np-master/bgc-gcd/train.py)
- [sampler.py](/Users/musk/Documents/code/np-master/bgc-gcd/data/sampler.py)

---

## 2. What The Numbers Mean

All "known-BGC" numbers below are **recovery against antiSMASH annotations**, not a gold-standard head-to-head benchmark against antiSMASH itself.

So the right interpretation is:

- higher recall = we recover more antiSMASH-annotated BGCs
- higher prediction-level precision = more of our predicted regions overlap antiSMASH BGCs

This is still the right safety benchmark for now, but it is not yet a final claim of "beating antiSMASH."

---

## 3. Key antiSMASH-Recovery Results

### 3.1 Original default Stage 4

At the original default operating point:

- `test_id` recall: `0.2707`
- `test_ood` recall: `0.2132`
- precision at that stage was reported under the old antiSMASH-count-based metric:
  - `test_id`: `0.5729`
  - `test_ood`: `0.8901`

Reference:

- [2026-04-08-bgc-system-redesign.md](/Users/musk/Documents/code/np-master/docs/2026-04-08-bgc-system-redesign.md#L105)

### 3.2 Current official mainline

`H_hardneg_r10`, full-window, `score_maxtype`, 3-seed mean ± std:

| Metric | `test_id` | `test_ood` |
|---|---:|---:|
| Recall | **0.731 ± 0.012** | **0.566 ± 0.041** |
| Prediction-level precision | **0.650 ± 0.032** | **0.531 ± 0.080** |
| `<5kb` recall | 0.222 ± 0.005 | 0.171 ± 0.014 |
| `5-10kb` recall | 0.352 ± 0.022 | 0.380 ± 0.009 |
| Truncated recall | 0.636 ± 0.011 | 0.555 ± 0.030 |

Reference:

- [stability_confirmation.md](stability-3seed.md)

### 3.3 Cumulative gain from the original system

On `test_id`:

- recall improved from `0.271` to `0.731`
- absolute gain: **`+0.460`**

On `test_ood`:

- recall improved from `0.213` to `0.566`
- absolute gain: **`+0.353`**

This entire gain was obtained with **zero architecture changes**.

---

## 4. Discovery-Side Results

Current discovery picture:

- `F_cap08` had the strongest top-end purity:
  - signature precision@20: `95%`
  - signature precision@50: `92%`
- `G_edgepad` preserved `@50` quality while improving short-BGC coverage:
  - signature precision@20: `90%`
  - signature precision@50: `92%`
- `H_hardneg_r10` remained acceptable while improving robustness:
  - signature precision@20: `90%`
  - signature precision@50: `86%`
  - signature diversity@50: `38`

Interpretation:

- discovery quality did **not collapse** as the safety side improved
- current mainline is more selective than `G_edgepad`
- some top-50 purity was traded for better OOD robustness and higher diversity

References:

- [p2_phase_summary.md](phase2-ood-robustness.md#L81)
- [ws_discovery_benchmark.md](discovery-benchmark.md)

---

## 5. What Has Been Proven

The project has now established five points clearly.

1. **The original low performance was not mainly an architecture failure.**  
   A large part of the gap came from objective mismatch, threshold mismatch, prior shift, and label/input-coverage defects.

2. **The current route is scientifically real.**  
   Discovery-side candidates remain enriched for biosynthetic signature domains, so the model is not just learning trivial background composition.

3. **Short-BGC coverage was a real bottleneck.**  
   Edge padding moved `<5kb` recall from effectively zero to about `22%`, and `5-10kb` recall to about `35%`.

4. **OOD false positives were also a real bottleneck.**  
   Hard-negative training reduced OOD unmatched predictions strongly while keeping most recall.

5. **The current mainline is reproducible.**  
   `test_id` results are stable across 3 seeds; `test_ood` still has moderate variance, but central tendency is clear.

---

## 6. What Is Still Not Good Enough

The project is in a much better state, but it is not finished.

The main unresolved problems are:

1. **Known-BGC recovery is improved but still incomplete.**
   - `test_id` still misses about `27%` of antiSMASH-annotated BGCs
   - `test_ood` still misses about `43%`

2. **Very short BGCs remain difficult.**
   - `<5kb` recall is only about `22%`
   - edge padding fixed visibility, but not full separability

3. **OOD robustness is better but not fully settled.**
   - `test_ood` precision is much better than the edge-padding-only stage
   - but seed variance remains noticeably higher than in `test_id`

4. **We still cannot claim parity with antiSMASH.**
   The project can now claim a credible and much stronger system, but not "antiSMASH matched" in a strict sense.

---

## 7. Why It Is Reasonable To Start Architecture Discussion Now

It is reasonable to start architecture discussion **now**, because the main pre-architecture confounders have been substantially reduced:

- score contract aligned
- threshold contract aligned
- train/deploy prior mismatch reduced
- short-BGC coverage defect partly fixed
- OOD hard-negative deficiency partly fixed

That means future failure is more likely to reflect a real limitation of the current representation stack rather than a simple pipeline mistake.

This does **not** mean the current architecture is already proven to be the bottleneck.

It means:

**we now have a clean enough baseline that architecture work can be discussed seriously and judged fairly.**

---

## 8. Current Decision Boundary

The current project status should be stated as:

- **Official mainline:** `H_hardneg_r10`
- **Current claim:** strong recovery and discovery baseline, not yet antiSMASH parity
- **Current bottlenecks:** very-short BGCs, remaining OOD instability, likely boundary/local-structure limits
- **Next strategic topic:** architecture redesign

Recommended framing for the next discussion:

1. What failure modes remain after alignment, coverage, and robustness fixes?
2. Which of those failure modes are plausibly caused by global mean pooling and loss of position information?
3. What is the smallest architecture change that targets those failure modes directly?

---

## 9. Code/Artifact Map For The Next Discussion

Use these as the minimum set of canonical materials:

- [train.py](/Users/musk/Documents/code/np-master/bgc-gcd/train.py)
- [sampler.py](/Users/musk/Documents/code/np-master/bgc-gcd/data/sampler.py)
- [ws_checkpoint_selection_contract.md](docs/_archive/ws_execution_history/ws_checkpoint_selection_contract.md)
- [ws1_t2_threshold_contract.md](threshold-contract.md)
- [p1_phase_summary.md](phase1-short-bgc-coverage.md)
- [p2_phase_summary.md](phase2-ood-robustness.md)
- [stability_confirmation.md](stability-3seed.md)

These are enough to start the architecture redesign discussion without going back through the entire execution history.
