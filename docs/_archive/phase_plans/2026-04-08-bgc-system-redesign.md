# BGC Discovery System Redesign Memo

**Date:** 2026-04-08  
**Project:** `np-master`  
**Status:** Draft for multi-party review

---

## 1. Executive Summary

This memo redefines the project from the top down.

The current project should not be framed as "window classification with good AUROC." Its real task is:

1. detect as many biologically meaningful BGC regions as possible on whole genomes,
2. maintain acceptable precision on known BGCs,
3. produce credible antiSMASH-missed candidates for downstream validation.

At present, even the best available Stage 4 evidence should still be read as evidence on a **reduced deployment proxy**, because ambiguous boundary windows are excluded from key parts of the current evaluation stack.

Under that framing, the current system has produced **strong but still preliminary evidence that the novel-candidate path is real**, but it has **not yet achieved acceptable known-BGC coverage relative to antiSMASH annotations**.

The most important conclusion is:

- the overall direction is still valid,
- the current system is not yet optimizing the final task directly,
- the immediate priority is **system alignment**, not an immediate switch to LoRA.

LoRA or other backbone changes may become necessary later, but the current evidence does **not** justify treating representation learning as the first bottleneck. Several higher-level design mismatches must be corrected first.

---

## 2. Root Task Definition

### 2.1 Primary Objective

Build a genome-scale BGC discovery system that is useful for **novel candidate discovery**, not merely for reproducing antiSMASH.

The system should satisfy two constraints at the same time:

- **Known-BGC coverage constraint:** antiSMASH-annotated BGCs should not be missed at an unacceptable rate. This is a safety constraint, not the final scientific north star.
- **Novel-discovery constraint:** the system should identify antiSMASH-missed regions that are enriched for real biosynthetic evidence and rank them well under fixed downstream validation budget. This is the primary scientific objective.

These two constraints are both required. A system that only recovers known BGCs is not enough. A system that only emits exciting novel candidates but misses most known BGCs is also not enough.

### 2.2 Operational Task

At deployment time, the task is:

1. scan a genome,
2. score local regions for BGC-likeness,
3. convert local scores into continuous candidate regions,
4. separate likely known-like regions from likely antiSMASH-missed regions,
5. rank candidates for manual or orthogonal validation.

This is a **region discovery** problem, not a pure window classification problem.

Importantly, the current implementation only approximates this task imperfectly. In particular, the current Stage 4 pipeline excludes ambiguous windows from threshold selection and genome-level calling, so the present evidence does **not** yet represent a full-window approximation to whole-genome deployment.

### 2.3 What Success Looks Like

The project should be evaluated on three axes, with the first axis treated as the primary discovery objective and the latter two treated as required safeguards:

1. **Novel candidate quality**
   - fixed-budget ranking quality
   - enrichment for biosynthetic Pfam/domain evidence
   - support from independent tools where applicable
   - distance/boundary analysis to distinguish new candidates from boundary expansion
   - explicit separation between clean antiSMASH-missed candidates and merely unmatched predictions
   - de-duplicated family diversity and coverage across phyla/types
   - incremental yield relative to antiSMASH, GECCO, and DeepBGC under the same validation budget
2. **Known BGC coverage**
   - region-level recall against antiSMASH annotations
   - especially on short BGCs, truncated BGCs, and OOD phyla
3. **Known BGC quality**
   - region-level precision, with explicit metric definition
   - type concordance
   - overcalling behavior in BGC-rich genomes

The project should no longer use a single headline metric as its main success criterion.

---

## 3. Current Evidence Base

### 3.1 Evidence That the Current Direction Works

The system has already produced evidence that it is not merely learning trivial GC-content artifacts.

Current evidence includes:

- Window-level discrimination is strong in Stage 3.
- MIBiG external validation suggests the model is learning biologically relevant BGC signal.
- User-provided Pfam validation on top novel candidates is especially strong:
  - `50/50` contain biosynthetic Pfam domains
  - `46/50` contain core BGC signature domains
  - dominant signatures include canonical NRPS/PKS-related domains such as `PF00550`, `PF00501`, `PF00668`, `PF00975`, `PF00109`, `PF00698`

However, this evidence currently applies to the present high-confidence unmatched candidate set, not yet to a fully cleaned set of **clean antiSMASH-missed candidates** under a stricter interval-aware definition. The correct takeaway is narrower: unmatched predictions already contain strong BGC-like biological signal, but the clean antiSMASH-missed discovery path is not yet fully established.

This is already enough to conclude that the project direction is **not** a dead end.

### 3.2 Evidence That the Current System Is Not Yet Sufficient

Server-side Stage 4 results show poor recall against antiSMASH-annotated known BGCs at the **current default operating point**:

- `test_id`
  - antiSMASH BGCs: `3121`
  - matched: `845`
  - missed: `2276`
  - recall: `0.2707`
  - precision: `0.5729` under the current antiSMASH-count-based metric, not yet strict prediction-level precision
- `test_ood`
  - antiSMASH BGCs: `1937`
  - matched: `413`
  - missed: `1524`
  - recall: `0.2132`
  - precision: `0.8901` under the current antiSMASH-count-based metric, not yet strict prediction-level precision

These numbers are not compatible with the claim that "antiSMASH-annotated BGCs are largely recovered."

However, these numbers should **not** be interpreted as the backbone's final achievable recall frontier. They describe the current Stage 4 operating point, which is tied to an extremely conservative `is_negative` threshold from the current validation policy.

This distinction matters:

- current known-BGC coverage is clearly insufficient,
- but current low recall is not by itself proof that the representation stack has hit its ceiling.

Therefore the current system state is:

- **novel-discovery evidence: strong but still preliminary**
- **known-BGC coverage: insufficient**

---

## 4. Current System Map

### 4.1 Stage 1: Data and Labels

Current Stage 1 builds:

- a GTDB-based bacterial genome panel,
- antiSMASH-derived BGC annotations,
- fixed-size DNA windows,
- 8-dimensional labels:
  - 7 BGC classes
  - 1 `is_negative` bit

Important properties of the current labeling scheme:

- positive window: overlap with a BGC of at least `0.5`
- negative window: no overlap and at least `10 kb` away from any BGC
- ambiguous window: all hard boundary cases in between

This is a useful weakly supervised setup, but it encodes a **core-window discrimination task**, not a full region-calling task.

It also introduces a hard blind spot for short BGCs. Under the current geometry:

- window size = `16,384 bp`
- positive threshold = overlap ratio `>= 0.5`

any BGC shorter than roughly `8 kb` cannot produce a positive window under ordinary sliding-window placement. Server-side error analysis already shows that this is not theoretical:

- `<5 kb` BGCs: `123/123` had no positive windows
- `5-10 kb` BGCs: `135/144` had no positive windows

This is a structural labeling problem, not a downstream decoder problem.

### 4.2 Stage 2: Frozen Evo2 Representation

Each `16,384 bp` window is passed through frozen Evo2, and a mean-pooled `4096`-dimensional embedding is extracted.

This design has one major strength:

- it can learn directly from raw DNA without depending on gene calling or Pfam preprocessing.

It also has an important potential limitation:

- mean pooling suppresses local structure and boundary detail that are important for region detection.

### 4.3 Stage 3: Window-Level Classifier

Current training stack:

- projection MLP
- 8-way independent sigmoid multi-label head
- optional contrastive loss
- optional DANN

This stage is evaluated mainly through:

- per-type AUROC
- mean AUROC
- a derived BGC binary AUROC

### 4.4 Stage 4: Genome-Level Calling

Genome-level calling is currently implemented as:

1. threshold the per-window score,
2. mark windows as BGC or non-BGC,
3. merge adjacent predicted windows,
4. compare merged regions against antiSMASH regions.

This is the first place where the system actually resembles the real deployment task.

The current decoder is still simple, but it is not literally just a bare merge:

- adjacent windows are merged contig-wise,
- a minimum region length filter is applied,
- per-region max probabilities and merged-window counts are tracked.

Even so, it remains a lightweight baseline rather than a strong region-decoding system.

The current Stage 4 implementation also excludes ambiguous windows entirely from:

- threshold selection,
- confusion analysis,
- genome-level calling.

Therefore the current Stage 4 outputs should be interpreted as a **reduced deployment proxy**, not yet as a full contig-wide scan that includes the hardest boundary regime.

---

## 5. Primary System Misalignments

This section identifies the most important reasons the current system underperforms on known-BGC recovery.

### 5.1 Training-Time Detection Score and Deployment-Time Detection Score Are Not the Same

This is the single most important issue.

Current Stage 3 reports a BGC binary metric derived from:

- `max(type_probs)`

But Stage 4 genome scanning uses:

- the independent `is_negative` head

These are not equivalent because the classifier is implemented as **8 independent sigmoid outputs**, not a constrained joint distribution.

Consequences:

- a model can look strong under Stage 3 validation while being poorly calibrated for Stage 4 calling,
- checkpoint selection is not tied to the actual deployment score,
- threshold tuning is applied to a head that was not the one implicitly emphasized during model selection.

This is not a small detail. It means the system has no single, stable contract for "BGC detection score."

### 5.2 The Proxy Task Is Easier Than the Real Task

The current label construction favors interior windows of BGCs and excludes many boundary windows as ambiguous.

This teaches the model to separate:

- clear BGC core windows
- from clearly distant non-BGC background windows

It does **not** teach the model to:

- detect BGC boundaries,
- preserve short clusters,
- connect weak edge windows into full regions.

So the model may truly learn BGC signal while still failing at complete BGC recovery.

For short BGCs, the issue is even more severe than "boundary difficulty." A non-trivial subset of short clusters never creates any positive training window at all under the current geometry and overlap rule. That means some recall loss is baked into the supervision before model training even starts.

Ambiguous-window exclusion makes the proxy-task gap even larger. The current system drops many of the most difficult boundary-adjacent windows from both training-time evaluation and Stage 4 calling, even though those windows are unavoidable in real deployment.

### 5.3 Region Calling Is Too Weak

The current region caller is essentially:

- threshold -> merge adjacent positives

This is a minimal baseline, not a strong region-decoding system.

It currently lacks:

- score smoothing along contigs,
- edge-aware handling near contig ends,
- explicit short-BGC handling,
- boundary refinement,
- confidence-aware region aggregation,
- separation between "known-like recovery mode" and "novel-candidate mode"

As a result, the system is structurally biased toward:

- under-calling known BGCs when thresholds are strict,
- over-calling large swaths of genomes when thresholds are relaxed.

There is also a metric-definition problem in the current implementation. The reported Stage 4 "precision" is not yet a strict prediction-level precision. Matching is performed against antiSMASH BGCs, and the final aggregate precision uses `n_matched / n_pred_total`. Under that setup, very large merged regions can be favored in ways that do not correspond to a clean one-prediction-one-match precision definition.

### 5.4 Training Distribution and Deployment Distribution Are Mismatched

Current training uses a heavily balanced batch sampler with approximately `50% positive / 50% negative`.

Whole-genome scanning does not look like that.

Real deployment conditions are:

- strongly imbalanced,
- dominated by background windows,
- highly genome-dependent,
- especially difficult in BGC-rich Actinomycetota genomes.

This mismatch degrades calibration and makes threshold behavior unstable.

This should be treated as a major design issue, not a secondary implementation detail. The extreme gap between the intuitive sigmoid midpoint (`0.5`) and the current selected `is_negative` threshold is strong evidence that the current training distribution, validation target, and deployment operating point are misaligned. It is also suggestive of calibration failure, but that stronger claim should be reserved for cases where explicit calibration diagnostics are run.

### 5.5 Split and Leakage Controls Are Not Sufficiently Hard

The current split design includes a GCF proxy leak check, but the actual implementation is looser and more optimistic than the conceptual goal.

The main consequence is not necessarily catastrophic leakage. The more important issue is:

- window-level generalization is easier than the true downstream task.

That means Stage 3 numbers should not be treated as direct evidence that the full system is close to deployment quality.

### 5.6 Window Geometry Is an Independent First-Class Variable

Current system behavior depends heavily on a set of geometric choices that are currently treated as background constants:

- `window size = 16,384 bp`
- `stride = 8,192 bp`
- no tail padding for short contig ends

These are not implementation details. They directly affect:

- short-BGC recoverability,
- edge-window dilution,
- contig-edge truncation behavior,
- and the shape of the downstream region aggregation layer's search space.

Therefore window geometry should be treated as a redesign variable in its own right, not merely as part of Stage 1 preprocessing.

### 5.7 Representation Limits May Exist, but They Are Not Yet Proven to Be the Main Bottleneck

It is plausible that frozen mean-pooled DNA embeddings do not preserve enough local biosynthetic structure for optimal known-BGC recovery.

However, this is currently a **second-order hypothesis**, not the primary diagnosis.

Why:

- the system already shows strong biological enrichment in novel candidates,
- multiple larger design mismatches exist upstream,
- those mismatches are sufficient by themselves to explain a large part of the known-recall failure.

Therefore, switching to LoRA immediately would be premature.

At the same time, decoder work should not be overclaimed. Because mean pooling suppresses positional detail, decoder work should initially be framed as improving **region aggregation stability**, not assumed to solve boundary recovery by itself. If boundary fidelity becomes a primary target, the system may need earlier position-preserving pooling, subwindow aggregation, or other representation changes.

### 5.8 Threshold Policy Is a First-Class Contract, Not Just a Hyperparameter

The current Stage 4 threshold is selected by:

- excluding ambiguous validation windows,
- optimizing per-head window-level F1,
- then transferring the resulting threshold directly into genome-scale region calling.

This procedure entangles several issues:

- window-level objective mismatch,
- class-prior mismatch,
- calibration mismatch,
- and decoder sensitivity.

Therefore the redesign must treat threshold-selection policy as part of the top-level system contract, not as a downstream tuning detail.

### 5.9 "Novel Candidate" Is Not Yet a Clean Semantic Category

In the current implementation, a novel candidate is effectively:

- an unmatched predicted region

rather than a fully cleaned "antiSMASH-missed BGC candidate."

This distinction matters because:

- the current matching logic is not one-to-one,
- oversized merged predictions may remain unmatched in ways that are semantically closer to boundary overreach than to true novelty,
- and the current nearest-BGC distance is not a strict interval distance with overlap-aware zeroing.

Therefore the current system should distinguish clearly between:

- **unmatched predictions**
- **boundary-expansion candidates**
- **clean antiSMASH-missed candidates**

---

## 6. Design Principles for the Next System

The redesign should follow the principles below.

### 6.1 One Canonical Detection Score

The system must define a single canonical **BGC detection score** that is used consistently for:

- validation,
- checkpoint selection,
- threshold analysis,
- genome scanning,
- region construction

No future stage should be allowed to silently substitute a different score.

This does **not** mean the entire system should collapse to a single scalar for all purposes. It means known-BGC recovery and region construction must stop switching between incompatible detection heads.

### 6.2 One Explicit Threshold-Selection Contract

The system must also define an explicit threshold-selection policy for deployment-facing evaluation.

That policy must specify:

- which split is used,
- whether ambiguous windows are included,
- which objective is optimized,
- and how a window-level threshold is translated into a region-level operating point.

Without this, a canonical detection score is still insufficient for reproducible deployment evaluation.

### 6.3 Region Discovery, Not Just Window Discrimination

The model+decoder stack should be treated as a region detector.

This means:

- labels,
- losses,
- calibration,
- and post-processing

must all be designed around region recovery, not only around window-level separability.

### 6.4 Dual-Mode Objective: Novel Ranking First, Known Coverage as Guardrail

The system should explicitly support two operating modes:

1. **Novel-discovery mode**
   - stricter ranking
   - lower-volume, higher-confidence candidates
   - optimized for fixed validation budget
   - primary scientific mode
2. **Known-recovery mode**
   - maximize recall on known antiSMASH BGCs
   - acceptable precision
   - safety/guardrail mode

These modes can share a backbone and many computations, but they should not be conflated into one threshold choice.

### 6.5 Two Score Contracts, Not One

The redesigned system should distinguish between:

1. **`bgc_detection_score`**
   - used for genome scanning
   - used for region construction
   - used for known-BGC recovery evaluation
2. **`novelty_priority_score`**
   - used to rank antiSMASH-missed candidates
   - may combine model confidence with orthogonal evidence such as Pfam/domain support, distance to known BGCs, contig-edge penalties, and cross-tool support

These two scores serve different tasks and should not be conflated.

### 6.6 Orthogonal Validation Is Mandatory

Novel candidates should be judged by evidence outside the model itself:

- Pfam/domain enrichment
- support from external tools
- boundary-distance analyses
- truncation checks
- optional gene architecture review

This is already partially implemented and should become a first-class system component.

### 6.7 Improve System Alignment Before Increasing Model Complexity

Before spending major effort on LoRA, larger backbones, or more complex heads, the project should first align:

- detection score
- labels
- evaluation
- region aggregation
- threshold policy

Only after alignment should representation bottlenecks be re-evaluated.

---

## 7. Revised Top-Level Architecture

The recommended architecture for the next iteration is:

1. **Window encoder**
   - current frozen Evo2 embedding as baseline
   - with future option for position-preserving pooling or subwindow aggregation if boundary fidelity becomes a main bottleneck
2. **Canonical BGC detection head**
   - one deployable scalar score
3. **Auxiliary type head**
   - used for type evidence and interpretation
4. **Contig-level region aggregation layer**
   - smooth scores
   - aggregate windows
   - improve region stability under realistic scanning conditions
   - not assumed to fully solve fine boundary recovery without stronger positional representations
5. **Candidate cleaning layer**
   - separates unmatched predictions from boundary expansions and clean antiSMASH-missed candidates
6. **Novelty ranking layer**
   - combines candidate-level evidence into a novelty priority score
7. **Dual output layer**
   - known-BGC recovery report
   - novel-candidate ranked list
8. **Orthogonal validation layer**
   - Pfam/domain checks
   - optional multi-tool cross-support

Under this design, the system is no longer forced to choose between:

- "recover known BGCs"
- and "generate exciting novel candidates"

Instead, both are part of the same stack.

---

## 8. Evaluation Framework for the Redesigned System

### 8.1 Known-BGC Safety Dashboard

The following metrics should be standard on every serious run:

- region-level recall vs antiSMASH on `test_id`
- region-level recall vs antiSMASH on `test_ood`
- region-level precision with explicit matching definition
- prediction-level precision with explicit matching definition
- type concordance
- per-type recall
- recall by length bucket
- recall by truncation
- recall by phylum
- full-window vs ambiguous-excluded evaluation side by side where applicable
- explicit statement of whether the run is still a reduced deployment proxy or a true full-window deployment approximation

### 8.2 Novel-Candidate Dashboard

The following metrics should be standard on candidate reports:

- total candidate count
- high-confidence candidate count
- precision@`20`
- precision@`50`
- hit rate under fixed review budget
- Pfam/domain enrichment at `top 20 / 50 / 100 / 200`
- fraction with core signature domains
- de-duplicated family diversity at `top 20 / 50 / 100`
- distribution by type and phylum
- cross-phylum and cross-type coverage under fixed budget
- distance to nearest antiSMASH region
- fraction near contig edges
- enrichment versus matched null regions from the same phylum/background regime
- incremental yield relative to antiSMASH, GECCO, and DeepBGC at matched review budget
- counts of:
  - unmatched predictions
  - boundary-expansion candidates
  - clean antiSMASH-missed candidates

### 8.3 Fairness Constraint

No claim of "matching antiSMASH" should be made unless:

- known-BGC recovery is explicitly reported,
- precision-recall tradeoffs are shown,
- the comparison is on the same region-level task,
- and the same operating point is clearly stated.

### 8.4 Decision Gates

The next major design decisions should be made through explicit gates:

Global rule:

- no gate counts as passed on the reduced ambiguous-excluded proxy alone
- reduced-proxy results can be used for diagnosis, but promotion decisions must ultimately be made on the full-window deployment approximation once that evaluation is available
- every major gate must satisfy both:
  - a **safety condition** on known-BGC recovery
  - a **discovery condition** on fixed-budget novel ranking quality

#### Discovery Gate Requirement

For all major system changes, discovery-side reporting must include at least:

- precision@`20`
- precision@`50`
- de-duplicated family diversity at fixed budget
- incremental yield relative to antiSMASH, GECCO, and DeepBGC at matched review budget

No phase should be considered successful if it only improves antiSMASH recovery while leaving fixed-budget novel ranking flat or worse.

#### Gate A: Is the score mismatch the dominant problem?

Run the same checkpoint with different candidate scores:

- `max(type_probs)`
- `1 - is_negative`
- simple score fusion

Hold fixed:

- decoder
- split
- evidence source
- threshold-selection policy, or explicitly compare threshold-selection policies as a separate axis

Pass condition:

- without changing the backbone or decoder, region-level `test_id` recall improves by at least `+0.05` absolute over the current canonical baseline.
- and fixed-budget novel ranking metrics do not regress materially.

Interpretation:

- if this gate passes, score contract mismatch is a first-order problem.
- this gate should report both the current reduced-proxy evaluation and the full-window deployment approximation, but it should only count as a formal pass once the full-window result is available.

#### Gate B: Are prior, calibration, and sampling the dominant problem?

Keep the score contract fixed and bring training-time class priors closer to deployment.

Pass condition:

- under the full-window deployment approximation, safety metrics improve materially after calibration and/or sampling redesign
- and fixed-budget novel ranking metrics also improve materially.

Interpretation:

- if this gate passes, prior mismatch and training-distribution mismatch are first-order problems and should be addressed before investing heavily in downstream aggregation complexity.

#### Gate C: Is region aggregation the dominant problem?

Keep the model fixed and strengthen the region aggregation layer.

Pass condition:

- with the canonical detection score and sampling policy fixed, region aggregation improvements raise region-level `test_id` recall to at least `0.50` while keeping precision at or above `0.30`.
- and fixed-budget novel ranking metrics improve or stay meaningfully non-inferior.

Metric requirement:

- this gate must use the full-window deployment approximation and an explicitly defined prediction-level precision, not only the current antiSMASH-count-based matched rate.

Secondary check:

- short-BGC recall and truncated-BGC recall each improve by at least `+0.15` absolute.
- improvements should be interpreted primarily as region aggregation gains, not as proof that boundary recovery has been fundamentally solved.

Interpretation:

- if this gate passes, current representation remains a viable baseline and region aggregation is a major bottleneck.

#### Gate D: Is representation the dominant problem?

Only after Gates A, B, and C are addressed:

- if known recall remains poor,
- and novel precision is also flattening,
- then stronger representation learning becomes the main priority.

Trigger condition:

- after Gates A, B, and C are completed, region-level `test_id` recall remains below `0.50`
- and novel-candidate enrichment relative to a matched null distribution is no longer improving materially.

That is the correct point to consider LoRA seriously.

---

## 9. Prioritized Optimization Roadmap

### Phase 0: Audit and Alignment

Objective:

- remove ambiguity about what the system is optimizing

Tasks:

- run a zero-cost score swap experiment on existing checkpoints:
  - `max(type_probs)`
  - `1 - is_negative`
  - simple fused detection score
- define one canonical detection score
- define one canonical threshold-selection policy
- re-run Stage 3/4 evaluation using the same score
- confirm all threshold analyses and checkpoint selection use the same deployment metric
- define canonical evidence sources for future memos and reports
- document where ambiguous windows are included or excluded in every evaluation stage

Expected outcome:

- an honest baseline for known-BGC recovery

### Phase 0.5: Calibration

Objective:

- determine how much current failure is due to score calibration rather than representation

Tasks:

- test low-cost calibration methods on the current model outputs
- re-evaluate recall/precision tradeoffs under calibrated scores
- quantify threshold sensitivity around the current operating region
- separate calibration effects from threshold-policy effects
- quantify current fixed-budget novel ranking quality before retraining

Expected outcome:

- a cleaner estimate of how much recall can be recovered without retraining

### Phase 1: Sampling and Training Distribution Redesign

Objective:

- train the model on a task closer to real deployment before investing heavily in decoder work

Tasks:

- redesign positive/negative sampling to reflect genome-scan conditions more faithfully
- revisit calibration-aware training objectives where justified
- evaluate whether prior-aligned training changes fixed-budget novel ranking quality

Expected outcome:

- better calibration
- fewer threshold pathologies
- cleaner signal for downstream label/decoder experiments

### Phase 2: Label and Geometry Redesign

Objective:

- remove structural blind spots before upgrading downstream aggregation complexity

Tasks:

- redesign short-BGC labeling so short clusters can generate positive supervision
- revisit ambiguous-window policy near BGC boundaries
- evaluate alternative window geometry:
  - smaller stride
  - alternate window sizes
  - tail padding or edge-aware handling
- support dual evaluation modes: known-recovery and novel-discovery
- define how ambiguous windows participate in both training and deployment-time evaluation

Expected outcome:

- reduced structural supervision gaps
- better support for short, edge-near, and truncated BGCs

### Phase 3: Split Hardening and Benchmark Hardening

Objective:

- stop overinterpreting optimistic intermediate metrics

Tasks:

- harden leakage logic
- define stricter downstream benchmarks
- make full-window deployment evaluation the promotion standard
- re-baseline Stage 3 and Stage 4 under those rules

Expected outcome:

- trustworthy performance estimates

### Phase 4: Region Aggregation Upgrade

Objective:

- improve region stability without overclaiming boundary recovery

Tasks:

- add contig-level smoothing
- add short-BGC-aware decoding
- add conservative boundary refinement only where the representation supports it
- test multiple region aggregation strategies
- define clean candidate classes:
  - unmatched predictions
  - boundary expansions
  - clean antiSMASH-missed candidates
- replace heuristic nearest-BGC distance with overlap-aware interval distance

Expected outcome:

- significant recall and candidate-cleaning improvement without assuming the decoder alone can recover fine boundaries

### Phase 5: Representation Upgrade

Objective:

- improve what the encoder can represent after system alignment is fixed

Candidate directions:

- LoRA on top Evo2 layers
- richer pooling than mean pooling
- local-structure-aware heads
- hybrid raw-DNA + domain-evidence models

Expected outcome:

- additional gains after the major system mismatches are removed

---

## 10. LoRA Decision

Current recommendation:

- **Do not treat LoRA as the immediate next step.**

Rationale:

1. the current system has already demonstrated biologically meaningful novel-candidate enrichment,
2. multiple higher-level task-alignment problems remain unresolved,
3. those problems are sufficient to explain much of the known-BGC recall failure,
4. jumping to LoRA now would make it hard to know whether gains come from better representation or from merely masking a design flaw.

LoRA should be promoted only if:

- the score contract is fixed,
- region aggregation is improved,
- label/evaluation alignment is improved,
- and known-BGC recall is still unacceptably low.

At that point, LoRA becomes a justified response to a demonstrated representation bottleneck.

---

## 11. Why the Current Misalignment Happened

This is not best explained as a single bug.

### 11.1 Main Cause: The Top-Level Contract Was Never Made Explicit

The stage documents are detailed, but they are written as **execution manuals per stage**, not as a unified system specification.

What was missing at the top level:

- one canonical detection score plus one explicit novelty-ranking contract,
- one explicit deployment threshold-selection contract,
- one deployment operating mode definition,
- one explicit contract connecting Stage 3 validation to Stage 4 genome scanning,
- one explicit success dashboard balancing known recovery and novel precision.

Without that contract, each stage can be individually reasonable while the end-to-end system is misaligned.

### 11.2 Secondary Cause: Local Execution Drift

There are also implementation-level drifts:

- metric definitions differ across stages,
- comments and actual behavior are not always fully aligned,
- some fallback design choices became sticky defaults.

These execution drifts matter, but they are downstream of the missing top-level contract.

### 11.3 Final Attribution

The fairest diagnosis is:

- **primarily a specification/design problem**
- **secondarily an execution consistency problem**

The stage documents were not "bad," but they were not written to prevent this exact class of cross-stage misalignment.

---

## 12. Recommended Project Reframe

The project should now be framed as:

> A DNA-foundation-model-based BGC discovery system that aims to jointly improve known-BGC recovery and antiSMASH-missed candidate discovery, with orthogonal validation used to establish biological credibility.

This framing is stronger and more accurate than:

- "a window classifier with high AUROC"
- or "an antiSMASH replacement"

It preserves the genuine novelty of the project while honestly acknowledging the current gap in known-BGC recall.

---

## 13. Immediate Next Deliverables

The next formal deliverables should be:

1. a **system alignment audit** documenting score, metric, and threshold inconsistencies,
2. a **known-BGC recovery benchmark spec**,
3. a **novel-candidate validation benchmark spec**,
4. a **label and geometry redesign plan**,
5. a **region aggregation redesign plan**,
6. a **LoRA decision gate memo** to be used only after the above are completed.

---

## 14. Review Questions

Reviewers should answer the following:

1. Is the project’s root task defined correctly here?
2. Is the diagnosis of current failure modes plausible and complete?
3. Is the recommendation to delay LoRA justified?
4. Are the proposed optimization phases in the right order?
5. Is the known-BGC vs novel-discovery dual-objective framing the right long-term contract?
6. Are the proposed gate thresholds the right ones for this project?

---

## Appendix A: Key Evidence Sources

Primary repo sources:

- [stage1_detailed.md](/Users/musk/Documents/code/np-master/stage1_detailed.md)
- [stage2_detailed.md](/Users/musk/Documents/code/np-master/stage2_detailed.md)
- [stage3_detailed.md](/Users/musk/Documents/code/np-master/stage3_detailed.md)
- [stage4_detailed.md](/Users/musk/Documents/code/np-master/stage4_detailed.md)
- [STAGE1_REPORT.md](/Users/musk/Documents/code/np-master/results/STAGE1_REPORT.md)
- [STAGE2_REPORT.md](/Users/musk/Documents/code/np-master/results/STAGE2_REPORT.md)
- [STAGE3_REPORT.md](/Users/musk/Documents/code/np-master/results/STAGE3_REPORT.md)
- [STAGE4_REPORT.md](/Users/musk/Documents/code/np-master/results/STAGE4_REPORT.md)
- [evaluate.py](/Users/musk/Documents/code/np-master/bgc-gcd/evaluate.py)
- [train.py](/Users/musk/Documents/code/np-master/bgc-gcd/train.py)
- [dataset.py](/Users/musk/Documents/code/np-master/bgc-gcd/data/dataset.py)
- [sampler.py](/Users/musk/Documents/code/np-master/bgc-gcd/data/sampler.py)
- [classifier.py](/Users/musk/Documents/code/np-master/bgc-gcd/model/classifier.py)
- [run_stage4.py](/Users/musk/Documents/code/np-master/scripts/stage4/run_stage4.py)
- [error_analysis.py](/Users/musk/Documents/code/np-master/scripts/stage4/error_analysis.py)
- [validate_novel_candidates.py](/Users/musk/Documents/code/np-master/scripts/stage4/validate_novel_candidates.py)

Server-side metrics referenced in this memo (Config B era, now archived):

- `/data/muskliu/np_master/data/_archive/stage4/step4_2/genome_level_metrics.json` (archived)
- `/data/muskliu/np_master/data/_archive/stage4/stage4_summary.json` (archived)
- `/data/muskliu/np_master/data/stage4_results/error_analysis/error_analysis.json`
- `/data/muskliu/np_master/data/_archive/stage4/recall_targeted_thresholds.json` (archived)
- `/data/muskliu/np_master/data/stage4_results/pfam_validation/validation_summary.json`

Current mainline (Config F_cap08) canonical artifacts:

- `/data/muskliu/np_master/data/stage4_results/ws3_cap08_eval/score_swap_results.json`
- `/data/muskliu/np_master/data/stage4_results/ws_discovery_benchmark_cap08/discovery_benchmark.json`

Canonical evidence policy for future revisions:

- canonical quantitative evidence should come from server-side JSON artifacts in `/data/muskliu/np_master/data/stage4_results/`
- local markdown reports are interpretive summaries unless explicitly marked as canonical for a given run
- any future memo should state:
  - checkpoint
  - run date
  - threshold policy
  - artifact paths used for its quantitative claims

User-provided orthogonal validation evidence referenced in this memo:

- top-50 high-confidence current candidate set Pfam validation:
  - `50/50` biosynthetic Pfam support
  - `46/50` core signature support
