# BGC System Execution Roadmap

**Date:** 2026-04-09  
**Project:** `np-master`  
**Purpose:** Convert the redesign memo into a high-throughput execution plan with small tasks, explicit inputs/outputs, and review gates.

---

## 1. Execution Model

This project should now move from memo-writing to **spec-driven execution**.

The working model is:

1. define one narrowly scoped task,
2. define exact inputs,
3. define exact outputs,
4. define exact verification,
5. execute in parallel where dependencies allow,
6. review centrally against the same contract.

The main rule is:

- no task is "done" because code was written,
- a task is only done when its declared output artifact exists and passes its declared review checks.

---

## 2. Roles

### 2.1 Execution Side

Execution workers can be:

- Claude Code sessions
- other AI agents
- human contributors

Workers should be given:

- one task ID
- one task spec
- one bounded write scope
- one expected artifact list

Workers should **not** be asked to "improve Stage 4" or "fix recall" in the abstract.

### 2.2 Review Side

Central review should remain with one reviewer of record.

Current recommendation:

- you dispatch execution
- I review task specs, outputs, metrics, and claims

The reviewer of record checks:

- whether the task matched the spec
- whether the artifact is reproducible
- whether metrics use the correct contract
- whether the claimed conclusion is justified

---

## 3. Global Rules

These rules apply to every execution task.

### 3.1 Canonical Evidence Rule

Quantitative claims must cite canonical artifacts.

Canonical quantitative artifacts currently mean:

- server-side JSON artifacts in `/data/muskliu/np_master/data/stage4_results/`
- new run artifacts produced under the same standard

Markdown summaries do not count as canonical unless explicitly promoted for a specific run.

### 3.2 Full-Window Promotion Rule

No task can be promoted based only on the current ambiguous-excluded reduced proxy.

Reduced-proxy evaluation may be used for:

- diagnosis
- debugging
- quick iteration

But any task that changes project priority, roadmap order, or representation strategy must eventually report:

- reduced-proxy result
- full-window deployment approximation result

### 3.3 Discovery-First Rule

`antiSMASH` recovery is a guardrail, not the scientific north star.

Every major workstream must report both:

- **safety metrics**
  - known-BGC recovery
  - strict prediction-level precision
- **discovery metrics**
  - precision@`20`
  - precision@`50`
  - de-duplicated family diversity
  - incremental yield vs `antiSMASH`, `GECCO`, `DeepBGC` where applicable

### 3.4 Claim Discipline Rule

Workers must distinguish:

- unmatched prediction
- boundary-expansion candidate
- clean antiSMASH-missed candidate

No result should call all unmatched predictions "novel BGCs".

---

## 4. Workstream Order

The execution order should be:

1. `WS0`: Contract audit and evidence freeze
2. `WS1`: Score and threshold alignment
3. `WS2`: Calibration and prior mismatch
4. `WS3`: Label and geometry redesign
5. `WS4`: Benchmark hardening and full-window evaluation
6. `WS5`: Region aggregation upgrade
7. `WS6`: Candidate cleaning and novel ranking
8. `WS7`: Representation upgrade decision

This order is deliberate.

- `WS1` and `WS2` come before decoder/aggregation work because current scores and class priors are not yet trustworthy.
- `WS4` comes before promotion decisions because reduced-proxy success is not enough.
- `WS7` should not begin as a main line of work until earlier system-alignment tasks have completed.

---

## 5. Workstream Specs

Each workstream is broken into small execution tasks.

## 5.1 WS0: Contract Audit and Evidence Freeze

### `WS0-T1` Canonical artifact inventory

Objective:

- freeze the list of canonical artifacts used by future reviews

Inputs:

- current local memo
- server artifact directory

Outputs:

- one inventory file listing:
  - artifact path
  - run date
  - checkpoint
  - metric meaning
  - canonical/non-canonical status

Verification:

- every metric cited in the redesign memo can be traced to one listed artifact

Review focus:

- no orphan metrics
- no report-only metrics masquerading as canonical

### `WS0-T2` Metric contract inventory

Objective:

- enumerate every metric definition currently used across Stage 3 and Stage 4

Inputs:

- `bgc-gcd/evaluate.py`
- `bgc-gcd/train.py`
- `scripts/stage4/run_stage4.py`

Outputs:

- one table containing:
  - metric name
  - code location
  - exact formula
  - split used
  - whether ambiguous windows are included
  - whether metric is window-level or region-level

Verification:

- the inventory explicitly covers:
  - `bgc_binary_auroc`
  - `is_negative` threshold selection
  - current Stage 4 precision

Review focus:

- hidden metric drift
- unspoken evaluation assumptions

---

## 5.2 WS1: Score and Threshold Alignment

### `WS1-T1` Score swap benchmark

Objective:

- test whether score mismatch is a first-order failure mode

Inputs:

- one fixed checkpoint
- one fixed split
- one fixed aggregation implementation

Variants:

- `max(type_probs)`
- `1 - is_negative`
- simple fused score

Outputs:

- one comparison artifact with:
  - reduced-proxy metrics
  - full-window approximation metrics if available
  - known-BGC safety metrics
  - discovery metrics at fixed budget

Verification:

- all variants use the same checkpoint and same aggregation code

Review focus:

- fair comparison
- whether score mismatch alone materially moves safety or discovery metrics

### `WS1-T2` Threshold policy spec

Objective:

- define one explicit threshold-selection contract

Inputs:

- current threshold code
- results from `WS1-T1`

Outputs:

- one short spec stating:
  - validation split
  - inclusion/exclusion of ambiguous windows
  - objective optimized
  - operating points to report
  - how window threshold maps to region decision

Verification:

- the spec is implementable without ambiguity

Review focus:

- threshold policy is deployment-facing, not just a convenience heuristic

---

## 5.3 WS2: Calibration and Prior Mismatch

### `WS2-T1` Calibration-only benchmark

Objective:

- estimate how much can be recovered without retraining

Inputs:

- frozen model outputs from the current checkpoint

Methods:

- temperature scaling
- Platt scaling or similarly cheap alternatives
- threshold sweeps around current operating region

Outputs:

- one artifact reporting:
  - safety metrics before/after calibration
  - discovery metrics before/after calibration
  - reliability-style diagnostics if implemented

Verification:

- no retraining
- score contract held fixed

Review focus:

- separate score-shape issues from representation issues

### `WS2-T2` Sampling redesign benchmark

Objective:

- test whether train/deploy prior mismatch is a first-order problem

Inputs:

- same architecture
- revised sampler or revised class weighting

Outputs:

- one ablation report comparing:
  - current 50/50 setup
  - one or more deployment-closer priors

Verification:

- model architecture held fixed
- training budget and split documented

Review focus:

- whether gains are real or just threshold artifacts
- whether discovery ranking improves, not just recall

---

## 5.4 WS3: Label and Geometry Redesign

### `WS3-T1` Short-BGC supervision redesign

Objective:

- remove the current structural blind spot for short BGCs

Inputs:

- current window geometry
- Stage 1 label code
- short-BGC error analysis

Outputs:

- one label redesign spec
- one concrete implementation
- one short-BGC-focused benchmark

Verification:

- explicitly report short-BGC positive-window recoverability before/after

Review focus:

- whether the redesign actually makes short clusters learnable

### `WS3-T2` Geometry sweep

Objective:

- measure the impact of window size, stride, and edge handling

Inputs:

- current preprocessing pipeline

Variants:

- smaller stride
- alternate window sizes
- edge-aware handling or tail padding

Outputs:

- one geometry comparison table

Verification:

- same downstream metric contract across variants

Review focus:

- short-BGC recall
- truncation behavior
- compute cost vs gain

---

## 5.5 WS4: Benchmark Hardening and Full-Window Evaluation

### `WS4-T1` Full-window deployment evaluation spec

Objective:

- define what counts as a real deployment-facing evaluation

Inputs:

- current Stage 4 evaluation path
- current ambiguous-window behavior

Outputs:

- one spec stating:
  - whether all windows are evaluated
  - how ambiguous windows are handled
  - what gets reported as reduced proxy vs full-window approximation

Verification:

- reviewers can tell exactly whether a reported metric is promotion-grade

Review focus:

- no ambiguity between debugging evaluation and deployment evaluation

### `WS4-T2` Prediction-level precision implementation

Objective:

- replace or supplement the current antiSMASH-count-based precision

Inputs:

- current matching logic

Outputs:

- strict prediction-level precision implementation
- matching-definition note
- backward comparison with old metric

Verification:

- a reader can see exactly why old precision and new precision differ

Review focus:

- oversized merged region bias
- one-to-one vs one-to-many matching behavior

---

## 5.6 WS5: Region Aggregation Upgrade

### `WS5-T1` Conservative aggregation baseline

Objective:

- improve region stability without overclaiming boundary recovery

Inputs:

- canonical detection score
- canonical threshold policy

Methods:

- contig-level smoothing
- short-BGC-aware aggregation
- conservative boundary refinement

Outputs:

- one aggregation benchmark

Verification:

- safety metrics improve or hold
- discovery metrics do not regress materially

Review focus:

- aggregation stability
- whether gains depend on reduced-proxy evaluation only

### `WS5-T2` Boundary sensitivity analysis

Objective:

- determine whether aggregation has hit a representation ceiling

Inputs:

- best aggregation variant from `WS5-T1`

Outputs:

- one analysis splitting performance by:
  - boundary proximity
  - short BGCs
  - contig edges
  - overlap ratio

Verification:

- shows whether remaining misses are aggregation-limited or representation-limited

Review focus:

- whether to continue aggregation work or escalate to representation work

---

## 5.7 WS6: Candidate Cleaning and Novel Ranking

### `WS6-T1` Candidate class cleaner

Objective:

- split current unmatched predictions into semantically cleaner buckets

Inputs:

- Stage 4 prediction regions
- antiSMASH annotations

Outputs:

- implementation that labels candidates as:
  - unmatched prediction
  - boundary expansion
  - clean antiSMASH-missed candidate

Verification:

- overlap-aware distance logic
- partial overlaps handled explicitly

Review focus:

- no inflated novel counts due to merge artifacts

### `WS6-T2` Fixed-budget ranking benchmark

Objective:

- evaluate the project on the metric that matters for real validation budget

Inputs:

- cleaned candidate set
- ranking score

Outputs:

- one benchmark report with:
  - precision@`20`
  - precision@`50`
  - de-duplicated family diversity
  - phylum/type spread
  - incremental yield vs `antiSMASH`, `GECCO`, `DeepBGC`

Verification:

- compared tools use matched review budget

Review focus:

- actual discovery value, not just total candidate count

---

## 5.8 WS7: Representation Upgrade Decision

### `WS7-T1` Representation decision memo

Objective:

- decide whether LoRA or other representation changes are now justified

Inputs:

- outputs from `WS1` through `WS6`

Outputs:

- one decision memo answering:
  - what was fixed without changing representation
  - what still fails under full-window evaluation
  - whether failures are now representation-limited

Verification:

- no representation escalation without completed upstream evidence

Review focus:

- whether LoRA is solving a real bottleneck or merely compensating for unresolved system design debt

---

## 6. Minimal Task Packet Format

Every execution packet should fit this template.

### Task Header

- `Task ID`
- `Owner`
- `Goal`
- `Write scope`
- `Dependencies`

### Task Body

- `Inputs`
- `Method`
- `Outputs`
- `Verification`
- `Non-goals`

### Review Footer

- `Safety metrics reported`
- `Discovery metrics reported`
- `Reduced proxy or full-window`
- `Canonical artifact paths`
- `Conclusion`

If a worker cannot fill this template cleanly, the task is still too vague.

---

## 7. Parallelization Strategy

The following tasks can run early and mostly in parallel:

- `WS0-T1`
- `WS0-T2`
- `WS1-T1`
- `WS2-T1`
- `WS4-T1`
- `WS4-T2`

The following tasks should wait for earlier evidence:

- `WS2-T2` after `WS1-T2`
- `WS3-T1` after `WS0-T2`
- `WS5-T1` after `WS1-T2` and initial `WS2` results
- `WS6-T2` after `WS6-T1`
- `WS7-T1` after most earlier workstreams

This structure is intended for high-throughput execution without losing review discipline.

---

## 8. What To Do Next

The immediate next batch should be:

1. `WS0-T1` Canonical artifact inventory
2. `WS0-T2` Metric contract inventory
3. `WS1-T1` Score swap benchmark
4. `WS4-T1` Full-window deployment evaluation spec
5. `WS4-T2` Prediction-level precision implementation

This batch is the best first cut because it does not require committing to LoRA, large retraining, or heavy architecture work.

It answers the most important questions first:

- what is the real metric contract
- what is the real deployment contract
- how much is score mismatch hurting us
- how much of current precision is metric artifact

---

## 9. Reviewer Standard

The central reviewer should reject any task output that has one of the following problems:

- metric definition is unclear
- reduced proxy is presented as deployment truth
- unmatched predictions are presented as clean novel candidates
- discovery metrics are missing
- known-BGC safety metrics are missing
- claim strength exceeds what the artifact supports

This is how execution speed stays high without allowing conceptual drift.
