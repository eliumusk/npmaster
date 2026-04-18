# 9-Genome Sweep Design

**Goal:** Build a scientifically defensible experiment system for NP-Master that searches for configurations able to beat BGC-Prophet on the external 9-genome benchmark without turning the benchmark into a de facto training set.

**Scope:** This document defines the evaluation protocol, parameter taxonomy, staged search order, local-vs-server execution model, experiment metadata, and the task-execution contract for subagents. It does not define individual run commands yet; that belongs in the implementation plan.

## 1. Objective

The final target is external performance, not internal proxy performance.

The ship criterion is:

- `9-genome lockbox recall >= BGC-Prophet recall`
- `9-genome lockbox precision > BGC-Prophet precision`

At the time of writing, the reference BGC-Prophet result on the shared `bench` metric is:

- recall `0.3475`
- precision `0.4529`
- F1 `0.3933`

These values are the external bar, not the development ranking metric.

## 2. Benchmark Split Policy

The 9-genome benchmark must no longer be treated as a single report set during development.

It will be split once into:

- `9-genome-dev`: 5 genomes
- `9-genome-lockbox`: 4 genomes

This split must be stratified, not random. The split file must be versioned in the local repo and reused across all future experiments until explicitly replaced by a new versioned split definition.

The stratification objective is to keep `dev` and `lockbox` roughly balanced on:

- number of GT BGCs
- GT BGC length distribution
- overall difficulty under existing tools
- prediction-length behavior under current NP-Master operating points

`lockbox` is frozen. It is not allowed to participate in:

- hyperparameter selection
- threshold selection
- decoder tuning
- architecture selection
- early stopping decisions
- negative-sampling policy selection
- inference tiling selection

`lockbox` is only opened for final frozen candidates.

## 3. Evaluation Contract

All development and final reporting must use the same formal external metric family already adopted in this repo:

- scorer: `bench`
- matching:
  - recall TP if `overlap / gt_len >= 0.5`
  - precision TP if `overlap / pred_len >= 0.5`

This preserves comparability with the current comparison table and prevents hidden metric drift during the sweep.

The benchmark contract must stay fixed while searching. If the metric changes, that is a new protocol version and old results stop being directly comparable.

## 4. Development Ranking

Development ranking must happen on `9-genome-dev`, never on full `9-genome`, and never on `lockbox`.

The ranking rule is lexicographic:

1. maximize recall first
2. break ties with precision

This reflects the actual project goal: the model must not solve precision by collapsing recall below the external bar. Precision-only optimization is therefore disallowed as the primary development objective.

To move a candidate forward, it must:

- improve the `dev` ranking metric under the fixed protocol
- not regress catastrophically on internal validation
- keep enough audit information to reproduce the result exactly

## 5. Anti-Overfitting Rules

- Do not iterate on `lockbox`.
- Do not use `lockbox` to choose among multiple candidates.
- Do not patch decoder settings after seeing `lockbox`.
- Do not change data construction policy after seeing `lockbox`.
- Do not report a `lockbox` win unless the candidate was frozen before the run.
- Record rejected candidates as well as promoted candidates so the search history stays auditable.

If a candidate fails on `lockbox`, the next candidate must be selected using `dev` again. `lockbox` does not become a second development set.

## 6. Parameter Taxonomy

The search space must be organized by what kind of recomputation each parameter requires.

### 6.1 Decoder / Postprocess Parameters

These are the cheapest parameters. They require no retraining and no new embeddings if per-window CSV scores already exist.

Parameters:

- `score_family`
- `start_threshold`
- `extend_threshold`
- `max_gap`
- `length_cap`
- `min_support_windows`
- `valley_threshold`
- `valley_min_windows`

Recompute required:

- re-decode regions from saved per-window predictions
- re-evaluate benchmark metrics

### 6.2 Inference Tiling Parameters

These parameters change what the model sees at inference time and directly affect boundary resolution.

Parameters:

- window size
- stride
- edge padding policy
- edge-window retention policy

Recompute required:

- rerun inference
- regenerate per-window prediction CSVs
- rerun decoder sweep on the regenerated predictions

Retraining is not automatically required if the checkpoint stays fixed, but comparable training-side experiments will usually require matching data artifacts later.

### 6.3 Data Construction and Sampling Parameters

These parameters change the learning problem itself.

Parameters:

- hard-negative definition
- hard-negative mining threshold and top-k policy
- hard-negative weighting
- positive/negative sampling ratio
- phylum-aware weighting or balancing
- ambiguous-window inclusion policy
- chunk construction settings
- chunk overlap logic
- edge-padding policy for training data
- dataset mixing policy across sources

Recompute required:

- rebuild dataset artifacts such as `meta_arrays.pkl` and `chunk_index.pkl`
- regenerate hard-negative annotations if used
- retrain affected models

### 6.4 Training Hyperparameters

These parameters do not change the data contract but do change optimization.

Parameters:

- learning rates
- warmup schedule
- weight decay
- batch size
- gradient clipping
- epoch budget
- loss weights
- dropout
- seed

Recompute required:

- retraining from the current dataset artifacts

### 6.5 Architecture Parameters

These are the most expensive and most confounded parameters.

Parameters:

- CNN depth
- dilation schedule
- hidden width / projection width
- receptive field design
- gating initialization
- head design
- warm-start strategy

Recompute required:

- retraining
- often rechecking data/tiling assumptions afterward

## 7. Staged Search Strategy

The sweep must be staged. A full joint sweep across all dimensions is not acceptable because it explodes compute, hides causality, and makes results hard to trust.

### Stage 0: Split and Instrumentation Freeze

Before any new search:

- freeze the stratified `5/4` benchmark split
- freeze the benchmark scorer version
- freeze artifact and metadata schema
- freeze the local-vs-server execution contract

This stage exists so later comparisons are real comparisons.

### Stage 1: Decoder Sweep

Search only decoder/postprocess parameters on already generated per-window predictions.

Purpose:

- measure how much remaining gain is still available without retraining
- establish a cheap baseline before touching training

Output:

- best decoder setting on `dev`
- estimate of remaining decoder headroom

If decoder changes no longer move `dev` materially, decoder tuning is considered near-saturated.

### Stage 2: Inference Tiling Sweep

Search window size and stride next.

Purpose:

- test whether boundary failures come from coarse tiling
- test whether smaller windows or denser stride improve boundary precision without immediate retraining

Output:

- a small number of promoted tiling configurations

### Stage 3: Data Construction Sweep

Only after tiling is narrowed down:

- sweep hard-negative policy
- sweep negative-sampling policy
- sweep edge-handling policy
- sweep chunk/index construction variants

Purpose:

- determine whether the current bottleneck is label construction and training distribution rather than decoder geometry

This is the first stage where systematic data rebuilding is expected.

### Stage 4: Training Hyperparameter Sweep

With data and tiling fixed for the stage:

- sweep optimizer and regularization settings
- run shortlisting on `dev`

Purpose:

- isolate optimization gains from data gains

### Stage 5: Architecture Sweep

Architecture changes come last and only for already strong pipelines.

Purpose:

- avoid wasting compute on model redesign before lower-cost sources of gain are exhausted

Only a small number of architecture candidates should be explored.

### Stage 6: Multi-Seed Promotion and Lockbox

For promoted candidates only:

- rerun multi-seed
- compare on `dev`
- freeze one or a few final candidates
- run `lockbox` once

No broad sweep is allowed at this stage.

## 8. Why This Search Order

This order follows standard ML discipline:

- cheap, high-interpretability knobs first
- expensive, confounded knobs later
- freeze data contracts before heavy model tuning
- keep external evaluation separate from daily search

In this repo specifically, it also matches the evidence already seen:

- decoder changes help, but likely do not explain the whole gap
- tiling appears to affect boundaries
- hard negatives and data policy are plausible bottlenecks
- architecture should not be the first place to spend large compute

## 9. Local vs Server Operating Model

The local repo is the only source of truth for:

- code
- configs
- docs
- orchestration logic
- split definitions
- experiment manifests

The two servers are execution-only.

Servers may:

- receive packaged run bundles
- execute commands
- emit artifacts

Servers may not:

- become canonical code state
- receive ad hoc manual code fixes
- serve as a hidden branch of development

If code must change, the change happens locally first and is then resynced to servers as a new execution bundle.

## 10. Run Packaging Model

Every run starts locally from:

- an explicit git commit
- a resolved config snapshot
- a versioned split definition
- a versioned command specification

The local controller prepares a run bundle containing:

- code snapshot reference
- config snapshot
- dataset/input manifest
- execution command
- output target paths

That bundle is synced one-way to a server workspace for execution.

Server outputs flow back as artifacts only:

- logs
- checkpoints
- predictions
- plots
- tables
- summaries

If a rerun is needed, it is generated locally as a new bundle. There is no server-side “fix in place”.

## 11. Required Experiment Metadata

Every experiment must record:

- `experiment_id`
- `task_id`
- optional `parent_experiment_id`
- git commit hash
- dirty-state flag
- split version
- resolved config file(s)
- expanded config snapshot
- exact entrypoint command
- server identifier
- start and end timestamps
- seeds
- dataset/input paths
- stable dataset checksums or version tags
- environment snapshot
- primary and secondary metrics
- run status
- failure reason if failed
- short human summary

Without this metadata, a run is not considered valid evidence.

## 12. Artifact Layout

Each experiment gets one immutable directory:

```text
experiments/<experiment_id>/
  manifest.json
  config.resolved.yaml
  env.txt
  command.txt
  logs/
    stdout.log
    stderr.log
  metrics/
    train.json
    eval.json
    summary.json
  artifacts/
    checkpoints/
    predictions/
    plots/
    tables/
  inputs/
    manifest.json
```

Rules:

- `manifest.json` is the authoritative index
- `metrics/summary.json` is the human-facing summary
- outputs are immutable after completion
- updates create a new experiment ID, not an overwrite

## 13. Reproducibility Rules

A run is reproducible only if it can be reconstructed from:

- the local repo
- the recorded manifest
- the captured inputs

A run is invalid if it depends on:

- shell history
- untracked server edits
- manual hotfixes
- undocumented environment differences

Cross-server differences must be logged as separate experiments, not silently merged.

## 14. Task Execution Contract

Execution will be task-based and subagent-driven.

The main agent is responsible for:

- defining the task graph
- packaging context per task
- assigning tasks to subagents
- collecting structured outputs
- reviewing results against protocol
- promoting or rejecting candidates
- maintaining the canonical local documentation

Subagents are responsible for:

- executing scoped tasks independently
- returning structured outputs only for their task
- not redefining protocol rules on the fly

This keeps the search scalable without fragmenting repository state.

## 15. Immediate Next Deliverables

After this design is approved, the next plan should implement:

- stratified `5/4` benchmark split file
- experiment manifest schema
- immutable artifact directory convention
- Stage 1 decoder sweep runner
- Stage 2 tiling sweep runner
- candidate promotion and lockbox gate
- subagent task templates for execution-only workers

The implementation plan should preserve the rule that local code is canonical and servers are disposable execution backends.
