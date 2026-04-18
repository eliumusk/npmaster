# 9-Genome Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local experiment framework for the stratified `5/4` 9-genome sweep, then launch the first decoder and tiling batches on the execution servers with full artifact tracking.

**Architecture:** Add a small local experiment layer around the existing benchmark tooling instead of rewriting the training stack. The local repo will own split definitions, experiment manifests, sweep configs, and launch scripts; servers will only receive snapshots and execute generated commands, returning immutable artifacts.

**Tech Stack:** Python 3, existing repo scripts under `scripts/`, unittest, JSON/YAML manifests, SSH/rsync-based remote execution.

---

### Task 1: Benchmark Split and Experiment Schema

**Files:**
- Create: `configs/splits/nine_genome_dev_lockbox_v1.json`
- Create: `configs/experiments/stage0_manifest_schema.json`
- Create: `src/experiments/__init__.py`
- Create: `src/experiments/manifest.py`
- Test: `tests/test_experiment_manifest.py`

- [ ] **Step 1: Write the failing tests**

Add tests that require:
- the split file to contain all 9 benchmark genomes exactly once
- `dev` to contain 5 genomes and `lockbox` to contain 4 genomes
- `manifest.py` to validate a minimal experiment manifest with required fields
- `manifest.py` to reject missing `experiment_id`, `git_commit`, `server`, or `command`

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_experiment_manifest.py'`
Expected: FAIL because the split file and `src/experiments/manifest.py` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create the split JSON with a stratified `5/4` layout and a version tag. Create `manifest.py` with:
- `load_split(path)`
- `validate_split(split_data)`
- `validate_manifest(manifest_dict)`
- `write_manifest(path, manifest_dict)`

Keep the validator minimal and strict: required top-level keys only, plus basic type/emptiness checks.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_experiment_manifest.py'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/splits/nine_genome_dev_lockbox_v1.json configs/experiments/stage0_manifest_schema.json src/experiments/__init__.py src/experiments/manifest.py tests/test_experiment_manifest.py
git commit -m "feat: add benchmark split and manifest validation"
```

### Task 2: Local Artifact Layout and Experiment Helpers

**Files:**
- Create: `src/experiments/layout.py`
- Create: `src/experiments/runbook.py`
- Test: `tests/test_experiment_layout.py`

- [ ] **Step 1: Write the failing tests**

Add tests that require:
- creation of the canonical experiment directory structure under `experiments/<experiment_id>/`
- generation of `manifest.json`, `command.txt`, and `config.resolved.yaml`
- deterministic experiment path resolution from `experiment_id`

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_experiment_layout.py'`
Expected: FAIL because the helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- `create_experiment_layout(root_dir, experiment_id)`
- `write_command(path, command)`
- `write_resolved_config(path, config_dict)`
- `build_runbook_entry(...)`

The layout must match the spec document exactly:
- `logs/`
- `metrics/`
- `artifacts/checkpoints`
- `artifacts/predictions`
- `artifacts/plots`
- `artifacts/tables`
- `inputs/`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_experiment_layout.py'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/layout.py src/experiments/runbook.py tests/test_experiment_layout.py
git commit -m "feat: add experiment artifact layout helpers"
```

### Task 3: Stage 1 Decoder Sweep Runner

**Files:**
- Create: `configs/sweeps/stage1_decoder_v1.yaml`
- Create: `scripts/run_stage1_decoder_sweep.py`
- Modify: `scripts/evaluate_nine_genome.py`
- Test: `tests/test_stage1_decoder_sweep.py`

- [ ] **Step 1: Write the failing tests**

Add tests that require:
- loading a sweep YAML with lists of decoder parameters
- generating concrete decoder trial combinations
- filtering evaluation to the `dev` subset from the split file
- writing a ranked summary JSON/CSV for the generated trials

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_stage1_decoder_sweep.py'`
Expected: FAIL because the runner and dev-subset support do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement `run_stage1_decoder_sweep.py` as a local orchestration script that:
- reads the split file
- takes `--split-part dev|lockbox`
- enumerates decoder combinations from `stage1_decoder_v1.yaml`
- invokes the existing decode/eval logic without retraining
- ranks trials lexicographically by recall then precision
- writes `summary.json` and `summary.csv`

Only add the smallest extension needed to `evaluate_nine_genome.py` to support explicit genome subsets cleanly.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_stage1_decoder_sweep.py'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/sweeps/stage1_decoder_v1.yaml scripts/run_stage1_decoder_sweep.py scripts/evaluate_nine_genome.py tests/test_stage1_decoder_sweep.py
git commit -m "feat: add stage1 decoder sweep runner"
```

### Task 4: Stage 2 Tiling Sweep Runner

**Files:**
- Create: `configs/sweeps/stage2_tiling_v1.yaml`
- Create: `scripts/run_stage2_tiling_sweep.py`
- Modify: `scripts/inference_stride.py`
- Test: `tests/test_stage2_tiling_sweep.py`

- [ ] **Step 1: Write the failing tests**

Add tests that require:
- loading a tiling sweep config with multiple `(window, stride)` trials
- generating unique trial IDs
- validating that `stride <= window`
- building remote-safe command specs for each trial

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_stage2_tiling_sweep.py'`
Expected: FAIL because the tiling sweep runner does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement `run_stage2_tiling_sweep.py` as a local planner that:
- reads the split file and a tiling sweep YAML
- generates trial manifests and concrete inference/eval command specs
- stores each trial under the canonical experiment layout
- prepares GPU-friendly server commands for later dispatch

Keep `inference_stride.py` changes minimal: only add any missing flag plumbing needed for generated commands.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_stage2_tiling_sweep.py'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/sweeps/stage2_tiling_v1.yaml scripts/run_stage2_tiling_sweep.py scripts/inference_stride.py tests/test_stage2_tiling_sweep.py
git commit -m "feat: add stage2 tiling sweep planner"
```

### Task 5: Server Packaging and Launch Orchestration

**Files:**
- Create: `scripts/launch_experiment_batch.py`
- Create: `scripts/sync_snapshot_to_server.py`
- Create: `configs/servers/execution_hosts.json`
- Test: `tests/test_server_launch.py`

- [ ] **Step 1: Write the failing tests**

Add tests that require:
- turning a local experiment directory into a server run bundle spec
- assigning trials across multiple servers / GPUs
- rejecting bundles that do not include commit, config snapshot, and command

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_server_launch.py'`
Expected: FAIL because the server launch helpers do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement:
- `sync_snapshot_to_server.py` to package and copy the local snapshot plus experiment bundle
- `launch_experiment_batch.py` to read trial manifests and emit/execute remote commands
- GPU assignment logic that spreads independent jobs across available devices before stacking multiple jobs on one GPU

Use config-driven hosts and leave room for two execution servers, but do not hardcode mutable server-side paths outside config.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_server_launch.py'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/launch_experiment_batch.py scripts/sync_snapshot_to_server.py configs/servers/execution_hosts.json tests/test_server_launch.py
git commit -m "feat: add server sync and launch orchestration"
```

### Task 6: Documentation and First Batch Execution

**Files:**
- Modify: `README.md`
- Modify: `docs/mainline.md`
- Modify: `docs/superpowers/specs/2026-04-18-nine-genome-sweep-design.md`
- Create: `experiments/README.md`

- [ ] **Step 1: Write the failing test**

This is a documentation and execution task; no unit test required. Instead, define the verification commands before editing:
- `python -m unittest discover -s tests`
- one dry-run Stage 1 planner invocation
- one dry-run Stage 2 planner invocation

- [ ] **Step 2: Run verification pre-check**

Run:
```bash
python -m unittest discover -s tests
```
Expected: PASS before the doc-only changes.

- [ ] **Step 3: Update docs and run the first batch**

Document:
- where split files live
- how experiments are laid out
- how local-to-server execution works
- how Stage 1 and Stage 2 are launched

Then:
- generate the first Stage 1 decoder batch
- generate the first Stage 2 tiling batch
- sync the snapshot to execution servers
- launch the jobs
- collect the first result manifests locally

- [ ] **Step 4: Run verification**

Run:
```bash
python -m unittest discover -s tests
python scripts/run_stage1_decoder_sweep.py --help
python scripts/run_stage2_tiling_sweep.py --help
python scripts/launch_experiment_batch.py --help
```
Expected: all commands succeed.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/mainline.md docs/superpowers/specs/2026-04-18-nine-genome-sweep-design.md experiments/README.md
git commit -m "docs: add sweep execution workflow"
```

