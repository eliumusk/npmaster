# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

BGC (Biosynthetic Gene Cluster) detection/discovery on bacterial genomes and metagenomes. Two-stage pipeline:

1. **Scorer**: frozen Evo2 7B embeddings (layer `blocks.20.mlp.l3`, mean-pooled) → light head (either LR_A1 logistic regression, or a dilated-CNN context model `ContextFullModel`).
2. **Decoder**: per-window scores → BGC region calls.

**Two mainline families, pick by target** (`docs/REGRESSION_LEDGER.md` has frozen numbers, `configs/operating_points.yaml` has parameters):

| Target | Candidate | Decoder | Entrypoint |
|---|---|---|---|
| 9-genome benchmark | `v4.1 multi-scale` (Evo2 + LR_A1, w=8192 primary + w=4096 short-arm, union-dedup IoU=0.3) | hand-tuned hysteresis | `scripts/decode/v4_1.py`, `scripts/evaluate/v4_1_multiscale.py` |
| Metagenomes (OER004256…) | `v4.1-learned` | trained 1D-CNN decoder (`experiments/evo2_first_principles/decoder_v1/runs/c00_cnn3_h32_antismash_bce/decoder.pt`) | `scripts/decode/v4_1_learned.py` |
| Type classification | `defrag_v2` CNN head (top-1 82.7% vs v4.1 Stage-2 LR 70.5%) | — | `scripts/classify/defrag_v2_regions.py` |

`defrag_v2` detection itself is retired (v4.1-learned supersedes it on metagenomes); the CNN survives only as the type labeler.

## Architecture conventions (enforced — see `docs/CONVENTIONS.md`)

- `src/` is a **library**: no argparse, no CLIs, no `__main__` that does real work. Subpackages: `data/`, `model/`, `evo2_lr/`, `decoding/`, `benchmark/`, `classification/`, `baselines/`, `plotting/`, plus top-level `train.py` / `evaluate.py` / `utils.py`.
- `scripts/` is **CLI entrypoints**, grouped by verb: `extract/ inference/ decode/ classify/ evaluate/ train/ plot/ diagnose/ utils/`. **No loose `.py` files at the top level** of `scripts/`. Scripts never import scripts — shared logic lifts to `src/`.
- `experiments/` is **outputs only** (JSON/CSV/plots), one dir per target (`nine_genome/`, `evo2_first_principles/`, `OER004256/`). Every results dir has a `README.md` saying what model/OP/config produced the numbers. No orphan JSONs.
- `configs/` = YAML only. No code, no absolute paths.
- `_backups/` = local-only ckpt backups (never pushed, never deleted). `_archive/` = retired provenance (gitignored).

### Import pattern

Each script inserts `src/` onto `sys.path` exactly once at top of file — this is the **sole allowed `sys.path.insert`** in the repo:

```python
REPO_ROOT = Path(__file__).resolve().parents[N]
sys.path.insert(0, str(REPO_ROOT / "src"))
from decoding.postprocess import decode_regions_hysteresis
```

Intra-`src/` imports use the same flat form (`from model.context_full_model import ...`).

### Naming rules (non-negotiable)

- Version suffix: `v4_1` not `v41`. Benchmark: `nine_genome` not `9genome` (must be valid Python identifier).
- Scripts are verb-first: `extract_embeddings.py`, not `embedding_extractor.py`.
- **Forbidden suffixes in committed filenames**: `_FIXED`, `_v1/_v2` (for iteration), `_tmp`, `_old`, `_new`, `_debug`, `_final`, `_final2`. Iterate in place and let git hold history.
- Hyperparameter-tagged names (`defrag_v2_ms10_s099`) only go in config `run_name`s / result filenames — never in module paths.

## Evaluation contract (`docs/evaluation-protocol.md`)

Per-window score vector is `probs ∈ ℝ⁸` (7 BGC types + 1 non-BGC). Three families, computable post-hoc from one CSV:

- `max_type = max(probs[:7])`
- `1_minus_isneg = 1 - probs[7]`
- `fused = 0.5·max_type + 0.5·(1 - probs[7])` — **default formal family**.

9-genome matching rule is `bench` = **asymmetric 50% overlap**: recall TP = `overlap/gt_len ≥ 0.5`, precision TP = `overlap/pred_len ≥ 0.5`. The older `any-overlap` precision is deprecated.

**Report as primary**: `dedup_f1`, `bench_precision`, `bench_recall`. Raw `bench_f1` alone is fragmentation-gameable — a decoder that fragments one GT into many short preds inflates precision without improving boundaries. Sweep CSVs must also carry `dedup_precision`, `unique_gt_precision_tp`, `duplicate_precision_tp`, `max_preds_per_gt`, `median_pred_len`. Gate: `dedup_p / bench_p >= 0.75` to reject the short-core hack.

**Never tune thresholds, OPs, or hyperparameters on the 9-genome set.** It is the held-out formal benchmark; 9-genome-tuned OPs (e.g. `tuned_v2` in `operating_points.yaml`) are diagnostic upper bounds, never formal claims. Internal val is the only tuning surface.

## CNN class order (`memory/reference_cnn_class_order.md`)

`ContextFullModel` outputs 8 columns in this order — **verify before indexing CNN probs**, a class-order bug was fixed 2026-04-23:

```
0 Polyketide, 1 NRP, 2 Terpene, 3 RiPP, 4 Saccharide, 5 Alkaloid, 6 Other, 7 non-BGC
```

## Local ↔ server split

This repo is the single source of truth for **code + configs + docs + small promoted JSON/CSV outputs**. Everything else (FASTA, Evo2 embeddings, `.pt` checkpoints, wandb, raw results) lives on the server and is gitignored (`/data/`, `*.pt`, `*.npy`, `*.fna`, `*.gbk`, etc.).

- **Edit only locally.** The server is execution-only. `zhanglin/` `syh/` `sunyuhong/` trees on the server are other people's work — ignore them when diagnosing issues in this code.
- Sync to node7 A800 via: `rsync -a --exclude=__pycache__ --exclude=.pytest_cache --exclude=.DS_Store /Users/musk/Documents/code/npmaster/ Voc_gpu_node7_a800:/data/muskliu/npmaster/`
- Server conda env: `conda activate bgc`. Run commands from repo root.

## Running things

When handling large tasks, break them down into smaller subtasks and delegate them to subagents whenever possible.并且所有已经做好的实验和结果都要记录下来。
All canonical reproduce commands live in **`docs/REGRESSION_LEDGER.md`** (§ "Reproduce commands"). That file is also the list of numbers any refactor must preserve — if a commit reports a value outside the listed 95% CI or changes a point estimate by more than rounding, revert.

High-frequency entrypoints:

```bash
# v4.1 honest readout on 9-genome (→ dedup_F1 0.4878, bench_F1 0.5192)
python scripts/evaluate/v4_1_multiscale.py \
  --primary-preds experiments/evo2_first_principles/9g_preds_lrA1_w8192 \
  --short-preds   experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096 \
  --gt-json /data/syh/benchmark/nine_genome/ground_truth.json \
  --out-json experiments/evo2_first_principles/v4_1_final_readout.json

# defrag_v2 9-genome (→ dedup_F1 0.4385, precision 0.4714, n_pred 140)
python scripts/evaluate/reproduce_defrag_v2_nine_genome.py \
  --checkpoint /data/muskliu/npmaster/data/phase3/stage3_hardneg_v1/best/best_tk100-ms0-hnw4.pt \
  --window 8192 --stride 2048 \
  --start-threshold 0.99 --extend-threshold 0.95 --min-support-windows 10 \
  --gt-json /data/syh/benchmark/nine_genome/ground_truth.json

# Stride inference (per-window CSVs)
python scripts/inference/stride.py --fna-dir ... --ckpt ... --window 8192 --stride 2048 --out-dir ...

# Train Route B mainline
python -u src/train.py --config configs/mainline.yaml --seed 0 \
  --data-dir data/mibig_v2 --out-dir data/phase3/route_b/s0
```

Use the single-OP evaluator `scripts/evaluate/nine_genome.py --operating-point <name>` to read an OP from `configs/operating_points.yaml`. Override with explicit flags (`--start-threshold`, `--extend-threshold`, `--min-support-windows`) for precision-alternate variants (e.g. `defrag_v2_ms10_s0995`).

## Tests

```bash
python -m pytest tests/ -q                        # everything
python -m pytest tests/test_nine_genome_tools.py -x -q   # single file
```

Tests are **local-runnable** (no server data required) for the pure-logic modules: baselines parsing, benchmark math, layout/manifest, plotting. Detection/classification number regressions (`docs/REGRESSION_LEDGER.md` #1–8) require server-side data and are verified in the next server session after any refactor. Local post-refactor smoke test: `python -m py_compile` on all moved scripts + `--help` on each renamed entrypoint.

## Known blind spot — do not try to fix

`GCA_000154945.1` recall is ~0.03 under every model (shared with BGC-Prophet); root cause is a concentration of <5 kb BGCs needing native `w=2048` training (~20 A800-h). Not an architecture problem. See `docs/reports/genome_154945_diagnostic.md`.
