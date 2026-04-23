# npmaster Conventions

Rules for this repo. Pick one way, stick to it. If you want to break a rule, change the rule first.

## 1. Directory Responsibilities

| Dir | Contains | Does NOT contain |
|---|---|---|
| `src/` | Importable library code: models, data loaders, decoders, evaluation math. Pure Python, no `argparse`, no `__main__` that does real work. | CLI entrypoints, output files, configs. |
| `scripts/` | CLI entrypoints. One verb per file. Imports from `src/`. Never imports from another script. | Shared logic (lift to `src/`), outputs, configs. |
| `experiments/` | Evaluation/inference OUTPUTS only (JSON, CSV, plots, per-target `README.md`). Organized by benchmark/target. | Code. Raw sweep trial debris. |
| `configs/` | YAML/JSON parameterizing runs. | Code, credentials, absolute paths. |
| `docs/` | Human-facing Markdown. | Auto-generated reports. |
| `_backups/` | Local-only model checkpoint backups. Never deleted, never pushed to remote. | Anything else. |
| `_archive/` | Retired code/results kept for provenance. Single folder at repo root. Gitignored. | Active work. |

No `tmp/`, `misc/`, `scratch/` anywhere.

## 2. Naming (non-negotiable)

- **Version suffix: `v4_1`, not `v41`.** Matches `evo2_first_principles_v4_1_frozen.yaml`. `v4_10` vs `v41_0` is a trap we're avoiding.
- **Benchmark: `nine_genome`, not `9genome`.** Python identifiers cannot start with a digit.
- **Tool/model names:** bare name for the artifact family (`defrag_v2`), hyperparameter-tagged form (`defrag_v2_ms10_s099`) only in result filenames or config `run_name`s, never in module paths.
- **Scripts: verb-first.** `extract_embeddings.py`, not `embedding_extractor.py`. Scripts are actions; `src/` holds the nouns.
- **Forbidden suffixes in committed names:** `_FIXED`, `_v1`, `_v2` (for iteration), `_tmp`, `_old`, `_new`, `_debug`, `_final`, `_final2`. Iterate with git. If a file is wrong, rewrite it in place and commit.

## 3. `scripts/` Subgroups

Group by verb, not by model:

- `extract/` — produce embeddings/features from raw sequence.
- `inference/` — run a trained model over inputs, emit raw scores.
- `decode/` — turn scores into BGC region calls (postprocessing, tiling).
- `classify/` — per-type classification (Stage 2 LR, CNN head).
- `evaluate/` — score predictions against ground truth.
- `train/` — fit models.
- `baselines/` — third-party tool wrappers (antiSMASH, DeepBGC, etc.).
- `plot/` — figure generation only.
- `diagnose/` — one-off investigation scripts.
- `utils/` — sync, snapshot, housekeeping.

**If a script doesn't fit a group, it goes in `utils/`, not the top level. Top-level `scripts/` has zero loose `.py` files.**

## 4. Hardcoded Paths

- No absolute `/data/...`, `/root/...`, `/Users/...` in committed code.
- Genome IDs, species names, sample lists live in `configs/` or get passed as `argparse` args.
- Repo-relative paths: `pathlib.Path(__file__).resolve().parents[N]`. Never `os.getcwd()`.
- Checkpoint paths: resolved via a config key, not string-concatenated in code.

## 5. `experiments/` Layout

- Top level = one directory per active target: `nine_genome/`, `OER004256/`, `evo2_first_principles/`.
- Sweep trial subdirs (100+ `*-trial-NNN`) go to `_archive/` once the sweep has a summary CSV + README.
- Every results directory contains a `README.md` stating: what model, what config, what operating point, what the numbers mean. No orphan JSONs.
- Frozen operating points (lockbox-validated) are committed; exploratory runs are gitignored unless promoted.

## 6. Anti-Patterns

- `sys.path.insert(0, ...)` — fix the `src/` layout instead. Never patch paths at runtime.
- Scripts importing scripts — shared logic belongs in `src/`.
- Silent suffix mutation (`_FIXED`, `_v2`, `_final`) — rewrite in place, let git hold history.
- Two directories with the same name and different roles. The `src/` copy keeps the name; `scripts/` sides become verb-based.
- Committing `__pycache__/`.
