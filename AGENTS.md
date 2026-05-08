# AGENTS.md

This repository is for BGC region detection/discovery on bacterial genomes and metagenomes.

## Project Definition

This is not a generative model. The core task is supervised genome mining:

- **Input:** DNA contigs/genomes, usually FASTA, converted to frozen Evo2 features on the server.
- **Output:** BGC region calls (`genome`, `contig`, `start`, `end`, score/source), with optional BGC type labels and novelty-triage fields.

## Active Mainlines

| Mainline | What it is | Status | Entrypoints |
|---|---|---|---|
| `evo2_lr_multiscale` | Renamed original v4.1 line: Evo2 mean-pooled `blocks.20.mlp.l3` embeddings + LR scorer + primary w=8192 and short-rescue w=4096 decoder. | Formal 9-genome mainline. | `scripts/decode/evo2_lr_multiscale.py`, `scripts/evaluate/evo2_lr_multiscale.py` |
| `evo2_lr_learned_decoder` | Metagenome variant of the Evo2-LR line using a trained 1D CNN decoder over LR score tracks. | OER-focused production handoff, not the 9-genome formal line. | `scripts/decode/evo2_lr_learned_decoder.py` |
| `evo2_per_token_unet` | Per-token Evo2 features + U-Net head. | Exploratory candidate. Do not present E10/E11 as fully clean 9-genome claims; the family has 9g-exposed selection history. | `scripts/train/evo2_per_token_unet/`, `scripts/inference/evo2_per_token/`, `scripts/decode/evo2_per_token_regions.py` |

Retired experiments and diagnostics live only in `_archive/2026-04-26-mainline-reset/` and are ignored by git.

## Conventions

- `src/` is importable library code only.
- `scripts/` contains CLI entrypoints grouped by verb.
- `experiments/` contains promoted active outputs only.
- `configs/` contains YAML configs only.
- Do not tune thresholds, operating points, model selection, or hyperparameters on the 9-genome benchmark. 9-genome is a held-out formal benchmark.
- If a result has 9g-exposed selection history, label it diagnostic or exploratory.
- Use `evo2_lr_multiscale`, not `v4_1`/`v41`, for active names.
- Use `nine_genome`, not `9genome`, in committed names.
- Do not use committed iteration suffixes like `_v1`, `_v2`, `_final`, `_tmp`, `_old`, `_new`, or `_debug`.

## Evaluation Contract

Per-window score vector is `probs in R^8`: seven BGC types plus non-BGC.

Primary 9-genome metrics:

- `dedup_f1`
- `bench_precision`
- `bench_recall`

The 9-genome `bench` matching rule is asymmetric 50% overlap: recall TP uses `overlap / gt_len >= 0.5`, precision TP uses `overlap / pred_len >= 0.5`. Raw `bench_f1` alone is fragmentation-gameable; always report dedup metrics.

## Local vs Server

The local repo is the source of truth for code, configs, docs, and small promoted CSV/JSON artifacts. FASTA, Evo2 embeddings, `.pt` checkpoints, large arrays, wandb, and raw server outputs remain server-side and gitignored.

Sync to node7 A800:

```bash
rsync -a --exclude=__pycache__ --exclude=.pytest_cache --exclude=.DS_Store /Users/musk/Documents/code/npmaster/ Voc_gpu_node7_a800:/data/muskliu/npmaster/
```

Server env: `conda activate bgc`; run commands from repo root.

## Tests

```bash
python -m pytest tests/ -q
python -m py_compile $(git ls-files '*.py')
```

Number regressions that require server-side data must be checked in the next server session after refactors.

## Known Blind Spot

`GCA_000154945.1` recall remains poor under current models because many GT regions are under 5 kb. This needs native smaller-window training, not a repository cleanup fix.
