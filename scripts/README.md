# scripts/

CLI entry points. Each file is an action. Grouped by verb, not by model.

| Dir | What |
|---|---|
| `extract/` | FASTA → Evo2 mean-pool embeddings. Shard management + 16-GPU dispatch launcher live here. |
| `inference/` | Apply a trained head (CNN / LR) to cached embeddings → per-window score CSV. |
| `decode/` | Per-window scores → BGC region calls. Model-specific decoder drivers (e.g. defrag_v2 fused-score OP). |
| `classify/` | Add per-type labels to regions (Stage-2 LR heads or CNN per-type output). |
| `evaluate/` | Score predictions against ground truth. Includes `v4_1_multiscale.py` (canonical ensemble readout), `nine_genome.py` (single-OP evaluator), `baselines_nine_genome.py` (full table), per-type classification evals, and `reproduce_defrag_v2_nine_genome.py`. |
| `train/` | Fit models (`lr.py`) + training data prep (`build_chunks.py`, `mine_hard_negatives.py`). |
| `plot/` | Figure generation for 9-genome comparison, boundary plots, per-case studies. |
| `diagnose/` | One-off investigation scripts (precision-gap analysis, specific blind-spot genomes). Expect turnover here. |
| `utils/` | Housekeeping (snapshot sync, provisioning). |

Third-party tool parsers live in `src/baselines/` (library, not CLI). Plotting utilities with shared symbols live in `src/plotting/`.

## Rules (from `docs/CONVENTIONS.md`)

- Top-level `scripts/` has **zero** loose `.py` files. Every script lives in a verb dir (or `utils/` if it doesn't fit).
- Scripts never import other scripts. Shared logic lives in `src/`.
- Exactly one `sys.path.insert(0, REPO_ROOT / "src")` per script. No other runtime path patching.
- Verb-first naming: `extract_embeddings.py`, not `embedding_extractor.py`.

## Canonical invocations

See `docs/REGRESSION_LEDGER.md` for the full set of commands that produce the frozen numbers any refactor must preserve.
