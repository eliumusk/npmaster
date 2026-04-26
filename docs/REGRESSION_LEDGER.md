# Regression Ledger

This ledger lists the active numbers after the 2026-04-26 mainline reset. Historical experiments live in `_archive/2026-04-26-mainline-reset/` and are not active claims.

## Formal 9-Genome Mainline

| Mainline | Metric | Value | Artifact | Reproduce |
|---|---|---:|---|---|
| `evo2_lr_multiscale` | dedup F1 | **0.4878** (CI95 0.345-0.609) | `experiments/evo2_lr_multiscale/nine_genome_readout.json` | `scripts/evaluate/evo2_lr_multiscale.py` |
| `evo2_lr_multiscale` | bench precision | **0.5976** | same | same |
| `evo2_lr_multiscale` | bench recall | **0.4590** | same | same |
| `evo2_lr_multiscale` | bench F1 | **0.5192** | same | same |
| `evo2_lr_multiscale` | strict F1 | **0.4299** (CI95 0.284-0.564) | same | same |

Formal benchmark rule: no thresholds, operating points, model families, or hyperparameters may be selected on 9-genome.

## OER004256 Handoff

| Mainline | Metric | Value | Artifact |
|---|---|---:|---|
| `evo2_lr_learned_decoder` | detected regions | **1,230** | `experiments/evo2_lr_multiscale/oer004256_learned_decoder/regions.csv` |
| `evo2_lr_learned_decoder` | antiSMASH-missed candidates | **429** | `experiments/evo2_lr_multiscale/oer004256_learned_decoder/novel_triage/novel_triage_full.csv` |
| `evo2_lr_learned_decoder` | OER same-script AS-recall | **0.2065** (408 / 1,976) | `experiments/evo2_lr_multiscale/oer004256_learned_decoder/oer_compare.json` |
| `evo2_lr_learned_decoder` | canonical 9g dedup F1 | **0.3323** | `experiments/evo2_lr_multiscale/learned_decoder_nine_genome_readout.json`; metagenome tradeoff, not formal 9g replacement |

The older learned-decoder `0.461` number used a different greedy-pairing metric and must not be compared to the formal `0.4878` number.

## Per-Token U-Net Diagnostic Branch

| Candidate | Metric | Value | Artifact |
|---|---|---:|---|
| E10 threshold 0.3 | OER AS-recall | **0.3117** | `experiments/evo2_per_token_unet/oer004256/readouts/e10_thr03.json` |
| E10 threshold 0.4 | OER AS-recall | 0.2687 | `experiments/evo2_per_token_unet/oer004256/readouts/e10_thr04.json` |
| E10 threshold 0.5 | OER AS-recall | 0.2485 | `experiments/evo2_per_token_unet/oer004256/readouts/e10_thr05.json` |
| E11 threshold 0.3 / 0.4 / 0.5 | OER AS-recall | 0.2738 / 0.2389 / 0.1690 | `experiments/evo2_per_token_unet/oer004256/readouts/` |
| E10 threshold 0.3 | 9g dedup F1 | **0.4561** | `experiments/evo2_per_token_unet/nine_genome/E10_h64_lr1e3_b32/readouts/` |
| E10 threshold 0.4 | 9g dedup F1 | **0.4664** | same |
| E10 threshold 0.5 | 9g dedup F1 | **0.4751** | same |
| E11 threshold 0.3 / 0.4 / 0.5 | 9g dedup F1 | 0.4470 / 0.4500 / 0.4544 | `experiments/evo2_per_token_unet/nine_genome/E11_h64_lr1e3_b64/readouts/` |

Status: threshold-clean diagnostic, not a formal 9-genome claim. Thresholds 0.3 and 0.4 are OER-driven; threshold 0.5 is a default reference. The U-Net model family was advanced after 9-genome Round-1 readouts, so promotion requires a fresh pre-registered selection protocol.

## Reproduce Commands

```bash
python scripts/evaluate/evo2_lr_multiscale.py \
  --primary-preds <server-primary-preds-dir> \
  --short-preds <server-short-preds-dir> \
  --gt-json /data/syh/benchmark/nine_genome/ground_truth.json \
  --out-json experiments/evo2_lr_multiscale/nine_genome_readout.json
```

```bash
python scripts/evaluate/evo2_per_token_unet_nine_genome.py   --regions-csv <regions.csv>   --method-name evo2_per_token_unet_E10_thr0.3   --gt-json /data/syh/benchmark/nine_genome/ground_truth.json   --out-json <readout.json>
```

The E10/E11 per-token region CSVs are server-side input caches. If absent after a cleanup,
regenerate them from the promoted checkpoints/probability shards before rerunning this command.

## Local Verification After Refactors

```bash
python -m py_compile $(git ls-files '*.py')
python -m pytest tests/ -q
```

Full number reproduction requires server-side FASTA/features/checkpoints.
