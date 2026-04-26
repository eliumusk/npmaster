# NP-Master

NP-Master is a BGC region detection and discovery pipeline for bacterial genomes and metagenomes. It is not a generative model: given DNA sequences, it predicts genomic intervals that look like biosynthetic gene clusters, then optionally adds BGC type labels and novelty triage metadata.

## System I/O

Input: one or more DNA contigs/genomes, usually FASTA. The server-side pipeline extracts frozen Evo2 7B `blocks.20.mlp.l3` features.

Output: BGC region calls with `genome`, `contig`, `start`, `end`, score/source fields, and optional type/novelty annotations.

## Active Mainlines

| Mainline | Purpose | Status | Key entrypoints | Results |
|---|---|---|---|---|
| `evo2_lr_multiscale` | Original v4.1 line renamed: Evo2 mean-pool embeddings + LR scorer + multi-scale decoder. | Formal 9-genome mainline. OER learned-decoder variant is metagenome-focused. | `scripts/decode/evo2_lr_multiscale.py`, `scripts/evaluate/evo2_lr_multiscale.py`, `scripts/decode/evo2_lr_learned_decoder.py` | `experiments/evo2_lr_multiscale/` |
| `evo2_per_token_unet` | Per-token Evo2 features + U-Net decoder for short metagenomic contigs. | Exploratory candidate. The E10/E11 thresholds are not selected from 9-genome, but the model family has 9g-exposed history and is not a formal benchmark claim. | `scripts/train/evo2_per_token_unet/`, `scripts/inference/evo2_per_token/`, `scripts/decode/evo2_per_token_regions.py` | `experiments/evo2_per_token_unet/` |

`defrag_v2` detection and other probes are retired from active development. Historical files were moved to `_archive/2026-04-26-mainline-reset/` and are intentionally ignored by git.

## Canonical Numbers

See `docs/REGRESSION_LEDGER.md` for the current claims and caveats. Headline active numbers:

- `evo2_lr_multiscale` on 9-genome: dedup F1 `0.4878`, bench precision `0.5976`, bench recall `0.4590`.
- `evo2_lr_learned_decoder` on OER004256: 1,230 regions, 429 antiSMASH-missed candidates in the committed handoff artifact.
- `evo2_per_token_unet` E10 on OER004256: AS-recall `0.3117` at threshold `0.3`; diagnostic only until a clean pre-registered selection rule is run.

## Qualitative Boundaries

The current E10 U-Net conservative point (`threshold=0.4`) has per-genome 9-genome boundary plots in `figures/evo2_per_token_unet_boundaries/`.

![Evo2 per-token U-Net 9-genome boundaries](figures/evo2_per_token_unet_boundaries/overview_3x3.png)

## Layout

- `configs/`: active YAML configs only.
- `src/`: importable library code for active mainlines.
- `scripts/`: CLI entrypoints grouped by verb.
- `experiments/`: promoted active outputs only.
- `docs/`: current conventions, evaluation protocol, and regression ledger.
- `_archive/`: retired local provenance; not tracked.

## Quick Commands

```bash
python scripts/evaluate/evo2_lr_multiscale.py \
  --primary-preds <server-primary-preds-dir> \
  --short-preds <server-short-preds-dir> \
  --gt-json /data/syh/benchmark/nine_genome/ground_truth.json \
  --out-json experiments/evo2_lr_multiscale/nine_genome_readout.json
```

```bash
python scripts/decode/evo2_per_token_regions.py \
  --probs-dir <per-token-probs-dir> \
  --out-regions-csv <regions.csv> \
  --threshold 0.3
```
