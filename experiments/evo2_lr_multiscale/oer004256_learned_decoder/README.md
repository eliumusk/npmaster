# OER004256 / evo2_lr_learned_decoder

Production OER004256 handoff for the Evo2-LR mainline with a learned 1D CNN decoder over score tracks.

This is the metagenome-focused variant of `evo2_lr_multiscale`: the scorer is the same Evo2-LR line, but the hand-tuned hysteresis rule is replaced by a learned decoder to recover short-contig BGCs.

## Frozen Settings

| Parameter | Value |
|---|---|
| primary arm | w=8192 / s=2048 |
| short arm | w=4096 / s=1024 |
| primary threshold | 0.95 |
| short threshold | 0.98 |
| short length cap | 10,000 bp |
| union-dedup IoU | 0.3 |
| decoder | 1D CNN, depth=3, hidden=32 |

## Headline Artifacts

| File | Content |
|---|---|
| `regions.csv` | 1,230 decoded regions. |
| `regions_fully_typed.csv` | Region calls plus Evo2-LR and optional defrag type labels. |
| `novel_triage/novel_triage_full.csv` | HMM/MIBiG triage for antiSMASH-missed candidates. |

## Status

OER production handoff. This is not the formal 9-genome benchmark line; use `evo2_lr_multiscale` for formal 9-genome reporting.
