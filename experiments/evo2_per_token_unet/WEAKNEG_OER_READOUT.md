# Weak-Negative U-Net OER Readout

Date: 2026-04-27

Model:

- Checkpoint: `experiments/evo2_per_token_unet/train_runs/full_weakneg_bce_w05_ddp4_gb64_seed0/best.pt`
- Loss: `weak_negative_bce`, `neg_weight=0.5`
- OER probability cache on server: `experiments/evo2_per_token_unet/oer004256/weakneg_bce_w05_full/probs/`
- Local synced readouts: `experiments/evo2_per_token_unet/oer004256/weakneg_bce_w05_full/`

Evaluation target:

- OER antiSMASH reference: `/data/muskliu/oer_bgc_scan/results/antismash`
- FASTA length source: `/data/muskliu/oer_bgc_scan/filtered`
- Type labels: post-hoc Evo2-LR per-type heads from `data/evo2_lr_multiscale/type_lr`

## OER Detection Results

| Operating point | Regions | AS-recall | antiSMASH-missed candidates | Median length |
|---|---:|---:|---:|---:|
| `thr0.50_min2000` | 2,199 | 0.5703 | 579 | 8,192 |
| `thr0.60_min8000` | 775 | 0.2961 | 67 | 10,122 |
| `thr0.70_min8000` | 617 | 0.2510 | 33 | 10,240 |

The `thr0.50_min2000` setting is the current discovery-oriented OER readout:
it recovers substantially more antiSMASH regions than the Evo2-LR learned decoder
reference in the same evaluator, while producing a larger antiSMASH-missed bucket.
The `thr0.70_min8000` setting is the conservative setting aligned with the best
current 9-genome diagnostic operating point, but it is not the best OER discovery
setting.

Reference in the same OER evaluator:

- `evo2_lr_learned_decoder`: 1,230 regions, AS-recall 0.2065, 429 antiSMASH-missed candidates, median length 5,973 bp.

## Post-Hoc Type Labels

Type labels were attached with the Evo2-LR per-type classifier over cached
w=8192 mean embeddings. This is a region annotation layer, not part of the
U-Net detector. It should be treated as useful triage for wet-lab handoff, not
as a final multi-type U-Net readout.

Type distributions:

- `thr0.50_min2000`: Terpene 1000, RiPP 474, Other 373, Polyketide 224, NRP 125, Saccharide 3.
- `thr0.60_min8000`: Terpene 338, Polyketide 160, RiPP 116, NRP 92, Other 69.
- `thr0.70_min8000`: Terpene 262, Polyketide 146, RiPP 86, NRP 81, Other 42.

## Artifacts

- `oer004256/weakneg_bce_w05_full/thr050_min2000/regions.csv`
- `oer004256/weakneg_bce_w05_full/thr050_min2000/regions_typed.csv`
- `oer004256/weakneg_bce_w05_full/thr050_min2000/oer_compare.json`
- `oer004256/weakneg_bce_w05_full/thr050_min2000/novel_triage/novel_triage_full.csv`
- `oer004256/weakneg_bce_w05_full/thr050_min2000/novel_triage/wetlab_candidates_top75.csv`
- `oer004256/weakneg_bce_w05_full/thr060_min8000/regions.csv`
- `oer004256/weakneg_bce_w05_full/thr060_min8000/regions_typed.csv`
- `oer004256/weakneg_bce_w05_full/thr060_min8000/oer_compare.json`
- `oer004256/weakneg_bce_w05_full/thr070_min8000/regions.csv`
- `oer004256/weakneg_bce_w05_full/thr070_min8000/regions_typed.csv`
- `oer004256/weakneg_bce_w05_full/thr070_min8000/oer_compare.json`

## Interpretation

This run is a usable OER readout for the current weak-negative model. The type
classification path is operational, but it is inherited from Evo2-LR and should
not be over-interpreted as a U-Net multi-task model. The next handoff table for
wet-lab candidates should start from `thr0.50_min2000/regions_typed.csv`, then
filter antiSMASH-overlap status, region length, type confidence, biosynthetic
HMM evidence, and MIBiG similarity.

## HMM / MIBiG Triage

The first triage pass used the `thr0.50_min2000` antiSMASH-missed bucket.

| Step | Count |
|---|---:|
| antiSMASH-missed candidates | 579 |
| HMM strong biosynthetic evidence | 357 |
| HMM moderate biosynthetic evidence | 78 |
| Pfam-only weak evidence | 49 |
| no informative HMM hits | 95 |
| MIBiG variant-level matches | 275 |
| MIBiG likely-novel matches | 170 |
| MIBiG novel matches | 134 |
| first-pass wet-lab PASS | 171 |

The wet-lab PASS gate is intentionally conservative about evidence but not yet
final: `n_biosynth_hits >= 3`, `n_biosynth_orfs >= 2`, region length at least
5 kb, and no near-identical MIBiG hit (`max_mibig_pident < 70`). The top handoff
table is `novel_triage/wetlab_candidates_top75.csv`; the full auditable table is
`novel_triage/novel_triage_full.csv`.
