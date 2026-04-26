# OER004256 Per-Token U-Net Readout

**Date**: 2026-04-26
**Status**: completed readout; not yet promoted to mainline
**Dataset**: OER004256 metagenome, contigs >= 10 kb, 16 FASTA shards
**Features**: Evo2 layer `blocks.20.mlp.l3`, window=8192, stride=2048, downsample=8, `d_reduced=128`
**Total windows**: 755,512

## Why This Run Exists

The first per-token head (`head_v1`) failed the 9-genome benchmark, mainly from low recall and fragmentation. It did, however, show a small short-BGC signal. This run tests the OER-first hypothesis: a per-token U-Net may help on short metagenomic contigs even if it is not an isolate-genome mainline replacement.

## Run Summary

Two Round-2 U-Net checkpoints were applied to the full OER feature set:

| tag | checkpoint | GPU | inference output |
|---|---|---:|---|
| E10 | `/data/muskliu/npmaster/experiments/evo2_per_token_unet/round2/E10_h64_lr1e3_b32/best.pt` | node7 GPU0 | `e10_probs/` |
| E11 | `/data/muskliu/npmaster/experiments/evo2_per_token_unet/round2/E11_h64_lr1e3_b64/best.pt` | node7 GPU1 | `e11_probs/` |

Inference completed for both checkpoints:

- 16 / 16 shard probability files written for E10 and E11.
- Runtime: 703.2 s per checkpoint, run in parallel (about 11.7 min wall time).
- Decode thresholds: 0.3, 0.4, 0.5.
- Decode parameters: `min_len_bp=2000`, `max_gap_bp=500`.

The original `extract_evo2_per_token_oer.sh --stage infer-decode` decode loop failed after successful inference because GNU `tr` rejects `tr '.' ''`. The local script has been fixed to use `tr -d '.'`. Decode was then run manually from the completed probability files; inference did not need to be repeated.

## AntiSMASH Readout

Evaluator: `scripts/evaluate/evo2_per_token_unet_oer.py`

Reference antiSMASH set:

- Path: `/data/muskliu/oer_bgc_scan/results/antismash`
- Regions: 1,976
- Median antiSMASH region length: 13,573 bp
- Matching rule: antiSMASH recall under asymmetric 50% overlap (`overlap / AS_region_len >= 0.5`)

| method | threshold | n_pred | AS TP | AS-recall | novel | AS-overlap rate of preds | median_len_bp |
|---|---:|---:|---:|---:|---:|---:|---:|
| E10 U-Net | 0.3 | 1,377 | 616 | **0.3117** | 470 | 0.6587 | 9,828 |
| E10 U-Net | 0.4 | 1,271 | 531 | 0.2687 | 388 | 0.6947 | 9,384 |
| E10 U-Net | 0.5 | 1,194 | 491 | 0.2485 | 325 | 0.7278 | 8,192 |
| E11 U-Net | 0.3 | 1,154 | 541 | 0.2738 | 382 | 0.6690 | 10,240 |
| E11 U-Net | 0.4 | 1,083 | 472 | 0.2389 | 321 | 0.7036 | 10,033 |
| E11 U-Net | 0.5 | 1,007 | 334 | 0.1690 | 265 | 0.7368 | 6,182 |
| v4.1-learned | production OP | 1,230 | 408 | 0.2065 | 429 | 0.6512 | 5,973 |
| v4.1-hysteresis | frozen OP | 254 | 187 | 0.0946 | 15 | 0.9409 | 12,288 |

Per-contig-length recall buckets:

| method | threshold | 10-20 kb | 20-50 kb | >=50 kb |
|---|---:|---:|---:|---:|
| E10 U-Net | 0.3 | 87/212 (0.410) | 23/108 (0.213) | 4/35 (0.114) |
| E10 U-Net | 0.4 | 72/212 (0.340) | 20/108 (0.185) | 4/35 (0.114) |
| E10 U-Net | 0.5 | 70/212 (0.330) | 18/108 (0.167) | 4/35 (0.114) |
| E11 U-Net | 0.3 | 74/212 (0.349) | 22/108 (0.204) | 4/35 (0.114) |
| E11 U-Net | 0.4 | 61/212 (0.288) | 18/108 (0.167) | 4/35 (0.114) |
| E11 U-Net | 0.5 | 53/212 (0.250) | 8/108 (0.074) | 3/35 (0.086) |
| v4.1-learned | production OP | 58/212 (0.274) | 13/108 (0.120) | 3/35 (0.086) |
| v4.1-hysteresis | frozen OP | 23/212 (0.108) | 4/108 (0.037) | 0/35 (0.000) |

## Interpretation

E10 at threshold 0.3 is the best OER recall point in this run: it lifts AS-recall from v4.1-learned's 0.2065 to 0.3117 (+10.5 percentage points), with a modestly larger candidate set (1,377 vs 1,230 regions).

E10 at threshold 0.4 is the conservative operating point: AS-recall remains above v4.1-learned (0.2687 vs 0.2065), while novel calls drop from 470 to 388 and the fraction of predictions overlapping antiSMASH rises to 0.6947.

E11 is weaker than E10 across all tested thresholds. It should be kept as a comparison run, not promoted.

## 9-Genome Readout

The same E10/E11 checkpoints were also scored on 9-genome with `scripts/evaluate/evo2_per_token_unet_nine_genome.py` and canonical `benchmark.nine_genome.eval_regions`.

| method | threshold | selection status | dedup_F1 (CI95) | bench_P | bench_R | strict_F1 | n_pred |
|---|---:|---|---:|---:|---:|---:|---:|
| E10 U-Net | 0.3 | OER recall-first | 0.4561 (0.3161-0.5621) | 0.4444 | 0.4984 | 0.4044 | 333 |
| E10 U-Net | 0.4 | OER conservative | 0.4664 (0.3240-0.5733) | 0.4809 | 0.4852 | 0.4136 | 314 |
| E10 U-Net | 0.5 | default threshold reference | 0.4751 (0.3305-0.5833) | 0.4983 | 0.4852 | 0.4243 | 303 |
| E11 U-Net | 0.3 | comparison | 0.4470 (0.2997-0.5525) | 0.4424 | 0.4852 | 0.3898 | 321 |
| E11 U-Net | 0.4 | comparison | 0.4500 (0.3111-0.5537) | 0.4870 | 0.4623 | 0.3850 | 308 |
| E11 U-Net | 0.5 | default threshold reference | 0.4544 (0.3171-0.5560) | 0.5067 | 0.4492 | 0.3914 | 298 |
| v4.1-hysteresis | formal mainline | no 9g tuning | 0.4878 (0.345-0.609) | 0.5976 | 0.4590 | 0.4299 | 246 |

Interpretation: E10 is close to v4.1 on 9-genome without choosing thresholds from 9g. The OER-preselected conservative point (`thr=0.4`) is 0.4664 dedup_F1, and the default threshold (`thr=0.5`) is 0.4751. However, the checkpoint/config family came from a round2 grid that was later read out on 9g, so these are threshold-clean diagnostics. A fully formal claim still needs a pre-registered checkpoint + threshold selection rule that does not inspect 9g.

## Caveats

- This is an antiSMASH-overlap readout, not a true OER F1/AUC benchmark. OER does not have manually curated positive/negative GT.
- "Novel" means no overlap with antiSMASH on the same contig. It is a candidate bucket, not a false-positive count.
- The candidate has 9-genome readouts under the same checkpoint and decoder settings, but not yet a fully pre-registered checkpoint + threshold protocol.
- No type-classification readout has been attached to these E10/E11 regions yet.

## Artifact Map

Remote server (`Voc_gpu_node7_a800`):

- Round-2 checkpoints and 9g regions: `/data/muskliu/npmaster/experiments/step2_per_token/round2/`
- Feature manifests and parts: `/data/muskliu/npmaster/experiments/step2_per_token/oer_w8192_s2048/OER004256_shard*.manifest.json`
- E10 probabilities: `/data/muskliu/npmaster/experiments/step2_per_token/oer_w8192_s2048/e10_probs/`
- E11 probabilities: `/data/muskliu/npmaster/experiments/step2_per_token/oer_w8192_s2048/e11_probs/`
- Decoded regions: `/data/muskliu/npmaster/experiments/step2_per_token/oer_w8192_s2048/e{10,11}_regions_thr{03,04,05}.csv`
- Readout JSONs: `/data/muskliu/npmaster/experiments/step2_per_token/oer_w8192_s2048/readouts/e{10,11}_thr{03,04,05}_compare.json`
- Logs: `/data/muskliu/npmaster/experiments/step2_per_token/oer_w8192_s2048/logs/`

Local repo:

- Launch script: `scripts/extract/extract_evo2_per_token_oer.sh`
- Inference script: `scripts/inference/evo2_per_token/infer_shards.py`
- Decode script: `scripts/decode/evo2_per_token_regions.py`
- OER readout script: `scripts/evaluate/evo2_per_token_unet_oer.py`
- This summary: `experiments/evo2_per_token_unet/OER_UNET_READOUT.md`
- 9g readout JSONs: `experiments/evo2_per_token_unet/nine_genome/E10_h64_lr1e3_b32/readouts/`, `experiments/evo2_per_token_unet/nine_genome/E11_h64_lr1e3_b64/readouts/`

## Recommended Next Steps

1. Pre-register a clean per-token/U-Net selection rule that uses internal validation and/or OER proxy metrics but not 9-genome.
2. Use E10 threshold 0.3 as the recall-first OER point and threshold 0.4 as the conservative OER point until that formal rule is written.
3. Add type labels and HMM/MIBiG triage only for the chosen E10 operating point, not all six CSVs.
4. Keep v4.1-hysteresis as the 9-genome formal mainline until the pre-registered E10-style protocol is executed end-to-end.
