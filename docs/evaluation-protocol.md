# Evaluation Protocol

## Task

Input DNA contigs are converted to BGC region predictions. The model does not generate new DNA or chemical structures.

## 9-Genome Benchmark

9-genome is held out for formal reporting. Do not tune thresholds, model families, hyperparameters, or operating points on it.

Primary metrics:

- `dedup_f1`
- `bench_precision`
- `bench_recall`

`bench` is asymmetric 50% overlap:

- Recall TP: `overlap / gt_len >= 0.5`
- Precision TP: `overlap / pred_len >= 0.5`

Dedup F1 is mandatory because raw precision can be inflated by fragmenting one GT region into many short predictions.

## OER / Metagenome Readouts

OER004256 does not have manually curated full ground truth. AntiSMASH overlap is a proxy readout:

- AS-recall: fraction of antiSMASH regions covered by a prediction under asymmetric 50% overlap.
- Novel candidates: predictions that do not overlap antiSMASH on the same contig.

Novel here means antiSMASH-missed candidate, not validated new chemistry.

## Formal vs Diagnostic

- `evo2_lr_multiscale` is the current formal 9-genome mainline.
- `evo2_per_token_unet` is diagnostic until checkpoint and threshold selection are pre-registered without inspecting 9-genome.
