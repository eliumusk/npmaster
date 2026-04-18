# 9-Genome Baseline Factor Analysis

Date: 2026-04-18
Model: `Route B s0` (`/data/muskliu/np_master/data/phase3/route_b/best_route_b_s0.pt`)
Predictions directory: `/data/syh/benchmark/nine_genome/results/npmaster`
Evaluator: [scripts/phase4/eval_nine_genome_operating_point.py](/Users/musk/Documents/code/np-master/scripts/phase4/eval_nine_genome_operating_point.py)

## Summary

Using the same `stride=16384` window predictions, threshold tightening alone does **not** fix strict 9-genome precision. The dominant gain comes from preventing oversized merged regions with a `length_cap`.

## Aggregate Results

| Operating point | Loose F1 | Strict F1 | IoU F1 |
| --- | ---: | ---: | ---: |
| `thr=0.5 gap=16384 cap=None` | 0.6352 | 0.0125 | 0.0125 |
| `thr=0.98 gap=0 cap=None` | 0.5825 | 0.0198 | 0.0198 |
| `thr=0.5 gap=16384 cap=50000` | 0.4752 | 0.1117 | 0.0968 |
| `thr=0.5 gap=0 cap=50000` | 0.4670 | 0.1107 | 0.0959 |
| `thr=0.98 gap=0 cap=50000` | 0.4858 | 0.1323 | 0.1161 |

## Interpretation

- `length_cap=50 kb` is the main lever that lifts strict performance from near-zero to a usable range.
- Removing `max_gap=16384` while keeping the same `length_cap=50 kb` changes the result only marginally.
- Tightening the threshold from `0.5` to `0.98` helps only when region length is already controlled.
- Current 9-genome failure mode is therefore dominated by **oversized merged regions**, not by a simple threshold mismatch.

## Current Tuned Contract

After rerunning 9-genome inference with `stride=8192`, the best v1 sweep point is:

- `score = max_type`
- `stride = 8192`
- `threshold = 0.95`
- `max_gap = 0`
- `length_cap = 30000`

At this point:

- `strict F1 = 0.1450`
- `IoU F1 = 0.1258`

Compared with the best `stride=16384` point (`thr=0.98 gap=0 cap=50000`):

- `strict F1`: `0.1323 -> 0.1450`
- `IoU F1`: `0.1161 -> 0.1258`

Interpretation:

- The main problem is still oversized merged regions.
- Finer stride does help, but as a second-order gain on top of a stricter region-construction contract.

This v1 contract is now superseded by the postprocess-v2 decoder in
[nine-genome-postprocess-v2.md](/Users/musk/Documents/code/np-master/docs/nine-genome-postprocess-v2.md),
which keeps the same stride and cap but adds `min_support_windows = 3`.

## Ongoing Experiment

- Full `stride=8192` re-inference for `Route B s0` is running on `Voc_gpu_node7_a800`.
- Goal: isolate whether stricter postprocessing is enough, or whether boundary resolution improves materially when the inference stride is made finer.

## Route B Seed Stability

Saved 9-genome sweeps for the three official Route B seeds show the same pattern:

| Seed | Window AUROC | Best strict F1 | Best strict precision | Best strict recall | Best point |
| --- | ---: | ---: | ---: | ---: | --- |
| `s0` | 0.8010 | 0.1310 | 0.1063 | 0.1705 | `thr=0.8 gap=0 cap=50000` |
| `s1` | 0.7985 | 0.1374 | 0.1076 | 0.1902 | `thr=0.1 gap=0 cap=50000` |
| `s2` | 0.7998 | 0.1333 | 0.1082 | 0.1738 | `thr=0.8 gap=0 cap=50000` |

Interpretation:

- The ranking signal is stable across seeds (`window AUROC ~ 0.80`).
- Strict region performance remains low across seeds even after sweeping operating points.
- All three seeds want `length_cap=50 kb`, reinforcing that oversized merged regions are the dominant current failure mode.

## Selected Visualization Case

For the boundary-comparison figure, use:

- Genome: `GCA_000568915.1`
- Contig: `KK037233.1`
- Zoom: `120 kb - 510 kb`

Reason:

- Under the old default NP-Master contract, this genome has `strict precision = 0.000`.
- Under the tuned `stride=8192` contract above, this rises to `0.151`.
- The selected zoom contains an old NP-Master merged region (`131072-507904`) that swallows 8 GT BGCs, making the boundary improvement visually obvious.
