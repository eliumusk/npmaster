# 9-Genome Postprocess V2

Date: 2026-04-18
Status: completed

## Goal

Second-stage postprocessing experiments on the completed `stride=8192` 9-genome predictions.

This stage tests whether a slightly more structured region decoder can improve
strict/IoU region quality beyond the current tuned contract:

- `score = max_type`
- `stride = 8192`
- `threshold = 0.95`
- `max_gap = 0`
- `length_cap = 30000`

## Decoder Additions

- `start_threshold`: high-confidence seed threshold
- `extend_threshold`: lower threshold for region growth
- `min_support_windows`: minimum number of supporting windows
- `valley_threshold`: optional score floor used to split sustained weak bridges
- `valley_min_windows`: minimum valley width before splitting

## Implementation

- Decoder module: [postprocess_v2.py](/Users/musk/Documents/code/np-master/scripts/phase4/postprocess_v2.py)
- Sweep entrypoint: [sweep_nine_genome_postprocess_v2.py](/Users/musk/Documents/code/np-master/scripts/phase4/sweep_nine_genome_postprocess_v2.py)
- Tests: [test_nine_genome_tools.py](/Users/musk/Documents/code/np-master/tests/test_nine_genome_tools.py)

## Experiment Grid

- `start_thresholds = 0.90, 0.95, 0.98`
- `extend_thresholds = 0.80, 0.85, 0.90, 0.95`
- `max_gaps = 0, 8192, 16384`
- `length_caps = 30000, 50000`
- `min_support_windows = 1, 2, 3`
- `valley_thresholds = none, 0.85, 0.90`
- `valley_min_windows = 1, 2`

## Current Notes

- The old problem is clearly oversized merged regions.
- The current tuned contract fixes that, but still produces many window-shaped short calls.
- This stage asks whether we can remove some of that fragmentation without falling back to giant merged regions.

## Results

Remote output:

- `/tmp/stride8192_postprocess_v2_sweep.json`

Best strict and best IoU point are the same:

- `start_threshold = 0.95`
- `extend_threshold = 0.95`
- `max_gap = 0`
- `length_cap = 30000`
- `min_support_windows = 3`
- `valley_threshold = none`
- `valley_min_windows = 1`

Aggregate metrics:

- `strict F1 = 0.1475`
- `IoU F1 = 0.1280`
- `strict precision = 0.1026`
- `strict recall = 0.2623`

Compared with the previous tuned contract (`stride=8192`, `threshold=0.95`, `max_gap=0`, `length_cap=30000`):

- `strict F1`: `0.1450 -> 0.1475`
- `IoU F1`: `0.1258 -> 0.1280`
- `strict precision`: `0.1002 -> 0.1026`

## Interpretation

- The useful gain comes from `min_support_windows = 3`.
- Dual-threshold hysteresis does not help beyond the best single-threshold point in this grid.
- Valley splitting also does not help in this first sweep.
- The current best decoder is therefore simpler than expected:
  - single threshold at `0.95`
  - `min_support_windows = 3`
  - `length_cap = 30000`

## Current Recommendation

For the next round of frozen 9-genome evaluation, use:

- `score = max_type`
- `stride = 8192`
- `start_threshold = 0.95`
- `extend_threshold = 0.95`
- `max_gap = 0`
- `length_cap = 30000`
- `min_support_windows = 3`

This supersedes the previous `v1` tuned contract that had the same thresholds and cap but `min_support_windows = 1`.

## Internal-Val Reality Check

We then ran the same decoder family on the cached internal `val` predictions:

- cache: `/data/muskliu/np_master/data/phase4/postprocess/cached_preds_val_s0.npz`
- output: `/data/muskliu/np_master/data/phase4/postprocess/sweep_val_postprocess_v2_s0.json`

Best `val` point for `max_type`:

- `start_threshold = 0.98`
- `extend_threshold = 0.95`
- `max_gap = 0`
- `length_cap = 50000`
- `min_support_windows = 1`
- `strict F1 = 0.5606`
- `IoU F1 = 0.5174`

Best `val` point for `fused` is slightly better:

- `start_threshold = 0.98`
- `extend_threshold = 0.95`
- `max_gap = 0`
- `length_cap = 50000`
- `min_support_windows = 1`
- `strict F1 = 0.5610`
- `IoU F1 = 0.5210`

This is important:

- internal `val` does **not** choose the 9-genome-tuned `min_support_windows = 3` + `length_cap = 30000` point
- instead it prefers the earlier, simpler `length_cap = 50000`, `min_support_windows = 1` decoder

## Frozen Benchmark Check

We froze the internal-`val`-selected `max_type` point and applied it back to 9-genome:

- `start_threshold = 0.98`
- `extend_threshold = 0.95`
- `max_gap = 0`
- `length_cap = 50000`
- `min_support_windows = 1`

9-genome result under this frozen point:

- `strict F1 = 0.1216`
- `IoU F1 = 0.0726`

Interpretation:

- the stronger 9-genome-only postprocess point does **not** generalize from internal validation
- therefore it should be treated as a benchmark-diagnostic point, not the formal frozen contract
- for reporting and downstream model work, the validation-selected contract is more trustworthy even if its 9-genome score is lower
