# Route B Hard-Negative V1

Date: 2026-04-18

## Goal

Raise `9-genome` region-level `precision` under the newer `50% overlap`
contract without falling back to benchmark-specific postprocess tuning.

## Current Reading

Under the newer region-level metric:

- `NP-Master (orig)`: `R=0.6361`, `P=0.0171`, `F1=0.0334`
- `stride8192_tuned_v2`: `R=0.5311`, `P=0.2513`, `F1=0.3412`
- `stride8192_val_frozen_fused`: `R=0.5934`, `P=0.1649`, `F1=0.2580`

Interpretation:

- postprocess tuning already fixed the largest decoder failure
- remaining gain is more likely to come from training-side precision work
- `internal val frozen` remains the formal operating point

## V1 Intervention

Two changes only:

1. **Fresh Route B hard-negative mining**
   - script: `scripts/mine_hard_negatives.py`
   - source: current Route B checkpoint scores on `train` windows labeled `negative`
   - default score family: `fused`
   - default selection: `top_k=5000`, `min_score=0.95`

2. **Route B loss-side hard-negative weighting**
   - dataset: `src/data/chunk_dataset.py`
   - trainer: `src/train.py`
   - mechanism: keep the same chunk traversal, but assign extra loss weight to flagged
     hard-negative positions inside `loss_mask`

This avoids changing:

- chunk construction
- validation protocol
- decoder contract
- backbone / architecture

## Logging Contract

Every run should write:

- mining:
  - `summary.json`
  - `hardneg_candidates.csv`
  - patched `meta_arrays.pkl`
- training:
  - `config_<tag>.json`
  - `train_log_<tag>.csv`
  - `summary_<tag>.json`
  - `best_<tag>.pt`
  - `last_<tag>.pt`

The training CSV now also records:

- `hard_neg_frac`

so we can verify the run is actually seeing flagged windows.

## First Experiment

Recommended first run:

- mine from:
  - checkpoint: `data/phase3/route_b/best_route_b_s0.pt` (server)
- train:
  - seed `0`
  - same Route B settings as mainline
  - `hard_neg_weight = 4.0`

Reason:

- one seed is enough to test direction
- weight-only is the smallest change from the current mainline
- if this does not improve precision/F1 under the `internal val frozen` and
  `9-genome` tables, we should not spend multi-GPU budget on 3-seed replication

## Decision Rule

Promote to 3-seed only if the first seed shows at least one of:

- better `9-genome` region-level `precision`
- better `9-genome` region-level `F1`
- no material regression on the internal mainline benchmark
