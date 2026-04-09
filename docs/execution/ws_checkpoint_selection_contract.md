# Checkpoint Selection Contract

**Date:** 2026-04-09
**Task ID:** A_checkpoint_selection_contract

---

## 1. Problem

The previous checkpoint selection used `val_mean_auroc` — the average per-type AUROC across 7 BGC classes. This optimized for **type discrimination**, not for **BGC detection**.

The canonical detection score is `score_maxtype = max(sigmoid(logit[0:7]))`, and its ranking quality is measured by `bgc_binary_auroc = AUROC(is_bgc, max(probs[:, :7]))`.

Selecting by `mean_auroc` can pick a checkpoint that discriminates types well but ranks BGC-vs-background poorly, which is what actually matters for deployment.

## 2. New Selection Metric

| Parameter | Old | New |
|-----------|-----|-----|
| **Selection metric** | `val_mean_auroc` | **`val_bgc_binary_auroc`** |
| **What it measures** | Average per-type ranking quality | BGC-vs-background ranking quality using `score_maxtype` |
| **Code location** | `train.py:214-216` | Same location, now configurable |
| **Config key** | (hardcoded) | `train.checkpoint_metric` (default: `"bgc_binary_auroc"`) |

## 3. Why `bgc_binary_auroc`

1. **Directly measures canonical score quality.** `bgc_binary_auroc` is computed as `AUROC(is_bgc, max(probs[:, :7]))` — exactly the ranking of `score_maxtype` on val. A checkpoint with higher `bgc_binary_auroc` will produce better region-level recall/precision under the canonical threshold contract.

2. **Aligned with deployment task.** The deployment task is "detect BGC regions," not "classify BGC types." Type classification is secondary and measured separately by `type_concordance`.

3. **Empirically justified.** In WS2-T2 training, `bgc_binary_auroc` and `mean_auroc` tracked closely but diverged at key epochs. The best `mean_auroc` epoch was not always the best `bgc_binary_auroc` epoch.

## 4. Implementation

### Code changes

**`bgc-gcd/train.py`:**

```python
# Line ~115: Read configurable metric
ckpt_metric = cfg["train"].get("checkpoint_metric", "bgc_binary_auroc")

# Line ~214: Select by deployment-facing metric
current_metric = val_metrics.get(ckpt_metric, val_metrics["mean_auroc"])
if current_metric > best_val:
    best_val = current_metric
    torch.save(ckpt, out_dir / f"best_{log_tag}.pt")
```

**Checkpoint now includes metadata:**
```python
ckpt = {"model": ..., "epoch": ..., "config": ..., "val": ...,
        "checkpoint_metric": ckpt_metric}  # records which metric was used
```

### Config

Add to any training config:
```yaml
train:
  checkpoint_metric: bgc_binary_auroc
```

If omitted, defaults to `bgc_binary_auroc` (new default). To restore old behavior, set `checkpoint_metric: mean_auroc`.

### Backward compatibility

- Old configs without `checkpoint_metric` will now default to `bgc_binary_auroc` instead of `mean_auroc`. This is an intentional default change.
- Old checkpoints are unaffected (they don't store `checkpoint_metric`; the field simply won't be present).
- The training log CSV still records both `val_mean_auroc` and `val_bgc_auroc`, so both are visible for comparison.

## 5. Verification

- [x] Training log shows `checkpoint_metric=bgc_binary_auroc` at startup
- [x] Best checkpoint saved when `bgc_binary_auroc` improves (marked with `★`)
- [x] Final summary records `best_val_bgc_binary_auroc` instead of `best_val_mean_auroc`
- [x] Old training flow unbroken (setting `checkpoint_metric: mean_auroc` restores old behavior)

## 6. Changed Files

| File | Change |
|------|--------|
| `bgc-gcd/train.py` | Configurable `checkpoint_metric`, default `bgc_binary_auroc` |
| `bgc-gcd/configs/ws2_t2_prior_aligned.yaml` | Added `checkpoint_metric: bgc_binary_auroc` |
