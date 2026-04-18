# Score Family Diagnosis

Date: 2026-04-18

## Question

After the 9-genome postprocessing diagnosis, the next question is whether the
main limitation is still in the region decoder or whether the score family
itself should change.

We focused on:

- `max_type = max(probs[:7])`
- `fused = 0.5 * max_type + 0.5 * (1 - is_negative)`

## Internal-Val Postprocess V2 Sweep

Using the cached Route B `val` predictions:

- cache: `/data/muskliu/np_master/data/phase4/postprocess/cached_preds_val_s0.npz`
- output: `/data/muskliu/np_master/data/phase4/postprocess/sweep_val_postprocess_v2_s0.json`

Best `max_type` point:

- `start_threshold = 0.98`
- `extend_threshold = 0.95`
- `max_gap = 0`
- `length_cap = 50000`
- `min_support_windows = 1`
- `strict F1 = 0.5606`
- `IoU F1 = 0.5174`

Best `fused` point:

- `start_threshold = 0.98`
- `extend_threshold = 0.95`
- `max_gap = 0`
- `length_cap = 50000`
- `min_support_windows = 1`
- `strict F1 = 0.5610`
- `IoU F1 = 0.5210`

Interpretation:

- `fused` is only slightly better than `max_type` on internal `val`.
- Both score families prefer the same decoder shape.
- Internal validation does **not** support the more aggressive 9-genome-tuned
  `length_cap = 30000`, `min_support_windows = 3` point.

## Window-Level Score Diagnosis

To separate decoder issues from score-family issues, we added:

- script: [diagnose_score_families.py](/Users/musk/Documents/code/np-master/scripts/phase4/diagnose_score_families.py)
- remote output: `/tmp/score_family_diagnosis_s0.json`

Config:

- `internal val`: Route B `s0`, split `val`, `ambiguous` windows excluded
- `9-genome`: `stride=8192` merged CSVs at `/tmp/npmaster_stride8192_s0_merged`
- window-positive definition on `9-genome`: at least `1 bp` overlap with GT

### Internal `val`

Sample counts:

- `n_total = 88,990`
- `n_positive = 7,245`
- `n_negative = 81,745`

| score | AUROC | AUPRC | neg q99 | neg rate @ 0.95 | pos rate @ 0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `max_type` | `0.9802` | `0.9432` | `0.1934` | `0.00486` | `0.8678` |
| `1 - is_negative` | `0.9802` | `0.9400` | `0.1578` | `0.00429` | `0.8540` |
| `fused` | `0.9812` | `0.9424` | `0.1983` | `0.00428` | `0.8584` |

Interpretation:

- Internal `val` is already very well separated at the window level.
- `fused` has the best `AUROC`, but the gain over `max_type` is small.
- At a hard threshold such as `0.95`, all three score families still keep the
  negative tail very low on `val` (`~0.4%` false-positive rate).

### 9-Genome

Sample counts:

- `n_total = 9,460`
- `n_positive = 1,276`
- `n_negative = 8,184`

| score | AUROC | AUPRC | neg q95 | neg q99 | neg rate @ 0.95 | pos rate @ 0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `max_type` | `0.8214` | `0.5345` | `0.999975` | `0.999998` | `0.1232` | `0.6959` |
| `1 - is_negative` | `0.8244` | `0.5270` | `0.999933` | `0.999997` | `0.1211` | `0.6967` |
| `fused` | `0.8256` | `0.5380` | `0.999939` | `0.999996` | `0.1193` | `0.6951` |

Interpretation:

- `fused` is again slightly best overall, but the gain is still modest.
- The real problem is the negative tail collapse on `9-genome`.
- On `internal val`, the `0.95` threshold lets through only `~0.4%` of
  negatives. On `9-genome`, the same threshold lets through `~12%` of
  negatives.
- More importantly, the `9-genome` negative tail is already almost saturated:
  `neg q95 ~= 0.9999`, `neg q99 ~= 1.0`.

This is a different failure mode from the decoder-only story. Even before
region merging, many benchmark-negative windows already receive near-maximum
scores.

### Tail Behavior

`9-genome` high-score negatives are not uniformly spread.

For `max_type`, the top-20 highest-score negatives are concentrated in:

- `GCA_000154945.1` (`11/20`)
- `GCA_000158895.1` (`8/20`)
- `GCA_000158915.1` (`1/20`)

For `fused`, the concentration broadens slightly, but the same genomes still
dominate:

- `GCA_000158895.1` (`7/20`)
- `GCA_000154945.1` (`7/20`)
- `GCA_000158915.1` (`2/20`)
- `GCA_000568915.1` (`2/20`)
- `GCA_000568255.1` (`2/20`)

This suggests we should inspect those genomes as hard-negative sources instead
of treating the problem as uniform random noise.

The lowest-score benchmark-positive windows are mixed:

- some are marginal overlap windows with very small GT overlap (`strict < 0.05`)
- but several have meaningful overlap (`strict ~0.40-0.56`) and still get
  scores near zero

So the benchmark miss story is not just label noise at the boundary.

## Cross-Tool Support For Benchmark-Negative High-Score Regions

To distinguish "model false positives" from "benchmark-negative but biologically
plausible regions", we checked the `9-genome` high-score negative regions under:

- score: `fused`
- threshold: `0.95`
- merge gap: `8192`
- source predictions: `/tmp/npmaster_stride8192_s0_merged`

For each merged region that had **no GT overlap**, we asked whether it overlaps
with any region called by:

- `antiSMASH`
- `DeepBGC` (`deepbgc_score >= 0.5`)
- `BGC-Prophet`

### Aggregate Result

- total GT-negative high-score regions: `90`
- with at least one external-tool overlap: `89`
- fraction with any support: `0.9889`

For long regions (`>= 30 kb`):

- long GT-negative high-score regions: `75`
- long regions with any support: `74`
- fraction with any support: `0.9867`

Tool-support counts across the 90 regions:

- `antiSMASH`: `78`
- `DeepBGC`: `67`
- `BGC-Prophet`: `13`
- `none`: `1`

The single unsupported region is:

- `GCA_000154945.1`, `CM000950.1:1441792-1474560`, `32.8 kb`, `3 windows`,
  `peak fused score = 0.992457`

### Interpretation

This is the strongest evidence so far that the remaining failure is **not
primarily decoder tuning**.

If almost every benchmark-negative high-score region is also supported by
`antiSMASH` or another external tool, then those regions are unlikely to be
uniform random false positives. More likely explanations are:

- the `9-genome` GT is narrower than the region sets emitted by the tools
- the GT omits regions that antiSMASH / DeepBGC / BGC-Prophet still consider
  BGC-like
- our model is learning a broader "BGC-like" signal than the benchmark GT
  accepts

So the next problem is better described as a **data / label / benchmark-contract
problem**, not just a postprocessing-parameter problem.

## Frozen 9-Genome Check

We then froze the internal-`val`-selected operating point and applied it to the
completed `stride=8192` 9-genome predictions.

### Frozen `max_type`

- `strict F1 = 0.1216`
- `IoU F1 = 0.0726`

### Frozen `fused`

- `strict F1 = 0.1276`
- `IoU F1 = 0.0757`

Interpretation:

- `fused` also improves slightly on 9-genome under the frozen validation point.
- The gain is real but small.
- The main story remains unchanged: score family matters less than the region
  construction contract, and the 9-genome-only optimum does not generalize from
  internal validation.

## Working Conclusion

For formal frozen evaluation, the most defensible contract is currently:

- `score = fused`
- `start_threshold = 0.98`
- `extend_threshold = 0.95`
- `max_gap = 0`
- `length_cap = 50000`
- `min_support_windows = 1`

For benchmark diagnosis only, the stronger 9-genome-specific point is still:

- `score = max_type`
- `stride = 8192`
- `threshold = 0.95`
- `max_gap = 0`
- `length_cap = 30000`
- `min_support_windows = 3`

These two points should not be conflated.

## Updated Working Conclusion

The score-family diagnosis changes the roadmap in one important way:

- `fused` should remain the default formal score, because it is consistently
  slightly better than `max_type`
- but postprocessing is no longer the only bottleneck
- the next real bottleneck is cross-benchmark score calibration / hard
  negatives

The strongest evidence is the gap between:

- `internal val`: excellent ranking, low negative tail
- `9-genome`: only moderate ranking, but extremely heavy negative tail

and the external-support result:

- `9-genome` GT-negative high-score regions are almost all supported by at least
  one external tool (`89 / 90`)

So the next step should move from decoder tuning to data / score-layer work:

1. define a cleaner diagnostic subset:
   high-score negatives with no external-tool support vs high-score negatives
   with antiSMASH/DeepBGC/BGC-Prophet support
2. inspect the unsupported or weakly supported subset first, because that is
   closer to true model error
3. revisit hard-negative mining / sampling using those windows, instead of
   treating all 9-genome benchmark negatives as equally reliable negatives
4. only after that revisit loss design or model architecture
