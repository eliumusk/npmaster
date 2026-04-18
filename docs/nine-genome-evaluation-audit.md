# 9-Genome Evaluation Audit

Date: 2026-04-18

## Scope

This audit checks whether our current `9-genome` conclusions are being made
under the same metric contract as the benchmark code on the server.

Official benchmark scripts on server:

- `/data/syh/benchmark/nine_genome/run_eval.py`
- `/data/syh/benchmark/nine_genome/evaluate.py`
- `/data/syh/benchmark/nine_genome/gene_level_eval.py`
- `/data/syh/benchmark/nine_genome/run_npmaster.py`

Local comparison scripts:

- [eval_nine_genome_operating_point.py](/Users/musk/Documents/code/np-master/scripts/phase4/eval_nine_genome_operating_point.py)
- [diagnose_score_families.py](/Users/musk/Documents/code/np-master/scripts/phase4/diagnose_score_families.py)

## Audit Result

## 1. The current benchmark is using 305 GT regions, not 291

Server ground truth:

- file: `/data/syh/benchmark/nine_genome/ground_truth.json`
- total GT regions: `305`

Per-genome counts:

- `GCA_000158915.1`: `47`
- `GCA_000158875.1`: `32`
- `GCA_000154945.1`: `34`
- `GCA_000158895.1`: `35`
- `GCA_000156435.1`: `35`
- `GCA_000568915.1`: `31`
- `GCA_000158815.1`: `27`
- `GCA_000568255.1`: `38`
- `GCA_000156475.1`: `26`

Implication:

- If we verbally refer to the literature `291`-BGC `9-genome` benchmark, that is
  **not exactly the same object** as the current server benchmark.
- All current server-side benchmark numbers should be described as:
  `the current /data/syh/benchmark/nine_genome implementation with 305 GT regions`

## 2. Official region-level precision is a very loose metric

Official region-level logic in `run_eval.py` and `evaluate.py`:

- recall:
  a GT BGC is counted as found if the **sum of overlaps** from all predictions
  on that contig reaches `>= 50%` of the GT BGC length
- precision:
  a prediction is counted as correct if it has **any positive overlap at all**
  with any GT BGC

This is not a boundary-sensitive metric.

Consequences:

- oversized predictions are weakly penalized
- fragmented predictions can still recover recall if their overlaps add up
- the published `precision` in `results.json` is closer to a very loose
  prediction-level hit rate than to strict localization precision

## 3. Our local `loose` metric exactly reproduces the official NP-Master number at the default point

Official server `results.json` for NP-Master:

- recall: `0.6361`
- precision: `0.6343`
- f1: `0.6352`
- `tp_bgc = 194 / 305`
- `tp_pred = 111 / 175`

Using our local evaluator on the same remote predictions:

- command:
  `threshold=0.5`, `max_gap=16384`, `length_cap=None`
- [eval_nine_genome_operating_point.py](/Users/musk/Documents/code/np-master/scripts/phase4/eval_nine_genome_operating_point.py)

Result:

- `loose recall = 0.6361`
- `loose precision = 0.6343`
- `loose f1 = 0.6352`

Implication:

- for the default NP-Master point, our local `loose` numbers are directly
  comparable to the official benchmark numbers
- our `strict` and `IoU` metrics are **additional diagnostics**, not the metric
  used by the official benchmark

## 4. Official NP-Master benchmark runner is fixed to `max_type`, `threshold=0.5`, `stride=16384`

Official server runner:

- file: `/data/syh/benchmark/nine_genome/run_npmaster.py`

Current contract in that script:

- `WINDOW = 16384`
- `STRIDE = 16384`
- per-window score:
  `bgc_score = max(sigmoid(type_logits[:7]))`
- region calling:
  keep windows with `bgc_score >= 0.5`
- merge rule:
  contiguous windows with allowed extra gap `16384`

Implication:

- the official NP-Master benchmark numbers do **not** use:
  - `stride=8192`
  - `fused`
  - `threshold=0.95`
  - `length_cap`
  - `postprocess v2`

So all of our improved numbers are valid as diagnosis / tuned variants, but they
are **not** the official benchmark contract.

## 5. Official gene-level evaluation is a separate benchmark and should not be mixed with region results

Official gene-level logic:

- file: `/data/syh/benchmark/nine_genome/gene_level_eval.py`
- genes are built from feature tables in `/data/syh/benchmark/nine_genome/ft_cache`
- GT labels are gene-overlap labels derived from region GT
- region predictions from each tool are projected down to gene labels

Current official gene-level overall results:

- antiSMASH:
  `precision = 0.2614`, `recall = 0.6493`, `f1 = 0.3728`
- DeepBGC:
  `precision = 0.1711`, `recall = 0.8303`, `f1 = 0.2838`
- BGC-Prophet:
  `precision = 0.4924`, `recall = 0.4999`, `f1 = 0.4961`
- NP-Master:
  `precision = 0.1497`, `recall = 0.7107`, `f1 = 0.2473`

Implication:

- region-level `results.json` and gene-level `gene_level_results.json` are
  measuring different things
- we should never compare:
  - region `precision = 0.6343` for NP-Master
  - with gene-level `precision = 0.1497`
  as if they were the same metric

## 6. Our score-family diagnosis is useful, but it is not itself the official benchmark metric

In [diagnose_score_families.py](/Users/musk/Documents/code/np-master/scripts/phase4/diagnose_score_families.py),
we label `9-genome` windows positive if they have at least `1 bp` overlap with GT.

This is useful for:

- window-level AUROC
- window-level AUPRC
- checking negative-tail collapse
- comparing `max_type` vs `fused`

But it is not the same as either official benchmark:

- not the official loose region-level benchmark
- not the official gene-level benchmark

So window-level score diagnosis should be used only to guide model development,
not as the main reported benchmark table.

## 7. Which current conclusions are trustworthy

These conclusions are trustworthy:

- official default NP-Master benchmark:
  `recall = 0.6361`, `precision = 0.6343`, `f1 = 0.6352`
  under the current loose region-level contract
- our claim that default NP-Master collapses under strict/IoU localization:
  our local evaluator shows `strict f1 = 0.0125`, and this is a valid extra
  diagnostic because it is computed on the same predictions
- our claim that `stride=8192`, `length_cap`, and `min_support` improve
  strict/IoU localization:
  these are valid as tuned diagnostic results
- our claim that `fused` is only slightly better than `max_type`:
  valid for score diagnosis and frozen tuned comparisons

These conclusions should be stated more carefully:

- any claim of direct paper parity against the canonical `291`-BGC benchmark
- any claim that our strict/IoU numbers are the official benchmark metric
- any claim that `9-genome` "negatives" are clean negatives for training
  without qualification

## 8. Practical reporting rule going forward

For `9-genome`, we should report three clearly separated tables:

1. official region benchmark:
   the current loose contract from server `results.json`
2. official gene benchmark:
   the current `gene_level_results.json`
3. our diagnostic localization table:
   `loose / strict / IoU` under frozen or tuned operating points

This keeps benchmark reporting honest while still allowing us to improve the
decoder and inspect localization quality.

## Working Conclusion

Right now, the most defensible reading is:

- the current benchmark implementation is internally consistent
- but it is looser than the boundary-sensitive question we actually care about
- our stricter metrics are useful and scientifically important
- however they are **our added diagnostics**, not the official benchmark

So the immediate next step should be:

- keep the official loose benchmark as the baseline table
- keep strict/IoU as our localization audit table
- stop conflating region benchmark, gene benchmark, and window-score diagnosis
