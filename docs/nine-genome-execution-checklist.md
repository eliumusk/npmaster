# 9-Genome Execution Checklist

**Date:** 2026-04-18
**Purpose:** exact execution order for the current 9-genome debug phase

---

## 1. Files To Watch

- Contract: [nine-genome-evaluation-contract.md](/Users/musk/Documents/code/np-master/docs/nine-genome-evaluation-contract.md)
- 9-genome benchmark root: `/data/syh/benchmark/nine_genome`
- Internal sweep script: [scripts/phase4/postprocess_sweep.py](/Users/musk/Documents/code/np-master/scripts/phase4/postprocess_sweep.py)
- 9-genome sweep script: [scripts/phase4/nine_genome_sweep_s1.py](/Users/musk/Documents/code/np-master/scripts/phase4/nine_genome_sweep_s1.py)

---

## 2. Current Facts

Known current benchmark outputs:

- region-level summary: `/data/syh/benchmark/nine_genome/evaluation/results.json`
- gene-level summary: `/data/syh/benchmark/nine_genome/evaluation/gene_level_results.json`
- 9-genome sweep (existing): `/data/muskliu/np_master/data/phase4/nine_genome_sweep_muskliu_s1/sweep_muskliu_s1_max_type.json`
- internal postprocess sweep (existing test-only): `/data/muskliu/np_master/data/phase4/postprocess/sweep_results.json`

Known protocol problem:

- `/data/syh/benchmark/nine_genome/evaluate.py` currently evaluates NP-Master using fixed `threshold=0.5`
- it also merges across one full-window gap
- this is diagnostic only and must not be treated as the final frozen operating point

---

## 3. Mandatory Run Order

### Step A: Confirm Existing 9-genome Baseline

Run:

```bash
ssh Voc_gpu_node7_a800 'python - <<'"'"'PY'"'"'
import json
with open("/data/syh/benchmark/nine_genome/evaluation/results.json") as f:
    print(json.load(f))
PY'
```

What this answers:

- what the current default benchmark says today

### Step B: Confirm Existing 9-genome Window/Postprocess Diagnostic

Run:

```bash
ssh Voc_gpu_node7_a800 'python - <<'"'"'PY'"'"'
import json
p="/data/muskliu/np_master/data/phase4/nine_genome_sweep_muskliu_s1/sweep_muskliu_s1_max_type.json"
with open(p) as f:
    d=json.load(f)
print(d["window_diagnostic"])
print(sorted(d["sweep_results"], key=lambda r: r["f1_strict"], reverse=True)[:5])
PY'
```

What this answers:

- is the score informative at window level
- how sensitive 9-genome region metrics are to postprocessing

### Step C: Select Frozen Operating Point On Internal Val

Run:

```bash
ssh Voc_gpu_node7_a800 "bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate bgc && python /data/muskliu/np_master/code/np-master/scripts/phase4/postprocess_sweep.py --cache /data/muskliu/np_master/data/phase4/postprocess/cached_preds_val_s0.npz --out /data/muskliu/np_master/data/phase4/postprocess/sweep_val_results.json --splits val --device cuda'"
```

Then summarize:

```bash
ssh Voc_gpu_node7_a800 'python - <<'"'"'PY'"'"'
import json
p="/data/muskliu/np_master/data/phase4/postprocess/sweep_val_results.json"
with open(p) as f:
    d=json.load(f)
rows=d["sweep"]["val"]["max_type"]["rows"]
for r in rows:
    r["f1_strict"]=2*r["p_strict"]*r["recall_strict"]/(r["p_strict"]+r["recall_strict"]+1e-9)
top=sorted(rows,key=lambda r:r["f1_strict"],reverse=True)[:10]
for r in top:
    print({k:r[k] for k in ["threshold","max_gap","length_cap","n_regions","recall_loose","p_loose","recall_strict","p_strict","f1_strict","median_len_kb"]})
PY'
```

What this answers:

- which operating point should be frozen without touching 9-genome test

### Step D: Run Final Frozen 9-genome Comparison

Do not tune here. Re-run only with the frozen operating point from Step C.

Minimum comparison:

- current default benchmark result
- frozen conservative operating point

Required outputs:

- region-level recall
- prediction-level precision
- region-level F1
- total predicted regions
- gene-level precision / recall / F1

---

## 4. Decision Rules

### If Step B shows:

- window AUROC/AUPRC is reasonable
- but region-level metrics move a lot across `threshold/gap/cap`

Then conclude:

- the main problem is postprocessing, not backbone quality

### If Step C returns a much more conservative point than the current benchmark default:

Then conclude:

- the current benchmark is using an overly permissive operating point

### If Step D with the frozen point still looks bad:

Then move to model/data interventions:

- score family redesign
- checkpoint selection alignment
- hard-negative policy
- label geometry
- context model changes

---

## 5. Working Candidate

This is not final, but current internal evidence suggests the frozen point will likely be more conservative than the benchmark default, probably in the neighborhood of:

- `score = max_type`
- `threshold >= 0.9`
- `max_gap = 0`
- `length_cap = None or 80k`

This is only a working hypothesis until Step C completes.
