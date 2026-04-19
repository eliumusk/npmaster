# Stage 5 overnight autonomous run — status snapshot

_Started 2026-04-20 17:31 UTC; this doc written while trials were still running._

## What's running

| Wave | Trials | Seeds | Node(s) | GPUs | Status |
|---|---|---|---|---|---|
| 1 | 6 variants (baseline_v1, tight_pos_075_v1, tight_pos_090_v1, truncated_exclude_v1, short_bgc_upweight_v1, hardpos_hardneg_v2) | 0 | node7 | 0-5 | launched, training |
| 2 | same 6 variants | 1 | node8 | 0-5 | launched, training |
| 3 | 4 high-signal variants (baseline, tight_pos_075, tight_pos_090, hardpos_hardneg_v2) | 2 | split: node7 GPU 6-7 + node8 GPU 6-7 | 6-7 per node | launched, training |
| 4 | tight_pos_085/095/099 + tight_pos_090 at 120 epochs | mixed (see `configs/sweeps/stage5_no9_data_v1_wave4.yaml`) | auto — picks up GPUs as wave-1/2/3 finish | variable | queued via `scripts/launch_stage5_wave4_auto.sh` |

24 trials total. All 16 A800s saturated; wave-4 auto-launcher polls `nvidia-smi` every 30 s and dispatches the next queued trial the moment any GPU drops below 500 MiB of usage.

## Autonomous orchestrator

`scripts/stage5_orchestrator.sh` is running on node7 under nohup. When ≥7 of node7's GPUs are idle for 2 consecutive 2-minute polls (i.e. training phase is done), it will:

1. Run `scripts/summarize_stage5_results.py` → `experiments/stage5_no9_data_v1-summary.csv` + `.md`.
2. Run `scripts/stage5_pick_winners.py --top 3` → `experiments/stage5_no9_data_v1-selection_decision.md` freezing 3 winner variants (averaged across completed seeds, min 30 epochs).
3. Parse the freeze file for the best-seed checkpoint of each winner.
4. Launch `scripts/launch_stage5_9genome_eval.sh` on those checkpoints — inference (`evo2_bgc` env) + `evaluate_nine_genome.py` (bgc env, `val_frozen_fused` operating point, `--window 8192 --stride 2048`).

Every action appends to `experiments/stage5_no9_data_v1-orchestrator.log`.

## Where results land

| artefact | path |
|---|---|
| per-trial train log | `experiments/stage5_no9_data_v1-<variant>_s<seed>/artifacts/checkpoints/train_log_*.csv` |
| patched meta arrays | `experiments/stage5_no9_data_v1-<variant>_s<seed>/inputs/meta_arrays_<variant>.pkl` |
| best checkpoint | `experiments/stage5_no9_data_v1-<variant>_s<seed>/artifacts/checkpoints/best_<variant>_s<seed>.pt` |
| ranked summary | `experiments/stage5_no9_data_v1-summary.csv` + `.md` |
| selection freeze | `experiments/stage5_no9_data_v1-selection_decision.md` |
| 9-genome eval (per winner) | `experiments/<winner>/metrics/final_9genome_<tag>.json` and `.../artifacts/predictions_9genome_<tag>/` |

## Early signal from epoch 4-9 (not the decisive metric)

```
tight_pos_090_v1     peak val F1 0.8905 (s0) 0.8826 (s1) 0.8491 (s2)
tight_pos_075_v1     peak val F1 0.8731 (s0) 0.8761 (s1) 0.8589 (s2)
short_bgc_upweight_v1 peak val F1 0.8357 (s0) 0.8294 (s1)
baseline_v1          peak val F1 0.8306 (s0) 0.8214 (s1) 0.8084 (s2)
hardpos_hardneg_v2   peak val F1 0.8199 (s0) 0.7925 (s1) 0.8153 (s2)
truncated_exclude_v1 peak val F1 0.8199 (s0) 0.7925 (s1)
```

**Important caveat:** every variant is evaluated against its *own* tight labels — a tight_pos_090 model is scored on val windows with overlap ≥ 0.90. That's an easier precision bar than evaluating the baseline on overlap ≥ 0.50. So the internal-val ranking is suggestive but not decisive. The real comparison happens on 9-genome, where GT is a fixed human-curated tight core and all variants compete head-to-head.

## Why these particular variants?

See `docs/stage5-data-audit.md §5` for the full intervention ladder and §1–§2 for the data-side findings that motivated it:

- **Training pool is 100 % antiSMASH**; zero MIBiG clusters. Plan's `mibig_boost_v1` is not implementable from current data.
- **Positive label rule is overlap ≥ ~0.5** on antiSMASH regions — a window whose centre sits ~4 kb *past* an antiSMASH edge still counts as positive. This is exactly the source of wide predictions relative to 9-genome tight-core GT.
- **Tight-threshold variants** (`tight_pos_075/085/090/095/099`) directly shrink the "fire zone" during training without needing any new data.
- **`truncated_exclude_v1`** masks the 14 321 positives whose source BGC hits a contig edge (31.7 % of bgc_annotations rows are truncated) — their boundaries are unreliable by construction.
- **`short_bgc_upweight_v1`** and **`hardpos_hardneg_v2`** reweight the 511 short (< 5 kb) BGCs + the 8 304 mined hard negatives via the existing `--hard-neg-weight` loss path.

## Failure modes the user should check when they wake up

- **Orchestrator hung or crashed**: read `experiments/stage5_no9_data_v1-orchestrator.log`. Expected final lines look like `orchestrator done — 9-genome eval running in background`.
- **Wave-4 didn't fire**: check `experiments/stage5_no9_data_v1-launch_wave4_node{7,8}.log`. Auto-launcher has a 6-hour deadline; if wave-1/2/3 stalled, wave-4 may time out.
- **Training divergence**: any row in `summary.md` with status `done` but `peak_val_f1 < 0.5` — look at that trial's `logs/run_s*.log`.
- **9-genome eval failed**: look at `experiments/<winner>/logs/final_9genome_*.log`. Most common cause: Evo2 model path / env activation issue.

## What to do next

1. `cat experiments/stage5_no9_data_v1-summary.md` — see final internal-val ranking across all 24 trials.
2. `cat experiments/stage5_no9_data_v1-selection_decision.md` — see frozen winners.
3. `for f in experiments/stage5_no9_data_v1-*/metrics/final_9genome_*.json; do jq -r '"\(.trial_id // input_filename)  f1=\(.bench_f1)  dedup=\(.dedup_f1)  n_pred=\(.n_pred)"' "$f"; done` — scan 9-genome outcomes.
4. Compare to `defrag_v2_ms10_s099` (the 9-genome-tuned upper bound at F1 0.4385). Anything at or near that number, **without ever inspecting 9-genome during selection**, is a legitimate Stage 5 success.
