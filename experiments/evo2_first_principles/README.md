# evo2_first_principles/

The design campaign that produced v4.1, run 2026-04-21 → 2026-04-22.

Principle: declare operating points and architecture choices from first principles on internal-val, then read out ONCE on 9-genome under `docs/evaluation-protocol.md`. No OP tuning on 9-genome.

## Files

### Final frozen readout
- `v4_1_final_readout.json` — v4.1 multi-scale ensemble, 9-genome honest readout. dedup_F1 = 0.4878 (CI95 0.345–0.609). Three blocks: `primary_only (v4 baseline)`, `short_only (L<10kb)`, `v4.1 ensemble`.

### Short-BGC rescue arm
- `short_bgc/w4096_honest.json` — w=4096/s=1024 scale honest readout (the "short rescue" fed into v4.1).

### Stage 0 ablation archive (`stage0/`)
30 JSON files covering architecture + label variants tested before v4.1 was frozen. See `docs/reports/overnight_exploration_2026-04-22.md` for the full log.

| Prefix | What was tested | Winner? |
|---|---|---|
| `exp_a_*` | Baseline mean-LR (w=8192) | yes — became v4 baseline |
| `exp_c_*_gene_context__*` | Gene-level context heads (bilstm/cnn/lr/mlp/transformer) loose-overlap | no — gene-level caps dedup_F1 ~0.43 |
| `exp_d_variants__*` | Variant heads with hardneg + proto_core labels | no — training-val AUROC gap is the bottleneck |
| `exp_e_window_context__*` | Window-level context heads (cnn/transformer) | no — no win over plain LR |
| `exp_f_ensemble.json` | Learned score fusion | no — meta-AUROC 0.96 on test_id, 0.31 dedup on 9g (overfit) |
| `exp_g_{boundary,protocore,region,soft}.json` | Label recipe variants | no — proto_core precision win too narrow |
| `fusion_probe.json`, `honest_readout*.json`, `internal_val_sweep_dense.json` | Readout protocol validations | — |
| `route_b_*_honest.json` | Route B (CNN head) honest readouts for context | — |
| `stage2_type_lr_metrics.json` | Per-type LR training metrics (7 binary heads) | — |
| `selected_training_genomes_60.csv` | Training split manifest | — |

Stage 0 is design-log material — null results matter. Do not delete without preserving the finding.
