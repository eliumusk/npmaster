# experiments/

Evaluation/inference OUTPUTS only. No code. Organized by target:

| Dir | What it contains |
|---|---|
| `nine_genome/` | 9-genome benchmark results — the "held-out test" numbers. Includes sweep summaries, OP winners, and classification eval against antiSMASH. |
| `evo2_first_principles/` | First-principles design campaign (2026-04-21 → 2026-04-22) that produced v4.1. Stage 0 ablations + v4.1 frozen readout. |
| `OER004256/` | Production handoff for the biology advisor. 254 v4.1 regions + 93 defrag_v2 regions on 7 metagenomic samples. See `OER004256/README_HANDOFF.md`. |

Trial-level sweep debris (`*-trial-NNN` directories) retire to `_archive/experiments/` once the sweep has a summary CSV. Current working tree contains only promoted artifacts.

Frozen operating points are codified in `configs/operating_points.yaml` and `configs/experiments/evo2_first_principles_v4_1_frozen.yaml`.

For the numbers anyone is claiming, see `docs/REGRESSION_LEDGER.md`.
