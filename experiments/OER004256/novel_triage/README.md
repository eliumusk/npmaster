# OER004256 / novel_triage — 15-candidate triage artifacts

Validation artifacts for the 15 "v4.1 called, antiSMASH didn't call" candidates from the **v4.1 multi-scale hysteresis** output (254 regions) on OER004256.

See [`docs/oer004256-novel-triage.md`](../../../docs/oer004256-novel-triage.md) for method + three-layer rationale.

## Files

| File | Content |
|---|---|
| `candidates_novel.csv` | 15 candidates (v4.1 regions with no antiSMASH overlap on same contig) |
| `novel_candidates_triage.csv` | Per-candidate HMM-scan verdict, biosynth class counts, top biosynth + Pfam hits, nearest antiSMASH distance, final STRONG / MODERATE / LIKELY_FP verdict |
| `novel_candidates_orf_hits.tsv` | Every ORF × HMM hit (biosynth DB + Pfam), E-values and per-hit class tags |
| `mibig_novelty.csv` | Per-candidate MIBiG 4.0 similarity — best BGC, max pident, n_ORFs ≥ 30% / 70%, top-3 aggregated hits, NOVEL / LIKELY_NOVEL / VARIANT / KNOWN verdict |
| `mibig_blastp_raw.tsv` | DIAMOND blastp raw hits (ORF-level, top-25 per query) |
| `prior_fdr_nine_genome.json` | 9-genome calibration: for each score bin, how many v4.1 calls antiSMASH missed AND GT hit — our **prior FDR** |
| `prior_fdr_nine_genome_table.md` | Human-readable table of the same |
| `coverage_diag/` | **OER vs 9g coverage diagnostic** — diagnosis of the 7.8× antiSMASH:v4.1 gap that led to building the learned decoder (see `comparison.md`) |
| `sweep_summary.{csv,md}` | 16-config learned-decoder sweep results at fixed threshold=0.5 (seeded `docs/learned-decoder.md`) |

## Headline numbers

- **14/15 STRONG** biosynth HMM evidence. 1/15 MODERATE (cand7 — FabA primary-metabolism, excluded).
- **0/15 KNOWN**, 3/15 LIKELY_NOVEL, 12/15 VARIANT in MIBiG 4.0. Max pident 67.6% to any known BGC.
- **Prior FDR on 9g** at score ≥ 0.98: ≥ 53% of "antiSMASH-missed" calls are real (lower bound).

Script entrypoints: `scripts/diagnose/oer_novel_candidates.py`, `scripts/classify/hmm_biosynth_scan.py`, `scripts/classify/mibig_novelty.py`, `scripts/diagnose/prior_fdr_nine_genome.py`, `scripts/diagnose/antismash_coverage.py`.
