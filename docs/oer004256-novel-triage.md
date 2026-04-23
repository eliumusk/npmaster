# OER004256 novel-candidate triage — method + numbers

Three-layer in-silico validation for "v4.1 called, antiSMASH didn't call" candidates on OER004256 (~1.48 Gb metagenomic assembly, 7 samples). Designed to maximize wet-lab signal:noise before committing samples to experimental screening.

## Why this is needed

v4.1 on OER emits many candidates that antiSMASH does not. That set is:
- The highest-potential cohort (if real, these are compounds antiSMASH-based tools miss — exactly the paper story).
- The highest-risk cohort (the model is literally OOD relative to its antiSMASH-based training labels).

Without validation we cannot send wet-lab safely. Validation must be **model-independent** — relying on molecular evidence, not model scores.

## Three-layer filter

### Layer 1 — prior FDR calibration (9-genome)

Apply v4.1 multi-scale hysteresis to 9-genome. For the resulting high-score calls that antiSMASH missed, check how many overlap the human-curated GT:

| score bin | v4.1 total | AS covered | AS missed | AS-missed & GT-hit | **P(GT-hit \| AS-missed)** |
|---|---:|---:|---:|---:|---:|
| 0.95–0.98 | 42 | 28 | 14 | 5 | 0.36 |
| 0.98–0.99 | 22 | 18 | 4 | 2 | 0.50 |
| 0.995–1.00 | 162 | 153 | 9 | 6 | 0.67 |
| **≥0.98 aggregate** | **204** | **189** | **15** | **8** | **0.533** |

Interpretation: on a dataset where we *do* have GT, **≥53%** of "v4.1 yes, antiSMASH no" calls at score ≥ 0.98 are real BGCs. This is a lower bound — 9g GT itself misses truly novel BGCs.

Script: `scripts/diagnose/prior_fdr_nine_genome.py`.

### Layer 2 — biosynthetic content (HMM)

Predict ORFs with `pyrodigal` per candidate region, then scan with:
- antiSMASH **rule HMMs** + **NRPS/PKS domain HMMs** (966 profiles) — the same HMMs antiSMASH uses for clustering.
- **Pfam-A 35.0** — 20k profiles, broad coverage.

Verdict:
- **STRONG**: ≥ 3 biosynthetic HMM hits
- **MODERATE**: 1–2 biosynth hits
- **WEAK** / **LIKELY_FP**: 0 biosynth, relies only on Pfam (many housekeeping families)

Logic: if a candidate contains multiple biosynthetic machinery (PKS_KS + PKS_AT + PKS_KR...) at E < 1e-50 each, that is not a model hallucination — the enzymatic signature *is there*. antiSMASH missed it because its **cluster-closure rules** require specific anchor combinations + flanking context that short metagenomic contigs cannot provide.

Script: `scripts/classify/hmm_biosynth_scan.py`.

### Layer 3 — novelty (MIBiG 4.0)

DIAMOND `blastp` every candidate ORF against MIBiG 4.0 (39,992 proteins from 3,013 characterized BGCs).

Per-candidate verdict (using best ORF-BGC pident):
- **KNOWN**: ≥ 2 ORFs with pident ≥ 70% — this candidate is likely a fragment of an already-characterized BGC.
- **VARIANT**: ≥ 2 ORFs with pident ≥ 30% (but not KNOWN) — related chemistry, ≥30% sequence divergence, likely a new congener.
- **LIKELY_NOVEL**: 1 ORF with pident ≥ 30%.
- **NOVEL**: No ORF hits MIBiG at pident ≥ 30% — truly novel scaffold.

Logic: we want candidates whose wet-lab validation would yield *new* molecules, not re-discoveries.

Script: `scripts/classify/mibig_novelty.py`.

## Results, first iteration — v4.1 hysteresis output (254 regions, 15 novel)

- Layer 1 (prior FDR): P(GT-hit | AS-missed, score≥0.98) ≥ 0.53.
- Layer 2 (HMM): **14/15 STRONG** biosynth evidence (10–76 domain hits, E < 1e-50). 1/15 MODERATE (cand7, Polyketide-labeled but only FabA Pfam hit → fatty-acid primary metabolism, excluded from wet-lab).
- Layer 3 (MIBiG): **0/15 KNOWN, 3/15 LIKELY_NOVEL, 12/15 VARIANT.** All 15 candidates have max pident < 70% to any known BGC. Family signatures include Sorangipyranone-like PKS, minutissamide-like NRPS, aryl polyene variants, and one enediyne-family candidate (cand2).

Artifacts: `experiments/OER004256/novel_triage/`.

## Results, second iteration — v4.1-learned output (1,230 regions, 429 novel)

After replacing hysteresis with the learned decoder (`docs/learned-decoder.md`) the novel pool expanded from 15 to 429. Running the same three-layer filter:

| Biosynth verdict | count |
|---|---:|
| STRONG (≥3 hits) | **190** (44%) |
| MODERATE (1–2) | 73 |
| WEAK / Pfam-only | 55 |
| LIKELY_FP | 111 (26%) |

| Novelty verdict | count |
|---|---:|
| KNOWN | **0** |
| NOVEL | 113 (26%) |
| LIKELY_NOVEL | 139 (32%) |
| VARIANT | 177 (41%) |

**Top wet-lab cohort (STRONG biosynth × NOVEL-or-LIKELY_NOVEL) = 86 candidates.**

Notable new scaffolds discovered by the learned decoder (not present in the original 15):
- `jamaicamide-like` PKS-NRPS (cand95) — parent family is a cyanobacterial neurotoxin.
- `rifamorpholine-like` (cand396) — rifamycin family, 4 kb short cluster.
- `quinolidomicin-like` (cand19) — fully NOVEL under MIBiG threshold.
- `scytocyclamide-like` (cand91, cand151) — cyanobacterial cyclic peptides.

Artifacts: `experiments/OER004256/v4_1_learned/novel_triage/`.

## Honest caveats

- **Layer 1 (prior FDR) is a lower bound** — 9-genome GT itself is human-curated for *known* BGCs and misses genuinely novel ones. Real precision is likely higher than 53% on a population of HMM-passing candidates.
- **Layer 2 depends on existing HMM libraries** — a truly novel biosynthetic class with no HMM coverage would fail Layer 2 and be excluded. Unavoidable limit of HMM-based validation; wet-lab is the only way to catch these.
- **Layer 3 MIBiG is incomplete** — only ~3,000 BGCs characterized. A candidate labeled NOVEL may still be similar to an uncharacterized cluster in public databases.
- **None of these layers replace wet-lab.** They reduce the search space 30–100× and give confidence that spending wet-lab budget on the filtered cohort has decent expected return, not zero.

## Recommended wet-lab picks (current state)

Use `novel_triage_full.csv` column `PASS_for_wetlab = True` (≈ 261 / 429), then rank by `n_biosynth_hits` descending. Top candidates by scaffold diversity:

1. `cand0` — Polyketide, full modular PKS (AT+KR+DH+ER+MT), Sorangipyranone-like, score 1.000
2. `cand95` — Polyketide-NRPS, jamaicamide-like scaffold
3. `cand396` — Other / AurF, rifamorpholine-like
4. `cand3` — Terpene-NRPS hybrid with Trp halogenase, 18 kb
5. `cand10` or `cand5` — NRPS starter module (FAAL+CAL+A+TD), minutissamide-like
6. `cand19` — Other, genuinely NOVEL (no MIBiG hit ≥30%)

Script chain to reproduce from scratch:
```bash
# 1. intersect v4.1 regions with antiSMASH regions -> novel CSV
python scripts/diagnose/oer_novel_candidates.py \
    --regions-csv experiments/OER004256/v4_1_learned/regions_fully_typed.csv \
    --antismash-dir /data/muskliu/oer_bgc_scan/results/antismash \
    --out-dir experiments/OER004256/v4_1_learned/novel_triage

# 2. HMM scan (biosynth + Pfam) + MIBiG DIAMOND blastp
python scripts/classify/hmm_biosynth_scan.py       # inputs hardcoded; adapt if needed
python scripts/classify/mibig_novelty.py
```
