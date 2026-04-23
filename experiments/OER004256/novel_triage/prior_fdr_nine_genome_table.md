# 9-genome score-stratified prior FDR

- v4.1 total regions: **246**  (decoded from existing preds, frozen OP)
- antiSMASH regions:  **283**
- 9-genome GT:        **305**  (human-curated tight-core)

## Calls broken down by max_score bin

| score bin | total | AS covered | AS missed | AS miss & GT-hit | AS miss & no GT | **P(GT-hit \| AS-missed)** |
|---|---:|---:|---:|---:|---:|---:|
| 0.95-0.98 | 42 | 28 | 14 | 5 | 9 | **0.357** |
| 0.98-0.99 | 22 | 18 | 4 | 2 | 2 | **0.500** |
| 0.99-0.995 | 20 | 18 | 2 | 0 | 2 | **0.000** |
| 0.995-1.0 | 162 | 153 | 9 | 6 | 3 | **0.667** |

## Aggregated for score >= 0.98 (OER-candidate threshold)
- Total v4.1 calls: **204**
- AS-missed: **15**
- AS-missed & GT-hit: **8**
- **P(GT-hit | AS-missed, score>=0.98) = 0.533**

## Interpretation for OER 15 candidates
Assuming 9-genome calibration transfers: of 15 OER candidates at score>=0.98 AS-missed,
expected GT-hit-rate-analogue = 53.3% → ~8.0 are true novel-style BGCs,
~7.0 may be model calls outside curated BGC space.
**Caveats:** 9-genome GT itself may miss novel BGCs (lower-bound), and OER is metagenomic
(different composition). Use as *directional* prior, not exact FDR.
