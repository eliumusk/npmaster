# Coverage diagnostic — OER vs 9-genome

## Top-line ratio

| dataset | antiSMASH regions | v4.1 regions | AS/v4.1 | DNA (Gb) | AS density/Gb | v4.1 density/Gb |
|---|---:|---:|---:|---:|---:|---:|
| OER | 1976 | 254 | 7.78× | 0.00 | 12820679.2 | 1648002.3 |
| 9g | 283 | 246 | 1.15× | 0.00 | 4042857142.9 | 3514285714.3 |

## antiSMASH region length distribution

| bucket | OER | 9g | OER % | 9g % |
|---|---:|---:|---:|---:|
| <5kb | 0 | 1 | 0.0% | 0.4% |
| 5-10kb | 134 | 1 | 6.8% | 0.4% |
| 10-20kb | 1420 | 36 | 71.9% | 12.7% |
| 20-50kb | 419 | 175 | 21.2% | 61.8% |
| >=50kb | 3 | 70 | 0.2% | 24.7% |
| **median length** | **13577** | **29782** | | |

## Contig length where antiSMASH regions live

| contig bucket | OER | 9g | OER % | 9g % |
|---|---:|---:|---:|---:|
| <15kb | 1976 | 283 | 100.0% | 100.0% |
| 15-30kb | 0 | 0 | 0.0% | 0.0% |
| 30-100kb | 0 | 0 | 0.0% | 0.0% |
| 100-500kb | 0 | 0 | 0.0% | 0.0% |
| >=500kb | 0 | 0 | 0.0% | 0.0% |

## Category split

| category | OER | 9g |
|---|---:|---:|
| BOTH_from_antismash_side | 237 | 166 |
| AS_ONLY | 1739 | 117 |
| V41_ONLY | 17 | 80 |

## AS_ONLY length distribution (v4.1 missed these)

| bucket | OER AS_ONLY | 9g AS_ONLY | OER % | 9g % |
|---|---:|---:|---:|---:|
| <5kb | 0 | 0 | 0.0% | 0.0% |
| 5-10kb | 134 | 0 | 7.7% | 0.0% |
| 10-20kb | 1264 | 33 | 72.7% | 28.2% |
| 20-50kb | 340 | 81 | 19.6% | 69.2% |
| >=50kb | 1 | 3 | 0.1% | 2.6% |

## AS_ONLY — ORF count distribution

| #CDS bucket | OER AS_ONLY | 9g AS_ONLY |
|---|---:|---:|
| 1-3_CDS | 0 | 0 |
| 4-7_CDS | 60 | 1 |
| 8-15_CDS | 1030 | 33 |
| >15_CDS | 649 | 83 |

## AS_ONLY — did v4.1 see them?

| case | OER | 9g |
|---|---:|---:|
| never_covered_no_windows | 0 | 0 |
| covered_low_score(<0.5) | 135 | 0 |
| covered_mid_score(0.5-0.95) | 1155 | 77 |
| covered_high_score(>=0.95)_but_decoder_rejected | 449 | 40 |

## AS_ONLY — top 10 antiSMASH product types

**OER:**
- terpene-precursor: 460
- terpene: 331
- RiPP-like: 186
- NRPS-like: 159
- T3PKS: 113
- RRE-containing: 87
- phosphonate: 73
- betalactone: 47
- hydrogen-cyanide: 46
- ranthipeptide: 39

**9g:**
- terpene: 31
- RiPP-like: 20
- terpene-precursor: 11
- ectoine: 5
- lanthipeptide-class-iii: 4
- melanin: 4
- T3PKS: 4
- NI-siderophore: 4
- CDPS: 4
- redox-cofactor: 4

## antiSMASH regions per contig (clustering on same contig)

| regions/contig | OER | 9g |
|---|---:|---:|
| 1 | 1956 | 4 |
| 2-3 | 10 | 2 |
| 4-10 | 0 | 1 |
| >10 | 0 | 9 |
| total contigs with AS calls | 1966 | 16 |

## v4.1 max-score percentiles on AS_ONLY regions

| percentile | OER | 9g |
|---|---:|---:|
| p10 | 0.529 | 0.755 |
| p50 | 0.846 | 0.917 |
| p90 | 0.981 | 0.983 |
| p99 | 0.999 | 0.993 |