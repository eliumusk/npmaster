# Stage 0 — Honest Readout (no 9-genome tuning)

## Protocol

Two operating points, BOTH chosen in advance from first-principles,
NOT tuned on any evaluation corpus:

- **Primary**: `s=0.90 e=0.90 ms=1 max_gap=0` — LR 0.9-calibrated cutoff, single threshold.
- **Alt**:     `s=0.95 e=0.80 ms=3 max_gap=0` — conservative seeds + relaxed extend + spike filter.

Plus 5-OP sensitivity band around those choices.

## Results

### w16384

| OP | dedup_F1 (CI95) | strict_F1 | bench_F1 | P | R | n_pred |
|---|---|---|---|---|---|---|
| **primary** `s=0.9/e=0.9/ms=1` | 0.463 (0.326..0.577) | 0.362 | 0.470 | 0.529 | 0.423 | 170 |
| **alt** `s=0.95/e=0.8/ms=3` | 0.387 (0.254..0.496) | 0.276 | 0.387 | 0.671 | 0.272 | 79 |
| sens `s=0.88/e=0.88/ms=1` | 0.458 | 0.352 | 0.461 | 0.473 | 0.449 | 184 |
| sens `s=0.92/e=0.92/ms=1` | 0.469 | 0.363 | 0.471 | 0.572 | 0.400 | 152 |
| sens `s=0.9/e=0.7/ms=3` | 0.431 | 0.272 | 0.434 | 0.560 | 0.354 | 100 |
| sens `s=0.95/e=0.9/ms=1` | 0.443 | 0.358 | 0.445 | 0.618 | 0.348 | 131 |
| sens `s=0.93/e=0.8/ms=3` | 0.402 | 0.282 | 0.404 | 0.659 | 0.292 | 85 |

Per-genome (primary OP):

| genome | n_gt | n_pred | dedup_F1 | strict_F1 | bench_P | bench_R |
|---|---:|---:|---:|---:|---:|---:|
| GCA_000158915.1 | 47 | 30 | 0.656 | 0.545 | 0.700 | 0.617 |
| GCA_000158875.1 | 32 | 14 | 0.498 | 0.391 | 0.643 | 0.406 |
| GCA_000154945.1 | 34 | 13 | 0.043 | 0.043 | 0.154 | 0.029 |
| GCA_000158895.1 | 35 | 21 | 0.290 | 0.179 | 0.286 | 0.371 |
| GCA_000156435.1 | 35 | 19 | 0.210 | 0.111 | 0.158 | 0.314 |
| GCA_000568915.1 | 31 | 17 | 0.484 | 0.458 | 0.647 | 0.387 |
| GCA_000158815.1 | 27 | 18 | 0.667 | 0.667 | 0.889 | 0.556 |
| GCA_000568255.1 | 38 | 23 | 0.613 | 0.492 | 0.652 | 0.579 |
| GCA_000156475.1 | 26 | 15 | 0.483 | 0.293 | 0.467 | 0.500 |

### w8192

| OP | dedup_F1 (CI95) | strict_F1 | bench_F1 | P | R | n_pred |
|---|---|---|---|---|---|---|
| **primary** `s=0.9/e=0.9/ms=1` | 0.432 (0.299..0.552) | 0.383 | 0.488 | 0.475 | 0.502 | 358 |
| **alt** `s=0.95/e=0.8/ms=3` | 0.487 (0.344..0.611) | 0.419 | 0.511 | 0.581 | 0.456 | 222 |
| sens `s=0.88/e=0.88/ms=1` | 0.408 | 0.355 | 0.461 | 0.423 | 0.505 | 392 |
| sens `s=0.92/e=0.92/ms=1` | 0.448 | 0.395 | 0.500 | 0.519 | 0.482 | 318 |
| sens `s=0.9/e=0.7/ms=3` | 0.460 | 0.389 | 0.478 | 0.444 | 0.518 | 288 |
| sens `s=0.95/e=0.9/ms=1` | 0.467 | 0.400 | 0.506 | 0.591 | 0.443 | 247 |
| sens `s=0.93/e=0.8/ms=3` | 0.483 | 0.416 | 0.511 | 0.548 | 0.479 | 250 |

Per-genome (primary OP):

| genome | n_gt | n_pred | dedup_F1 | strict_F1 | bench_P | bench_R |
|---|---:|---:|---:|---:|---:|---:|
| GCA_000158915.1 | 47 | 57 | 0.605 | 0.549 | 0.614 | 0.745 |
| GCA_000158875.1 | 32 | 28 | 0.474 | 0.367 | 0.500 | 0.531 |
| GCA_000154945.1 | 34 | 35 | 0.000 | 0.000 | 0.086 | 0.000 |
| GCA_000158895.1 | 35 | 56 | 0.314 | 0.220 | 0.321 | 0.486 |
| GCA_000156435.1 | 35 | 48 | 0.224 | 0.206 | 0.188 | 0.343 |
| GCA_000568915.1 | 31 | 27 | 0.498 | 0.498 | 0.593 | 0.452 |
| GCA_000158815.1 | 27 | 32 | 0.628 | 0.576 | 0.844 | 0.667 |
| GCA_000568255.1 | 38 | 49 | 0.552 | 0.518 | 0.612 | 0.632 |
| GCA_000156475.1 | 26 | 26 | 0.596 | 0.577 | 0.692 | 0.615 |

## Interpretation

A 4096-d linear probe on frozen Evo2 blocks.20.mlp.l3 mean-pool embeddings,
with a decoder operating point chosen from first-principles (no 9-genome data touched),
recovers the night-run performance. The internal-val OP-sweep (see night report)
was a methodological detour: because test_ood BGC density differs structurally from
the 9-genome Actinomycetes, sweep-chosen OPs collapsed to spurious low-threshold
regimes. A principled fixed OP sidesteps this and lands on the same numbers.