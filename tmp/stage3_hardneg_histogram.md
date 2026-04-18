# Stage 3 Hardneg Score Histogram — Training Negatives

- Checkpoint: `/data/muskliu/npmaster/data/phase3/route_b/best_route_b_s0.pt` (epoch 94)
- N total windows: 1,064,568
- N train (any label): 626,449
- **N train negative: 544,831**
- Inference time: 197.0s on 1× A800
- Execution: node7 GPU0, 2026-04-18. No training, no repo changes. Raw `probs` cached at `/data/muskliu/npmaster/tmp_hist/all_probs_route_b_s0.npy`.

## TL;DR — Stage 3 Axis B Planning Guidance

The hardneg pool under the current mainline checkpoint is **extremely sparse on the high-score tail**:

| threshold (fused) | # train-negatives at or above | % of 544,831 |
|------------------:|------------------------------:|------------:|
| 0.99 | 190 | 0.035% |
| 0.95 | 368 | 0.068% |
| 0.90 | 459 | 0.084% |
| 0.80 | 563 | 0.103% |
| 0.70 | 658 | 0.121% |
| 0.50 | 871 | 0.160% |

Translated into top-k floors under `fused`:

- top_k = 100 → floor score 0.9973
- top_k = 368 → floor score 0.9503 (matches current mine's `n_hard_negative=368` @ `min_score=0.95`)
- top_k = 500 → floor score 0.8732
- top_k = 1,000 → floor score 0.392 (half the pool is already below 0.5!)
- top_k = 5,000 → floor score 0.0055 (essentially easy negatives)
- top_k = 15,000 → floor score 0.0005 (indistinguishable from random negatives)
- top_k = 50,000 → floor score 6e-5 (mostly easy negatives; noise)

**Risk R2 from `tmp/stage3_data_plan.md` is confirmed**: with current score family / current checkpoint, any top_k ≥ ~900 is dominated by near-zero-score negatives that are not "hard" by any meaningful definition. The existing Axis B sweep values {1000, 5000, 15000, 50000} are collinear in the sense that they all collapse to "~900 actually-hard samples + a long tail of easy negatives". The model won't see meaningfully different hardneg sets past ~k=900.

### Recommended top_k sweep for Stage 3a Axis B

Replace `{1000, 5000, 15000, 50000}` with tiers that actually change the pool composition:

- **k = 100** — ultra-hard only (floor ~0.997). Tiny set; probably dominated by label noise / GT-narrowness (see `docs/next-phase.md:14`, R1 in the Stage 3 plan).
- **k = 368** — current operating point (`fused @ min=0.95`, what `mine_s0_fused95_k5000` produced). Baseline.
- **k = 900** — "all with score ≥ 0.5" (extends to include the shoulder of the distribution).
- **k = 2,000** — stretches into the mid-score (floor ~0.064). Tests whether "ambiguously hard" negatives help.
- (optional) **k = 5,000** — to probe whether diluting with easy negatives hurts or is neutral; treat as a control, not a hypothesis.

All overlaps are nested (smaller set ⊂ larger set) because mining is deterministic top-k descending; the sweep cleanly isolates "add more low-score negatives" as the axis.

### Cross-score-family note

`max_type` and `1_minus_isneg` give near-identical distributions to `fused` (counts above 0.95: 367 and 419 respectively). Axis A (score family) won't change the pool size much; it only re-ranks within the top few hundred. Axis A is lower priority than Axis B+C.

### Should mining threshold / score family be revisited?

Given the sparsity of the high-score tail, consider a fourth Stage 3a sub-trial: re-mine on a **different checkpoint** (e.g. earlier epoch snapshot, or warm-start `H_hardneg_r10` itself). The current checkpoint has already been hardneg-trained, which may explain the thin tail — the model has learned to push confident-false-positives down. Mining on a less-hardneg-saturated model could surface a larger pool.

---


## Score family: `fused`

- min=0.0000 max=1.0000 mean=0.0019 median=0.0000 std=0.0361

### Quantiles

| q | value |
|---|------:|
| 0.01 | 0.0000 |
| 0.05 | 0.0000 |
| 0.1 | 0.0000 |
| 0.25 | 0.0000 |
| 0.5 | 0.0000 |
| 0.75 | 0.0000 |
| 0.9 | 0.0001 |
| 0.95 | 0.0002 |
| 0.99 | 0.0045 |
| 0.995 | 0.0261 |
| 0.999 | 0.8280 |

### Count above threshold

| thr | count | % of train-neg |
|----:|------:|---------------:|
| 0.5 | 871 | 0.160% |
| 0.7 | 658 | 0.121% |
| 0.8 | 563 | 0.103% |
| 0.9 | 459 | 0.084% |
| 0.95 | 368 | 0.068% |
| 0.99 | 190 | 0.035% |

### Histogram (50 bins over [0,1])

| bin_low | bin_high | count |
|--------:|---------:|------:|
| 0.000 | 0.020 | 541,852 |
| 0.020 | 0.040 | 647 |
| 0.040 | 0.060 | 276 |
| 0.060 | 0.080 | 189 |
| 0.080 | 0.100 | 136 |
| 0.100 | 0.120 | 124 |
| 0.120 | 0.140 | 89 |
| 0.140 | 0.160 | 71 |
| 0.160 | 0.180 | 46 |
| 0.180 | 0.200 | 46 |
| 0.200 | 0.220 | 48 |
| 0.220 | 0.240 | 44 |
| 0.240 | 0.260 | 36 |
| 0.260 | 0.280 | 48 |
| 0.280 | 0.300 | 26 |
| 0.300 | 0.320 | 32 |
| 0.320 | 0.340 | 30 |
| 0.340 | 0.360 | 25 |
| 0.360 | 0.380 | 42 |
| 0.380 | 0.400 | 39 |
| 0.400 | 0.420 | 25 |
| 0.420 | 0.440 | 23 |
| 0.440 | 0.460 | 26 |
| 0.460 | 0.480 | 17 |
| 0.480 | 0.500 | 23 |
| 0.500 | 0.520 | 34 |
| 0.520 | 0.540 | 21 |
| 0.540 | 0.560 | 18 |
| 0.560 | 0.580 | 15 |
| 0.580 | 0.600 | 17 |
| 0.600 | 0.620 | 18 |
| 0.620 | 0.640 | 14 |
| 0.640 | 0.660 | 30 |
| 0.660 | 0.680 | 24 |
| 0.680 | 0.700 | 22 |
| 0.700 | 0.720 | 15 |
| 0.720 | 0.740 | 27 |
| 0.740 | 0.760 | 20 |
| 0.760 | 0.780 | 18 |
| 0.780 | 0.800 | 15 |
| 0.800 | 0.820 | 8 |
| 0.820 | 0.840 | 23 |
| 0.840 | 0.860 | 19 |
| 0.860 | 0.880 | 25 |
| 0.880 | 0.900 | 29 |
| 0.900 | 0.920 | 29 |
| 0.920 | 0.940 | 40 |
| 0.940 | 0.960 | 51 |
| 0.960 | 0.980 | 84 |
| 0.980 | 1.000 | 255 |

### Top-k score floors

| top_k | score floor (k-th largest) |
|-----:|---------------------------:|
| 100 | 0.9973 |
| 368 | 0.9503 |
| 500 | 0.8732 |
| 1000 | 0.3924 |
| 2000 | 0.0636 |
| 5000 | 0.0055 |
| 10000 | 0.0012 |
| 15000 | 0.0005 |
| 50000 | 0.0001 |

### Top-k pairwise overlaps (smaller ⊂ larger)

| pair | intersection |
|------|-------------:|
| 100_vs_368 | 100 |
| 100_vs_500 | 100 |
| 100_vs_1000 | 100 |
| 100_vs_2000 | 100 |
| 100_vs_5000 | 100 |
| 100_vs_10000 | 100 |
| 100_vs_15000 | 100 |
| 100_vs_50000 | 100 |
| 368_vs_500 | 368 |
| 368_vs_1000 | 368 |
| 368_vs_2000 | 368 |
| 368_vs_5000 | 368 |
| 368_vs_10000 | 368 |
| 368_vs_15000 | 368 |
| 368_vs_50000 | 368 |
| 500_vs_1000 | 500 |
| 500_vs_2000 | 500 |
| 500_vs_5000 | 500 |
| 500_vs_10000 | 500 |
| 500_vs_15000 | 500 |
| 500_vs_50000 | 500 |
| 1000_vs_2000 | 1,000 |
| 1000_vs_5000 | 1,000 |
| 1000_vs_10000 | 1,000 |
| 1000_vs_15000 | 1,000 |
| 1000_vs_50000 | 1,000 |
| 2000_vs_5000 | 2,000 |
| 2000_vs_10000 | 2,000 |
| 2000_vs_15000 | 2,000 |
| 2000_vs_50000 | 2,000 |
| 5000_vs_10000 | 5,000 |
| 5000_vs_15000 | 5,000 |
| 5000_vs_50000 | 5,000 |
| 10000_vs_15000 | 10,000 |
| 10000_vs_50000 | 10,000 |
| 15000_vs_50000 | 15,000 |

## Score family: `max_type`

- min=0.0000 max=1.0000 mean=0.0019 median=0.0000 std=0.0363

### Quantiles

| q | value |
|---|------:|
| 0.01 | 0.0000 |
| 0.05 | 0.0000 |
| 0.1 | 0.0000 |
| 0.25 | 0.0000 |
| 0.5 | 0.0000 |
| 0.75 | 0.0000 |
| 0.9 | 0.0000 |
| 0.95 | 0.0001 |
| 0.99 | 0.0033 |
| 0.995 | 0.0213 |
| 0.999 | 0.8457 |

### Count above threshold

| thr | count | % of train-neg |
|----:|------:|---------------:|
| 0.5 | 876 | 0.161% |
| 0.7 | 695 | 0.128% |
| 0.8 | 588 | 0.108% |
| 0.9 | 472 | 0.087% |
| 0.95 | 367 | 0.067% |
| 0.99 | 193 | 0.035% |

### Histogram (50 bins over [0,1])

| bin_low | bin_high | count |
|--------:|---------:|------:|
| 0.000 | 0.020 | 542,050 |
| 0.020 | 0.040 | 594 |
| 0.040 | 0.060 | 264 |
| 0.060 | 0.080 | 166 |
| 0.080 | 0.100 | 137 |
| 0.100 | 0.120 | 94 |
| 0.120 | 0.140 | 80 |
| 0.140 | 0.160 | 58 |
| 0.160 | 0.180 | 56 |
| 0.180 | 0.200 | 61 |
| 0.200 | 0.220 | 41 |
| 0.220 | 0.240 | 32 |
| 0.240 | 0.260 | 30 |
| 0.260 | 0.280 | 30 |
| 0.280 | 0.300 | 27 |
| 0.300 | 0.320 | 26 |
| 0.320 | 0.340 | 31 |
| 0.340 | 0.360 | 26 |
| 0.360 | 0.380 | 23 |
| 0.380 | 0.400 | 21 |
| 0.400 | 0.420 | 19 |
| 0.420 | 0.440 | 17 |
| 0.440 | 0.460 | 18 |
| 0.460 | 0.480 | 25 |
| 0.480 | 0.500 | 29 |
| 0.500 | 0.520 | 21 |
| 0.520 | 0.540 | 15 |
| 0.540 | 0.560 | 13 |
| 0.560 | 0.580 | 13 |
| 0.580 | 0.600 | 12 |
| 0.600 | 0.620 | 19 |
| 0.620 | 0.640 | 21 |
| 0.640 | 0.660 | 18 |
| 0.660 | 0.680 | 26 |
| 0.680 | 0.700 | 23 |
| 0.700 | 0.720 | 17 |
| 0.720 | 0.740 | 19 |
| 0.740 | 0.760 | 20 |
| 0.760 | 0.780 | 29 |
| 0.780 | 0.800 | 22 |
| 0.800 | 0.820 | 19 |
| 0.820 | 0.840 | 15 |
| 0.840 | 0.860 | 30 |
| 0.860 | 0.880 | 23 |
| 0.880 | 0.900 | 29 |
| 0.900 | 0.920 | 44 |
| 0.920 | 0.940 | 37 |
| 0.940 | 0.960 | 51 |
| 0.960 | 0.980 | 74 |
| 0.980 | 1.000 | 266 |

### Top-k score floors

| top_k | score floor (k-th largest) |
|-----:|---------------------------:|
| 100 | 0.9975 |
| 368 | 0.9488 |
| 500 | 0.8801 |
| 1000 | 0.3866 |
| 2000 | 0.0533 |
| 5000 | 0.0041 |
| 10000 | 0.0008 |
| 15000 | 0.0003 |
| 50000 | 0.0000 |

### Top-k pairwise overlaps (smaller ⊂ larger)

| pair | intersection |
|------|-------------:|
| 100_vs_368 | 100 |
| 100_vs_500 | 100 |
| 100_vs_1000 | 100 |
| 100_vs_2000 | 100 |
| 100_vs_5000 | 100 |
| 100_vs_10000 | 100 |
| 100_vs_15000 | 100 |
| 100_vs_50000 | 100 |
| 368_vs_500 | 368 |
| 368_vs_1000 | 368 |
| 368_vs_2000 | 368 |
| 368_vs_5000 | 368 |
| 368_vs_10000 | 368 |
| 368_vs_15000 | 368 |
| 368_vs_50000 | 368 |
| 500_vs_1000 | 500 |
| 500_vs_2000 | 500 |
| 500_vs_5000 | 500 |
| 500_vs_10000 | 500 |
| 500_vs_15000 | 500 |
| 500_vs_50000 | 500 |
| 1000_vs_2000 | 1,000 |
| 1000_vs_5000 | 1,000 |
| 1000_vs_10000 | 1,000 |
| 1000_vs_15000 | 1,000 |
| 1000_vs_50000 | 1,000 |
| 2000_vs_5000 | 2,000 |
| 2000_vs_10000 | 2,000 |
| 2000_vs_15000 | 2,000 |
| 2000_vs_50000 | 2,000 |
| 5000_vs_10000 | 5,000 |
| 5000_vs_15000 | 5,000 |
| 5000_vs_50000 | 5,000 |
| 10000_vs_15000 | 10,000 |
| 10000_vs_50000 | 10,000 |
| 15000_vs_50000 | 15,000 |

## Score family: `1_minus_isneg`

- min=0.0000 max=1.0000 mean=0.0020 median=0.0000 std=0.0373

### Quantiles

| q | value |
|---|------:|
| 0.01 | 0.0000 |
| 0.05 | 0.0000 |
| 0.1 | 0.0000 |
| 0.25 | 0.0000 |
| 0.5 | 0.0000 |
| 0.75 | 0.0000 |
| 0.9 | 0.0001 |
| 0.95 | 0.0002 |
| 0.99 | 0.0044 |
| 0.995 | 0.0235 |
| 0.999 | 0.8712 |

### Count above threshold

| thr | count | % of train-neg |
|----:|------:|---------------:|
| 0.5 | 918 | 0.168% |
| 0.7 | 733 | 0.135% |
| 0.8 | 629 | 0.115% |
| 0.9 | 516 | 0.095% |
| 0.95 | 419 | 0.077% |
| 0.99 | 243 | 0.045% |

### Histogram (50 bins over [0,1])

| bin_low | bin_high | count |
|--------:|---------:|------:|
| 0.000 | 0.020 | 541,932 |
| 0.020 | 0.040 | 658 |
| 0.040 | 0.060 | 278 |
| 0.060 | 0.080 | 162 |
| 0.080 | 0.100 | 120 |
| 0.100 | 0.120 | 110 |
| 0.120 | 0.140 | 81 |
| 0.140 | 0.160 | 68 |
| 0.160 | 0.180 | 52 |
| 0.180 | 0.200 | 36 |
| 0.200 | 0.220 | 39 |
| 0.220 | 0.240 | 36 |
| 0.240 | 0.260 | 35 |
| 0.260 | 0.280 | 33 |
| 0.280 | 0.300 | 35 |
| 0.300 | 0.320 | 25 |
| 0.320 | 0.340 | 32 |
| 0.340 | 0.360 | 16 |
| 0.360 | 0.380 | 25 |
| 0.380 | 0.400 | 21 |
| 0.400 | 0.420 | 24 |
| 0.420 | 0.440 | 25 |
| 0.440 | 0.460 | 22 |
| 0.460 | 0.480 | 24 |
| 0.480 | 0.500 | 24 |
| 0.500 | 0.520 | 20 |
| 0.520 | 0.540 | 23 |
| 0.540 | 0.560 | 17 |
| 0.560 | 0.580 | 16 |
| 0.580 | 0.600 | 22 |
| 0.600 | 0.620 | 20 |
| 0.620 | 0.640 | 15 |
| 0.640 | 0.660 | 17 |
| 0.660 | 0.680 | 12 |
| 0.680 | 0.700 | 23 |
| 0.700 | 0.720 | 22 |
| 0.720 | 0.740 | 21 |
| 0.740 | 0.760 | 23 |
| 0.760 | 0.780 | 20 |
| 0.780 | 0.800 | 18 |
| 0.800 | 0.820 | 19 |
| 0.820 | 0.840 | 25 |
| 0.840 | 0.860 | 18 |
| 0.860 | 0.880 | 32 |
| 0.880 | 0.900 | 19 |
| 0.900 | 0.920 | 26 |
| 0.920 | 0.940 | 43 |
| 0.940 | 0.960 | 52 |
| 0.960 | 0.980 | 87 |
| 0.980 | 1.000 | 308 |

### Top-k score floors

| top_k | score floor (k-th largest) |
|-----:|---------------------------:|
| 100 | 0.9987 |
| 368 | 0.9697 |
| 500 | 0.9141 |
| 1000 | 0.4279 |
| 2000 | 0.0568 |
| 5000 | 0.0054 |
| 10000 | 0.0013 |
| 15000 | 0.0006 |
| 50000 | 0.0001 |

### Top-k pairwise overlaps (smaller ⊂ larger)

| pair | intersection |
|------|-------------:|
| 100_vs_368 | 100 |
| 100_vs_500 | 100 |
| 100_vs_1000 | 100 |
| 100_vs_2000 | 100 |
| 100_vs_5000 | 100 |
| 100_vs_10000 | 100 |
| 100_vs_15000 | 100 |
| 100_vs_50000 | 100 |
| 368_vs_500 | 368 |
| 368_vs_1000 | 368 |
| 368_vs_2000 | 368 |
| 368_vs_5000 | 368 |
| 368_vs_10000 | 368 |
| 368_vs_15000 | 368 |
| 368_vs_50000 | 368 |
| 500_vs_1000 | 500 |
| 500_vs_2000 | 500 |
| 500_vs_5000 | 500 |
| 500_vs_10000 | 500 |
| 500_vs_15000 | 500 |
| 500_vs_50000 | 500 |
| 1000_vs_2000 | 1,000 |
| 1000_vs_5000 | 1,000 |
| 1000_vs_10000 | 1,000 |
| 1000_vs_15000 | 1,000 |
| 1000_vs_50000 | 1,000 |
| 2000_vs_5000 | 2,000 |
| 2000_vs_10000 | 2,000 |
| 2000_vs_15000 | 2,000 |
| 2000_vs_50000 | 2,000 |
| 5000_vs_10000 | 5,000 |
| 5000_vs_15000 | 5,000 |
| 5000_vs_50000 | 5,000 |
| 10000_vs_15000 | 10,000 |
| 10000_vs_50000 | 10,000 |
| 15000_vs_50000 | 15,000 |
