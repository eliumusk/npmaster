# Stage 3 Hardneg Mined Sample Inspection

Baseline meta: `/data/muskliu/npmaster/data/phase3/route_a/meta_arrays.pkl`
Baseline train-negative count: 544831
Baseline pre-existing hardneg count: 8304
BGC annotations: 19762 BGCs across 11874 contigs
Proximity threshold: +/- 50000 bp (window len 8192)

Classification policy:
- **clean_FP** = no annotated BGC within 50 kb -> hardneg training safe
- **ambiguous** = nearby BGC but >0 and <=10 kb away (flanking context)
- **GT-gap** = window OVERLAPS an annotated BGC in bgc_annotations.csv
  (window is labeled 'negative' but GT itself contains a BGC in that region -> labeling conflict;
   training on these as hardneg WOULD punish the model for correctly detecting a real BGC)

## Trial tk2000-hnw1
- total hardneg in trial: 2000
- overlap with baseline hardneg: 606
- NEW hardnegs (not in baseline): 1394
- score range across candidates: min=0.0636 max=1.0000

### 20 random NEW hardneg samples (tk2000-hnw1, deep inspection)

| # | global_idx | genome_id | contig_id | window [start,end) | score | top_type | (1-isneg) | BGC proximity | class |
|---|-----------|-----------|-----------|--------------------|-------|----------|-----------|---------------|-------|
| 1 | 1056467 | GCF_040678115.1 | NZ_JBFATB010000174.1 | 0-8192 | 0.5918 | t2:0.465 | 0.719 | none within 50kb | clean_FP |
| 2 | 379550 | GCF_003752025.1 | NZ_RJKR01000011.1 | 90112-98304 | 0.1291 | t3:0.225 | 0.033 | downstream 19211bp GCF_003752025.1_BGC_006 (Polyketide) | clean_FP |
| 3 | 78451 | GCA_023151225.1 | JAKLJK010000011.1 | 98304-106496 | 0.8748 | t6:0.776 | 0.974 | none within 50kb | clean_FP |
| 4 | 865600 | GCF_040678775.1 | NZ_JBFAQP010000015.1 | 278528-286720 | 0.0787 | t3:0.131 | 0.027 | none within 50kb | clean_FP |
| 5 | 823233 | GCF_040638135.1 | NZ_JBEZXR010000065.1 | 0-8192 | 0.1274 | t3:0.001 | 0.254 | downstream 21165bp GCF_040638135.1_BGC_018 (NRP) | clean_FP |
| 6 | 775077 | GCF_039541095.1 | NZ_BAABFO010000001.1 | 327680-335872 | 0.1311 | t6:0.228 | 0.034 | upstream 15755bp GCF_039541095.1_BGC_001 (Other) | clean_FP |
| 7 | 454808 | GCF_010365055.1 | NZ_JAAFHA010000058.1 | 8192-16384 | 0.9603 | t1:0.928 | 0.993 | none within 50kb | clean_FP |
| 8 | 362406 | GCF_003344965.1 | NZ_CP031194.1 | 4562944-4571136 | 0.2758 | t6:0.424 | 0.127 | none within 50kb | clean_FP |
| 9 | 1064295 | GCF_964110445.1 | NZ_CAXLAL010000137.1 | 0-8192 | 0.9952 | t1:0.994 | 0.996 | none within 50kb | clean_FP |
| 10 | 1039694 | GCF_039521845.1 | NZ_BAAACW010000083.1 | 0-8192 | 0.9713 | t2:0.947 | 0.996 | none within 50kb | clean_FP |
| 11 | 308765 | GCF_001905005.1 | NZ_LWKZ01000024.1 | 475136-483328 | 0.1311 | t6:0.074 | 0.188 | upstream 15641bp GCF_001905005.1_BGC_034 (Other) | clean_FP |
| 12 | 1046465 | GCF_040612395.1 | NZ_JBEXQA010000124.1 | 0-8192 | 0.9136 | t1:0.917 | 0.911 | none within 50kb | clean_FP |
| 13 | 999266 | GCA_963608365.1 | CAVDFM010000071.1 | 0-8192 | 0.9877 | t3:0.984 | 0.991 | none within 50kb | clean_FP |
| 14 | 114515 | GCA_035441325.1 | DAOROA010000010.1 | 0-8192 | 0.1976 | t5:0.174 | 0.221 | none within 50kb | clean_FP |
| 15 | 109098 | GCA_033794685.1 | JARAYZ010000002.1 | 2293760-2301952 | 0.4188 | t6:0.298 | 0.540 | downstream 21104bp GCA_033794685.1_BGC_015 (Alkaloid;Terpene) | clean_FP |
| 16 | 326681 | GCF_002285495.1 | NZ_CP016786.1 | 729088-737280 | 0.1151 | t3:0.185 | 0.046 | upstream 15102bp GCF_002285495.1_BGC_002 (RiPP) | clean_FP |
| 17 | 751712 | GCF_038374745.1 | NZ_JBANSM010000001.1 | 589824-598016 | 0.8213 | t3:0.867 | 0.775 | none within 50kb | clean_FP |
| 18 | 800018 | GCF_040300525.1 | NZ_JBEPDU010000030.1 | 32768-40960 | 0.1107 | t6:0.164 | 0.057 | downstream 31290bp GCF_040300525.1_BGC_020 (Terpene) | clean_FP |
| 19 | 1024311 | GCF_015209575.1 | NZ_JADFCP010000059.1 | 0-8192 | 0.8009 | t3:0.792 | 0.810 | none within 50kb | clean_FP |
| 20 | 1050567 | GCF_040637235.1 | NZ_JBEZVX010000006.1 | 393216-401408 | 0.6882 | t1:0.806 | 0.570 | upstream 14001bp GCF_040637235.1_BGC_011 (NRP) | clean_FP |
## Trial tk100-hnw1
- total hardneg in trial: 100
- overlap with baseline hardneg: 68
- NEW hardnegs (not in baseline): 32

## Trial tk368-hnw1
- total hardneg in trial: 368
- overlap with baseline hardneg: 187
- NEW hardnegs (not in baseline): 181

## Trial tk900-hnw1
- total hardneg in trial: 900
- overlap with baseline hardneg: 352
- NEW hardnegs (not in baseline): 548


## Classification totals (20 samples from tk2000-hnw1)
- clean_FP: 20
- GT-gap: 0
- ambiguous: 0

## Flagged surprises

## Full-population BGC proximity across ALL NEW hardnegs in tk2000-hnw1
- Total NEW hardnegs: 1394
- OVERLAP with annotated BGC (GT-gap): 0 (0.0%)
- 0 < dist <= 10 kb (ambiguous/flanking): 0 (0.0%)
- 10 kb < dist <= 50 kb: 256 (18.4%)
- No BGC within 50 kb (clean_FP): 1138 (81.6%)