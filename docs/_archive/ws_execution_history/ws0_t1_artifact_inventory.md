# WS0-T1: Canonical Artifact Inventory
> **Archive note:** Artifact paths in this inventory reflect the Config B era state. Many paths have since been archived to `_archive/`. Current mainline artifacts are in `ws3_cap08_eval/` and `ws_discovery_benchmark_cap08/`.

**Date:** 2026-04-09
**Task ID:** WS0-T1
**Server:** Voc_gpu_node7_a800
**Base path:** `/data/muskliu/np_master/`

---

## Canonical Artifacts (Stage 4 Results)

All under `/data/muskliu/np_master/data/stage4_results/`.

| Path | Type | Size | Modified | Contents | Canonical? |
|------|------|------|----------|----------|------------|
| `stage4_summary.json` | JSON | 5.5K | Apr 8 05:05 | Master summary: per-type thresholds, genome-level metrics (test_id, test_ood), FN breakdown by type/length/truncation/phylum, novel candidate summary, MiBIG validation (AUROC, recall@0.5, type concordance). **Primary source of all headline metrics.** | YES |
| `step4_1/optimal_thresholds.json` | JSON | 1.3K | Apr 8 05:05 | Per-type F1-optimal thresholds + precision/recall/F1 at those thresholds. 8 entries (7 types + is_negative). | YES |
| `step4_1/false_positive_analysis.json` | JSON | 36K | Apr 8 05:05 | Detailed FP analysis per window. | YES |
| `step4_1/false_negative_analysis.json` | JSON | 17K | Apr 8 05:05 | Detailed FN analysis per window. | YES |
| `step4_2/genome_level_metrics.json` | JSON | 874B | Apr 8 05:05 | Genome-level detection: test_id (recall=0.2707, prec=0.5729) and test_ood (recall=0.2132, prec=0.8901). Per-type recall breakdown. | YES |
| `step4_2/predicted_regions.json` | JSON | 884K | Apr 8 05:05 | All predicted contiguous BGC regions across genomes. | YES |
| `step4_2/matching_results.json` | JSON | 342K | Apr 8 05:05 | Predicted region to antiSMASH BGC matching details. | YES |
| `step4_2/novel_candidates.json` | JSON | 207K | Apr 8 05:05 | Novel candidate regions not matching any antiSMASH annotation. | YES |
| `step4_2/per_genome_summary.csv` | CSV | 38K | Apr 8 05:05 | Per-genome stats: n_predicted, n_matched, n_missed, precision, recall. | YES |
| `step4_3/novel_candidates/candidate_summary.json` | JSON | 526B | Apr 8 05:05 | Novel candidate breakdown: 683 total (565 high, 118 medium confidence). By type: NRP 398, Polyketide 195, RiPP 34, Other 17, Saccharide 16, Terpene 13, Alkaloid 10. Median distance to nearest BGC: 43,943 bp. | YES |
| `step4_3/novel_candidates/candidates_ranked.csv` | CSV | 69K | Apr 8 05:05 | Ranked list of all novel candidates with scores and metadata. | YES |
| `step4_3/false_negative_analysis/fn_summary.json` | JSON | 1.8K | Apr 8 05:05 | FN breakdown by type, truncation, length, phylum (top 10). Total FN: 20,654 windows. Key: <10kb BGCs have 100% miss rate. | YES |
| `step4_4/mibig_metrics.json` | JSON | 503B | Apr 8 05:05 | MiBIG benchmark: 2502 BGCs, 99.88% detection rate, mean AUROC 0.7293, type concordance 0.775. Per-type AUROC ranges from 0.43 (Saccharide) to 0.94 (RiPP). | YES |
| `fp_spatial_analysis.json` | JSON | 864B | Apr 8 05:17 | FP spatial analysis: 14,614 total FPs, 88.5% far from BGC (>20kb), median distance 78kb. By predicted type: NRP dominates (8,968). 286 genomes have FPs, median 7 per genome. | YES |
| `recall_targeted_thresholds.json` | JSON | 373B | Apr 8 05:10 | Recall-targeted thresholds: at 70% recall precision drops to 0.357; at 80% recall precision drops to 0.263. | YES |
| `error_analysis/error_analysis.json` | JSON | 6.1K | Apr 8 06:49 | Detailed error analysis at threshold 0.128: overall miss rate 37.2%. By length: <5kb 100% miss, 5-10kb 97.9%, 10-20kb 54.6%, 20-50kb 30.7%, >50kb 1.4%. By type, truncation, overlap ratio, score distribution. | YES |
| `error_analysis/bgc_detection_details.json` | JSON | 564K | Apr 8 06:49 | Per-BGC detection details for error analysis. | YES |
| `pfam_validation/validation_summary.json` | JSON | 84K | Apr 8 05:37 | Pfam validation of top 50 novel candidates: 100% have biosynthetic Pfam, 92% have BGC signature domains. Top domains: PCP/T (38), AMP-binding (36), condensation (34). | YES |
| `pfam_validation/biosyn_pfam.hmm` | HMM | 214M | Apr 8 05:32 | HMMER database for biosynthetic Pfam domains. | NO (tool asset) |
| `pfam_validation/biosyn_pfam.hmm.h3i` | HMM index | 162K | Apr 8 05:32 | HMMER index file. | NO (tool asset) |
| `pfam_validation/biosyn_pfam.hmm.h3f` | HMM data | 46M | Apr 8 05:32 | HMMER filter file. | NO (tool asset) |
| `pfam_validation/biosyn_pfam.hmm.h3p` | HMM data | 105M | Apr 8 05:32 | HMMER profile file. | NO (tool asset) |
| `pfam_validation/biosyn_pfam.hmm.h3m` | HMM data | 89M | Apr 8 05:32 | HMMER model file. | NO (tool asset) |
| `pfam_validation/biosyn_names.txt` | Text | 24K | Apr 8 05:32 | List of biosynthetic Pfam family names. | NO (tool asset) |

---

## Data Artifacts (Stage 1/2)

All under `/data/muskliu/np_master/data/`.

### Stage 1: Genome Selection + antiSMASH Annotation

| Path | Type | Size | Modified | Contents |
|------|------|------|----------|----------|
| `genome_metadata.csv` | CSV | 424K | Apr 5 17:45 | 2000 genomes: species, phylum, class, order, family, genus, genome_size, n_contigs, checkm stats, assembly level, accession, fasta path. |
| `bgc_annotations.csv` | CSV | 2.3M | Apr 5 17:46 | antiSMASH BGC annotations: bgc_id, genome_id, contig_id, start/end_bp, length, types, product, is_truncated, phylum. |
| `selected_genomes.tsv` | TSV | 360K | Apr 5 14:04 | Initial genome selection from GTDB metadata. |
| `bac120_metadata.tsv.gz` | TSV.gz | 225M | Apr 5 14:02 | Full GTDB bac120 metadata (source data). |
| `antismash_summary.json` | JSON | 6.5K | Apr 5 17:46 | antiSMASH run summary across all genomes. |
| `genome_metadata_report.json` | JSON | 5.4K | Apr 5 17:45 | Genome metadata generation report. |
| `filter_report.json` | JSON | 3.2K | Apr 5 14:04 | Genome filtering criteria and counts. |
| `genomes/` | Dir (FASTA) | ~large | Apr 5 17:32 | 2000 genome FASTA files (.fna.gz). |
| `antismash_results/` | Dir | 1999 subdirs | Apr 5 17:30 | Per-genome antiSMASH output directories. |
| `withdrawn_accessions.txt` | Text | 64B | Apr 5 15:23 | NCBI accessions withdrawn during pipeline. |
| `STAGE1_REPORT.md` | Markdown | 8.1K | Apr 6 03:16 | Stage 1 narrative report. |

### Stage 1.5: Window Creation + Data Split

| Path | Type | Size | Modified | Contents |
|------|------|------|----------|----------|
| `windows/windows_metadata.csv` | CSV | 106M | Apr 6 03:13 | Full window metadata: 948,362 windows with genome_id, contig, position, labels, split assignment. |
| `windows/split_summary.json` | JSON | 7.6K | Apr 6 03:13 | Split statistics: train 1190 genomes / 562K windows, val 170 / 86K, test_id 340 / 158K, test_ood 300 / 142K. 0 GCF-leaked genomes. Class distributions per split. |
| `windows/train_windows.txt` | Text | 5.9M | Apr 6 03:13 | Train split window IDs. |
| `windows/val_windows.txt` | Text | 923K | Apr 6 03:13 | Val split window IDs. |
| `windows/test_id_windows.txt` | Text | 1.7M | Apr 6 03:13 | Test-ID split window IDs. |
| `windows/test_ood_windows.txt` | Text | 1.5M | Apr 6 03:13 | Test-OOD split window IDs. |

### Stage 2: Evo2 Embedding Extraction

| Path | Type | Size | Modified | Contents |
|------|------|------|----------|----------|
| `embeddings/embeddings.npy` | NPY | 15G | Apr 8 02:01 | 948,362 x 4,096 float32 embedding matrix. Layer: `blocks.20.mlp.l3`. |
| `embeddings/nll_features.npy` | NPY | 29M | Apr 8 02:01 | 948,362 x 8 NLL-derived features per window. |
| `embeddings/extraction_summary.json` | JSON | 813B | Apr 8 02:01 | Extraction report: 1996 genomes processed, 0 failed. 948,362 windows, dim=4096, no NaN/Inf. NLL diff var: positive mean 0.978, negative mean 0.946. |
| `embeddings/per_genome/` | Dir | ~large | Apr 7 23:49 | Per-genome embedding shards (intermediate). |

### Legacy Pickle Artifacts (pre-Stage 3 refactor)

| Path | Type | Size | Modified | Contents |
|------|------|------|----------|----------|
| `train.pkl` | Pickle | 72M | Apr 4 09:02 | Legacy train data (pre-refactor). |
| `val.pkl` | Pickle | 7.4M | Apr 4 09:02 | Legacy val data. |
| `test.pkl` | Pickle | 11M | Apr 4 09:02 | Legacy test data. |
| `train_emb.pkl` | Pickle | 143M | Apr 4 11:17 | Legacy train embeddings. |
| `val_emb.pkl` | Pickle | 15M | Apr 4 10:00 | Legacy val embeddings. |
| `test_emb.pkl` | Pickle | 21M | Apr 4 09:48 | Legacy test embeddings. |
| `classification_results.pkl` | Pickle | 571B | Apr 4 12:28 | Legacy classification output. |

### Stage 3: GCD Ablation Results

| Path | Type | Size | Modified | Contents |
|------|------|------|----------|----------|
| `stage3_results/run/` | Dir | 2 checkpoints | Apr 8 02:25 | Smoke test run. |
| `stage3_results/A_baseline/` | Dir | 6 checkpoints | Apr 8 02:46 | Config A: baseline (3 seeds x best/last). |
| `stage3_results/B_contrastive/` | Dir | 6 checkpoints | Apr 8 03:10 | Config B: +contrastive loss (3 seeds x best/last). |
| `stage3_results/C_full/` | Dir | 6 checkpoints | Apr 8 03:34 | Config C: +contrastive+entropy (3 seeds x best/last). |
| `stage3_results/D_with_nll/` | Dir | 6 checkpoints | Apr 8 03:59 | Config D: +NLL features (3 seeds x best/last). |
| `stage3_ablation.log` | Log | 209K | Apr 8 03:59 | Full Stage 3 ablation training log. |
| `STAGE3_REPORT.md` | Markdown | 7.3K | Apr 8 04:06 | Stage 3 narrative report. |

---

## Checkpoint Information

### Stage 3 Checkpoints (used by Stage 4)

The Stage 4 pipeline used checkpoints from the Stage 3 ablation. Based on the `stage4_summary.json` metrics and config D being the final ablation variant:

**Primary checkpoint path:** `/data/muskliu/np_master/data/stage3_results/D_with_nll/best_D_with_nll_s0.pt`
(and seeds s1, s2 at same location)

All checkpoints are 35M each, MLP-based GCD heads on frozen Evo2 embeddings.

### Earlier Checkpoints (bgc-gcd/results/)

| Path | Size | Modified | Contents |
|------|------|----------|----------|
| `bgc-gcd/results/run/best_smoke.pt` | 35M | Apr 5 14:19 | Earlier smoke test. |
| `bgc-gcd/results/A_baseline/best_A_baseline_s{0,1,2}.pt` | 35M ea. | Apr 5 14:20-22 | Earlier baseline (pre-refactor). |
| `bgc-gcd/results/B_contrastive/best_B_contrastive_s{0,1,2}.pt` | 35M ea. | Apr 5 14:22-24 | Earlier contrastive (pre-refactor). |
| `bgc-gcd/results/C_full/best_C_full_s{0,1,2}.pt` | 35M ea. | Apr 5 14:25-27 | Earlier full (pre-refactor). |

These are from the earlier Stage 3 run (Apr 5) before the data pipeline was re-run. The Apr 8 `stage3_results/` checkpoints are the canonical ones.

### Evo2 Backbone

Evo2 model itself is not stored in this project directory. The extraction used layer `blocks.20.mlp.l3` to produce 4096-dim embeddings. The backbone was frozen (no fine-tuning).

---

## Evidence Traceability

For each key metric cited in the redesign memo (`docs/2026-04-08-bgc-system-redesign.md`), the source artifact is mapped below.

| Metric / Claim in Memo | Source Artifact | Key in JSON |
|-------------------------|-----------------|-------------|
| Genome-level recall test_id = 0.2707 | `step4_2/genome_level_metrics.json` | `test_id.recall` |
| Genome-level recall test_ood = 0.2132 | `step4_2/genome_level_metrics.json` | `test_ood.recall` |
| Genome-level precision test_id = 0.5729 | `step4_2/genome_level_metrics.json` | `test_id.precision` |
| Genome-level precision test_ood = 0.8901 | `step4_2/genome_level_metrics.json` | `test_ood.precision` |
| Type concordance test_id = 0.8864 | `step4_2/genome_level_metrics.json` | `test_id.type_concordance` |
| MiBIG detection rate = 99.88% | `step4_4/mibig_metrics.json` | `bgc_detection_rate` |
| MiBIG mean AUROC = 0.7293 | `step4_4/mibig_metrics.json` | `mean_auroc` |
| MiBIG type concordance = 0.775 | `step4_4/mibig_metrics.json` | `type_concordance` |
| 683 novel candidates (565 high conf) | `step4_3/novel_candidates/candidate_summary.json` | `total`, `by_confidence` |
| 100% Pfam validation on top 50 | `pfam_validation/validation_summary.json` | `pct_with_biosynthetic` |
| 92% BGC signature domain rate | `pfam_validation/validation_summary.json` | `pct_with_bgc_signature` |
| FN: <10kb BGCs 100% miss rate | `step4_3/false_negative_analysis/fn_summary.json` | `fn_by_length.<10kb.miss_rate` |
| FN: truncated BGCs 5797/6270 missed | `step4_3/false_negative_analysis/fn_summary.json` | `fn_by_truncation.truncated` |
| FP spatial: 88.5% far from any BGC | `fp_spatial_analysis.json` | `far_from_bgc_pct` |
| Recall@70% requires prec drop to 0.357 | `recall_targeted_thresholds.json` | `recall_70%` |
| Error analysis: overall miss rate 37.2% at t=0.128 | `error_analysis/error_analysis.json` | `overall_bgc_miss_rate` |
| 948,362 total windows, 4096-dim | `embeddings/extraction_summary.json` | `total_windows_extracted`, `embedding_dim` |
| 2000 genomes, 0 GCF leakage | `windows/split_summary.json` | `split_genomes_final`, `gcf_leakage_check` |
| Per-type recall: Terpene 0.098 (test_id) | `step4_2/genome_level_metrics.json` | `test_id.per_type_recall.Terpene` |

---

## Summary Statistics

- **Total canonical JSON artifacts:** 17 files containing reproducible quantitative metrics
- **Total non-canonical tool assets:** 6 files (Pfam HMM database + index)
- **Stage 4 results total disk:** ~500M (dominated by HMM database)
- **Embeddings disk:** ~15G (single NPY file)
- **Genome data disk:** large (2000 genomes + antiSMASH results)
- **Checkpoint disk:** ~35M per checkpoint, 26 checkpoints total across all configs = ~910M
