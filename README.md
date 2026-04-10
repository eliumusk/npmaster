# np-master: BGC Discovery with Evo2

DNA foundation model (Evo2 7B) for bacterial biosynthetic gene cluster (BGC) detection and novel BGC discovery.

---

## 🔎 审查者导航（学姐请先看这里）

如果你是来帮我审这份工作的，**请先读 [REVIEW.md](REVIEW.md)** —— 里面有专门给审查者的导览：

- 数据集是怎么构造的 / 有什么已知坑
- 训练流程 + 合理性判断要点
- 当前 mainline 的核心 claim 和对应证据文件
- 可以重点质疑的地方

其它关键文档（按重要性排序）：

| 文档 | 内容 |
|------|------|
| [docs/mainline.md](docs/mainline.md) | **当前 mainline 完整概要**（指标 / 架构 / 数据 / 评估）|
| [docs/phase3-route-b-results.md](docs/phase3-route-b-results.md) | 当前 mainline 模型（Route B）的 3-seed 完整结果 |
| [docs/phase3-context-model-plan.md](docs/phase3-context-model-plan.md) | Route B 完整设计文档（含 4 轮 Codex 审查记录）|
| [docs/phase4-morning-report.md](docs/phase4-morning-report.md) | Phase 4 晨报：GECCO/DeepBGC 对比 + discovery 证据升级 + 湿实验候选 |
| [docs/phase4-comparison-and-discovery-plan.md](docs/phase4-comparison-and-discovery-plan.md) | Phase 4 设计文档（含 Codex 审查记录）|
| [docs/threshold-contract.md](docs/threshold-contract.md) | Detection score + threshold policy 规范 |

---

## 当前 Mainline: Phase 3 Route B (context model)

**3-seed mean ± std** on antiSMASH annotations:

| | test_id | test_ood |
|---|---------|----------|
| Recall | **0.820 ± 0.001** | **0.594 ± 0.006** |
| Precision (pred-level) | **0.857 ± 0.002** | **0.653 ± 0.009** |
| `<5kb` recall | **0.751 ± 0.004** | **0.556 ± 0.012** |
| `5-10kb` recall | **0.706 ± 0.003** | **0.651 ± 0.007** |
| `10-20kb` recall | **0.691 ± 0.005** | **0.411 ± 0.009** |

Discovery: sig precision@20 = 90%, sig precision@50 = 92%, diversity@50 = 45 profiles.

**Comparison with DL baselines** (on the same antiSMASH proxy benchmark, native operating points):

| Method | test_id recall | test_id P(pred) | <5kb recall | 5-10kb recall |
|--------|---------------|-----------------|-------------|---------------|
| GECCO 0.10.3 | 0.479 | 0.727 | 0.057 | 0.215 |
| **Route B (ours)** | **0.820** | **0.857** | **0.751** | **0.706** |

Route B's biggest edge is on short BGCs (<10kb), where it recovers **3-13x** more BGCs than GECCO.

**Matched null discovery enrichment:** signature density in top candidates is **1.79x** higher than same-genome GC/density-matched null controls (Mann-Whitney U, p = 1.8e-5).

**Architecture:** end-to-end model on frozen Evo2 7B embeddings:
```
Evo2 7B (frozen) → mean pool → 4096-dim window embedding →
  ProjectionHead (warm-started, trainable) → 256-dim →
  Dilated 1D CNN (3 layers, d=1,2,4, receptive field 15 windows = 120kb) →
  GatedResidual (sigmoid gate, β init -4.0 → α ≈ 0.018) →
  Classifier (warm-started) → 8 labels
```

Trained on contig chunks (L=32 windows, overlap stride=16) with **center-region masked BCE loss** and **AdamW layered lr** (proj 1e-4, context/classifier 1e-3).

Previous mainline **H_hardneg_r10** (archived in `configs/mainline.yaml`): test_id recall 0.731, test_ood recall 0.566. Phase 3 Route B is the first architecture upgrade.

---

## Repository Structure

```
bgc-gcd/                              # Training + evaluation code
  configs/
    mainline.yaml                     # H_hardneg_r10 config (previous mainline, still works)
    mainline_route_b.yaml             # Route B config (current mainline)
  train.py                            # Old training loop (MLP only, H_hardneg_r10)
  train_context.py                    # Route A training loop (frozen projector, 256-dim)
  train_context_e2e.py                # Route B training loop (end-to-end, 4096-dim)
  evaluate.py                         # Window-level evaluation (AUROC per type)
  evaluate_context.py                 # Context model evaluation helpers
  model/
    projector.py                      # 4096 → 2048 → 256, L2-normalized
    classifier.py                     # 256 → 8 labels
    dann.py                           # GRL + DomainClassifier (phylum invariance, H_hardneg_r10 only)
    full_model.py                     # BGCModel: Projector + Classifier + DANN (H_hardneg_r10)
    context_model.py                  # ContextBGCModel: dilated CNN + gated residual (Route A)
    context_full_model.py             # ContextFullModel: end-to-end (Route B, current mainline)
  data/
    dataset.py                        # BGCDataset: loads per-window embeddings + labels
    sampler.py                        # BalancedBatchSampler with hard-neg upsampling
    augmentation.py                   # EmbeddingAugmentation (noise / dim dropout)
    context_dataset.py                # ContextDataset: K-window neighborhoods (Route A)
    chunk_dataset.py                  # ChunkDataset: overlapping contig chunks (Route B)
  losses/
    classification_loss.py            # MultiLabelBCE with per-class pos_weight
    contrastive_loss.py               # MultiLabelSupCon (cross-phylum weighted, H_hardneg_r10)
    dann_loss.py                      # DANN domain adversarial loss
  utils.py                            # seed, config loading, device helpers

scripts/eval/                         # Region-level evaluation tools (shared across phases)
  ws1_score_swap.py                   # Region-level eval: score → threshold → region calling → recall/precision
  p1_t3_detailed_eval.py              # Per-bucket eval (length, truncation, threshold sweep)
  ws_discovery_benchmark.py           # Pfam-based novel candidate validation
  p1_t2_edge_padding.py               # Generate N-padded windows for short contigs
  p1_t2_extract_single_gpu.py         # Evo2 extraction for new windows
  p1_t2_merge_shards.py               # Merge multi-GPU extraction shards
  p2_t2_hard_negative_mining.py       # Hard-negative mining script

scripts/phase3/                       # Phase 3 (context model) scripts
  extract_projections.py              # Pre-extract 256-dim features from H_hardneg_r10 ProjectionHead
  build_chunks.py                     # Build overlapping chunk index for Route B
  context_predict.py                  # Full-contig inference with Route A model + region calling

scripts/phase4/                       # Phase 4 (comparison + discovery) scripts
  reproduce_mainline.sh               # One-command reproduction of Route B 3-seed training
  eval_mainline.sh                    # Route B 3-seed evaluation
  analysis/                           # Phase 4 analysis scripts (see analysis/README.md)
    01_baselines/                     # GECCO + DeepBGC install, run, parse, compare
    02_discovery/                     # D1-D5: matched null, dual rankers, concordance, top candidates
    03_helpers/                       # Genome lists, eval wrappers, training launchers
    04_audit/                         # Route B eval path audit scripts

docs/                                 # Current documentation
  mainline.md                         # Start here: current system state + all numbers
  phase1-short-bgc-coverage.md        # What edge padding fixed
  phase2-ood-robustness.md            # What hard-neg mining fixed
  phase3-context-model-plan.md        # Route B design doc (4-round Codex review)
  phase3-route-a-results.md           # Route A (frozen projector) results
  phase3-route-b-results.md           # Route B (end-to-end) results — CURRENT MAINLINE
  phase4-comparison-and-discovery-plan.md   # Phase 4 design doc (Codex review)
  phase4-morning-report.md            # Phase 4 execution results summary
  stability-3seed.md                  # 3-seed variance confirmation (H_hardneg_r10)
  threshold-contract.md               # Detection score + threshold policy spec
  discovery-benchmark.md              # Novel candidate Pfam validation results
```

---

## How to Run

### Prerequisites

- Python 3.9+ with conda
- GPU with ≥40 GB VRAM (A100 80GB / A800 80GB recommended)
- Dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `biopython`
- For baselines: `gecco-tool` (pip), `deepbgc` (pip, requires Python 3.7)
- For Pfam validation: HMMER, prodigal, Pfam-A HMM database
- **Data**: the training data (~17 GB of Evo2 embeddings + 1.06M windows) lives on shared storage, not in git. See `docs/mainline.md` for data paths.

### Train current mainline (Route B, 3 seeds)

```bash
cd bgc-gcd

# All three seeds sequentially (~35 min each on 1x A800)
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed python -u train_context_e2e.py \
    --embeddings $EMB \
    --projections $PROJECTIONS \
    --chunk-index $CHUNK_INDEX \
    --meta-arrays $META_ARRAYS \
    --contig-index $CONTIG_INDEX \
    --bgc-csv $BGC_CSV \
    --ckpt $H_HARDNEG_CKPT \
    --out-dir results/route_b \
    --seed $seed --epochs 100 --batch-size 64 \
    --lr-proj 1e-4 --lr-context 1e-3 --lr-classifier 1e-3
done
```

Or use the wrapper:

```bash
bash scripts/phase4/reproduce_mainline.sh all
```

### Evaluate Route B

Region-level evaluation with per-bucket recall:

```bash
python scripts/phase4/analysis/03_helpers/eval_3seed.py
```

Outputs per-seed metrics and 3-seed mean ± std for both test_id and test_ood splits.

### Compare with baselines

```bash
# 1. Run GECCO on test genomes
bash scripts/phase4/analysis/01_baselines/run_gecco_batch.sh

# 2. Run DeepBGC on genomes containing top candidates
bash scripts/phase4/analysis/01_baselines/run_deepbgc_small.sh

# 3. Parse outputs to unified format
python scripts/phase4/analysis/01_baselines/parse_gecco.py
python scripts/phase4/analysis/01_baselines/parse_deepbgc.py

# 4. Generate comparison table
python scripts/phase4/analysis/01_baselines/p3_compare.py
```

### Discovery benchmark (Pfam validation + dual rankers + multi-tool concordance)

```bash
# D1: diagnose current discovery weakness
python scripts/phase4/analysis/02_discovery/d1_diagnosis.py

# D2: matched null construction and distribution comparison
python scripts/phase4/analysis/02_discovery/d2_matched_null.py
bash scripts/phase4/analysis/02_discovery/d2_run_hmmscan.sh
python scripts/phase4/analysis/02_discovery/d2_analyze.py

# D3: dual composite rankers (Conservative + Novelty)
python scripts/phase4/analysis/02_discovery/d3_rankers.py

# D4: multi-tool concordance (requires GECCO + DeepBGC outputs)
python scripts/phase4/analysis/02_discovery/d4_concordance.py

# D5: top candidate deep dive with Track 1 + Track 2 handoff lists
python scripts/phase4/analysis/02_discovery/d5_deep_dive.py
```

---

## Training data

**The 17 GB embedding matrix is not in git** — it lives on shared GPU server storage. Specifically:

- `genomes/` — 1996 bacterial genome `.fna.gz` files (GTDB v226 filtered)
- `bgc_annotations.csv` — 19,762 antiSMASH-annotated BGCs
- `embeddings/edge_padding/embeddings_merged.npy` — 1,064,568 × 4096 Evo2 embeddings (17 GB)
- `stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv` — window metadata with labels
- `stage3_results/H_hardneg_r10/best_H_hardneg_r10.pt` — previous mainline checkpoint (used for Route B warm start)
- `phase3/route_b/best_route_b_s{0,1,2}.pt` — Route B 3-seed checkpoints (current mainline)

For the data layout and how the dataset was built, see:
- [docs/mainline.md §Training data](docs/mainline.md)
- Stage reports in `docs/_archive/stage_reports/` (build pipeline: GTDB selection → genome download → antiSMASH annotation → window extraction → Evo2 embedding)

---

## Key design decisions (for reviewers)

The reasoning behind each phase is in the phase design docs. Very briefly:

1. **Why frozen Evo2**: extraction costs ~400 GPU-hours per full pass. Stage 2 diagnostic showed layer-20 embedding has weak-but-nonzero BGC signal (kNN enrichment 1.4x). MLP head was chosen as lowest-cost option to validate feasibility.
2. **Why MLP first, then context CNN**: originally the whole signal gain was from specification fixes (Phase 0-2), not architecture. Phase 3 is the first architecture upgrade, once we confirmed signal limits.
3. **Why overlap chunks + center-region loss**: reviewed by Codex — alternative "center-only neighborhood" design would double-count long contigs and distort class prior.
4. **Why dual rankers (Conservative + Novelty)**: single composite ranker would be tuned toward "looks like known BGC," systematically excluding genuinely novel candidates. See Phase 4 plan §5.3.
5. **Why matched null instead of random null**: random negative windows still hit BGC-rich regions in Streptomyces; enrichment is underestimated. Matched null (same-genome, GC-matched, density-matched, exclude all tool calls) gives 1.79x real enrichment vs ~1.3x random.

---

## Limitations (honest)

- **MIBiG Tier 1 external validation not done**: requires MIBiG metadata JSON which we don't have on local. Tier 2 species-level matching gives 405 approximate matches but is noisy.
- **GECCO / DeepBGC training set leakage not audited**: they might have been trained on some of our test genomes.
- **DeepBGC uses Pfam 35.0** (symlinked from antiSMASH) instead of the native Pfam 31.0 — minor version drift might cost a few percent.
- **Per-candidate z-score on matched null only available for 18/50 candidates** — sampling constraints limit null coverage per candidate.
- **Only 50/1006 clean-missed candidates have Pfam hits**: full-scale ranking needs a batch hmmscan pass (not yet done).
- **test_ood is mostly Cyanobacteriota** — OOD claim cannot be generalized to all phyla.

See each phase doc's "Limitations" section for details.

---

## GPU Servers

The training runs on shared GPU servers. Server access details and paths are in the project memory, not in this README (for security).

- Two 8× A800 80GB nodes
- Shared storage under `/data/muskliu/np_master/`
- conda env `bgc` for training / evaluation (PyTorch)
- conda env `evo2_bgc` for Evo2 extraction only
- Evo2 model cached offline; set `HF_HUB_CACHE=/data/public_models/hub`

Ask the project owner for SSH credentials.

---

## Acknowledgments

- Evo2 7B DNA foundation model (Arc Institute / Goodman lab)
- antiSMASH 7 (for training labels and baseline comparison)
- GECCO (Larralde et al., 2024) and DeepBGC (Hannigan et al., 2019) for baseline comparison
- GTDB r226 genome selection
- Codex review for Phase 3 + Phase 4 plan iterations
