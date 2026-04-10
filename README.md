# np-master: BGC Discovery with Evo2

DNA foundation model (Evo2 7B) for bacterial biosynthetic gene cluster detection and novel BGC discovery.

## Current Mainline: Phase 3 Route B (context model)

| | test_id | test_ood |
|---|---------|----------|
| Recall | **0.820 +/- 0.001** | **0.594 +/- 0.006** |
| Precision | **0.857 +/- 0.002** | **0.653 +/- 0.009** |
| <5kb recall | **0.751 +/- 0.004** | **0.556 +/- 0.012** |
| 5-10kb recall | **0.706 +/- 0.003** | **0.651 +/- 0.007** |

Sig precision@20: 90%, sig precision@50: 92%, diversity@50: 45 unique profiles.

End-to-end model: ProjectionHead (warm-started, trainable) + Dilated 1D CNN + GatedResidual + Classifier. Trained on contig chunks (L=32, overlap stride=16) with center-region masked BCE loss.

Previous mainline H_hardneg_r10 (archived): test_id recall 0.731, test_ood recall 0.566.

For full context read `docs/mainline.md` and `docs/phase3-route-b-results.md`.

## Repo Structure

```
bgc-gcd/                      # Training + evaluation code
  configs/mainline.yaml           # Current config (the only one that matters)
  train.py                        # Training loop (supports hard-neg upsampling)
  evaluate.py                     # Window-level evaluation
  model/                          # ProjectionHead + Classifier + DANN
  data/                           # Dataset, sampler, augmentation
  losses/                         # BCE, contrastive, DANN

scripts/eval/                  # Reusable evaluation tools
  ws1_score_swap.py               # Region-level eval (recall, precision, unmatched)
  p1_t3_detailed_eval.py          # Per-bucket eval (length, truncation, threshold sweep)
  ws_discovery_benchmark.py       # Pfam-based novel candidate validation
  p2_t2_hard_negative_mining.py   # Mine hard negatives from training data
  p1_t2_edge_padding.py           # Generate padded windows for short contigs
  p1_t2_extract_single_gpu.py     # Evo2 extraction for padded windows
  p1_t2_merge_shards.py           # Merge multi-GPU extraction shards

docs/                          # Current documentation
  mainline.md                     # Start here: current system state + numbers
  stability-3seed.md              # 3-seed variance confirmation
  phase1-short-bgc-coverage.md    # What edge padding fixed
  phase2-ood-robustness.md        # What hard-neg mining fixed
  discovery-benchmark.md          # Novel candidate Pfam validation results
  threshold-contract.md           # Detection score + threshold policy spec
```

---

## GPU Servers

Two 8-card A800 nodes share storage via the same `/data/` mount.

| Name | Host | Port | User | GPUs |
|------|------|------|------|------|
| node7 | 221.194.152.152 | 50000 | root | 8x A800 80GB |
| node8 | 221.194.152.152 | 50010 | root | 8x A800 80GB |

SSH key: `~/.ssh/id_rsa_voc`

```bash
# Connect
ssh -i ~/.ssh/id_rsa_voc -p 50000 root@221.194.152.152   # node7
ssh -i ~/.ssh/id_rsa_voc -p 50010 root@221.194.152.152   # node8
```

### Conda Environments

There are two relevant environments. Choose based on task:

| Env | Activate | Use for | Has |
|-----|----------|---------|-----|
| `bgc` | `source /root/miniconda3/etc/profile.d/conda.sh && conda activate bgc` | Training, evaluation, all non-Evo2 work | torch, sklearn, numpy |
| `evo2_bgc` | `source /data/conda/ourconda_bashrc && conda activate evo2_bgc` | Evo2 embedding extraction only | torch, evo2 |

### HuggingFace Offline Mode

The server has no internet. Evo2 model weights are cached locally. When running Evo2 extraction, you MUST set:

```bash
export HF_HUB_CACHE=/data/public_models/hub
```

Without this, `evo2.Evo2("evo2_7b")` will fail with a connection timeout.

### Server Data Layout

**No scripts or code live on the server.** Everything runs by syncing from local first.

```
/data/muskliu/np_master/
  data/
    genomes/                              # 1996 bacterial genome .fna.gz files
    bgc_annotations.csv                   # 19,762 antiSMASH BGC annotations
    genome_metadata.csv                   # Genome panel metadata
    selected_genomes.tsv                  # GTDB selection table

    embeddings/
      embeddings.npy                      # Original 948,362 x 4096 (15 GB)
      nll_features.npy                    # Original 948,362 x 8 (29 MB)
      edge_padding/
        embeddings_merged.npy             # Current mainline: 1,064,568 x 4096 (17 GB)
                                          # = original + 116,206 padded windows

    windows/
      windows_metadata.csv                # Original labels (no edge padding, no cap)
      windows_metadata_cap08.csv          # + cap=0.8 label rule
      edge_padding/
        windows_metadata_edgepad_filtered.csv  # + edge-padded windows

    stage3_results/
      H_hardneg_r10/                      # Mainline checkpoint (seed 0)
        best_H_hardneg_r10.pt
      H_hardneg_r10_s1/                   # seed 1
      H_hardneg_r10_s2/                   # seed 2

    stage4_results/
      p2_t2_hard_neg/
        windows_metadata_hardneg.csv      # Current mainline labels (with hard_negative flag)
      p2_t2_hardneg_r10_eval/             # Mainline eval results
        score_swap_results.json
      p2_t2_hardneg_r10_discovery/        # Mainline discovery benchmark
        discovery_benchmark.json
      p2_stability_s{0,1,2}/              # 3-seed stability eval
      pfam_validation/                    # Pfam HMM database for discovery benchmark
        biosyn_pfam.hmm
        biosyn_names.txt
      p1_t1_audit/                        # Short-BGC diagnostic (reference)

    _archive/                             # All historical checkpoints and evals (17 GB)
```

### Key Paths Quick Reference

```bash
# Current mainline checkpoint
CKPT=/data/muskliu/np_master/data/stage3_results/H_hardneg_r10/best_H_hardneg_r10.pt

# Current mainline embeddings (1,064,568 windows)
EMB=/data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy

# Current mainline labels (with hard_negative column)
META=/data/muskliu/np_master/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv

# BGC annotations
BGC=/data/muskliu/np_master/data/bgc_annotations.csv

# Genomes
GENOMES=/data/muskliu/np_master/data/genomes
```

---

## How to Run Things

### Principle: local is truth, server is compute

1. Edit code locally
2. `scp` to server
3. `ssh` to run
4. `scp` results back if needed

```bash
# Sync code to server
scp -r -P 50000 -i ~/.ssh/id_rsa_voc bgc-gcd/ scripts/ \
  root@221.194.152.152:/data/muskliu/np_master/
```

### Train

```bash
ssh -i ~/.ssh/id_rsa_voc -p 50000 root@221.194.152.152 "
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate bgc
  cd /data/muskliu/np_master/bgc-gcd
  CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --config configs/mainline.yaml --seed 0 --tag my_run
"
```

Training takes ~7 min for 100 epochs on one A800. Output goes to the dir specified in `log.out_dir` in the config yaml.

### Evaluate (region-level recall + precision)

```bash
ssh -i ~/.ssh/id_rsa_voc -p 50000 root@221.194.152.152 "
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate bgc
  cd /data/muskliu/np_master
  CUDA_VISIBLE_DEVICES=0 python -u scripts/eval/ws1_score_swap.py \
    --ckpt \$CKPT --emb \$EMB --meta \$META --bgc-csv \$BGC \
    --out data/stage4_results/my_eval --fullwindow
"
```

### Evaluate (per-bucket detail: short BGC, truncation, threshold sweep)

```bash
python -u scripts/eval/p1_t3_detailed_eval.py \
  --ckpt-new $CKPT --emb-new $EMB --meta-new $META \
  --baseline-eval data/stage4_results/p2_t2_hardneg_r10_eval/score_swap_results.json \
  --bgc-csv $BGC --out data/stage4_results/my_detail_eval
```

### Discovery benchmark (Pfam validation on novel candidates)

```bash
python -u scripts/eval/ws_discovery_benchmark.py \
  --results data/stage4_results/my_eval/score_swap_results.json \
  --genomes-dir $GENOMES \
  --pfam-db data/stage4_results/pfam_validation/biosyn_pfam.hmm \
  --biosynthetic-list data/stage4_results/pfam_validation/biosyn_names.txt \
  --meta $META --out data/stage4_results/my_discovery --top-n 50 --cpus 16
```

### Evo2 extraction (only needed if adding new windows)

Must use `evo2_bgc` env and set `HF_HUB_CACHE`:

```bash
ssh -i ~/.ssh/id_rsa_voc -p 50000 root@221.194.152.152 "
  source /data/conda/ourconda_bashrc && conda activate evo2_bgc
  export HF_HUB_CACHE=/data/public_models/hub
  cd /data/muskliu/np_master
  CUDA_VISIBLE_DEVICES=0 python -u scripts/eval/p1_t2_extract_single_gpu.py \
    --fasta INPUT.fasta --gpu-id 0 --n-gpus 1 \
    --out data/embeddings/new_extraction
"
```

Speed: ~0.64 seq/s per A800 (1.56s per 16384bp window). For multi-GPU, launch one process per GPU with `--gpu-id K --n-gpus N` and `CUDA_VISIBLE_DEVICES=K`.
