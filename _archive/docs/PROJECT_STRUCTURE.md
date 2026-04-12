# Project Structure — GCD Phase

**Status**: Stage 1 done, Stage 2 running, Stage 3 code ready (2026-04-07).

## Active Directories

```
np-master/
├── stage1_detailed.md            # Stage 1 执行手册
├── stage2_detailed.md            # Stage 2 执行手册
├── stage3_detailed.md            # Stage 3 执行手册
├── related_work_survey.md        # 文献调研 (~30篇)
├── PROJECT_STRUCTURE.md          # 本文件
│
├── scripts/
│   ├── diagnostics/              # Day-1 诊断脚本 (01-06_*.py)
│   └── stage1/                   # Stage 1 数据构建脚本
│       ├── 01_download_gtdb_metadata.py
│       ├── 02_filter_and_sample.py
│       ├── 03_download_genomes.py
│       ├── 04_run_antismash.py
│       ├── 05_parse_antismash.py
│       ├── 06_create_windows.py
│       ├── 07_split_data.py
│       └── 08_generate_report.py
│
├── bgc-gcd/                      # Stage 3 训练代码
│   ├── README.md                 # 使用说明
│   ├── train.py                  # 主训练入口
│   ├── run_ablation.py           # 消融实验 runner
│   ├── evaluate.py               # 评估脚本
│   ├── plot_curves.py            # 训练曲线绘图
│   ├── configs/
│   │   ├── base.yaml             # 基础超参
│   │   └── ablations/            # A_baseline, B_contrastive, C_full, D_with_nll
│   ├── model/
│   │   ├── projector.py          # MLP 4096→2048→256
│   │   ├── classifier.py         # 256→8 sigmoid
│   │   ├── dann.py               # GRL + phylum classifier
│   │   └── full_model.py         # 组装
│   ├── losses/
│   │   ├── classification_loss.py
│   │   ├── contrastive_loss.py   # Multi-label SupCon
│   │   └── dann_loss.py
│   └── results/                  # Diagnostic ablation 结果
│       ├── ablation_table.csv
│       ├── A_baseline/           # 3 seeds × (best/last .pt + eval .json + train_log .csv/.png)
│       ├── B_contrastive/
│       ├── C_full/
│       └── figures/
│
├── data/
│   ├── mibig_gbk_3.1/           # 2502 MIBiG BGC raw GBK files
│   ├── ref_genomes_batch/        # 10 genera reference genomes (old, for diagnostics)
│   ├── diagnostic_day1/          # Day-1 诊断数据 (pkl)
│   ├── genomes/                  # 1996 GTDB genomes (Stage 1 output)
│   ├── antismash_results/        # antiSMASH output per genome (Stage 1)
│   ├── windows/                  # windows_metadata.csv (Stage 1)
│   ├── genome_metadata.csv       # 基因组元数据 (Stage 1)
│   └── embeddings/               # Evo2 embeddings (Stage 2, 生成中)
│
├── results/
│   ├── STAGE1_REPORT.md          # Stage 1 完成报告
│   └── diagnostic_day1/          # Day-1 诊断结果
│       ├── REPORT.md
│       ├── umap_*.png
│       ├── *.json
│       └── *.png
│
├── wiki/                         # GCD 阶段知识库
│   ├── index.md
│   ├── 项目总览_GCD.md
│   ├── CLAUDE.md                 # Wiki 维护规范
│   ├── concepts/                 # BGC, Evo2, GC偏差, 数据泄露
│   ├── data/                     # MIBiG, 参考基因组, antiSMASH
│   └── refs/                     # BGC-Prophet 论文
│
├── BGC-Prophet/                  # Reference implementation (cloned)
│
└── archive/                      # RF/监督学习阶段 (已归档)
    ├── scripts/                  # 23 个旧脚本
    ├── data/                     # 旧数据集 (20 GB+)
    ├── wiki/                     # 旧 wiki 页面
    ├── results/
    └── project_docs/             # 原始笔记
```

## Server-side (node7/node8)

```
/data/muskliu/np_master/
├── data/
│   ├── genomes/                  # 1996 genomes (rsync from local)
│   ├── windows/                  # windows_metadata.csv
│   └── embeddings/               # Stage 2 output (生成中)
│       ├── *.npy                 # per-genome embedding files
│       └── done/                 # completion flags
├── scripts/stage2/               # Embedding 提取脚本
├── bgc-gcd/                      # 训练代码 (rsync from local)
└── evo2/                         # Evo2 model weights
```
