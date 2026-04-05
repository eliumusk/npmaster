# Project Structure — GCD Phase

**Status**: Cleanup complete (2026-04-05). Ready to start GCD experiments.

## Active Directories

```
np-master/
├── scripts/
│   └── diagnostics/          # Day-1 诊断脚本(GCD 起点)
│       ├── 01_prep_mibig.py
│       ├── 02_sample_bg_windows.py
│       ├── 03_combine_and_extract.py
│       ├── 04_part_a_analysis.py
│       ├── 04b_actino_only.py
│       ├── 04c_actino_umap.py
│       ├── 04d_loto_knn.py
│       ├── 05_part_b_nll.py
│       └── 06_part_b_analysis.py
│
├── data/
│   ├── mibig_gbk_3.1/        # 2502 MIBiG BGC raw GBK files (for labeled set L)
│   ├── mibig_gbk_3.1.tar.gz
│   ├── ref_genomes_batch/    # 10 genera NCBI genomes (for unlabeled set U)
│   └── diagnostic_day1/      # Day-1 诊断的 MIBiG + bg pkl
│
├── results/
│   └── diagnostic_day1/      # Day-1 分析结果
│       ├── REPORT.md         # 主报告
│       ├── umap_by_*.png
│       ├── umap_actino_*.png
│       ├── distance_distributions.png
│       ├── nll_features_distribution.png
│       ├── part_a_summary.json
│       ├── part_b_summary.json
│       └── loto_results.json
│
├── wiki/                     # GCD 阶段知识库
│   ├── index.md
│   ├── 项目总览_GCD.md
│   ├── CLAUDE.md
│   ├── concepts/
│   │   ├── BGC.md
│   │   ├── Evo2.md
│   │   ├── GC偏差.md
│   │   └── 数据泄露.md
│   ├── data/
│   │   ├── MIBiG数据库.md
│   │   ├── 参考基因组.md
│   │   └── antiSMASH.md
│   └── refs/
│       ├── BGC-Prophet.md
│       └── BGC-Prophet总结.md
│
├── refine-logs/              # 策略文档 (待写 FINAL_PROPOSAL / EXPERIMENT_PLAN)
│
├── BGC-Prophet/              # reference implementation (cloned)
├── external/, samples/, project_docs/  # raw materials
│
└── archive/                  # 历史工作(RF/监督学习阶段)
```

## Archive Layout

```
archive/
├── scripts/                  # 23 个 RF 相关脚本 (01-21_*.py)
├── data/                     # 旧数据集 (20 GB+)
│   ├── expanded_strep/       # RF 主数据集
│   ├── multigenus/           # 跨属数据
│   ├── genome_scan/          # 全基因组扫描
│   ├── antismash_result/     # antiSMASH 标注
│   ├── clean_strep/, clean/  # 旧版本
│   ├── processed/, processed_v2/, processed_v3/  # 迭代版本
│   ├── figshare/             # 参考数据
│   └── final/, ref_genomes/
├── wiki/                     # 旧 wiki 页面
│   ├── 项目总览.md           # RF 阶段总览
│   ├── log.md
│   ├── experiments/          # 7 个 RF 实验记录
│   ├── methods/              # pipeline/分类器/评估方法
│   ├── concepts/             # Cluster-Bootstrap, Leave-One-Class-Out
│   └── data/                 # 数据集演进、多属、expanded-strep
└── results/
    └── figures/              # 旧实验图表
```

## What Changed

**Archived** (RF/监督学习阶段):
- 所有以 `01_` - `21_` 开头的脚本
- 所有旧数据集版本(expanded_strep, multigenus, genome_scan, 等)
- 所有 RF 相关实验记录
- RF 阶段的 pipeline / 分类器 / 评估方法文档

**Kept** (GCD 相关):
- MIBiG 原始 GBK 文件(GCD labeled set)
- ref_genomes_batch(GCD unlabeled set 来源)
- Day-1 诊断实验(脚本 + 数据 + 结果)
- 核心概念页(BGC, Evo2, GC偏差, 数据泄露)
- 参考论文(BGC-Prophet)

## Next Step

等用户确认 GCD 方向(基于 `results/diagnostic_day1/REPORT.md`),然后:
1. 写 `refine-logs/FINAL_PROPOSAL.md` — GCD 阶段方法方案
2. 写 `refine-logs/EXPERIMENT_PLAN.md` — 8 周 weekly deliverables
3. 启动 GCD 实验第一步:数据构建(genome panel + antiSMASH 标注)
