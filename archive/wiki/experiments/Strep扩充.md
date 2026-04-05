---
type: experiment
updated: 2026-04-05
status: active
date: 2026-04-05
related: [[../data/expanded-strep]], [[../data/antiSMASH]], [[首次验证实验]]
---

# Streptomyces 负样本扩充：25 → 105 基因组

## 动机

原 clean_strep 数据集只有 25 个 Strep 基因组作负样本，test split 只有 3 个独立neg genome，存在 AUROC 虚高风险。

扩充目标：覆盖更多基因组，使test有足够独立neg genome做可靠推断。

## 数据准备

### antiSMASH 标注扩充
- 原有：25个Strep基因组（之前跑过）
- 新增：80个Strep基因组（Docker运行antiSMASH 8.0.4）
- 耗时：~90分钟（4并行）
- 结果：**105/105 成功**，零失败
- 总BGC region: **3465**

### 数据集重建
用 `scripts/14_build_expanded_strep.py`，同时构建三个窗口大小：w4096, w8192, w16384。

## 结果：新旧Strep对比（w8192, RF）

| 数据集 | AUROC | 95% CI | test neg genomes | F1 |
|---|---|---|---|---|
| 旧 (25 genome) | 0.940 | [0.919, 0.959] | 3 | 0.881 |
| **新 (105 genome)** | **0.926** | **[0.893, 0.952]** | **11** | 0.885 |

### 解读

AUROC 下降 0.014 是**好事，不是坏事**：
- 旧数据集test只有3个独立neg genome，模型可能记住了这3个特定基因组的特征
- 新数据集11个独立neg genome，测试更严格
- CI 重叠，差异不显著

**这个 0.926 是更可信的baseline**。

## 附带发现：跨属结果稳定

Strep训练→跨属测试（新旧模型对比）：

| 测试属 | 新模型 | 旧模型 | 差异 |
|---|---|---|---|
| Micromonospora | 0.915 | 0.908 | +0.007 |
| Amycolatopsis | 0.916 | 0.921 | -0.005 |
| Pseudomonas | 0.945 | 0.950 | -0.005 |

跨属结果几乎不受负样本扩充影响，**跨属泛化是真实的**。

## 负样本质量审计（尝试）

在2026-04-05尝试用antiSMASH注释审计负样本中的biosynthetic gene污染，**方法失败**：

- antiSMASH `--minimal` 模式只对 cluster 内 CDS 做功能注释（`gene_kind`, `gene_functions`）
- cluster 外 CDS 只有prodigal预测的基因边界，无功能标注
- 不能直接用antiSMASH数据判断cluster外是否有漏网的生物合成基因

正确方法：对cluster外CDS的蛋白质序列做 hmmscan + Pfam。**未完成**（本地无Pfam库，服务器GPU被占用）。

## 对项目主线的影响

1. 建立了当前主数据集 [[../data/expanded-strep]]
2. 提供了更可信的基线（0.926 CI比[0.893, 0.952]）
3. 下游所有实验基于此数据集

## 代码

- `scripts/run_antismash_strep_expand.sh` — 批量运行antiSMASH
- `scripts/14_build_expanded_strep.py` — 重建数据集
- `scripts/15_extract_expanded_embeddings.py` — 提取embedding

## 耗时

- antiSMASH (80个新基因组): ~90 min
- Embedding提取 (三种窗口大小, 两节点并行): ~4 hr
- 评估: ~5 min
