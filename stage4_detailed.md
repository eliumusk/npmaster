# 阶段4：评估 — 详细执行手册

> 本阶段用训练好的模型（Config B, 3个seeds的best checkpoint）做全面评估。
> 核心目标：量化模型性能，和antiSMASH做详细对比，用MIBiG做独立验证。

---

## 概述

```
阶段3产出 → 训练好的模型（Config B, 3 seeds）
                ↓
        ┌───────┴────────┐──────────────┐──────────────┐
        ↓                ↓              ↓              ↓
    步骤4.1          步骤4.2        步骤4.3        步骤4.4
  窗口级评估      基因组级评估    antiSMASH对比   MIBiG外部验证
  (已在stage3     (把窗口预测     (模型 vs        (独立金标准
   基本完成)       合并成BGC      antiSMASH的     验证集)
                   区域预测)      一致与不一致)
```

---

## 步骤4.1 窗口级评估（已完成，补充分析）

### 输入
- Config B的3个best checkpoint
- test_id和test_ood的embedding + 标签

### 需要补充的分析

Stage 3已经报告了AUROC。还需要补充以下分析，用于论文和后续理解：

**4.1a 阈值选择**

对Config B seed 0的val集，画8个类型各自的precision-recall曲线，选取最优F1对应的阈值。

```
输出：optimal_thresholds.json
{
  "Polyketide": {"threshold": 0.xx, "precision": 0.xx, "recall": 0.xx, "f1": 0.xx},
  "NRP": {...},
  ...
  "is_negative": {"threshold": 0.xx, ...}
}
```

这组阈值在后续所有"硬预测"场景中使用（基因组级评估、antiSMASH对比）。

**4.1b 混淆分析**

用上面的阈值，对test_id生成8类的混淆矩阵（多标签版本：每个类型一个2×2矩阵）。

重点看：
- 哪些类型之间容易互相混淆？（如PKS被误判为NRP，这在hybrid BGC中可能是合理的）
- is_negative的假阳性（非BGC被误判为某种BGC类型）主要集中在哪些基因组/门中？

```
输出：
  results/confusion_matrices_test_id.png    # 8个2×2矩阵可视化
  results/false_positive_analysis.csv       # 假阳性窗口的详细信息（genome_id, phylum, 预测类型, 预测概率）
  results/false_negative_analysis.csv       # 假阴性窗口的详细信息
```

### 验收标准
- optimal_thresholds.json中所有阈值在0.1-0.9之间（极端值说明标签或模型有问题）
- 混淆矩阵已生成，可视觉确认合理

---

## 步骤4.2 基因组级评估

### 输入
- Config B模型
- test_id和test_ood所有窗口的模型预测（8维概率）
- windows_metadata.csv（窗口坐标）
- bgc_annotations.csv（antiSMASH的BGC坐标，作为ground truth）

### 目标

窗口级别的AUROC告诉你"单个窗口分对的概率"。但用户真正关心的是：给我一个基因组，你能不能正确标出哪些区域是BGC？

这需要把窗口级预测聚合成**基因组上的连续区域预测**。

### 执行

**4.2a 窗口预测 → 区域预测**

```python
# 对每个基因组：
# 1. 取该基因组所有窗口的模型预测
# 2. 对每个窗口，判断是否为BGC：
#    is_bgc_pred = max(pred_Polyketide, pred_NRP, ..., pred_Other) > threshold_bgc
#    其中 threshold_bgc 从4.1a得到（用is_negative的阈值：pred_is_negative < threshold → 预测为BGC）
# 3. 把连续的（或重叠的）BGC预测窗口合并成区域
#    - 窗口步长8192bp，两个相邻窗口重叠8192bp
#    - 如果相邻窗口都预测为BGC，合并为一个连续区域
#    - 合并后的区域记录：start_bp, end_bp, 每个类型的最大概率
# 4. 把过短的预测区域过滤掉（<3000bp的单窗口预测很可能是噪声）
```

```
输出：predicted_regions.csv
schema:
  genome_id, contig_id, pred_start, pred_end, pred_length,
  pred_Polyketide, pred_NRP, pred_Terpene, pred_RiPP, 
  pred_Saccharide, pred_Alkaloid, pred_Other,
  pred_primary_type,     # 概率最高的类型
  n_windows_merged       # 该区域包含多少个预测窗口
```

**4.2b 预测区域 vs antiSMASH标注的匹配**

```python
# 对每个基因组：
# 1. 取antiSMASH标注的BGC区域（from bgc_annotations.csv）
# 2. 取模型预测的BGC区域（from predicted_regions.csv）
# 3. 做区域匹配：
#    - 如果预测区域和antiSMASH区域重叠 ≥ 50%（以较短者为基准） → matched
#    - antiSMASH区域未被任何预测区域匹配 → false_negative（模型漏掉了）
#    - 预测区域未被任何antiSMASH区域匹配 → novel_candidate（模型发现了antiSMASH没标注的区域）

# 注意：novel_candidate不一定是"新BGC"，也可能是假阳性。
# 但它们是后续分析的重点对象。
```

### 输出

```
results/genome_level/
├── predicted_regions.csv                  # 所有预测区域
├── matching_results.csv                   # 每个antiSMASH BGC是否被匹配
├── novel_candidates.csv                   # 模型预测但antiSMASH未标注的区域
├── genome_level_metrics.json              # 汇总指标
└── per_genome_summary.csv                 # 每个基因组的检测统计
```

`genome_level_metrics.json`：
```json
{
  "test_id": {
    "n_antismash_bgcs": ...,
    "n_matched": ...,
    "n_missed": ...,
    "recall": ...,                     # matched / n_antismash_bgcs
    "n_predicted_regions": ...,
    "n_novel_candidates": ...,
    "precision": ...,                  # matched / n_predicted_regions
    "type_concordance": ...,           # 匹配的区域中，类型预测和antiSMASH一致的比例
    "per_type_recall": {
      "Polyketide": ..., "NRP": ..., ...
    }
  },
  "test_ood": { ... }
}
```

`per_genome_summary.csv`：
```csv
genome_id, phylum, n_antismash_bgcs, n_predicted_regions, 
n_matched, n_missed, n_novel_candidates, recall, precision
```

### 验收标准

1. **recall ≥ 0.70**（test_id上，模型能找到antiSMASH标注的BGC中的至少70%）
2. **precision ≥ 0.50**（test_id上，模型预测的区域中至少50%和antiSMASH吻合）
3. **type_concordance ≥ 0.60**（匹配区域中，类型预测正确率≥60%）
4. novel_candidates数量已统计（不设阈值，只需要记录）

### 反馈给我的信息

- genome_level_metrics.json完整内容
- novel_candidates的数量和分布：按phylum、按预测类型、按区域长度分布
- 最差的10个基因组（recall最低的）的详细情况：是什么物种、BGC是什么类型、为什么模型漏掉了
- 最好的10个基因组（recall=1.0且有novel_candidate的）的详细情况

---

## 步骤4.3 与antiSMASH的不一致性分析

### 输入
- 步骤4.2的matching_results.csv和novel_candidates.csv

### 目标

这是论文的核心分析。回答两个问题：
1. 模型漏掉了什么样的BGC？（false negatives → 模型的弱点）
2. 模型发现了antiSMASH没标注的什么区域？（novel candidates → 模型的潜在价值）

### 执行

**4.3a False Negative分析**

```python
# 对所有被模型漏掉的antiSMASH BGC：
# - 按类型分组统计：哪些类型最容易被漏掉？
# - 按BGC长度分组：短BGC是不是更容易被漏掉？（窗口可能只覆盖了BGC的一小部分）
# - 按phylum分组：哪些门的BGC更难检测？
# - 按is_truncated分组：截断的BGC是不是更容易被漏掉？
# - 按antiSMASH的similarity_to_known分组：越新颖的BGC是不是越容易被漏掉？
```

```
输出：
  results/false_negative_analysis/
  ├── fn_by_type.csv          # 各类型的miss rate
  ├── fn_by_length.csv        # BGC长度 vs miss rate
  ├── fn_by_phylum.csv        # 各门的miss rate
  ├── fn_by_truncation.csv    # truncated vs intact的miss rate
  └── fn_summary.json
```

**4.3b Novel Candidate分析**

```python
# 对所有novel_candidates（模型预测BGC但antiSMASH没标注的区域）：
# - 总数和占预测区域的比例
# - 按预测类型分布
# - 按phylum分布
# - 预测概率分布（高概率的候选更可信）
# - 区域长度分布
# - 这些区域在基因组上的位置特征：
#   - 离最近的antiSMASH BGC有多远？（如果很近，可能是边界扩展而非新发现）
#   - 是否在contig边缘？（可能是truncated BGC的遗漏部分）

# 按可信度排序：
# high_confidence: 预测概率 > 0.8 且 区域长度 > 10kb 且 离最近antiSMASH BGC > 20kb
# medium_confidence: 预测概率 > 0.5 且 区域长度 > 5kb
# low_confidence: 其余
```

```
输出：
  results/novel_candidates/
  ├── candidates_ranked.csv             # 按可信度排序的候选列表
  ├── candidates_by_confidence.json     # high/medium/low的数量
  ├── candidates_by_type.csv            # 按预测类型分布
  ├── candidates_by_phylum.csv          # 按phylum分布
  ├── distance_to_nearest_bgc.png       # 候选区域到最近antiSMASH BGC的距离分布
  └── candidate_summary.json
```

### 验收标准

- false negative分析覆盖了类型/长度/门/truncation四个维度
- novel candidate排序表已生成
- high_confidence候选数量已统计（不设最低数量要求）

### 反馈给我的信息

- 哪些类型的BGC最容易被漏掉（miss rate最高的类型）
- 短BGC（<10kb）的miss rate是否显著高于长BGC
- novel candidate总数和confidence分布
- high_confidence候选的top 20列表（genome_id, 坐标, 预测类型, 概率）
- 这些high_confidence候选是真的"新发现"还是大部分是antiSMASH的边界扩展？（看距离分布）

---

## 步骤4.4 MIBiG外部验证

### 输入
- Config B模型
- 诊断实验中已有的MIBiG 2502个BGC的embedding（data/diagnostic_day1/embeddings/）
- MIBiG的类型标注

### 目标

MIBiG是实验验证的金标准BGC集合。模型训练时**没有**使用MIBiG数据（只用了antiSMASH在基因组panel上的标注）。如果模型能正确分类MIBiG的BGC，说明：
1. 模型学到的不只是antiSMASH的规则，而是和真实BGC相关的特征
2. Evo2的embedding空间中确实编码了BGC的生物学信号

### 执行

```python
# 1. 加载MIBiG的2502个embedding（4096维，诊断实验已提取）
# 2. 通过Config B的投影头+分类头做预测（8维概率）
# 3. 和MIBiG的官方类型标注对比

# 注意：MIBiG的标注schema可能和我们的7类映射不完全一致
# 需要做MIBiG类型 → 7类的映射（用步骤1.2中相同的映射规则）
# MIBiG中有些BGC的类型标注可能和antiSMASH的7类映射不完全对应
# 映射不上的标记为"unmapped"，不参与评估
```

### 输出

```
results/mibig_validation/
├── mibig_predictions.csv               # 2502个BGC的8维预测概率 + MIBiG真实标签
├── mibig_metrics.json                  # 评估指标
└── mibig_confusion.png                 # 混淆矩阵可视化
```

`mibig_metrics.json`：
```json
{
  "n_total": 2502,
  "n_mapped": ...,              # 能映射到7类的BGC数
  "n_unmapped": ...,            # 映射不上的
  "bgc_detection_rate": ...,    # 被模型预测为"是BGC"（is_negative < threshold）的比例
  "per_type_auroc": {
    "Polyketide": ..., "NRP": ..., ...
  },
  "per_type_recall_at_05": {    # 以0.5为阈值的各类型recall
    "Polyketide": ..., ...
  },
  "mean_auroc": ...,
  "type_concordance": ...        # 预测的primary_type和MIBiG标注一致的比例
}
```

### 验收标准

1. **bgc_detection_rate ≥ 0.80**：至少80%的MIBiG BGC被模型识别为"是BGC"
2. **mean_auroc ≥ 0.70**：MIBiG上的类型分类表现
3. 如果bgc_detection_rate < 0.80，分析哪些类型/哪些长度的BGC被漏掉了

### 反馈给我的信息

- bgc_detection_rate（最重要的单一数字：多少MIBiG BGC被模型认为是BGC）
- per_type的AUROC和recall
- 被漏掉的MIBiG BGC的特征（类型、长度、来源物种）
- 如果某个类型在MIBiG上的表现远低于在test_id上的表现，说明训练数据（antiSMASH标注）和真实BGC（MIBiG）之间有系统性差异——这本身是一个有意义的发现

---

## 阶段4完成后的检查清单

- [ ] 阈值已从val集选定
- [ ] 基因组级recall ≥ 0.70（test_id）
- [ ] novel candidate列表已生成并按可信度排序
- [ ] false negative分析已完成（按类型/长度/门/truncation）
- [ ] MIBiG外部验证bgc_detection_rate ≥ 0.80
- [ ] 所有输出文件已生成

---

## 风险与应对

| 风险 | 应对 |
|---|---|
| 基因组级recall偏低（<0.60） | 检查阈值是否合适；检查是否是短BGC（<10kb）被系统性漏掉 |
| novel candidate数量为0 | 说明模型完全复现了antiSMASH，没有额外发现——这本身也是一个结论 |
| novel candidate数量极多（>预测区域的30%） | 说明模型假阳性过多，阈值需要调高 |
| MIBiG detection_rate < 0.60 | 严重问题——说明模型和真实BGC之间有系统性gap，需要排查是embedding问题还是MIBiG的类型映射问题 |
| MIBiG和test_id的AUROC差距过大（>0.15） | 可能是MIBiG embedding（诊断实验提取）和正式embedding的一致性问题——检查是否用了相同的提取配置 |
