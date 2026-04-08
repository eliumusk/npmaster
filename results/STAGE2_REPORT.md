# Stage 2 Report: Evo2 Embedding提取

生成时间: 2026-04-08

---

## 概述

本阶段对Stage 1产出的948,362个DNA窗口(16384bp)通过frozen Evo2 7B提取了4096维embedding向量和8维NLL统计特征。提取本身成功完成,**但embedding质量诊断暴露了严重的来源偏差问题和偏弱的BGC区分信号**。

---

## 2.1-2.2 提取执行

| 指标 | 值 |
|---|---|
| 总窗口数 | 948,362 |
| 成功提取 | 948,362 (100%) |
| 失败基因组 | 0 |
| 提取层 | `blocks.20.mlp.l3` |
| embedding维度 | 4096 |
| NLL特征维度 | 8 |
| 推理模式 | BF16, model.eval(), torch.no_grad() |
| 单窗口速度 | 1.57s (稳态, A800 80GB) |
| GPU配置 | 16× A800 (node7 8卡 + node8 8卡) |
| 总GPU时间 | **413.5 GPU-hours** |
| 输出大小 | embeddings.npy 15.5GB + nll_features.npy 30MB |

---

## 2.3 合并与验证

### 数值健康检查

| 检查项 | 结果 | 判定 |
|---|---|---|
| 行数匹配 (948,362 = metadata行数) | ✓ | PASS |
| NaN | 无 | PASS |
| Inf | 无 | PASS |
| 全零行 | 0 | PASS |
| 维度均值范围 | [-0.795, 0.395] | PASS (阈值±10) |
| 维度标准差范围 | [0.0019, 0.050] | **NOTE** (min略低于0.01阈值) |
| 随机抽查非零 | 10/10非零 | PASS |

**关于std_in_range**: 部分维度的标准差低至0.0019,说明这些维度在所有窗口上近似常数。这在深层隐藏表示中是常见现象——不是所有维度都承载高方差信息。不影响下游训练(分类器会自动忽略低方差维度),但如果后续要做PCA/白化预处理,这些维度会被自然压缩。

### NLL特征检查

| 检查项 | 结果 |
|---|---|
| NaN/Inf | 无 ✓ |
| diff_var (positive窗口均值) | 0.978 |
| diff_var (negative窗口均值) | 0.946 |
| positive vs negative差异 | Δ=0.032 (3.4%) |

**解读**: diff_var在诊断实验Part B中effect size为0.52,当时positive和negative的NLL差异明显。现在正式数据上Δ仅3.4%,说明NLL特征的BGC区分能力在消除来源偏差后大幅减弱——这与embedding诊断的结果一致,不是独立的坏消息,而是同一个底层原因的不同表现。

---

## 2.4 Embedding质量诊断

### 2.4.1 UMAP可视化

采样5000 positive + 5000 negative窗口,30-neighbor, cosine metric。

**按phylum着色** (`umap_by_phylum.png`):
- UMAP被phylum完全主导。5个明显的cluster分别对应Actinomycetota(蓝)、Pseudomonadota(橙)、Bacillota(绿)、Cyanobacteriota(紫)、Bacteroidota(红)。
- 这和诊断实验一致:Evo2的layer 20 embedding主要编码物种/门级别的序列组成特征。

**按BGC类型着色** (`umap_by_type.png`):
- Polyketide(蓝)和NRP(橙)在Actinomycetota cluster的右上方有一定程度的聚集,但与negative(灰)混合严重。
- Terpene/RiPP/Other分散在各个phylum cluster中,无独立结构。
- BGC类型之间无清晰边界。

**按BGC vs 非BGC着色** (`umap_by_label.png`):
- positive(红)和negative(灰)在所有phylum cluster中高度重叠。
- **没有观察到BGC vs 非BGC的全局分离趋势**。
- 在Actinomycetota cluster中,positive点在某些子区域密度略高(右上),但分离远不充分。

### 2.4.2 kNN Enrichment

| 指标 | 值 | 阈值 | 判定 |
|---|---|---|---|
| 全局enrichment | **1.383** | ≥1.5 | **FAIL** |
| Actino控制phylum后 | **1.409** | ≥1.5 | **FAIL** |

**采样**: 3000 positive + 3000 negative, 5-NN, cosine距离。

**解读**:
- enrichment 1.38-1.41说明BGC窗口的5近邻中有约70%也是BGC(prior=50%时随机期望50%)。信号存在但弱。
- 控制phylum后(仅看Actinomycetota)enrichment从1.383微升到1.409,说明phylum不是唯一的混淆因素——即使在同一个门内,BGC信号也不够强。
- 对比诊断实验的enrichment 2.66x:正式数据消除了来源偏差后,信号预期降低,但降到1.4是偏低的。

### 2.4.3 距离分布

| 对比类型 | 余弦距离中位数 |
|---|---|
| 同类型 PKS-PKS | 0.0232 |
| 异类型 PKS-NRP | 0.0233 |
| BGC-背景 | 0.0230 |
| 背景-背景 | 0.0226 |

**解读**:
- 四种距离分布几乎完全重叠(中位数差异<0.001)。从距离直方图(`distance_distributions.png`)看,四条曲线无法区分。
- **frozen Evo2 layer 20的embedding空间中,BGC和非BGC窗口没有可被余弦距离度量的结构性差异**。
- 对比预期:如果BGC embedding有可区分信号,BGC-背景距离应明显大于同类BGC-BGC距离。实际上完全一样。

### 2.4.4 来源偏差检查 ⚠️ 最关键的验收项

| 指标 | 值 |
|---|---|
| 同基因组内 BGC vs 非BGC 余弦距离(中位数) | **0.0132** |
| 跨基因组 BGC vs 非BGC 余弦距离(中位数) | **0.0237** |
| 样本量 | intra=17,920对, inter=1,993对 |
| 判定 | **FAIL: intra < inter** |

**采样方法**:
- 选取同时拥有≥3个positive和≥3个negative窗口的基因组(1843个)
- intra: 从同一基因组中随机配对pos和neg窗口
- inter: 从不同基因组中随机配对pos(基因组A)和neg(基因组B)

**解读**:
- 同基因组内的BGC-vs-非BGC距离(0.0132)显著小于跨基因组的(0.0237)。
- 这意味着:**Evo2的embedding首先按"来自哪个基因组"聚类,其次才(微弱地)按"是否是BGC"区分**。
- 直觉上:同一个Streptomyces基因组里切出的两个窗口(一个落在BGC,一个落在背景),它们在embedding空间里的距离,小于两个不同Streptomyces基因组之间的随机窗口距离。
- **数据设计的成功面**: 我们的正负样本确实来自同一基因组,所以来源偏差不会导致"假阳性"分类——分类器不能通过区分"这个embedding来自哪个基因组"来作弊,因为同一基因组既有正样本又有负样本。
- **数据设计的局限面**: embedding本身编码的主要是基因组身份(物种、GC含量、k-mer组成),BGC vs 非BGC的信号被这个更强的信号淹没。分类器需要从这个被淹没的微弱信号中学习,这对模型容量和训练策略提出了更高要求。

---

## Stage 2 验收清单

| 检查项 | 结果 | 判定 |
|---|---|---|
| embeddings.npy行数 = metadata行数 | 948,362 = 948,362 | ✅ PASS |
| 无NaN/Inf | ✓ | ✅ PASS |
| kNN enrichment ≥ 1.5 (控制phylum后) | 1.409 | ❌ FAIL |
| 来源偏差检查通过 | intra < inter | ❌ FAIL |
| 失败基因组 < 1% | 0% | ✅ PASS |
| UMAP/距离分布图已生成 | ✓ | ✅ PASS |

**2/6 项未通过。**

---

## 与诊断实验的对比

| 指标 | 诊断实验 (Stage 0) | 正式数据 (Stage 2) | 变化 |
|---|---|---|---|
| 数据来源 | MIBiG BGC + 随机背景(不同基因组) | 同基因组切出的正负窗口 | 消除来源偏差 |
| kNN enrichment (Actino) | 2.66x | 1.41x | **↓47%** |
| NLL diff_var effect size | 0.52 | ~0.03 | **↓94%** |
| UMAP BGC分离 | 部分可见 | 几乎不可见 | 显著退化 |
| 来源偏差 | 未检查(已知存在) | 确认仍存在 | — |

**核心发现**: 诊断实验中观察到的"BGC信号"(enrichment 2.66x, NLL effect 0.52)的**相当大一部分来自来源偏差**(正样本来自MIBiG,负样本来自不同基因组)。消除来源偏差后,frozen Evo2 layer 20 embedding中的BGC vs 非BGC信号大幅减弱。

这不是数据处理的bug——这恰恰是来源偏差消除的预期结果。问题是:剩余信号(enrichment 1.4x)是否足以支撑一个有用的分类器。

---

## 可能的前进方向

### 方向A: 继续用当前embedding训练,看分类器能否放大微弱信号
- **理由**: enrichment 1.4x虽然弱,但不是1.0(完全没信号)。非线性分类器(MLP)可能能从4096维空间中找到线性kNN无法发现的非线性决策边界。
- **风险**: 信号太弱,分类器可能过拟合或无法泛化。
- **成本**: 低(直接进Stage 3,几小时训练)。

### 方向B: 换layer或pooling策略
- **理由**: layer 20是凭经验选的(约2/3深度)。浅层可能保留更多局部序列pattern(BGC的domain结构),深层可能过度smooth到全局genomic context。不同pooling(max pooling捕获极端特征,attention pooling选择性关注)也可能提取不同信号。
- **成本**: 需要重跑提取(但可以只跑子集验证,不需要全量)。
- **建议**: 先在500个基因组子集上试layer 10/15/25 + max pooling,看enrichment能否回到1.5+。

### 方向C: 微调Evo2
- **理由**: frozen embedding本质上是通用DNA语言模型的表示,没有被优化来区分BGC。微调(即使只微调top几层)可以让模型学习BGC-specific的特征。
- **成本**: 高(需要A800时间,训练代码开发)。
- **建议**: 作为备选,先验证A/B是否足够。

### 方向D: 放弃中间层embedding,直接用NLL/logits
- **理由**: NLL diff_var在诊断实验中有信号,虽然正式数据上减弱了。可以尝试更丰富的logits统计特征(不只8维)。
- **成本**: 低(已有logits数据)。
- **风险**: NLL信号同样大幅减弱了(effect size 0.52 → 0.03)。

### 建议优先级: A → B → C

先花最小成本(A: 直接训练)验证是否足够。如果不够,再投入B(换layer,中等成本)。C是最后手段。

---

## 输出文件

```
data/embeddings/
├── embeddings.npy                    # (948362, 4096), float32, 15.5GB
├── nll_features.npy                  # (948362, 8), float32, 30MB
├── extraction_summary.json           # 提取+验证汇总
├── per_genome/                       # 按基因组的备份 + checkpoint flags
│   ├── emb_{genome_id}.npy
│   ├── nll_{genome_id}.npy
│   └── {genome_id}_done.flag
└── diagnostics/
    ├── diagnostics_summary.json      # 诊断结果汇总
    ├── umap_by_type.png
    ├── umap_by_phylum.png
    ├── umap_by_label.png
    └── distance_distributions.png
```
