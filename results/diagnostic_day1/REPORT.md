# Day-1 诊断实验报告

**日期**: 2026-04-05
**Part A 状态**: 完成 ✓ | **Part B 状态**: 完成 ✓ | **LOTO 补充**: 完成 ✓

---

## 数据组成

- **MIBiG 2502 BGCs**(变长,中心截断至≤16384bp,无padding)
- **背景 3700 个窗口**(16384bp 固定):
  - Strep 1500(Actinobacteria,复用 expanded_strep)
  - Pseudomonas 1000(Proteobacteria)
  - Bacillus 1000(Firmicutes)
  - Nostoc 200(Cyanobacteria)
- **总计 6202 样本** → Evo2 7B blocks.20.mlp.l3 mean-pool → **4096 维**
- **提取耗时**: 8 × A800 并行 ~20 分钟,零错误

---

## 核心发现

### ⚠️ 主导信号是**物种/门**,不是 BGC-ness(坏消息部分)

看 `umap_by_phylum.png`:UMAP 空间清晰分成 4 个门的 cluster(Actinobacteria / Proteobacteria / Firmicutes / Cyanobacteria),BGC(红点)**集中在 Actinobacteria cluster 附近**(因为 MIBiG 大多来自放线菌)。

**后果**:
- 全局距离分析被门间差异主导
- 简单 kNN 的 2.18x enrichment 被"门内 vs 门外"虚高
- 这回答了你问卷里那个诊断:"**分开了 → 是坏消息**"

### ✓ 但去掉门/长度 confound 后,**BGC 信号是真实的**(好消息部分)

控制变量:**只看 Actinobacteria** + **长度 ≥ 14kb**(和 bg 的 16kb 可比):

| 指标 | 值 | 判定 |
|---|---|---|
| kNN 5近邻命中率 | 0.87 | ✓ 强 |
| 先验 P(BGC) | 0.327 | — |
| **Enrichment** | **2.66x** | ✓ **比全局 2.18x 更强** |

这说明:当控制了 phylum + length 之后,BGC 和 bg 在 embedding 空间中是**可以分开的**。信号是真的,只是被 confound 遮住了。

### 类型结构:弱,但存在

在 Actino-only UMAP(`umap_actino_only.png`)里:
- **bg(灰色)形成底部紧密 cluster**
- **BGCs 形成上方的分散区域**
- 部分 BGC 类型有聚集趋势(Saccharide、Terpene 轻度成簇)
- 多数类型在 BGC 区域内**混在一起**

**后果**: GCD 做 type discovery 要费力 —— 类型结构需要对比学习**放大**,目前 raw embedding 中很弱。

---

## 距离分布的诡异(解释)

| 距离类型 | 均值 |
|---|---|
| bg-bg(全 panel) | 0.0146 |
| same-type BGC-BGC | 0.039 |
| BGC-bg | 0.038 |
| diff-type BGC-BGC | 0.048 |

**表面读**: bg-bg 最近 → bg 高度相似;BGC-bg 和 same-type 一样近 → BGC 没区分度。

**实际解读**:
- bg 在 embedding 空间中是**紧密的小 cluster**(Strep 之间 + 同 phylum 之间都很像)
- BGC 是**分散的大云团**(7 类分散)
- 所以 BGC 间的平均距离 > BGC 到 bg 的平均距离,**这不是缺陷,是空间几何**
- 但**局部**邻居关系仍然正确(kNN 高) —— 说明分类边界存在,只是不是简单的"到 centroid 距离"

---

## ⭐ Leave-One-Type-Out kNN(关键 GCD 前提验证)

**直接回答**: frozen Evo2 空间里,"未知类型"BGC 是否仍然靠近"已知类型"BGC?

对每个类型 T:
- 从"library"中移除所有 T 类 BGC(模拟 T 是未知类型)
- 对 T 类样本查询 top-5 最近邻 → 看 BGC 命中率
- Retention ratio = LOTO 命中率 / baseline 命中率(越接近1越好)

**控制环境**: Actinobacteria + 长度≥14kb(去除 confound)

| Type | N | Baseline | LOTO | Retention | Library Prior | Enrichment |
|---|---|---|---|---|---|---|
| Alkaloid | 9 | 0.87 | 0.87 | **1.00** | 0.33 | 2.6x |
| Saccharide | 30 | 0.92 | 0.82 | **0.89** | 0.32 | 2.6x |
| Terpene | 11 | 0.66 | 0.56 | **0.86** | 0.32 | 1.7x |
| Other | 74 | 0.79 | 0.75 | **0.95** | 0.30 | 2.5x |
| Polyketide | 287 | 0.92 | 0.74 | **0.81** | 0.23 | 3.2x |
| NRP | 136 | 0.84 | 0.61 | **0.72** | 0.28 | 2.2x |
| RiPP | 35 | 0.70 | 0.44 | **0.63** | 0.32 | 1.4x |

**解读**:
- **GCD 对大多数类型前景良好**: 6/7 类型的 retention > 0.70,即使移除同类,novel 样本仍被其他 BGC 包围(而非落入 bg)
- **Enrichment 全部 > 1.4x**: 每个类型在 LOTO 后仍然比随机采样找到更多 BGC 邻居
- **RiPP 是警示**: retention=0.63,RiPP 在 frozen embedding 中相对"孤立",GCD 发现 novel RiPP 类型会困难
- **NRP 意外偏弱**: 0.72,可能因为 NRP 内部子类差异大

**对 GCD 框架的关键结论**:
> **frozen Evo2 空间中存在"跨类型的 BGC-ness"信号**,GCD 的核心前提**成立**。但信号强度因类型而异,需要对比学习进一步放大。

---

## Part B: NLL Profiling(autoregressive likelihood 诊断)

**设置**: 3 个 Strep 基因组(S_aquilus + S_deccanensis + S_longhuiensis)共 3719 个 16384bp 滑窗(348 BGC + 3371 non-BGC,按 antiSMASH 真值)。每窗口算 Evo2 per-position log-likelihood,提取 8 个统计特征。

| 特征 | BGC 均值 | non-BGC 均值 | Effect size | p-value |
|---|---|---|---|---|
| **diff_var**(一阶差分方差) | 0.954 | 0.904 | **+0.52** | 2e-20 |
| nll_std | 0.768 | 0.755 | +0.37 | 8e-11 |
| nll_q75 | 0.891 | 0.816 | +0.25 | 1e-05 |
| nll_mean | 0.620 | 0.577 | +0.24 | 2e-05 |
| nll_median | 0.350 | 0.294 | +0.21 | 2e-04 |
| autocorr_100 | 0.079 | 0.087 | -0.18 | 2e-03 |
| autocorr_1000 | 0.009 | 0.013 | -0.13 | 2e-02 |

**解读**:
- BGC 区域的 per-position NLL **更跳跃**(diff_var 高)—— Evo2 在 BGC 内部某些位置"惊讶度"变化大
- BGC 区域 **autocorr 更低** —— likelihood profile 不那么"平滑"
- BGC 区域**平均 NLL 更高** —— 略微比 non-BGC 不可预测

**单独 NLL 特征分类器性能**: Logistic Regression 5-fold CV **AUROC = 0.646 ± 0.09**

**判定**: NLL 信号是 **moderate** —— 存在,但比 embedding 弱。可作**互补通道**,不是主力。

---

## 决策树定位

原四种结果:

| # | 条件 | 路径 |
|---|---|---|
| 1 | embedding 强 + NLL 强 | GCD + 联合 NLL |
| **2** | **embedding 强 + NLL 弱** | **纯 embedding 走 GCD** |
| 3 | embedding 弱 + NLL 强 | LoRA 微调 + NLL 辅助 |
| 4 | 两个都弱 | Evo2 不适合 |

**当前定位**: **结果 2**(embedding moderate-strong, NLL moderate)

理由:
- **embedding 信号确认有**(kNN enrichment 2.66x,控制后;LOTO 大多数类型 retention > 0.8)
- **embedding 需要对比学习重塑**(phylum 主导、类型结构弱)
- **不需要 LoRA 微调 Evo2 本体**(先试 frozen + projector-only contrastive)
- **NLL 信号存在但弱**(AUROC 0.65)—— 可作 GCD 的辅助特征通道,不是主力

---

## 对 GCD 方向的影响

**好消息**:
- frozen Evo2 的 embedding **有 BGC 信号**,GCD 有戏
- 不需要一上来就 LoRA 微调 Evo2,可以先 frozen + projector 试
- kNN 作为初步 sanity check 已 pass

**需要警惕的**:
- Phylum signal 强烈 → 对比学习时必须**跨 phylum mixing**,让 projector 学会抑制 phylum 信号
- 类型结构弱 → GCD 的"类型监督"部分可能需要 strong augmentation + focal loss
- bg 紧密 → 可能需要显式 hard-negative mining,让 BGC 边缘样本不被吸进 bg cluster

**GCD 配方应调整**:
1. Contrastive loss 中加入 **phylum-aware negative sampling**(同 phylum 的 BGC-vs-bg 对作为硬负样本)
2. 考虑加 **phylum adversarial loss**:让 projector 输出**尽量无法预测 phylum**,只保留 BGC 相关信号
3. Augmentation 应跨 phylum(reverse-complement + 小突变,保持"BGC-ness"但消磨 phylum artifact)

---

## 最终结论

**GCD 框架的核心前提成立**:
1. ✓ frozen Evo2 embedding 空间有 BGC-vs-bg 信号(kNN enrichment 2.66x, 控制后)
2. ✓ 跨类型的 "BGC-ness" 存在(LOTO retention 大多 > 0.8)
3. ✓ NLL 信号存在(AUROC 0.65)可作为辅助特征

**需要警惕**:
- Phylum 信号主导 raw embedding → 对比学习必须 suppress phylum signal
- 类型结构弱 → 7 类 cluster 需要对比学习强化
- RiPP 在 frozen 空间相对孤立(retention=0.63)→ GCD 发现 novel RiPP 会困难
- bg 是紧密 cluster,BGC 是发散云团 → global distance 欺骗性强,必须靠 local neighborhood 判断

---

## GCD 框架的配方调整

基于诊断,修正 SimGCD 默认配方:

1. **Phylum-adversarial loss**: projector 输出**尽量无法预测 phylum**,强迫表示空间脱敏于物种
2. **Cross-phylum contrastive pairs**: 正 pair 从不同 phylum 的 BGC 中采样,避免学到 phylum artifact
3. **Length-aware augmentation**: 训练时随机截断 BGC 到不同长度,去除长度 confound
4. **可选 NLL 辅助通道**: 把 8 维 NLL 统计特征和 4096 维 embedding 一起作为 projector 输入
5. **RiPP 特殊处理**: 可能需要 oversampling 或额外的 RiPP-specific augmentation

---

## 产出文件(results/diagnostic_day1/)

**Part A(embedding):**
- `umap_by_class.png` — 7类 + bg,全数据
- `umap_by_length.png` — 长度 confound 检查
- `umap_by_phylum.png` — **phylum 主导的铁证**
- `umap_actino_only.png` — 去除 phylum confound 后,**BGC vs bg 分开**
- `umap_actino_lengthmatched.png` — 再去除长度 confound
- `distance_distributions.png` — 4组距离直方图
- `part_a_summary.json` — 量化

**Part B(NLL):**
- `nll_features_distribution.png` — 8 个 NLL 特征的 BGC vs non-BGC 分布
- `part_b_summary.json`

**LOTO:**
- `loto_results.json` — 7 类 leave-one-type-out 命中率

**服务器原始数据(可复用)**:
- `/data/muskliu/np_master/data/diagnostic_day1/embeddings/` — 6202×4096 embedding 全部保留
- `/data/muskliu/np_master/data/diagnostic_day1/nll/` — 3719 窗口 8 维 NLL 特征

---

## 下一步建议

1. **决定是否走 GCD** — 基于上面结论(我的判断: **GO**)
2. 如 GO → 我写详细的 `EXPERIMENT_PLAN.md`:SimGCD 改造 + 2个月 timeline + weekly deliverables
3. 如需要 LoRA 微调 Evo2 → 先跑一个简短的 LoRA PoC 确认可行性
