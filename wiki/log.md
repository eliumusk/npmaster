# Wiki 时间线

按时间倒序记录所有 ingest / 实验 / lint 事件。

格式：`## [YYYY-MM-DD] <type> | <summary>`

类型：`ingest`（新信息入库）/ `experiment`（实验完成）/ `lint`（健康检查）/ `refactor`（wiki结构调整）

---

## [2026-04-05] experiment | 全基因组扫描 - 核心实验暴露类型偏差

在3个未充分研究的Streptomyces基因组上做全基因组扫描，目标是找到antiSMASH漏掉的novel BGC。

**关键发现**：
- Evo2+RF对antiSMASH BGC的检测率只有 **30-35%**，GECCO达到 **85-87%**
- 类型偏差致命：terpene (0/16)、NI-siderophore (0/9)、ectoine (0/4)、RiPP-like (1/8) 完全或基本漏检
- 只对PKS/NRPS/butyrolactone等"大型模块化BGC"有效
- Evo2+GECCO共识novel候选只有 **3个**（跨3个基因组），GECCO独有novel候选 **48个**

**对项目核心claim的影响**：
- 项目核心问题是"比antiSMASH发现更多novel BGC"，当前方法未达成
- 之前的跨属AUROC 0.94不代表通用BGC检测能力（测试集也是PKS/NRPS为主）
- "Evo2学到通用BGC信号"是过度声明，需修正为"学到了大型模块化BGC的模式"

**新建**：[[experiments/全基因组扫描]]
**更新**：[[项目总览]] - 核心问题重新明确，当前状态评估诚实化
**更新**：[[index]] - 添加全基因组扫描链接，标记当前方法未达成核心目标

**待决策**：前进方向（类型平衡重训 / 微调Evo2 / Ensemble / 重新定位story 等）

---

## [2026-04-05] refactor | wiki 初始化

从 `project_docs/` 的零散笔记、`scripts/` 的代码逻辑、`gkaf305.md` 的参考论文，构建了结构化的 wiki。

**新建页面**（20个）：
- [[CLAUDE]], [[index]], [[log]], [[项目总览]]
- 概念：[[concepts/BGC]], [[concepts/Evo2]], [[concepts/GC偏差]], [[concepts/数据泄露]], [[concepts/Cluster-Bootstrap]], [[concepts/Leave-One-Class-Out]]
- 数据：[[data/MIBiG数据库]], [[data/参考基因组]], [[data/antiSMASH]], [[data/数据集演进]], [[data/expanded-strep]], [[data/多属数据集]]
- 方法：[[methods/pipeline]], [[methods/embedding提取]], [[methods/分类器]], [[methods/评估方法]]
- 实验：[[experiments/首次验证实验]], [[experiments/Strep扩充]], [[experiments/多属扩展]], [[experiments/窗口消融]], [[experiments/leave-one-class-out]], [[experiments/来源偏差诊断]]
- 参考：[[refs/BGC-Prophet论文]]

`project_docs/` 保留为原始笔记，未来新内容只进 wiki。

---

## [2026-04-05] experiment | 来源偏差诊断 Phase 1

MIBiG正样本 vs 负样本得分分布对比：
- MIBiG mean=0.901, median=0.950
- Negatives mean=0.109, median=0.071
- 两组完全分开，漏检率1.8%，误报率2.6%

**结论**：分布分离不能证明模型学到了BGC特征（也可能只是学到了来源差异）。等GPU空闲跑Phase 2（加入基因组原位BGC）才能鉴别。

in-situ BGC 数据已切好：18882个 w16384 窗口，105个Strep基因组。

见 [[experiments/来源偏差诊断]]。

---

## [2026-04-05] experiment | Polyketide 类型依赖对照

在 Leave-one-class-out 基础上，做随机去掉等量样本的对照：

- 随机去掉3196正样本（20次平均）: PK recall=0.791
- 按类型去掉所有Polyketide: PK recall=0.506

**类型依赖效应 -0.285 >> 数据量效应 -0.081**，类型依赖贡献了78%的recall下降。

见 [[experiments/leave-one-class-out]]。

---

## [2026-04-05] experiment | Leave-one-class-out 诊断

测试模型对7类BGC的泛化能力：
- Saccharide/Alkaloid 强泛化（recall=0.86/0.81）
- Polyketide/NRP 存在类型依赖（recall=0.51/0.58）
- RiPP 测试集太小无法判断

结论：模型是"通用BGC特征 + 类型特异特征"的混合体。

见 [[experiments/leave-one-class-out]]。

---

## [2026-04-05] experiment | Strep负样本扩充 25→105

antiSMASH标注扩充到105个Strep基因组，重建 expanded_strep 数据集：
- w8192 AUROC: 0.940 → 0.926（更可信的基线）
- 跨属结果稳定
- 窗口消融：16k > 8k > 4k

见 [[experiments/Strep扩充]], [[experiments/窗口消融]], [[experiments/多属扩展]]。

---

## [2026-04-04] experiment | 首次GPU验证

Evo2 7B 的 DNA embedding 在 clean_strep 数据集上：
- Evo2 + RF: AUROC=0.940
- k-mer + RF: AUROC=0.893
- **提升 +0.047**

方向验证成功。见 [[experiments/首次验证实验]]。

---

## [2026-04-04] refactor | 数据集修复 (clean_strep)

Codex review 抓住了 GC bias=0.097 + 基因组级泄露等致命问题。
修复方案：只用 Streptomyces 来源 + 按基因组分组split。

GC diff: 0.097 → 0.006 ✓

见 [[data/数据集演进]]。

---

## [2026-04-04] ingest | 项目启动

读完BGC-Prophet论文([[refs/BGC-Prophet论文]])，设计Evo2替代ESM-2的方案，下载MIBiG和参考基因组。
