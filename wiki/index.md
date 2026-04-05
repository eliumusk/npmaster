# Wiki — GCD 阶段

BGC 预测研究项目知识库(GCD 范式)。

> **历史记录**:RF/监督学习阶段的实验、方法、数据文档已归档到 `archive/wiki/`。当前阶段从 GCD 范式重新开始。

## 📍 当前阶段:Generalized Category Discovery (GCD)

**核心问题**:用 Evo2 DNA 基础模型 + GCD 框架,对已知 BGC 类型(MIBiG 7 类)保持检测能力的同时,从基因组中发现新类型 BGC。

**策略与决策**:`refine-logs/FINAL_PROPOSAL.md`, `refine-logs/REFINEMENT_REPORT.md`
**诊断结果**:`results/diagnostic_day1/REPORT.md`

## 📚 概念(保留)

- [[concepts/BGC]] — 生物合成基因簇:定义、7大类、检测挑战
- [[concepts/Evo2]] — DNA 基础模型:架构、层选择、likelihood scoring
- [[concepts/GC偏差]] — 负样本构造的首要陷阱
- [[concepts/数据泄露]] — 三类泄露(BGC级/基因组级/窗口重叠)


## 📄 参考论文

- [[refs/BGC-Prophet]] — 原论文,architecture reference
- [[refs/BGC-Prophet总结]] — 要点摘要

## 🗂 归档内容(`archive/wiki/`)

RF 阶段的以下内容已归档,**不再作为当前指导**:
- `experiments/` — 首次验证、多属扩展、窗口消融、Strep扩充、全基因组扫描、leave-one-class-out、来源偏差诊断
- `methods/` — pipeline、embedding提取、分类器、评估方法
- `concepts/` — Cluster-Bootstrap、Leave-One-Class-Out
- `data/` — 数据集演进、多属数据集、expanded-strep
- 项目总览(RF 阶段版本)、log.md

如需查阅历史判断,去 `archive/wiki/` 翻。
