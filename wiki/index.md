# Wiki — GCD 阶段

BGC 预测研究项目知识库(GCD 范式)。

> **历史记录**:RF/监督学习阶段的实验、方法、数据文档已归档到 `archive/wiki/`。当前阶段从 GCD 范式重新开始。

## 📍 当前状态 (2026-04-07)

- Stage 1 数据基座 ✅ | Stage 2 Embedding 🔄 | Stage 3 代码 ✅ | 文献调研 ✅
- 详见 [[项目总览_GCD]]

## 📋 执行手册 & 报告

- `stage1_detailed.md` — 数据基座构建 (基因组panel + antiSMASH + 窗口切片)
- `results/STAGE1_REPORT.md` — Stage 1 完成报告 (1996 genomes, 19,762 BGCs, 948k windows)
- `stage2_detailed.md` — Evo2 embedding 提取 (node7 运行中)
- `stage3_detailed.md` — 训练 pipeline 设计 (projector + classifier + DANN + SupCon)
- `bgc-gcd/README.md` — Stage 3 代码使用说明 + diagnostic ablation 结果
- `related_work_survey.md` — 文献调研 (~30篇, BGC工具/GCD/DNA-FM)

## 📚 概念

- [[concepts/BGC]] — 生物合成基因簇:定义、7大类、检测挑战
- [[concepts/Evo2]] — DNA 基础模型:架构、层选择、likelihood scoring
- [[concepts/GC偏差]] — 负样本构造的首要陷阱
- [[concepts/数据泄露]] — 三类泄露(BGC级/基因组级/窗口重叠)

## 📊 数据

- [[data/MIBiG数据库]] — MIBiG v3.1, 2502 BGCs, 7类
- [[data/参考基因组]] — GTDB v226 采样策略
- [[data/antiSMASH]] — antiSMASH 标注流程和类型映射

## 📄 参考论文

- [[refs/BGC-Prophet]] — 原论文, architecture reference
- [[refs/BGC-Prophet总结]] — 要点摘要
- `related_work_survey.md` — 完整文献调研 (BGC检测 + GCD + DNA-FM)

## 🗂 归档内容(`archive/wiki/`)

RF 阶段的以下内容已归档,**不再作为当前指导**:
- `experiments/` — 首次验证、多属扩展、窗口消融、Strep扩充、全基因组扫描、leave-one-class-out、来源偏差诊断
- `methods/` — pipeline、embedding提取、分类器、评估方法
- `concepts/` — Cluster-Bootstrap、Leave-One-Class-Out
- `data/` — 数据集演进、多属数据集、expanded-strep
- 项目总览(RF 阶段版本)、log.md

如需查阅历史判断,去 `archive/wiki/` 翻。
