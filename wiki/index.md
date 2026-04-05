# Wiki 目录

BGC 预测研究项目的知识库。入口：[[项目总览]] | 规范：[[CLAUDE]] | 时间线：[[log]]

## 🧭 Overview

- [[项目总览]] — 项目目标、pipeline、当前状态、核心结果

## 📚 Concepts（概念）

- [[concepts/BGC]] — 生物合成基因簇：定义、7大类、检测挑战
- [[concepts/Evo2]] — DNA基础模型：层选择、pooling、使用方式
- [[concepts/GC偏差]] — 负样本构造的首要陷阱，本项目的核心教训
- [[concepts/数据泄露]] — 三类泄露（BGC级/基因组级/窗口重叠）及检测
- [[concepts/Cluster-Bootstrap]] — BGC级置信区间的正确计算方式
- [[concepts/Leave-One-Class-Out]] — 诊断类型依赖vs通用特征的实验方法

## 💾 Data（数据）

- [[data/MIBiG数据库]] — 2502个实验验证BGC，正样本唯一来源
- [[data/参考基因组]] — 306+个NCBI基因组，负样本来源
- [[data/antiSMASH]] — BGC标注工具（只标cluster内CDS功能）
- [[data/数据集演进]] — v1→v2→v3→clean→clean_strep→expanded_strep 的迭代
- [[data/expanded-strep]] — **当前主数据集**（105 Strep genomes, 402 BGCs）
- [[data/多属数据集]] — Micro/Amyco/Pseudo 跨属测试集

## 🔬 Methods（方法）

- [[methods/pipeline]] — 端到端流程概览
- [[methods/embedding提取]] — Evo2推理代码、性能、环境配置
- [[methods/分类器]] — RF/LR/GBT/MLP 对比，为什么选RF
- [[methods/评估方法]] — AUROC/F1指标 + cluster bootstrap CI + sanity checks

## 🧪 Experiments（实验）

- [[experiments/首次验证实验]] — 2026-04-04 Evo2 vs k-mer, AUROC=0.940
- [[experiments/Strep扩充]] — 2026-04-05 负样本25→105基因组扩充
- [[experiments/多属扩展]] — 2026-04-05 跨属泛化, Strep→Pseudo=0.950
- [[experiments/窗口消融]] — 2026-04-05 w=16k > 8k > 4k
- [[experiments/leave-one-class-out]] — 2026-04-05 类型依赖诊断，PK依赖真实存在
- [[experiments/来源偏差诊断]] — 进行中 🚧 Phase 1完成，等GPU跑Phase 2
- [[experiments/全基因组扫描]] — 2026-04-05 ⚠️ 核心实验，暴露类型偏差，**当前方法未达成核心目标**

## 📄 References（外部参考）

- [[refs/BGC-Prophet论文]] — Lai Q. et al. 2025 NAR，本项目的主要对照方法

## 🗺️ 快速导航

**新人入门**：[[项目总览]] → [[concepts/BGC]] → [[concepts/Evo2]] → [[methods/pipeline]]

**跑实验**：[[data/expanded-strep]] → [[methods/pipeline]] → [[methods/评估方法]]

**理解数据**：[[data/数据集演进]] → [[concepts/GC偏差]] → [[concepts/数据泄露]]

**诊断问题**：[[experiments/leave-one-class-out]] + [[experiments/来源偏差诊断]]
