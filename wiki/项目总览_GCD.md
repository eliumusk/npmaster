---
type: overview
updated: 2026-04-07
status: active
stage: GCD
related: [[index]], [[refs/BGC-Prophet]], [[concepts/BGC]], [[concepts/Evo2]]
---

# 项目总览 — GCD 阶段

## 核心问题

> 用 Evo2 DNA 基础模型 + **Generalized Category Discovery(GCD)框架**,对已知 BGC 类型(MIBiG 7 类)保持检测能力,同时**从细菌基因组中发现新类型 BGC**。

**合作方**:导师 + 生物学院课题组。具备**高通量湿实验平台**,可批量验证候选。
**时间线**:2个月。
**硬件**:16 × A800-80GB(2 台节点: node7 + node8)。

## 合作方优先级(2026-04-05 确认)

1. **Priority 1(必达)**:对**已知 BGC** 的预测准确度对齐 antiSMASH
2. **Priority 2(锦上添花)**:发现未知 BGC 给湿实验验证

## 为什么选择 GCD 范式

前置的 RF/监督学习路径(archive/)遇到**结构性上限**:
- 训练标签来自 antiSMASH → 模型天花板 = antiSMASH
- 循环训练(antiSMASH label 定义正负) → 无法超越
- Frozen embedding + RF → 只能做"相似度检索",不能做"发现"

**GCD 同构于我们的任务**:
- Labeled set L = MIBiG 2502 BGC 的 7 个已知类型
- Unlabeled set U = 基因组窗口(混合已知 + 未知类型 + 非BGC)
- 目标 = 既分类已知,**又发现新 cluster**
- **不使用 antiSMASH 作训练标签** → 打破循环

## Day-1 诊断结果(2026-04-05)

详见 `results/diagnostic_day1/REPORT.md`。核心结论:

- ✅ **GCD 前提成立**:frozen Evo2 空间中存在跨类型的"BGC-ness"信号
- ⚠️ **需要对比学习**:raw embedding 被 phylum/物种信号主导
- 📊 kNN enrichment 2.66x(Actino + 长度控制后)
- 📊 LOTO retention 多数类型 > 0.8,RiPP 最弱(0.63)
- 📊 NLL-only AUROC=0.65(辅助通道,非主力)

**决策**: GO for GCD

## 当前进度 (2026-04-07)

### Stage 1: 数据基座 ✅ 已完成

详见 `results/STAGE1_REPORT.md`。

| 指标 | 数值 |
|------|------|
| 基因组 panel | **1,996** (GTDB v226, 30门) |
| antiSMASH BGC | **19,762** (31.7% truncated, 11.8% 多标签) |
| 窗口总数 | **948,362** (W=16384, S=8192, 50%重叠) |
| 划分 | train 1190 genomes / val 170 / test_id 340 / test_ood 300 |
| GCF 泄漏 | **0** (species-level proxy) |

BGC 类型分布: Terpene 5,296 > NRP 4,389 > Other 4,286 > RiPP 4,222 > PKS 4,122 >> Alkaloid 94 > Saccharide 84

### Stage 2: Evo2 Embedding 提取 🔄 进行中

- Node7: 8×A800 运行中, ~1.57s/window, 预计 04-08 上午完成
- Node8: 被占用, 需等空闲加入 (可将时间减半到 ~25h)
- 详见 `stage2_detailed.md`

### Stage 3: 训练 Pipeline ✅ 代码就绪 + Diagnostic Ablation 完成

代码在 `bgc-gcd/`, 用诊断数据 (6,202 样本) 验证了代码正确性。

**Ablation 结果 (diagnostic data, 3 seeds):**

| Config | Val mean AUROC | Val BGC AUROC | Test-ID mean AUROC | DANN acc |
|--------|---------------|---------------|-------------------|----------|
| A_baseline (CE only) | 0.821 ± 0.008 | 0.911 ± 0.005 | 0.832 ± 0.011 | ~50% (随机) |
| B_contrastive (+SupCon) | 0.839 ± 0.003 | 0.877 ± 0.058 | 0.803 ± 0.012 | ~55% |
| C_full (+SupCon+DANN) | 0.839 ± 0.002 | 0.871 ± 0.050 | 0.798 ± 0.008 | **~84%** |

**解读**:
- SupCon 提升了 val 指标，但 test 略降 → 诊断数据太小 (只有 ~600 BGC)，过拟合
- DANN 成功学会区分来源 (acc ~84%)，但对分类性能影响不明显 → 需等正式数据判断
- **这些数值不具参考价值**，仅验证代码运行正确。正式训练需等 Stage 2 的 948k embeddings

### Stage 4: Related Work 文献调研 ✅ 已完成

详见 `related_work_survey.md`。~30 篇论文，覆盖:
- BGC 检测工具 (antiSMASH, DeepBGC, GECCO, BGC-Prophet, BiGCARP 等)
- GCD 方法 (Vaze→SimGCD→PromptCAL) + 非CV应用 (AnnoGCD, LOOP, scGAD)
- DNA 基础模型 (DNABERT→NT→HyenaDNA→Evo→Evo2)

**三重 novelty 确认**: (1) 首次用 DNA FM 做 BGC 检测 (2) 首次将 GCD 应用于基因组学 (3) 16kb 窗口是 DNA FM 长 context 新应用

## Pipeline 架构

```
Stage 1 (✅ done)                  Stage 2 (🔄 running)
┌─────────────────────┐           ┌─────────────────────┐
│ 1996 genomes (GTDB)  │           │ Evo2 7B (frozen)    │
│ + antiSMASH → 19,762│           │ blocks.20.mlp.l3    │
│   BGCs, 948k windows │──────────▶│ mean pool → 4096d   │
│ + GCF-safe split     │           │ + NLL statistics    │
└─────────────────────┘           └─────────┬───────────┘
                                            │
                    Stage 3 (✅ code ready)  ▼
                    ┌─────────────────────────────┐
                    │ Projector: 4096→2048→256 (L2)│
                    │ Classifier: 256→8 (sigmoid)  │
                    │ DANN: 256→GRL→phylum (softmax)│
                    │ SupCon: multi-label positive  │
                    │ BalancedBatchSampler (50/50)  │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ 已知类型分类 + novel cluster │
                    │ → 候选 → 湿实验验证         │
                    └─────────────────────────────┘
```

## 时间线 (更新 2026-04-07)

| 周 | 计划 | 实际状态 |
|----|------|----------|
| W1 (04-05) | Day-1 诊断 | ✅ GCD 前提验证通过 |
| W1 (04-05~06) | Stage 1 数据基座 | ✅ 1996 genomes + 948k windows |
| W1 (04-05~06) | Stage 3 代码 + diagnostic ablation | ✅ A/B/C configs, 3 seeds |
| W1 (04-06~08) | Stage 2 Evo2 embedding 提取 | 🔄 Node7 运行中, ~04-08 完成 |
| W1 (04-07) | Related Work 文献调研 | ✅ ~30 篇论文整理完毕 |
| W2 (04-08~14) | 正式训练 + 调参 | ⏳ 等 Stage 2 完成 |
| W3-4 | SimGCD + phylum-adversarial 调优 | |
| W5 | Novel cluster 发现 + 分析 | |
| W6 | 与生物组对齐候选交付 | |
| W7-8 | 论文写作 + 实验收尾 | |

## 相关文档

- **执行手册**: `stage1_detailed.md`, `stage2_detailed.md`, `stage3_detailed.md`
- **Stage 1 报告**: `results/STAGE1_REPORT.md`
- **Stage 3 代码**: `bgc-gcd/` (README, configs, model, losses, evaluate)
- **文献调研**: `related_work_survey.md`
- **诊断实验**: `results/diagnostic_day1/REPORT.md`, `scripts/diagnostics/`
- **历史工作归档**: `archive/`
