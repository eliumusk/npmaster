---
type: overview
updated: 2026-04-05
status: active
stage: GCD
related: [[index]], [[refs/BGC-Prophet]], [[concepts/BGC]], [[concepts/Evo2]]
---

# 项目总览 — GCD 阶段

## 核心问题

> 用 Evo2 DNA 基础模型 + **Generalized Category Discovery(GCD)框架**,对已知 BGC 类型(MIBiG 7 类)保持检测能力,同时**从细菌基因组中发现新类型 BGC**。

**合作方**:导师 + 生物学院课题组。具备**高通量湿实验平台**,可批量验证候选。
**时间线**:2个月。
**硬件**:16 × A800-80GB(2 台节点)。

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

## Pipeline(规划中)

```
MIBiG 2502 BGCs (labeled, 7 classes)
  +                                  → Evo2 (frozen)
基因组 panel windows (unlabeled)        ↓
                                     4096-dim embedding
                                       ↓
                                     Projector MLP (学 phylum-invariant BGC-ness)
                                       ↓
                                     128-dim GCD space
                                       ↓
                                     SimGCD/ImbaGCD training
                                       │ (sup contrastive + self-sup contrastive + entropy reg)
                                       ↓
                                     Semi-supervised clustering
                                       ↓
                                     已知类型分类 + 新类型 cluster
                                       ↓
                                     novel candidates → 湿实验
```

## 下阶段任务

详细 plan: `refine-logs/EXPERIMENT_PLAN.md`(待写)

粗略 8 周:
1. W1: Day-1 诊断 ✓(已完成)
2. W2-3: 数据构建 + 全量 Evo2 embedding
3. W4-5: SimGCD 训练 + phylum-adversarial 调优
4. W6: novel cluster 发现 + 验证
5. W7: 与生物组对齐候选交付格式
6. W8: 实验收尾 + 结果整理

## 相关文档

- 策略决策过程:`refine-logs/` (FINAL_PROPOSAL, REFINEMENT_REPORT, REVIEW_SUMMARY)
- 诊断实验报告:`results/diagnostic_day1/REPORT.md`
- 诊断实验脚本:`scripts/diagnostics/`
- 历史工作归档:`archive/`
