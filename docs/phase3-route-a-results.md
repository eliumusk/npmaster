# Phase 3 Route A 结果报告

**日期:** 2026-04-10  
**模型:** Route A context model（冻结 ProjectionHead + 膨胀 CNN）  
**配置:** seed=0, K=11, 100 epochs, AdamW lr=1e-3, center-only BCE loss  
**Checkpoint:** best_route_a_s0.pt（epoch 84, val region F1 = 0.8222）

---

## 1. 训练概要

| 指标 | 值 |
|------|-----|
| 最终 α（sigmoid gate） | **0.1205** |
| α 轨迹 | 0.018 → 0.06 (ep10) → 0.10 (ep45) → 0.12 (ep84, 收敛) |
| 最佳 val region F1 | 0.8222 (epoch 84) |
| 最佳 val bgc_auroc | 0.9533 (epoch 50) |
| 训练时长 | ~8 分钟 (100 epochs × ~5s/epoch) |
| 参数量 | ~590K（ContextModule）+ 2K（Classifier）|

**关键观察：α 持续增长至 0.12 并收敛，证明上下文模块确实在学习有用信号。**

---

## 2. 核心指标对比

### 2.1 Overall Metrics

**口径说明：** 全部对比均使用基线 s0 单 seed（与 Route A seed=0 同口径）。3-seed 参考值仅供上下文理解，不用于 delta 计算。

| 指标 | 基线 s0 | **Route A** | Delta | 3-seed 参考 |
|------|---------|-------------|-------|-------------|
| test_id recall | 0.719 | **0.730** | **+1.1pp** | 0.731 ± 0.012 |
| test_id P(pred) | 0.686 | **0.664** | -2.2pp | 0.650 ± 0.032 |
| test_ood recall | 0.519 | **0.506** | -1.3pp | 0.566 ± 0.041 |
| test_ood P(pred) | 0.623 | **0.445** | **-17.8pp** | 0.531 ± 0.080 |
| 阈值 | 0.8635 | **0.9714** | +10.8pp | 0.8577-0.8670 |

### 2.2 Per-Bucket Recall（test_id）— **Phase 3 核心靶点**

基线 s0 无分段 bucket 数据，此处使用 3-seed mean 作为近似参考（bucket 级别 s0 单 seed 未单独存档）。

| Bucket | 基线 (3-seed mean) | **Route A** | Delta (近似) | 评估 |
|--------|-------------------|-------------|-------------|------|
| **<5kb** | 0.222 | **0.610** | **+38.8pp** | **巨大改善** |
| **5-10kb** | 0.352 | **0.528** | **+17.6pp** | **显著改善** |
| 10-20kb | 0.584 | 0.580 | -0.4pp | 持平 |
| 20-50kb | 0.784 | 0.749 | -3.5pp | 小幅退化 |
| >50kb | 0.993 | 0.986 | -0.7pp | 持平 |

### 2.3 Per-Bucket Recall（test_ood）

| Bucket | 基线 (3-seed mean) | **Route A** | Delta (近似) |
|--------|-------------------|-------------|-------------|
| <5kb | 0.171 | **0.487** | **+31.6pp** |
| 5-10kb | 0.380 | **0.609** | **+22.9pp** |
| 10-20kb | 0.360 | 0.329 | -3.1pp |
| 20-50kb | 0.562 | 0.466 | -9.6pp |
| >50kb | 0.980 | 0.969 | -1.1pp |

### 2.4 Per-Type Recall（test_id）

| Type | 基线 (approx) | **Route A** |
|------|--------------|-------------|
| NRP | ~0.88 | 0.900 |
| Polyketide | ~0.83 | 0.826 |
| Terpene | ~0.77 | 0.808 |
| RiPP | ~0.57 | 0.666 |
| Saccharide | ~0.65 | 0.765 |
| Alkaloid | ~0.33 | 0.417 |
| Other | ~0.58 | 0.607 |

---

## 3. 分析

### 3.1 核心发现：短 BGC recall 爆发性提升

Route A 的最显著结果是 **<5kb recall 从 22% 飙升至 61%**（+39pp），5-10kb 从 35% 到 53%（+18pp）。这直接验证了 Phase 3 的核心假设：**独立窗口分类是短 BGC 低召回的主因之一**，上下文模型通过看到邻近窗口的"neg→pos→neg"模式，能够更准确地识别这些被信号稀释的短 BGC。

在 OOD 上同样观察到短 BGC 的大幅改善（<5kb: +32pp, 5-10kb: +23pp）。

### 3.2 代价：中等长度 BGC 和 OOD precision 退化

这些收益伴随着明显代价：
- **20-50kb bucket 退化 3.5pp**（test_id）和 **9.6pp**（test_ood）
- **test_ood overall precision 暴跌 17.8pp**（0.623 → 0.445）
- 阈值大幅上升（0.8635 → 0.9714），说明模型的 score 分布发生了系统性偏移

### 3.3 阈值偏移分析

Route A 的阈值（0.9714）远高于基线（0.8635）。这可能源于：
1. 冻结 ProjectionHead 的 256-dim 表示是为独立分类优化的，加上 12% 的上下文扰动后，score 分布整体上移
2. center-only loss 只训练了中心窗口的判别能力，但推理时所有窗口都使用上下文增强的表示
3. 上下文对正样本的增益（短 BGC 的分数被拉高）大于对负样本的抑制

### 3.4 OOD 退化根因

OOD precision 大幅退化（-17.8pp）最可能的原因：
- 上下文模块在训练集 phyla（Actino/Pseudo/Bacillota）上学到了 contig 级别的模式
- OOD phyla（Cyanobacteriota）的 contig 模式不同，上下文信号变成噪声
- 基线通过 hard-neg mining 建立的 OOD 精度被上下文噪声部分抵消

### 3.5 α 学习行为

α 从 0.018 稳步增长到 0.12 并收敛。这说明：
- 上下文信号是真实的（不是噪声，否则 α 会回零）
- 但最优比例只有 ~12%——模型仍然主要依赖独立窗口信号
- Route B（端到端，projector 可适应上下文）有潜力让上下文贡献更大

---

## 4. Route A 成功标准评估

| 条件 | 阈值 | Route A 结果 | 通过？ |
|------|------|-------------|--------|
| short-bucket 改善 | <10kb 至少一个 bucket +3pp | **<5kb +38.8pp, 5-10kb +17.6pp** | **通过** |
| test_id 不退化 | recall ≥ 0.725 | **0.730** | **通过** |
| test_ood 不明显变差 | recall ≥ 0.54 | 0.506 | **未通过** (-1.3pp vs s0) |
| precision 安全 | test_id P(pred) ≥ 0.62 | **0.664** | **通过** |
| **OOD precision 安全** | 不崩溃 (下降 <10pp vs s0) | **0.445 (-17.8pp vs s0)** | **未通过（一级风险）** |
| discovery 不退化 | sig precision@20 ≥ 80% | 未运行 | 待定 |
| 单窗口 contig 退化检查 | 不劣于基线 | 未单独检查 | 待定 |

**Route A 未全面通过。** 两项未通过：
1. test_ood recall 0.506（门槛 ≥ 0.54），退化幅度 -1.3pp vs s0 尚在 3-seed 方差范围内
2. **test_ood P(pred) 0.445（-17.8pp vs s0）— 这是更严重的问题**，精度崩溃远超 recall 退化，说明上下文模型在 OOD 上产生了大量新增假阳性

---

## 5. 决策

### 按 Phase 3 文档决策矩阵（Section 7.6）

Route A 属于"**部分通过（有强正信号但未全满足）**"情形：
- 在冻结 projector 的条件下，上下文对 short-BGC recovery 有强正信号（<5kb: +39pp, 5-10kb: +18pp）
- test_id overall 持平（+1.1pp vs s0）
- **但 OOD precision 崩溃（-17.8pp vs s0）是一级风险**，不能被忽视

**决策：仍启动 Route B，正常优先级。**

理由：
1. 冻结 ProjectionHead 天然限制了上下文模型的适应能力（α 只到 0.12），阈值被迫上移至 0.97
2. 端到端训练可以让 ProjectionHead 适应上下文融合，预期缓解 score 分布偏移和阈值异常
3. Route B 使用 contig chunk + loss-level weighting（完整遍历真实分布），对 OOD 的处理可能更好
4. 短 BGC recall 的改善（22% → 61%）证明了上下文建模在冻结 projector 条件下对 short-BGC recovery 的价值

### Route B 必须解决的一级问题

1. **OOD precision 崩溃**（-17.8pp）：这是 Route A 最严重的退化。Route B 若不能将 OOD P(pred) 恢复到 ≥ 0.50，即使 short-BGC 改善再大也不可晋升。可能的缓解方向：端到端训练让 score 分布更合理、loss-level hard-neg weighting、OOD-aware regularization。
2. **阈值异常**（0.97 vs 基线 0.86）：端到端训练后检查阈值是否回归到合理范围。如果仍在 0.95+ 说明 score calibration 有系统性问题。
3. **20-50kb recall 退化**（-3.5pp ID, -9.6pp OOD）：监控中等长度 BGC 是否因上下文过度平滑而退化。

---

## 6. 技术备注

### 服务器端 Artifact

```
/data/muskliu/np_master/data/phase3/route_a/
  projections_256.npy              (1,064,568 × 256, 1.09 GB)
  contig_index.pkl                 (127,863 contigs)
  meta_arrays.pkl                  (labels, splits, metadata)
  best_route_a_s0.pt              (best by val region F1, epoch 84)
  best_auroc_route_a_s0.pt        (best by val bgc_auroc, epoch 50)
  last_route_a_s0.pt              (epoch 99)
  train_log_route_a_s0.csv        (full training log)
  summary_route_a_s0.json         (summary metrics)
  eval/context_eval_results.json  (full evaluation results)
```

### 代码文件

```
bgc-gcd/
  model/context_model.py           DilatedContextCNN + GatedResidual + ContextBGCModel
  data/context_dataset.py          ContextDataset (K-window neighborhoods)
  train_context.py                 Route A training loop
  evaluate_context.py              Window/region-level evaluation

scripts/phase3/
  extract_projections.py           256-dim extraction + contig index
  context_predict.py               Full evaluation pipeline
```
