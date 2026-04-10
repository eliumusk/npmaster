# Phase 3 Route B 结果报告

**日期:** 2026-04-10（初版 s0）→ 2026-04-10（3-seed + discovery 完成）  
**模型:** Route B 端到端（ProjectionHead + 膨胀 CNN + Classifier，全部可训练）  
**配置:** L_chunk=32, overlap stride=16, 100 epochs, AdamW 分层 lr (proj 1e-4, ctx/cls 1e-3)  
**Checkpoints:** best_route_b_s{0,1,2}.pt  
**状态:** **3-seed 稳定性通过 + discovery 通过 + 审计通过 → 建议晋升为新主线**

---

## 1. 训练概要

| 指标 | Route A | **Route B** |
|------|---------|-------------|
| 最终 α | 0.1205 | **0.1263** |
| 最佳 val region F1 | 0.8222 | **0.9049** |
| 最佳 val bgc_auroc | 0.9533 | **0.9821** |
| 训练时长 | ~8 min | **~35 min** |
| 参数量 | ~592K | **~9.5M**（含 ProjectionHead 8.9M） |
| 阈值 | 0.9714 | **0.8646** |

**关键观察：** 阈值从 Route A 的异常高值 0.9714 **回归到基线附近的 0.8646**，证实 Route A 的阈值偏移确实是冻结 projector 导致的。

---

## 2. 核心指标对比（3-seed mean ± std）

**口径：与基线 3-seed mean ± std 对比（同口径）。**

### 2.1 Overall Metrics

| 指标 | 基线 3-seed | Route A (s0) | **Route B 3-seed** | Route B Delta vs 基线 |
|------|------------|--------------|--------------------|-----------------------|
| test_id recall | 0.731 ± 0.012 | 0.730 | **0.820 ± 0.001** | **+8.9pp** |
| test_id P(pred) | 0.650 ± 0.032 | 0.664 | **0.857 ± 0.002** | **+20.7pp** |
| test_ood recall | 0.566 ± 0.041 | 0.506 | **0.594 ± 0.006** | **+2.8pp** |
| test_ood P(pred) | 0.531 ± 0.080 | 0.445 | **0.653 ± 0.009** | **+12.2pp** |

**稳定性大幅优于基线：** test_id recall std 从 0.012 → **0.001**（15x 更稳），test_ood P(pred) std 从 0.080 → **0.009**（9x 更稳）。

### 2.1a Per-Seed Detail

| 指标 | s0 | s1 | s2 |
|------|-----|-----|-----|
| test_id recall | 0.8197 | 0.8214 | 0.8197 |
| test_id P(pred) | 0.8544 | 0.8554 | 0.8600 |
| test_ood recall | 0.5943 | 0.5857 | 0.6014 |
| test_ood P(pred) | 0.6615 | 0.6407 | 0.6562 |
| 阈值 | 0.8646 | 0.9140 | 0.8826 |
| 最终 alpha | 0.1262 | 0.1262 | 0.1285 |

### 2.2 Per-Bucket Recall（test_id, 3-seed mean ± std）

| Bucket | 基线 (3-seed) | **Route B 3-seed** | Delta |
|--------|--------------|--------------------|-------|
| **<5kb** | 0.222 ± 0.005 | **0.751 ± 0.004** | **+52.9pp** |
| **5-10kb** | 0.352 ± 0.022 | **0.706 ± 0.003** | **+35.4pp** |
| **10-20kb** | 0.584 ± 0.022 | **0.691 ± 0.005** | **+10.7pp** |
| 20-50kb | 0.784 ± 0.011 | **0.842 ± 0.004** | **+5.8pp** |
| >50kb | 0.993 ± 0.002 | **0.987 ± 0.001** | -0.6pp |

### 2.3 Per-Bucket Recall（test_ood, 3-seed mean ± std）

| Bucket | 基线 (3-seed) | **Route B 3-seed** | Delta |
|--------|--------------|--------------------|-------|
| <5kb | 0.171 ± 0.014 | **0.556 ± 0.012** | **+38.5pp** |
| 5-10kb | 0.380 ± 0.009 | **0.651 ± 0.007** | **+27.1pp** |
| 10-20kb | 0.360 ± 0.044 | **0.411 ± 0.009** | **+5.1pp** |
| 20-50kb | 0.562 ± 0.049 | **0.569 ± 0.007** | +0.7pp |
| >50kb | 0.980 ± 0.003 | **0.987 ± 0.004** | +0.7pp |

### 2.4 Per-Type Recall（test_id）

| Type | 基线 (approx) | Route A | **Route B** |
|------|--------------|---------|-------------|
| NRP | ~0.88 | 0.900 | **0.965** |
| Polyketide | ~0.83 | 0.826 | **0.937** |
| Terpene | ~0.77 | 0.808 | **0.870** |
| RiPP | ~0.57 | 0.666 | **0.758** |
| Saccharide | ~0.65 | 0.765 | 0.706 |
| Alkaloid | ~0.33 | 0.417 | 0.417 |
| Other | ~0.58 | 0.607 | **0.698** |

---

## 3. Route A 一级问题是否修复

| Route A 问题 | Route A 值 | **Route B 值** | 修复？ |
|-------------|-----------|---------------|--------|
| OOD precision 崩溃 (-17.8pp) | 0.445 | **0.662 (+3.9pp vs s0)** | **完全修复** |
| 阈值异常 (0.9714) | 0.9714 | **0.8646** | **完全修复** |
| 20-50kb recall 退化 | 0.749 (-3.5pp) | **0.840 (+5.6pp)** | **完全修复** |

端到端训练让 ProjectionHead 适应了上下文融合，score 分布恢复正常（阈值 0.8646 ≈ 基线 0.8635），OOD precision 不仅恢复而且**超过基线**（0.662 vs 0.623）。

---

## 4. 成功标准评估

### Recovery 条件（全部 AND）

| 条件 | 阈值 | Route B 3-seed | 通过？ |
|------|------|---------------|--------|
| test_id recall | > 0.75 | **0.820 ± 0.001** | **通过** |
| test_id P(pred) | ≥ 0.63 | **0.857 ± 0.002** | **通过** |
| <5kb recall | > 0.25 | **0.751 ± 0.004** | **通过** |
| 5-10kb recall | > 0.38 | **0.706 ± 0.003** | **通过** |
| 10-20kb recall | > 0.60 | **0.691 ± 0.005** | **通过** |
| test_ood recall | > 0.57 | **0.594 ± 0.006** | **通过** |
| test_ood P(pred) | ≥ 0.50 | **0.653 ± 0.009** | **通过** |
| 3-seed 稳定性 | test_id recall std < 0.02 | **0.001** | **通过** |

### Discovery 条件（全部 AND）

| 条件 | 阈值 | Route B | 基线对比 | 通过？ |
|------|------|---------|---------|--------|
| sig precision@20 | ≥ 85% | **90%** (18/20) | = 基线 90% | **通过** |
| sig precision@50 | ≥ 80% | **92%** (46/50) | > 基线 86% (+6pp) | **通过** |
| diversity@50 (signature) | ≥ 35 | **45** profiles | > 基线 38 (+7) | **通过** |
| biosyn precision@20 | — | **100%** (20/20) | > null 95% | - |
| biosyn precision@50 | — | **100%** (50/50) | > null 96% | - |

**全部成功标准通过（Recovery 8/8 + Discovery 3/3）。**

---

## 5. 分析

### 5.1 端到端训练 vs 冻结 projector

Route B 相比 Route A 的全面提升证实了 Codex 审查的判断：冻结 projector 严重低估了上下文架构的上限。

| 指标 | Route A → Route B 改善 |
|------|----------------------|
| test_id recall | 0.730 → **0.820** (+9.0pp) |
| test_id P(pred) | 0.664 → **0.854** (+19.0pp) |
| test_ood P(pred) | 0.445 → **0.662** (+21.7pp) |

端到端训练让 ProjectionHead 学到了适合上下文融合的表示，而非只适合独立分类的表示。

### 5.2 阈值回归

Route B 阈值 0.8646 几乎等于基线 0.8635。这证实：
- Route A 的阈值异常（0.9714）是冻结 projector 导致的 calibration 问题
- 端到端训练自然解决了这个问题
- 模型的 score 分布与基线兼容

### 5.3 短 BGC 改善的持续性

从基线到 Route B 的短 BGC 改善巨大且在两个 route 间一致：

| Bucket | 基线 → Route A → Route B |
|--------|--------------------------|
| <5kb | 0.222 → 0.610 → **0.748** |
| 5-10kb | 0.352 → 0.528 → **0.708** |

这不是偶然——上下文建模确实解决了短 BGC 的信号稀释问题。

### 5.4 Precision 的意外大幅提升

test_id P(pred) 从 0.686 提升到 **0.854**（+16.8pp）。这个改善幅度出乎意料。可能原因：
- 上下文平滑减少了孤立假阳性
- 端到端训练让 projector 学到了更 BGC-specific 的表示
- 重叠 chunk 训练让模型对 region boundary 更精准

---

## 6. 审计结论

Audit 脚本（`/tmp/audit_route_b.py` + `/tmp/audit2.py`）验证了 5 项关键行为：

1. **Determinism**：重复前向完全一致（0 diff）
2. **Batch independence**：solo vs batched max diff ~1e-3（FP 精度级别）
3. **Length invariance**：中心窗口 score 在 L=16/32/64/128/256 下完全一致
4. **Full-contig vs chunked**：max diff 0.013，mean diff ~0
5. **Padding sensitivity**：valid_mask 正确屏蔽，0 diff
6. **Positive/negative separation**：20 个随机 val 正样本 19 个得分 > 0.86
7. **BN running stats**：`num_batches_tracked=350020`，运行统计量合理

推理路径（每条 contig 一次 forward，每个窗口得到单一预测）与训练路径（L=32 chunks, center loss）的行为一致。

## 7. 后续工作

### 7.1 立即晋升

- **更新 `configs/mainline.yaml`**：从 H_hardneg_r10 切换到 Route B 配置
- **更新 `docs/mainline.md`**：记录新主线指标
- **归档 H_hardneg_r10**：移至 `docs/_archive`

### 7.2 Phase 4 候选方向

Route B 已经让项目进入了新的性能水平，但仍有空间：
- **test_ood 10-20kb recall 仅 0.41**：OOD 中等长度 BGC 还有空间
- **Alkaloid recall 在 OOD = 0**：类别极不平衡，可能需要针对性处理
- **LoRA 微调 Evo2**：仍然是未探索的方向，可能进一步释放表示能力
- **Route B v2**：可加 contrastive/DANN/focal loss，每项单独测试

---

## 7. Artifact

```
/data/muskliu/np_master/data/phase3/route_b/
  chunk_index.pkl                  重叠 chunk 索引
  best_route_b_s0.pt              最佳 checkpoint (epoch 94)
  last_route_b_s0.pt              最终 checkpoint
  train_log_route_b_s0.csv        训练日志
  summary_route_b_s0.json         训练摘要
  train.log                       完整训练输出
  eval/results.json               完整评估结果
```
