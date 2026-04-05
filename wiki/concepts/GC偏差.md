---
type: concept
updated: 2026-04-05
status: active
related: [[../data/数据集演进]], [[数据泄露]]
---

# GC偏差

## 定义

正负样本的**GC含量均值差异**过大，导致模型可以仅靠GC含量就区分正负，而不需要学习真正的BGC特征。

## 为什么重要

GC含量是最简单的DNA统计特征，k-mer/LR分类器就能轻易利用它。**GC偏差是负样本构造中最常见的陷阱**。

一旦存在GC偏差，所有AUROC数字都是"artifact"，模型的真实能力被严重高估。

## 本项目的GC偏差历史

| 数据集版本 | GC差异 | 问题 | AUROC表现 |
|---|---|---|---|
| v2 (E.coli+B.subtilis负样本) | 0.197 | 负样本物种差异过大 | RF=1.00 (作弊) |
| v3 (Strep+E.coli+B.subtilis) | 0.102 | 多属混合 | RF=0.99 |
| clean (9属306基因组) | 0.097 | GC还是偏高 | LR=0.85 (作弊) |
| **clean_strep (仅Strep)** | **0.006** | **✓ 无偏差** | LR=0.66 (真实) |
| expanded_strep (105 Strep genome) | 0.005 | ✓ | RF=0.926 |

## 诊断方法

```python
def gc_content(seq):
    clean = ''.join(c for c in seq if c in 'ATCG')
    return (clean.count('G') + clean.count('C')) / len(clean)

pos_gc = np.mean([gc_content(s['dna']) for s in pos_samples[:1000]])
neg_gc = np.mean([gc_content(s['dna']) for s in neg_samples[:1000]])
diff = abs(pos_gc - neg_gc)

if diff > 0.02:
    print("⚠️ GC偏差严重，需要修复负样本")
elif diff > 0.01:
    print("△ 轻微偏差，注意")
else:
    print("✓ 无显著GC偏差")
```

## 如何避免

**核心原则**：正负样本必须来自**相同GC含量的基因组背景**。

最安全的做法（本项目采用）：
- 只从Streptomyces来源的MIBiG BGC做正样本
- 只从Streptomyces参考基因组做负样本
- 确保物种/属层级一致

不要做的：
- 正样本用全MIBiG（多物种），负样本只用Strep（单属）
- 正样本用细菌BGC，负样本用真菌/人类基因组

## 教训

"每次改动后都要检查GC差异"，这是最简单的偏差检测指标。见 [[../experiments/首次验证实验]] 中的迭代历史。

## 相关

- [[数据泄露]]：GC偏差的"姐妹问题"
- [[../data/数据集演进]]：GC偏差驱动的数据集迭代
