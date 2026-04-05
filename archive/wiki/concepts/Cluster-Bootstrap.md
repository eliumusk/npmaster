---
type: concept
updated: 2026-04-05
status: active
related: [[数据泄露]], [[../methods/评估方法]]
---

# Cluster Bootstrap（聚类级bootstrap）

## 为什么需要

常规bootstrap假设样本独立同分布（i.i.d.）。但本项目中：
- **窗口层面不独立**：同一BGC的多个窗口高度相关（序列重叠50%）
- **基因组层面不独立**：同一基因组的多个负样本窗口共享基因组特征

窗口级bootstrap会严重**低估**95% CI，产生过度自信的结论。

## 正确做法

Bootstrap的单位不是窗口，而是**BGC或基因组**：

```python
def cluster_bootstrap(y_true, y_prob, sources, n_boot=5000):
    source_to_idx = defaultdict(list)
    for i, src in enumerate(sources):
        source_to_idx[src].append(i)
    source_keys = list(source_to_idx.keys())
    n_sources = len(source_keys)
    aurocs = []
    for _ in range(n_boot):
        # 重采样BGC/基因组，而不是窗口
        boot_keys = np.random.choice(source_keys, n_sources, replace=True)
        boot_idx = []
        for k in boot_keys:
            boot_idx.extend(source_to_idx[k])
        yt = y_true[np.array(boot_idx)]
        yp = y_prob[np.array(boot_idx)]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, yp))
    return np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)
```

## 实际影响

**例子**：Pseudomonas跨属测试

| Bootstrap方法 | 95% CI |
|---|---|
| 窗口级 | [0.815, 0.952] |
| **BGC级 (cluster)** | **[0.788, 0.998]** |

窗口级CI宽度只有0.137，BGC级CI宽度0.210——因为Pseudomonas独立BGC只有8个，真实不确定度远高于窗口数量暗示的。

## 何时用哪种

| 数据特点 | Bootstrap单位 |
|---|---|
| 样本独立 | 窗口级 |
| 多窗口来自同BGC | BGC级 |
| 多窗口来自同基因组 | 基因组级 |
| 正负样本都有cluster结构 | 分开bootstrap再汇总 |

## 本项目的使用

所有实验的CI都用BGC级或基因组级cluster bootstrap。代码模板见 `scripts/16_expanded_eval.py`。

## 教训

**CI单位错误是"沉默杀手"**：数字看起来合理，但实际置信区间被低估2-10倍，可能导致错误的统计显著性判断。

详见 [[数据泄露]] 和 [[../experiments/多属扩展]] 的修正历程。
