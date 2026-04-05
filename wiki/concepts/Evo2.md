---
type: concept
updated: 2026-04-05
status: active
related: [[../项目总览]], [[../methods/embedding提取]]
---

# Evo2 DNA 基础模型

## 什么是 Evo2

Evo2 是 Arc Institute 发布的DNA基础模型（foundation model），在海量DNA序列上预训练，能生成per-nucleotide的上下文embedding。可以类比为"DNA版的BERT"。

## 模型规格

| 版本 | 参数量 | 上下文长度 | 硬件需求 |
|---|---|---|---|
| Evo2 7B | 7B | 最长1M bp | 单卡A800 (80GB) 够用 |
| Evo2 40B | 40B | 最长1M bp | 需要多卡推理 |

**本项目主要用 Evo2 7B**。

## 使用方式

### 推理（提取embedding）

```python
from evo2 import Evo2
model = Evo2("evo2_7b")

# DNA序列 → token ids
input_ids = model.tokenizer.tokenize(dna_seq)

# 前向推理，返回指定层的token-level embedding
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    _, embeddings = model(input_ids, return_embeddings=True,
                          layer_names=["blocks.20.mlp.l3"])

# shape: (1, seq_len, 4096)
```

### 层选择

经验证 `blocks.20.mlp.l3` 是最佳embedding层：

| 层 | |mean| | 状态 |
|---|---|---|
| blocks.0 | 0.002 | 太浅 |
| blocks.10 | 0.036 | 有信号 |
| **blocks.20** | **1.826** | **最佳** |
| blocks.28+ | ~0 | 可能精度问题 |
| unembed | 23.7 | logits层(512维) |

来源：project_docs/04_数据集问题与解决方案.md 的层选择测试。

### Pooling策略

Evo2输出 per-token 的embedding (seq_len × 4096)，需要pooling为固定维度：

- **Mean pooling**（当前方案）：对seq_len维度取平均 → 4096维向量
- 其他可尝试：max pooling、attention pooling、分段pooling

## 为什么用Evo2而非其他DNA模型

| 模型 | 参数量 | 特点 |
|---|---|---|
| DNABERT | 110M | 短序列(<512bp)，编码效率差 |
| Nucleotide Transformer | 50M-2.5B | 中等长度 |
| HyenaDNA | 6M-160M | 长序列但小 |
| **Evo2 7B** | **7B** | **长序列 + 大模型 + DNA特化** |

Evo2的优势是**同时支持长序列（100万bp）和大模型容量**。

## 在本项目的使用历史

| 日期 | 事件 |
|---|---|
| 2026-04-04 初期 | 通过NVIDIA NIM API用 Evo2 40B unembed层做可行性验证（AUROC=0.70 on 100样本） |
| 2026-04-04 后期 | 在GPU服务器上用 Evo2 7B blocks.20层，首次验证 AUROC=0.941 |
| 后续 | 用于所有实验的embedding提取 |

详见 [[../experiments/首次验证实验]]、[[../methods/embedding提取]]。

## 推理速度

- w4096: ~2.2 samples/s
- w8192: ~1.3 samples/s
- w16384: ~0.6 samples/s

（8×A800-80GB 分布式推理，每卡约9GB显存）

## 局限性

1. **推理慢**：比ESM-2慢得多（ESM-2 8M参数 vs Evo2 7B参数）
2. **显存需求高**：处理>10k序列需要做好batch管理
3. **per-token embedding不保存**：当前pipeline只保存mean pooling后的结果，想做分段pooling必须重跑
