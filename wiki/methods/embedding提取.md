---
type: method
updated: 2026-04-05
status: active
related: [[../concepts/Evo2]], [[pipeline]]
---

# Embedding 提取

## 概要

用 Evo2 7B 模型从DNA序列提取 4096 维embedding。

## 核心代码

```python
from evo2 import Evo2
import torch

model = Evo2("evo2_7b")

def extract_embedding(model, dna_seq, max_len=8192):
    # 1. 清理序列
    clean_seq = ''.join(c for c in dna_seq.upper() if c in 'ATCG')
    if len(clean_seq) < 10:
        return None
    
    # 2. 截断到max_len（取中间）
    if len(clean_seq) > max_len:
        start = (len(clean_seq) - max_len) // 2
        clean_seq = clean_seq[start:start + max_len]

    # 3. Tokenize
    input_ids = torch.tensor(
        model.tokenizer.tokenize(clean_seq), dtype=torch.int
    ).unsqueeze(0).to('cuda:0')

    # 4. 前向推理（bfloat16）
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, embeddings = model(
            input_ids, return_embeddings=True,
            layer_names=["blocks.20.mlp.l3"]
        )

    # 5. Mean pooling → 4096维
    emb = embeddings["blocks.20.mlp.l3"][0].float().mean(dim=0).cpu().numpy()
    return emb
```

详细脚本：`scripts/15_extract_expanded_embeddings.py`

## 层选择：blocks.20.mlp.l3

经测试，`blocks.20.mlp.l3` 是最佳embedding层：
- 信号强度 |mean|=1.826（比其他层更有信息量）
- 是 MLP 的第3个子层输出
- 详见 [[../concepts/Evo2#层选择]]

## Pooling：Mean pooling

```python
emb = token_level_output.mean(dim=0)  # (seq_len, 4096) → (4096,)
```

**当前实现只保存mean pooling后的结果**，per-token embedding被丢弃。

想做分段pooling或attention pooling必须重跑模型。

## 输出格式

```python
result = {
    'embeddings': np.array(embeddings),    # (N, 4096)
    'labels': np.array(labels),            # (N,)
    'categories': np.array(categories),    # (N, 7) 7维multi-label binary
    'ids': ids,                            # list of sample IDs
    'layer': 'blocks.20.mlp.l3',
    'embedding_dim': 4096,
    'num_samples': N,
    'num_errors': 0,
}
pickle.dump(result, open(f'{split}_emb.pkl', 'wb'))
```

## 性能

| 窗口大小 | 速度 | 每样本耗时 |
|---|---|---|
| 4096 bp | 2.2 samples/s | 0.45s |
| 8192 bp | 1.3 samples/s | 0.77s |
| 16384 bp | 0.6 samples/s | 1.67s |

单卡 A800-80GB，显存占用约 9GB。

## Embedding 文件大小

- `train_emb.pkl` (9201 samples × 4096维, float32): ~150 MB
- 不进git，只存服务器

## 错误处理

- 序列<10bp：跳过（errors++）
- 清理后的序列太短：跳过
- Tokenization/推理失败：跳过并打印

本项目embedding提取的错误率基本为0（Strep序列都很干净）。

## 环境变量

```bash
export HF_HUB_CACHE="/data/public_models/hub"     # 模型缓存
export HF_ENDPOINT="https://hf-mirror.com"        # HF镜像
```

## 分片运行（两节点并行）

```bash
# node7
python3 -u 15_extract_expanded_embeddings.py --shard 0

# node8 (并行)
python3 -u 15_extract_expanded_embeddings.py --shard 1
```

任务按数据大小平分，互不干扰。
