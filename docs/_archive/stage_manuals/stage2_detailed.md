# 阶段2：Evo2 Embedding提取 — 详细执行手册

> 本阶段将阶段1产出的约95万个DNA窗口，通过frozen Evo2 7B提取为4096维embedding向量。
> 这是整个pipeline中GPU时间消耗最大的阶段，需要仔细规划。

---

## 概述

```
输入：阶段1产出的基因组fasta文件 + windows_metadata.csv
  ↓
对每个窗口：从基因组中切出16384bp DNA序列
  ↓
送入frozen Evo2 7B forward pass
  ↓
提取 blocks.20.mlp.l3 层的隐藏状态 (L, 4096)
  ↓
mean pooling → 4096维向量
  ↓
输出：所有窗口的embedding矩阵 + 可选的NLL统计特征
```

---

## 步骤2.1 提取前准备

### 输入

```
data/genomes/                       # 1996个基因组fasta文件
data/windows/windows_metadata.csv   # 948,362行，包含window坐标和标签
```

### 预处理：生成窗口序列索引

不需要预先把所有窗口序列存成文件（会占几十GB）。用一个索引文件记录每个窗口的"基因组ID + contig + 起止坐标"，运行时动态从fasta中读取。

```python
# 索引格式（直接用windows_metadata.csv中已有的列）：
# genome_id, contig_id, window_start, window_end
# 运行时：用pysam或BioPython的SeqIO按坐标提取子序列
```

### 执行前检查

1. 确认Evo2 7B模型可以加载（`from evo2 import Evo2; model = Evo2('evo2_7b')`）
2. 确认单个样本forward pass正常：
   ```python
   import torch
   from evo2 import Evo2
   model = Evo2('evo2_7b')
   test_seq = 'ACGT' * 4096  # 16384bp
   input_ids = torch.tensor(model.tokenizer.tokenize(test_seq), dtype=torch.int).unsqueeze(0).to('cuda:0')
   with torch.no_grad():
       outputs, _ = model(input_ids)
   # 确认outputs shape: (1, 16384, vocab_size)
   ```
3. 确认中间层hook可以正常提取embedding：
   ```python
   # 提取 blocks.20.mlp.l3 的输出
   # 参考Evo2 repo的embedding提取示例
   layer_name = 'blocks.20.mlp.l3'
   # hook注册后，forward pass时自动捕获该层输出
   # 输出shape应为 (1, 16384, 4096)
   ```
4. 确认单卡显存占用：16384 token输入，BF16推理，预计单卡约30-40GB
5. 确认所有基因组fasta可被正常索引读取

### 输出

```
data/stage2_preflight.json
{
  "evo2_version": "...",
  "single_sample_time_sec": ...,
  "single_sample_gpu_memory_gb": ...,
  "embedding_shape": [1, 16384, 4096],
  "mean_pooled_shape": [1, 4096],
  "total_windows": 948362,
  "estimated_total_hours": ...,
  "estimated_output_size_gb": ...
}
```

### 验收标准

- preflight检查全部通过
- 单样本embedding shape正确 (1, 4096)
- 显存占用在80GB以内（单卡）
- 估算总时间和输出文件大小在可接受范围内

---

## 步骤2.2 批量Embedding提取

### 执行策略

**总量估算**：
- 948,362个窗口
- 但其中ambiguous窗口（约6.3% ≈ 59,800个）可以跳过——它们不参与训练也不参与评估
- 实际需要提取的窗口：约888,500个
- 如果资源紧张，可以进一步只提取train+val+test_id+test_ood中标记为positive或negative的窗口

**但建议全量提取（含ambiguous）**，理由：
- ambiguous窗口在后续全基因组扫描时需要预测
- 提取一次存下来，后续不需要重新跑Evo2
- 额外成本约6%的GPU时间，值得

**并行策略**：
- 按基因组分片，每张GPU处理一批基因组的所有窗口
- 16张A800，将1996个基因组均分为16组，每组约125个基因组
- 每组内顺序处理窗口（同一基因组的窗口顺序读取，利用磁盘顺序IO）

**容错设计**：
- 以基因组为单位做checkpoint：每处理完一个基因组，保存该基因组所有窗口的embedding
- 如果中途crash，从上次完成的基因组继续，不需要重头跑
- 每张GPU独立写自己的输出文件，最后合并

### 执行代码框架

```python
# extract_embeddings.py
# 参数：
#   --gpu_id: 当前GPU编号 (0-15)
#   --n_gpus: 总GPU数 (16)
#   --genomes_dir: 基因组fasta目录
#   --metadata_csv: windows_metadata.csv路径
#   --output_dir: embedding输出目录
#   --layer_name: 'blocks.20.mlp.l3'
#   --batch_size: 每次送入模型的窗口数（受显存限制，可能只能=1）
#   --extract_nll: 是否同时提取NLL统计特征（默认true）

# 伪代码：
def main(gpu_id, n_gpus):
    model = load_evo2('evo2_7b', device=f'cuda:{gpu_id}')
    register_hook(model, layer_name)
    
    # 分配基因组
    all_genomes = sorted(metadata['genome_id'].unique())
    my_genomes = all_genomes[gpu_id::n_gpus]  # 交错分配，均衡负载
    
    for genome_id in my_genomes:
        checkpoint_path = f'{output_dir}/{genome_id}_done.flag'
        if os.path.exists(checkpoint_path):
            continue  # 已处理，跳过
        
        genome_fasta = load_fasta(f'{genomes_dir}/{genome_id}.fasta')
        genome_windows = metadata[metadata['genome_id'] == genome_id]
        
        embeddings = []
        nll_features = []
        
        for _, row in genome_windows.iterrows():
            seq = extract_subsequence(genome_fasta, row['contig_id'], 
                                      row['window_start'], row['window_end'])
            
            # tokenize
            input_ids = tokenize(seq).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs, _ = model(input_ids)
                
                # 从hook获取中间层输出
                hidden = get_hooked_output()  # (1, L, 4096)
                emb = hidden.mean(dim=1)       # (1, 4096) mean pooling
                embeddings.append(emb.cpu().numpy())
                
                # 可选：提取NLL统计特征
                if extract_nll:
                    logits = outputs[0]  # (L, vocab_size)
                    nll = compute_nll_features(logits, input_ids)
                    nll_features.append(nll)
        
        # 保存该基因组的所有窗口embedding
        np.save(f'{output_dir}/emb_{genome_id}.npy', 
                np.vstack(embeddings))  # (N_windows, 4096)
        
        if extract_nll:
            np.save(f'{output_dir}/nll_{genome_id}.npy',
                    np.vstack(nll_features))  # (N_windows, 8)
        
        # 写checkpoint flag
        touch(checkpoint_path)
        
        log(f'GPU {gpu_id}: {genome_id} done, {len(embeddings)} windows')
```

### NLL统计特征提取（与embedding同时进行）

```python
def compute_nll_features(logits, input_ids):
    """
    从Evo2的logits输出中计算8个NLL统计特征。
    诊断实验Part B验证过这些特征有moderate信号(AUROC=0.646)。
    
    logits: (L, vocab_size) — 每个位置对下一个token的预测
    input_ids: (1, L) — 实际的token序列
    
    返回: (8,) 向量
    """
    # 计算per-position negative log-likelihood
    log_probs = F.log_softmax(logits[:-1], dim=-1)  # (L-1, vocab_size)
    targets = input_ids[0, 1:]                        # (L-1,)
    per_pos_nll = -log_probs[range(len(targets)), targets]  # (L-1,)
    
    nll = per_pos_nll.cpu().numpy()
    
    features = np.array([
        nll.mean(),                              # nll_mean
        nll.std(),                               # nll_std
        np.median(nll),                          # nll_median
        np.percentile(nll, 75),                  # nll_q75
        np.diff(nll).var(),                      # diff_var（一阶差分方差，最强信号）
        np.corrcoef(nll[:-100], nll[100:])[0,1] if len(nll)>100 else 0,  # autocorr_100
        np.corrcoef(nll[:-1000], nll[1000:])[0,1] if len(nll)>1000 else 0,  # autocorr_1000
        entropy(np.histogram(nll, bins=50, density=True)[0] + 1e-10)  # nll_entropy
    ])
    
    return features
```

### 启动命令

```bash
# 16卡并行，每张卡一个进程
for gpu_id in $(seq 0 15); do
    CUDA_VISIBLE_DEVICES=$gpu_id python extract_embeddings.py \
        --gpu_id $gpu_id \
        --n_gpus 16 \
        --genomes_dir data/genomes/ \
        --metadata_csv data/windows/windows_metadata.csv \
        --output_dir data/embeddings/per_genome/ \
        --layer_name blocks.20.mlp.l3 \
        --extract_nll true \
        &
done
wait
echo "All GPUs finished"
```

### 预计耗时

- 单窗口forward pass：约1-2秒（16384 token，Evo2 7B，A800，BF16）
- 888,500个窗口 / 16卡：约55,500个窗口/卡
- 每卡约55,500 × 1.5秒 ≈ 23小时
- 加上IO和开销：**预计总耗时约1.5-2天**（16卡满载）

### 输出

```
data/embeddings/
├── per_genome/                          # 按基因组分文件存储
│   ├── emb_{genome_id_1}.npy           # (N_windows_1, 4096)
│   ├── emb_{genome_id_2}.npy
│   ├── nll_{genome_id_1}.npy           # (N_windows_1, 8)，可选
│   ├── nll_{genome_id_2}.npy
│   ├── {genome_id_1}_done.flag         # checkpoint标记
│   └── ...
├── extraction_log.csv                   # 提取日志
└── extraction_summary.json              # 汇总统计
```

`extraction_log.csv` schema：
```csv
genome_id,gpu_id,n_windows,time_sec,status(success/error),error_msg
```

`extraction_summary.json`：
```json
{
  "total_genomes_processed": ...,
  "total_windows_extracted": ...,
  "total_gpu_hours": ...,
  "failed_genomes": [...],
  "embedding_dim": 4096,
  "nll_dim": 8,
  "layer_name": "blocks.20.mlp.l3"
}
```

---

## 步骤2.3 合并与验证

### 执行

提取完成后，将所有per_genome文件合并为单个矩阵，并与windows_metadata.csv对齐。

```python
# merge_embeddings.py
# 
# 按windows_metadata.csv中的行顺序，依次读取每个基因组的embedding，
# 拼接成一个大矩阵。确保行顺序和metadata完全一致。

all_embeddings = []
all_nll = []

for genome_id in metadata['genome_id'].unique():  # 按metadata中出现的顺序
    genome_mask = metadata['genome_id'] == genome_id
    n_windows = genome_mask.sum()
    
    emb = np.load(f'per_genome/emb_{genome_id}.npy')
    assert emb.shape == (n_windows, 4096), f"Shape mismatch for {genome_id}"
    all_embeddings.append(emb)
    
    if os.path.exists(f'per_genome/nll_{genome_id}.npy'):
        nll = np.load(f'per_genome/nll_{genome_id}.npy')
        assert nll.shape == (n_windows, 8)
        all_nll.append(nll)

embeddings = np.vstack(all_embeddings)   # (948362, 4096)
np.save('embeddings.npy', embeddings)

if all_nll:
    nll_features = np.vstack(all_nll)    # (948362, 8)
    np.save('nll_features.npy', nll_features)
```

### 输出

```
data/embeddings/
├── embeddings.npy              # (948362, 4096), float32, 约14.6GB
├── nll_features.npy            # (948362, 8), float32, 约29MB
├── per_genome/                 # 保留，作为备份
└── extraction_summary.json
```

### 验收标准

1. **完整性**：
   - `embeddings.npy` 行数 = `windows_metadata.csv` 行数 = 948,362
   - 如果不匹配，找出哪些基因组提取失败，报告失败原因
   - 允许少量失败（<1%），但需要在metadata中标记这些窗口

2. **数值健康**：
   - 无NaN值：`assert not np.isnan(embeddings).any()`
   - 无Inf值：`assert not np.isinf(embeddings).any()`
   - 数值范围合理：每个维度的均值在±10以内，标准差在0.01-10之间
   - 随机抽查10个窗口，确认embedding不全为0

3. **一致性校验**：
   - 从阶段1的诊断实验中取几个相同序列（如果有重叠的基因组），验证新提取的embedding和诊断实验中提取的一致（余弦相似度>0.99）。如果找不到重叠序列，跳过此项。

4. **NLL特征检查**（如果提取了）：
   - 无NaN/Inf
   - diff_var列（第5列）的positive vs negative窗口应有可检测的差异（参考诊断实验Part B的effect size 0.52）

---

## 步骤2.4 Embedding质量诊断

### 执行

用阶段0诊断实验的同一套分析流程，在正式数据上重跑，确认embedding信号质量。

```python
# embedding_diagnostics.py
#
# 使用阶段1的标签信息，对正式embedding做以下分析：

# 1. UMAP可视化（采样，全量太大画不了）
#    随机采样5000个positive + 5000个negative窗口
#    画三张图：按BGC类型着色、按phylum着色、按序列长度着色

# 2. 门内kNN enrichment
#    只看Actinomycetota的窗口
#    随机采样3000个positive + 3000个negative
#    5近邻命中率 → enrichment

# 3. 距离分布
#    同类型BGC-BGC、不同类型BGC-BGC、BGC-背景、背景-背景
#    各采样10000对，画直方图

# 4. 来源偏差检查（关键！）
#    在同一个基因组内，计算BGC窗口和非BGC窗口之间的平均距离
#    vs 不同基因组之间的BGC-非BGC距离
#    如果同基因组内的距离 < 跨基因组的距离，说明来源偏差已消除
```

### 输出

```
data/embeddings/diagnostics/
├── umap_by_type.png
├── umap_by_phylum.png
├── umap_by_length.png
├── knn_enrichment.json           # {"enrichment": ..., "hit_rate": ..., "prior": ...}
├── distance_distributions.png
├── source_bias_check.json        # 同基因组vs跨基因组距离对比
└── diagnostics_summary.json
```

### 验收标准

1. **kNN enrichment ≥ 1.5**（控制phylum后）。如果低于诊断实验的2.66x，可能是因为正式数据消除了来源偏差后信号变弱了——这是预期中的，但enrichment不应低于1.5，否则说明embedding对BGC的区分度不够。

2. **来源偏差消除**：同基因组内BGC-vs-非BGC的距离应该 ≥ 跨基因组的BGC-vs-非BGC距离。如果相反，说明来源偏差仍然存在（不应该发生，因为正负样本来自同一基因组）。

3. **phylum不再完全主导UMAP**：诊断实验中UMAP按phylum完美分离。正式数据中phylum仍然会是强信号，但同一个phylum的cluster内部应该能看到BGC和非BGC的分离趋势（至少部分分离）。

### 反馈给我的信息

- `diagnostics_summary.json` 完整内容
- 三张UMAP图
- kNN enrichment数值（全局 + 控制phylum后）
- 来源偏差检查结果
- 与诊断实验结果的对比（enrichment是升了还是降了、UMAP结构是否有明显变化）
- 如果enrichment < 1.5，详细说明可能的原因

---

## 阶段2完成后的检查清单

- [ ] embeddings.npy 行数 = windows_metadata.csv 行数
- [ ] 无NaN/Inf
- [ ] kNN enrichment ≥ 1.5（控制phylum后）
- [ ] 来源偏差检查通过
- [ ] extraction_summary.json 中失败基因组 < 1%
- [ ] UMAP图和距离分布图已生成，可视觉确认合理

全部通过后，进入阶段3（正式训练）。

---

## 风险与应对

| 风险 | 应对 |
|---|---|
| Evo2模型加载失败或版本不兼容 | 确认evo2包版本、CUDA版本、flash-attn版本匹配 |
| 单卡OOM | 确认用BF16推理（不用FP32），必要时减小输入长度到8192 |
| 某些基因组的窗口提取失败 | 按基因组checkpoint，失败的跳过，最后统计失败率 |
| 提取速度远慢于预期 | 检查是否开了梯度计算（必须torch.no_grad），检查数据IO瓶颈 |
| embeddings.npy太大无法一次性加载 | 用numpy的mmap模式（np.load(..., mmap_mode='r')）或分批读取 |
| kNN enrichment远低于诊断实验 | 不一定是问题——来源偏差消除后信号降低是预期的。如果<1.5再排查 |

---

## 磁盘空间预估

| 文件 | 大小 |
|---|---|
| embeddings.npy (948k × 4096 × 4 bytes) | ~14.6 GB |
| nll_features.npy (948k × 8 × 4 bytes) | ~29 MB |
| per_genome备份（和embeddings.npy等大） | ~14.6 GB |
| 合计 | ~30 GB |

加上阶段1的18GB，总磁盘占用约50GB。
