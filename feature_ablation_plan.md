# 特征表示消融实验 — 详细执行手册

> 本实验回答一个核心问题：当前模型的性能瓶颈是否来自mean pooling丢失位置信息？
> 通过控制变量法，系统测试窗口大小、特征表示方式、层深度三个因子的影响。

---

## 实验动机

当前pipeline（16kb窗口 + mean pooling + layer 20）在test_id上达到BGC binary AUROC 0.884。
模型的预测分数分布在BGC和非BGC之间高度重叠，导致无法同时达到高recall和高precision。
假设：mean pooling丢失了位置信息，而BGC的核心特征（特定基因的排列模式）恰好是位置相关的。
本实验验证这个假设，并寻找最优的特征表示配置。

---

## 实验矩阵

### 因子一：窗口大小（4个水平）

| 水平 | 窗口大小 | 步长(50%重叠) | 设计理由 |
|---|---|---|---|
| W4k | 4,096 bp | 2,048 bp | 小于大多数BGC，测试纯局部信号 |
| W16k | 16,384 bp | 8,192 bp | 当前baseline |
| W32k | 32,768 bp | 16,384 bp | 接近BGC median长度(~30kb) |
| W64k | 65,536 bp | 32,768 bp | 大于多数BGC，包含完整BGC+侧翼 |

### 因子二：特征表示（4个水平）

| 水平 | 表示方式 | 输出shape | 下游模型 | 位置信息 |
|---|---|---|---|---|
| MeanPool | 所有位置取平均 | (1, 4096) | MLP | 无 |
| MaxPool | 每维取最大值 | (1, 4096) | MLP | 无 |
| CNN | 每隔k个位置降采样 | (64, 256) | 1D CNN | 局部邻居 |
| Trans | 每隔k个位置降采样 | (64, 256) | 2层Transformer | 全局位置 |

### 因子三：层深度（2个水平）

| 水平 | 层 | 设计理由 |
|---|---|---|
| L10 | blocks.10.mlp.l3 | 浅层，约1/3深度，保留局部pattern |
| L20 | blocks.20.mlp.l3 | 当前baseline，约2/3深度，全局上下文 |

### 总计：4 × 4 × 2 = 32个组合，每个3个seed = 96次训练

---

## 数据

### 使用500个基因组子集

从阶段1的1996个基因组中，按phylum分层采样500个：
- Actinomycetota: 150
- Pseudomonadota: 125
- Bacillota: 125
- Bacteroidota: 50 (test_ood)
- Cyanobacteriota: 25 (test_ood)
- 其他门: 25

**重要**：这500个基因组的train/val/test_id/test_ood划分必须和阶段1的全量划分一致——即如果某个基因组在全量数据中是test_id，在子集中也必须是test_id。直接从全量的windows_metadata.csv中过滤这500个基因组的行即可。

### 每个窗口大小需要重新切片

不同窗口大小的切片结果不同。对每个窗口大小W：

```python
# 切片逻辑（复用阶段1.3，只改参数）
# W = 窗口大小 (4096 / 16384 / 32768 / 65536)
# S = W // 2 （50%重叠）
# 正样本：窗口与BGC重叠 ≥ 50%
# 负样本：窗口与所有BGC边界距离 ≥ 10kb
# ambiguous：其余（不参与训练和评估）
```

输出：
```
data/ablation/windows_{W}/
├── windows_metadata.csv      # 该窗口大小的所有窗口标签
└── split_summary.json        # 各split的窗口数和正负比例
```

### 预期的窗口数（粗略估算，500个基因组）

| 窗口大小 | 总窗口数 | 正样本数 | 正样本比例 |
|---|---|---|---|
| 4k | ~500k | ~20k | ~4% |
| 16k | ~120k | ~15k | ~12% |
| 32k | ~60k | ~12k | ~20% |
| 64k | ~30k | ~9k | ~30% |

窗口越大，总数越少但正样本比例越高（因为大窗口更容易覆盖BGC）。

---

## 特征提取

### 需要跑Evo2 forward pass的组合

4个窗口大小 × 2个层 = 8种提取配置。
每种配置需要保存3种输出（mean pooled、max pooled、downsampled raw），CNN和Trans共用downsampled输入。

### 提取逻辑

```python
def extract_features(model, sequence, layer_name, window_size):
    """
    一次forward pass，同时生成3种表示。
    
    必须确保: model.eval() 和 torch.no_grad() 已开启, BF16推理。
    
    输入: DNA序列 (长度=window_size)
    输出: 
      mean_pooled: (4096,)
      max_pooled: (4096,)
      downsampled_raw: (64, 4096)
    """
    input_ids = tokenize(sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs, _ = model(input_ids)
        hidden = get_hooked_output(layer_name)  # (1, L, 4096)
        hidden = hidden.squeeze(0)               # (L, 4096)
    
    # Mean pooling
    mean_pooled = hidden.mean(dim=0).cpu().numpy()  # (4096,)
    
    # Max pooling
    max_pooled = hidden.max(dim=0).values.cpu().numpy()  # (4096,)
    
    # Downsampled sequence: 均匀取64个位置
    L = hidden.shape[0]
    stride = L // 64
    indices = torch.arange(0, L, stride)[:64]
    downsampled_raw = hidden[indices].cpu().numpy()  # (64, 4096)
    
    return mean_pooled, max_pooled, downsampled_raw
```

**优化**：同时注册layer 10和layer 20的hook，一次forward pass提取两层。这样4种窗口大小只需要跑4轮，而不是8轮。

### PCA降维（在合并阶段统一做）

```python
# 对每个 (窗口大小, 层) 配置:
# 1. 加载所有窗口的downsampled_raw (N, 64, 4096)
# 2. reshape为 (N*64, 4096)
# 3. 从train集采样50000行拟合PCA(n_components=256)
# 4. 对所有数据apply PCA
# 5. reshape回 (N, 64, 256)
# 6. 保存PCA矩阵（后续推理时需要）
# 7. 检查: explained_variance_ratio的累计值，前256维应覆盖>90%方差
```

### 存储格式

```
data/ablation/features/{W}_{layer}/
├── mean_pooled.npy           # (N, 4096), float32
├── max_pooled.npy            # (N, 4096), float32
├── downsampled_pca.npy       # (N, 64, 256), float32 — CNN和Trans共用
├── pca_model.pkl             # sklearn PCA对象
├── pca_explained_var.json    # PCA的方差解释比例
└── extraction_log.json
```

### GPU时间预估

| 窗口大小 | 窗口数(500基因组) | 每窗口时间 | 总时间(单卡) | 16卡并行 |
|---|---|---|---|---|
| 4k | ~500k | ~0.5s | ~70h | ~4.4h |
| 16k | ~120k | ~1.5s | ~50h | ~3.1h |
| 32k | ~60k | ~3s | ~50h | ~3.1h |
| 64k | ~30k | ~6s | ~50h | ~3.1h |

4轮总计约14小时（16卡并行）。加上IO和重启开销，预计**1天**。

### 启动命令

```bash
for window_size in 4096 16384 32768 65536; do
    for gpu_id in $(seq 0 15); do
        CUDA_VISIBLE_DEVICES=$gpu_id python extract_ablation_features.py \
            --gpu_id $gpu_id \
            --n_gpus 16 \
            --window_size $window_size \
            --layers "blocks.10.mlp.l3,blocks.20.mlp.l3" \
            --genomes_dir data/genomes/ \
            --genome_list data/ablation/subset_500_genomes.txt \
            --bgc_annotations data/bgc_annotations.csv \
            --output_dir data/ablation/features/ \
            &
    done
    wait
    echo "Window size $window_size done"
done
```

---

## 下游模型

### MLP（用于MeanPool和MaxPool）

和阶段3完全相同:
```
Linear(4096, 2048) → BN → ReLU → Dropout(0.3)
Linear(2048, 256) → L2归一化
Linear(256, 8) → Sigmoid
```
参数量: ~9.5M

### 1D CNN（用于Downsampled序列）

```
输入: (batch, 64, 256)
    ↓ 转置为 (batch, 256, 64)  # Conv1d期望 (batch, channels, length)

Conv1d(256, 128, kernel_size=7, padding=3) → BN → ReLU → Dropout(0.2)
Conv1d(128, 128, kernel_size=7, padding=3) → BN → ReLU → Dropout(0.2)
Conv1d(128, 64, kernel_size=5, padding=2) → BN → ReLU → Dropout(0.2)
    ↓
GlobalAveragePooling → (batch, 64)
Linear(64, 8) → Sigmoid
```
参数量: ~0.5M

感受野:
- 3层卷积后感受野 = 17个降采样点
- W=16k时: 17 × 256bp ≈ 4.4kb
- W=64k时: 17 × 1024bp ≈ 17.4kb

### 2层Transformer（用于Downsampled序列）

```
输入: (batch, 64, 256)
    ↓
CLS token拼接 → (batch, 65, 256)  # 头部加一个可学习的CLS token
PositionalEncoding(max_len=65, d_model=256)  # 可学习的位置编码
TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1) × 2
    ↓
取CLS token输出 → (batch, 256)
Linear(256, 8) → Sigmoid
```
参数量: ~1.3M

Transformer和CNN的关键区别: Transformer的self-attention让任意两个位置可以直接交互（全局感受野），不需要通过多层卷积传递。如果BGC的信号涉及窗口两端的长距离依赖（比如边界特征），Transformer更有优势。

### 关于参数量差异的说明

MLP ~9.5M, CNN ~0.5M, Transformer ~1.3M。差异来自输入维度不同（4096 vs 256），不是设计缺陷。在论文中说明即可，不需要强行对齐参数量。

---

## 训练设置

所有32个组合统一使用以下配置:

```yaml
optimizer: SGD
lr: 0.1
momentum: 0.9
weight_decay: 1e-5
scheduler: CosineAnnealing
n_epochs: 100
batch_size: 128

loss:
  classification: MultiLabelBCE (α=1.0, per-class pos_weight, clamp=20)
  contrastive: MultiLabelSupCon (β=0.35, τ=0.07)

sampler: BalancedBatchSampler (50/50 pos/neg)
seeds: [0, 1, 2]
```

### 数据增强（按表示方式不同）

| 表示方式 | 增强方法 | 理由 |
|---|---|---|
| MeanPool/MaxPool | Gaussian noise σ=0.05 + dimension dropout 10% | 和阶段3一致 |
| CNN/Trans | Gaussian noise σ=0.05（逐元素）+ 位置mask 10%（随机将~6个位置整行置零）| 模拟序列中部分区域缺失 |

### 对比损失的适配

MeanPool/MaxPool: 和阶段3完全一致，对两个增强view的embedding做对比。

CNN/Trans: 对比损失作用在CNN/Transformer的倒数第二层输出上（CNN的global average pooling之后、分类头之前的64维向量；Transformer的CLS token 256维输出）。不在原始(64, 256)输入上做对比——维度太高。

---

## 评估指标

每个组合在val/test_id/test_ood上报告:

```json
{
  "val_mAUROC": "...",
  "val_bgc_binary_AUROC": "...",
  "test_id_mAUROC": "...",
  "test_id_bgc_binary_AUROC": "...",
  "test_id_precision_at_recall_70": "...",
  "test_ood_mAUROC": "...",
  "test_ood_bgc_binary_AUROC": "...",
  "train_time_sec": "..."
}
```

**最关键的两个指标**:
1. `test_id_bgc_binary_AUROC` — 排序能力，当前baseline 0.884
2. `test_id_precision_at_recall_70` — 实用性指标，当前baseline 0.281

---

## 输出

### 结果表

```
results/ablation/ablation_full_table.csv

columns:
  window_size, pooling, layer, seed,
  val_mAUROC, val_bgc_AUROC,
  test_id_mAUROC, test_id_bgc_AUROC, test_id_prec_at_rec70,
  test_ood_mAUROC, test_ood_bgc_AUROC,
  train_time_sec
```

96行（32组合 × 3 seed）。

### 汇总分析

```
results/ablation/
├── ablation_full_table.csv           # 完整96行结果
├── summary_by_pooling.csv            # 4种pooling的平均指标（固定其他变量）
├── summary_by_window.csv             # 4种窗口的平均指标
├── summary_by_layer.csv              # 2种层的平均指标
├── best_config.json                  # 最优组合及其指标
├── interaction_pooling_window.png    # pooling × window交互效应图
├── interaction_pooling_layer.png     # pooling × layer交互效应图
└── main_effects_plot.png             # 三个因子各自的主效应图
```

### 分析要点

分析一（核心问题）: 位置信息有没有用？
→ 固定窗口大小和层，比较4种pooling。如果CNN或Trans > MeanPool超过2%，位置信息有价值。

分析二: 最优窗口大小是什么？
→ 固定pooling和层，比较4种窗口。期望看到倒U型曲线。

分析三: 浅层还是深层更好？
→ 固定窗口和pooling，比较layer 10 vs layer 20。

分析四（交互效应）: 位置信息的价值是否随窗口增大而增大？
→ 看pooling × window的交互图。直觉上大窗口中mean pooling损失更多信息，所以CNN/Trans的优势应在大窗口中更明显。

分析五: CNN vs Transformer哪个更好？
→ 如果CNN ≈ Trans，说明局部pattern就够了。如果Trans > CNN，说明全局位置关系重要。

---

## 验收标准

1. **32个组合全部跑完**，ablation_full_table.csv有96行
2. **位置信息的价值被量化**: CNN或Trans vs MeanPool的差值及统计显著性
3. **至少一个组合 test_id_bgc_AUROC > 0.92**（当前baseline 0.884）
4. **至少一个组合 precision@recall70 > 0.40**（当前baseline 0.281）
5. 如果所有组合提升 < 2%，说明位置信息不是瓶颈，需要重新诊断

## 反馈给我的信息

- ablation_full_table.csv 完整内容
- 最优组合是什么，相比baseline(16k+MeanPool+L20)提升了多少
- 三个因子各自的主效应（哪个因子影响最大）
- pooling × window的交互效应
- CNN vs Transformer的对比
- PCA的explained_variance_ratio（确认256维保留了足够信息）
- 如果CNN/Trans没有显著提升，分析可能原因
- 总GPU时间消耗

---

## 风险与应对

| 风险 | 应对 |
|---|---|
| 64k窗口OOM | 确认BF16下显存占用；如果OOM则降到48k |
| PCA降维丢失关键信息 | 检查explained_variance_ratio；如果<90%则提高到512维 |
| CNN/Trans过拟合 | 监控train vs val gap；加重dropout或减小模型 |
| 500基因组子集不代表全量 | 确认子集的phylum和BGC类型分布与全量一致 |
| 提取时间超预期 | 先跑16k和32k（最可能最优），确认方向后再补其他 |
| 某些窗口大小的正样本太少 | 特别注意W4k（正样本比例~4%），BalancedSampler必须启用 |
