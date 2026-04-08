# 阶段3：训练Pipeline — 详细执行手册

> 本阶段可以用诊断实验的数据（6202样本）提前调通代码。
> 阶段2的正式embedding出来后，换数据重新训练即可。

---

## 概述

```
输入：N × 4096 的embedding矩阵 + N行的标签表
  ↓
投影头 MLP：4096 → 2048 → 256（归一化）
  ↓
分类头：256 → 8（sigmoid，多标签）
  ↓
DANN分支：256 → GRL → 线性层 → phylum类别数（softmax）
  ↓
输出：每个样本的8维预测概率 + 训练好的模型权重
```

---

## 数据接口

### 输入文件

```
embeddings.npy          # shape: (N, 4096), float32
labels.csv              # N行，包含以下列
```

`labels.csv` 必需列：
```csv
window_id,                          # 唯一标识
Polyketide,NRP,Terpene,RiPP,Saccharide,Alkaloid,Other,is_negative,  # 8维标签，每个0或1
phylum,                             # 字符串，用于DANN
split                               # train / val / test_id / test_ood
```

### 用诊断实验数据做原型开发时的适配

诊断实验的数据格式和正式数据不完全一样，需要一个适配脚本：

```python
# 适配逻辑：
# - MIBiG BGC样本：根据MIBiG的类型标注生成多标签向量，is_negative=0
# - 背景窗口：前7位全0，is_negative=1
# - phylum：MIBiG BGC标记为其来源phylum（多数是Actinobacteria），背景窗口用genome_metadata中的phylum
# - split：随机划分 70/10/20 为 train/val/test
# 注意：这批数据有来源偏差，训练结果不具参考价值，仅用于验证代码正确性
```

---

## 模型架构

### 文件：`model/projector.py`

```
投影头（ProjectionHead）:
  Linear(4096, 2048) → BatchNorm1d(2048) → ReLU → Dropout(0.3)
  Linear(2048, 256) → L2归一化

输入: (batch_size, 4096)
输出: (batch_size, 256)，L2范数=1
```

### 文件：`model/classifier.py`

```
多标签分类头（MultiLabelClassifier）:
  8个独立的线性层，每个 Linear(256, 1) → Sigmoid
  或等价地：Linear(256, 8) → Sigmoid

输入: (batch_size, 256)
输出: (batch_size, 8)，每个值在[0,1]之间

注意：不用softmax，各维度独立
```

### 文件：`model/dann.py`

```
梯度反转层（GradientReversalLayer）:
  前向传播：恒等变换（直接传递）
  反向传播：梯度乘以 -λ_dann

领域分类头（DomainClassifier）:
  GRL → Linear(256, 128) → ReLU → Linear(128, n_phyla) → Softmax

输入: (batch_size, 256)  # 和分类头共享投影头输出
输出: (batch_size, n_phyla)

λ_dann 初始=0.0，线性增长到1.0（前20个epoch从0增到1，之后保持1）
```

### 文件：`model/full_model.py`

```
完整模型（BGCModel）:
  self.projector = ProjectionHead()
  self.classifier = MultiLabelClassifier()
  self.dann = DomainClassifier(n_phyla)

  def forward(x):
    z = self.projector(x)           # (B, 256)
    pred = self.classifier(z)       # (B, 8)
    domain = self.dann(z)           # (B, n_phyla)
    return z, pred, domain
```

---

## 损失函数

### 文件：`losses/classification_loss.py`

```
多标签二值交叉熵（MultiLabelBCE）:

  loss = BCE(pred, target)  # 逐维度计算
  
  可选：per-class权重（处理类别不平衡）
  weight[i] = N_total / (n_classes * N_positive_for_class_i)

输入: pred (B, 8), target (B, 8)
输出: 标量loss
```

### 文件：`losses/contrastive_loss.py`

```
多标签感知的有监督对比损失（MultiLabelSupCon）:

核心逻辑：
  对batch中的每对样本(i, j)：
    - 如果 i 和 j 共享至少一个类型标签（任意一个类型位都=1） → 正pair
    - 如果 i 是positive且 j 是negative（或反过来） → 负pair
    - 如果 i 和 j 都是negative → 不构成有意义的pair，跳过
  
  loss = -log( exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
  其中sim是余弦相似度，τ是温度参数

特殊处理：
  - negative样本数量远大于positive → 每个batch中控制正负比例
    建议采样策略：每个batch中50% positive + 50% negative
  - 跨phylum正pair加权：如果两个同类型BGC来自不同phylum，权重×1.5
    （鼓励模型学习跨phylum的类型特征）

输入: z (B, 256), labels (B, 8), phylum (B,)
输出: 标量loss

温度参数 τ = 0.07（初始值，可调）
```

### 文件：`losses/dann_loss.py`

```
领域对抗损失（DANNLoss）:

  loss = CrossEntropy(domain_pred, phylum_target)
  
  注意：这个loss通过GRL反传，实际效果是让投影头输出无法预测phylum

输入: domain_pred (B, n_phyla), phylum_target (B,)
输出: 标量loss
```

### 总损失

```
L_total = α * L_classification + β * L_contrastive + γ * L_dann

初始超参：
  α = 1.0
  β = 0.35（SimGCD中supervised contrastive的默认权重）
  γ = 0.1（DANN权重，偏小，避免过度抑制有用信号）
```

---

## 数据加载

### 文件：`data/dataset.py`

```python
class BGCDataset(Dataset):
    """
    读取预提取的embedding和标签
    """
    def __init__(self, embeddings_path, labels_path, split='train'):
        self.embeddings = np.load(embeddings_path)  # (N, 4096)
        self.labels_df = pd.read_csv(labels_path)
        
        # 按split过滤
        mask = self.labels_df['split'] == split
        self.embeddings = self.embeddings[mask]
        self.labels_df = self.labels_df[mask].reset_index(drop=True)
        
        # 提取标签矩阵
        type_cols = ['Polyketide','NRP','Terpene','RiPP','Saccharide','Alkaloid','Other','is_negative']
        self.labels = self.labels_df[type_cols].values.astype(np.float32)  # (N_split, 8)
        
        # 提取phylum编码
        self.phyla = self.labels_df['phylum'].astype('category').cat.codes.values  # (N_split,)
    
    def __getitem__(self, idx):
        emb = self.embeddings[idx]  # (4096,)
        label = self.labels[idx]    # (8,)
        phylum = self.phyla[idx]    # int
        return emb, label, phylum
```

### 文件：`data/sampler.py`

```python
class BalancedBatchSampler:
    """
    每个batch中保证约50%的positive样本和50%的negative样本。
    positive样本中尽量均匀覆盖7种类型。
    
    这对对比学习很关键——如果batch中positive太少，
    几乎找不到正pair，对比loss退化。
    """
    def __init__(self, labels, batch_size=128):
        # labels: (N, 8)
        self.positive_indices = np.where(labels[:, 7] == 0)[0]  # is_negative=0的样本
        self.negative_indices = np.where(labels[:, 7] == 1)[0]
        self.batch_size = batch_size
        self.n_pos_per_batch = batch_size // 2
        self.n_neg_per_batch = batch_size - self.n_pos_per_batch
```

---

## 数据增强

### 文件：`data/augmentation.py`

```python
class EmbeddingAugmentation:
    """
    在embedding空间做增强，生成同一样本的两个"视图"。
    对比学习需要同一样本的两个增强版本作为正pair。
    
    方式1：加高斯噪声
      z_aug = z + N(0, σ²)，σ=0.05（embedding L2 norm的约5%）
    
    方式2：随机维度dropout
      随机将10%的维度置0，然后重新L2归一化
    
    方式3：混合
      以50%概率选方式1，50%概率选方式2
    
    注意：增强只在训练时使用，验证和测试时不做增强
    """
    def __init__(self, noise_std=0.05, dropout_rate=0.1):
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
    
    def __call__(self, embedding):
        # 返回两个增强版本
        view1 = self._augment(embedding)
        view2 = self._augment(embedding)
        return view1, view2
```

---

## 训练循环

### 文件：`train.py`

```
训练流程伪代码：

初始化：
  model = BGCModel(n_phyla=从数据中获取)
  optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
  scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

每个epoch：
  for batch in balanced_dataloader:
    embeddings, labels, phyla = batch                  # (B, 4096), (B, 8), (B,)
    
    # 数据增强（对比学习用）
    view1, view2 = augmentation(embeddings)            # 各(B, 4096)
    
    # 前向传播 — view1
    z1, pred1, domain1 = model(view1)                  # (B,256), (B,8), (B,n_phyla)
    
    # 前向传播 — view2
    z2, pred2, domain2 = model(view2)                  # (B,256), (B,8), (B,n_phyla)
    
    # 分类损失（两个view的平均）
    L_cls = (MultiLabelBCE(pred1, labels) + MultiLabelBCE(pred2, labels)) / 2
    
    # 对比损失（z1和z2之间）
    z_combined = concat(z1, z2)                        # (2B, 256)
    labels_combined = concat(labels, labels)           # (2B, 8)
    phyla_combined = concat(phyla, phyla)               # (2B,)
    L_con = MultiLabelSupCon(z_combined, labels_combined, phyla_combined)
    
    # DANN损失
    L_dann = (DANNLoss(domain1, phyla) + DANNLoss(domain2, phyla)) / 2
    
    # 总损失
    L_total = α * L_cls + β * L_con + γ * L_dann
    
    # 反向传播
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()
  
  scheduler.step()
  
  # 每个epoch结束后在val集上评估
  val_metrics = evaluate(model, val_dataloader)
  log(epoch, L_total, L_cls, L_con, L_dann, val_metrics)
  
  # 保存checkpoint
  if val_metrics['mean_auroc'] > best_val_auroc:
    save_checkpoint(model, 'best_model.pt')
  save_checkpoint(model, 'last_model.pt')

超参（初始值）：
  n_epochs = 100
  batch_size = 128
  lr = 0.1
  α = 1.0（分类损失权重）
  β = 0.35（对比损失权重）
  γ = 0.1（DANN损失权重）
  τ = 0.07（对比损失温度）
  noise_std = 0.05（增强噪声）
  dropout_rate = 0.1（增强dropout）
  λ_dann: 0→1线性增长（前20 epoch）
```

---

## 评估

### 文件：`evaluate.py`

```
评估流程：

对每个split（val / test_id / test_ood）：

1. 多标签分类指标（每个类型分别算）：
   - AUROC：每个类型的ROC曲线下面积
   - AP（Average Precision）：每个类型的PR曲线下面积
   - F1@0.5：以0.5为阈值的F1 score
   - mean_auroc：7个BGC类型的AUROC平均值（不含is_negative）

2. BGC二分类指标（把7类BGC合成一个"is BGC"标签）：
   - BGC_score = 1 - pred_is_negative（或 max(pred_type_1, ..., pred_type_7)）
   - 用BGC_score vs 真实is_bgc标签算 AUROC 和 PR-AUC

3. phylum不变性检查：
   - 在不同phylum上分别算BGC二分类AUROC
   - 如果DANN生效，各phylum的AUROC应该接近

输出：
  results/eval_{split}.json
  {
    "per_type_auroc": {"Polyketide": 0.xx, "NRP": 0.xx, ...},
    "per_type_ap": {...},
    "per_type_f1_at_05": {...},
    "mean_auroc": 0.xx,
    "bgc_binary_auroc": 0.xx,
    "bgc_binary_prauc": 0.xx,
    "per_phylum_auroc": {"Actinomycetota": 0.xx, ...},
    "dann_phylum_accuracy": 0.xx  # 越接近1/n_phyla越好
  }
```

---

## 消融实验

### 文件：`configs/ablations/`

```yaml
# A: 纯分类器（无对比学习，无DANN）
A_baseline:
  alpha: 1.0
  beta: 0.0     # 关闭对比损失
  gamma: 0.0    # 关闭DANN
  augmentation: false

# B: 分类器 + 对比学习
B_contrastive:
  alpha: 1.0
  beta: 0.35
  gamma: 0.0    # 关闭DANN
  augmentation: true

# C: 分类器 + 对比学习 + DANN（完整模型）
C_full:
  alpha: 1.0
  beta: 0.35
  gamma: 0.1
  augmentation: true

# D: 完整模型 + NLL辅助特征（输入维度4096+8=4104）
D_with_nll:
  alpha: 1.0
  beta: 0.35
  gamma: 0.1
  augmentation: true
  input_dim: 4104   # 拼接NLL统计特征
```

每个配置跑3次（不同随机种子），取平均。

### 消融实验输出

```
results/ablation_table.csv

config, seed, val_mean_auroc, val_bgc_auroc, test_id_mean_auroc, test_id_bgc_auroc, test_ood_bgc_auroc
A_baseline, 0, ..., ..., ..., ..., ...
A_baseline, 1, ..., ..., ..., ..., ...
A_baseline, 2, ..., ..., ..., ..., ...
B_contrastive, 0, ..., ..., ..., ..., ...
...
```

---

## 代码组织

```
bgc-gcd/
├── configs/
│   ├── base.yaml                  # 默认超参
│   └── ablations/
│       ├── A_baseline.yaml
│       ├── B_contrastive.yaml
│       ├── C_full.yaml
│       └── D_with_nll.yaml
├── data/
│   ├── dataset.py                 # BGCDataset
│   ├── sampler.py                 # BalancedBatchSampler
│   ├── augmentation.py            # EmbeddingAugmentation
│   └── adapt_diagnostic.py        # 适配诊断实验数据的脚本
├── model/
│   ├── projector.py               # ProjectionHead
│   ├── classifier.py              # MultiLabelClassifier
│   ├── dann.py                    # GRL + DomainClassifier
│   └── full_model.py              # BGCModel（组装以上三者）
├── losses/
│   ├── classification_loss.py     # MultiLabelBCE
│   ├── contrastive_loss.py        # MultiLabelSupCon
│   └── dann_loss.py               # DANNLoss
├── train.py                       # 主训练循环
├── evaluate.py                    # 评估脚本
├── run_ablation.py                # 跑全部消融实验
└── results/                       # 所有输出
```

---

## 用诊断实验数据调通代码的验收标准

用现有的6202样本数据跑：

### 代码正确性验证

1. **模型前向传播**：
   - 输入 (B, 4096) → 投影头输出 (B, 256)，L2范数≈1.0
   - 分类头输出 (B, 8)，每个值在(0, 1)之间
   - DANN输出 (B, n_phyla)，softmax概率和≈1.0

2. **损失函数**：
   - L_cls > 0，且随训练下降
   - L_con > 0，且随训练下降
   - L_dann：初始下降（模型学会预测phylum），然后上升或震荡（GRL开始生效）
   - L_total 无 NaN 或 Inf

3. **训练稳定性**：
   - 跑完100个epoch无崩溃
   - 梯度范数没有爆炸（< 10.0）
   - 学习率按cosine衰减正常工作

4. **评估流程**：
   - 评估脚本能正常产出 `eval_val.json`
   - 所有AUROC值在0.5-1.0之间（低于0.5说明标签可能搞反了）
   - 每种类型的AUROC都有值（没有因为样本量太少而报错）

### 期望的数值范围（仅用于验证代码，不代表最终性能）

在诊断实验数据上（有来源偏差），大致预期：
- 配置A（纯分类）：val mean_auroc 0.70-0.85
- 配置B（+对比学习）：比A高0-5个百分点
- 配置C（+DANN）：phylum预测准确率从90%+下降到接近随机（~25%）
- 配置D（+NLL）：和C差距不大（NLL信号弱）

如果某个配置的AUROC < 0.60，可能是代码bug，需要排查。
如果所有配置都 > 0.90，可能是来源偏差导致的虚高，不用担心。

### 反馈给我的信息

- 4个消融配置的完整 `ablation_table.csv`
- 每个配置的训练loss曲线（L_cls, L_con, L_dann 分别画）
- 配置C的DANN效果：phylum预测准确率随epoch的变化曲线
- 任何代码错误或异常情况的描述
- 训练总耗时（确认在合理范围内，预期单次<1小时）

---

## 正式训练时需要改的地方

当阶段2的正式embedding到位后，只需要改以下几处：

1. `embeddings.npy` 路径指向新的embedding文件
2. `labels.csv` 路径指向阶段1.3的 `windows_metadata.csv`
3. `n_phyla` 参数更新为正式数据中的phylum数量
4. `batch_size` 可能需要根据样本量调整（样本多了可以加大）
5. `n_epochs` 可能需要调整（样本多了可以少跑几个epoch）
6. class weight 根据正式数据的类别分布重新计算

模型架构、损失函数、评估逻辑全部不变。
