# Phase 3 执行计划：多窗口上下文模型

**日期:** 2026-04-10  
**状态:** 待执行  
**前置依赖:** H_hardneg_r10 已确认为主线（3-seed 稳定性验证通过）  
**决策门:** 本 phase 是 WS7（表示升级决策）的首次实验

---

## 1. 目的

Phase 1（边缘填充）和 Phase 2（hard-negative mining）已大幅缩小 specification/evaluation 层面的缺陷，但**尚不能确认系统已进入纯 representation-limited 状态**。当前剩余 gap 可能混合了三类因素：(1) 独立窗口分类的信息瓶颈（本 phase 的靶点），(2) operating-point / 阈值层面的残差（test_ood 高方差提示阈值仍对分布偏移敏感），(3) 短 BGC geometry 残差（<5kb recall 仅 22%，不排除仍有数据/标签层面可做的事）。

本 phase 选择先攻击因素 (1)——**每个窗口完全独立分类**的限制——因为它是当前最大且最可控的单一变量。但本 phase 的结论不应被解读为"所有剩余 gap 都是 representation 问题"。

本 phase 的目标是：在不改变 Evo2 backbone、不重新提取 embedding 的前提下，通过在窗口序列上引入上下文模型，验证**序列上下文是否能缓解剩余的失败模式**。

这是一个**架构变更**，是项目首次引入超越 MLP 的模型组件。

---

## 2. Phase 假设

### 主假设

当前剩余失败模式（短 BGC 低召回、边界窗口误判、OOD 孤立假阳性）**部分源于独立窗口分类的信息瓶颈**，而非 Evo2 embedding 本身的表示能力不足。

如果在窗口序列上引入上下文建模：
- 短 BGC 窗口可以借助邻居的信号辅助判断（neg→pos→neg 模式）
- 边界窗口可以看到内侧是 BGC、外侧是 background 的上下文
- 孤立假阳性在上下文平滑后应被抑制
- 以上改善应不依赖于增加训练数据或改变 Evo2

### 零假设

剩余失败模式主要由 Evo2 embedding 的信息瓶颈造成（例如 mean-pool 丢失了窗口内的位置信息），上下文模型无法显著帮助，需要转向 LoRA 或更深层的表示升级。

---

## 3. 主线基线

所有 Phase 3 实验必须与以下固定基线对比：

| 条目 | 值 |
|------|-----|
| 模型 | H_hardneg_r10（冻结 Evo2 + MLP） |
| 检测分数 | `score_maxtype = max(sigmoid(logit[0:7]))` |
| 阈值策略 | val F1-optimal，当前 0.8635 |
| 评估模式 | full-window（含 ambiguous 窗口） |
| 训练先验 | `pos_ratio=0.15`，`hard_neg_ratio=0.10` |
| 配置文件 | `bgc-gcd/configs/mainline.yaml` |

**基线指标（3-seed mean ± std）：**

| 指标 | test_id | test_ood |
|------|---------|----------|
| Recall | 0.731 ± 0.012 | 0.566 ± 0.041 |
| P(pred) | 0.650 ± 0.032 | 0.531 ± 0.080 |
| <5kb recall | 0.222 ± 0.005 | 0.171 ± 0.014 |
| 5-10kb recall | 0.352 ± 0.022 | 0.380 ± 0.009 |
| 10-20kb recall | 0.584 ± 0.022 | 0.360 ± 0.044 |

**服务器端基线 artifact：**
- Checkpoint: `/data/muskliu/np_master/data/stage3_results/H_hardneg_r10/best_H_hardneg_r10.pt`
- Embeddings: `/data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy` (1,064,568 × 4096)
- Labels: `/data/muskliu/np_master/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv`

---

## 4. 数据诊断：上下文模型的前提验证

在设计架构之前，必须确认数据结构是否支持上下文建模。以下是对 contig 结构和正样本分布的分析。

### 4.1 Contig 长度分布

| 统计量 | 值 |
|--------|-----|
| 总窗口数 | 1,064,568 |
| 总 contig 数 | 127,863 |
| 均值 | 8.3 窗口/contig |
| 中位数 | 1 窗口/contig |
| 最大值 | 1,482 窗口/contig |

| 分段 | contig 数 | 占比 |
|------|----------|------|
| ≤1 窗口 | 69,542 | 54.4% |
| ≤3 窗口 | 89,292 | 69.8% |
| ≤5 窗口 | 99,303 | 77.7% |
| ≤10 窗口 | 111,137 | 86.9% |

**注意：** 54.4% 的 contig 只有 1 个窗口（主要来自 edge padding 生成的短 contig 窗口）。分布极度右偏。

### 4.2 正样本的 contig 覆盖

| 条件 | 正样本窗口数 | 占全部正样本比例 |
|------|------------|----------------|
| contig ≥ 3 窗口 | 66,212 | **96.9%** |
| contig ≥ 5 窗口 | 64,326 | **94.1%** |
| contig = 1 窗口 | 1,550 | 2.3% |

**关键结论：几乎所有正样本（96.9%）位于有上下文可用的 contig 上。** 单窗口 contig 虽多（54%），但只占正样本的 2.3%。

### 4.3 正样本按 contig 长度的分布

| Contig 长度 | 正样本窗口数 | 占比 |
|------------|------------|------|
| 1w | 1,550 | 2.3% |
| 2w | 585 | 0.9% |
| 3w | 906 | 1.3% |
| 4-5w | 2,062 | 3.0% |
| 6-10w | 5,433 | 7.9% |
| 11-20w | 8,526 | 12.5% |
| 21-50w | 13,040 | 19.1% |
| 51-100w | 7,700 | 11.3% |
| >100w | 28,545 | **41.8%** |

### 4.4 BGC 连续正窗口 run-length 分析

| 统计量 | 值 |
|--------|-----|
| 总 run 数 | 19,275 |
| 均值 | 3.5 窗口 |
| 中位数 | 3 窗口 |
| 最大值 | 43 窗口 |

| Run 长度 | 占比 | 含义 |
|----------|------|------|
| 1 窗口 | 21.0% | 短 BGC / 边缘 — **最需要上下文帮助的群体** |
| 2 窗口 | 20.4% | 需要边界上下文 |
| 3 窗口 | 24.1% | 典型 BGC (~24kb) |
| 5 窗口 | 10.0% | 较大 BGC |
| >10 窗口 | 2.8% | 超大 BGC |

### 4.5 数据诊断结论

1. **前提条件满足：** 96.9% 的正样本有上下文可用。
2. **上下文窗口 K=11 合理：** 中位 run=3，K=11 可覆盖完整 BGC + 两侧各 4 个背景窗口。
3. **短 contig 退化可接受：** 单窗口 contig（54% 的 contig 但仅 2.3% 的正样本）通过残差连接退化为独立分类。
4. **主要收益来源：** >100 窗口 contig 上的 41.8% 正样本拥有丰富上下文。

### 4.6 各 split 窗口分布

| Split | positive | negative | ambiguous | total |
|-------|----------|----------|-----------|-------|
| train | 42,375 | 544,831 | 39,243 | 626,449 |
| val | 7,245 | 81,745 | 6,261 | 95,251 |
| test_id | 11,622 | 158,136 | 10,808 | 180,566 |
| test_ood | 7,105 | 148,524 | 6,673 | 162,302 |

**注意：** ambiguous 窗口约占 6%，它们是 BGC 边界区域。在上下文模型中，ambiguous 窗口应作为 context 输入参与前向传播，但**不参与 loss 计算**。

---

## 5. ML 最佳实践规则

### 规则 1：单轴变化优先

Route A（验证版）只改变一个变量：引入上下文模块。不同时改 loss、sampler、label 规则。

### 规则 2：不使用 test set 调参

上下文窗口 K、CNN 层数、学习率等超参数只在 val 上选择。`test_id` 和 `test_ood` 仅用于报告。

### 规则 3：不回归才能晋升 + OOD threshold 约束

任何上下文模型配置必须在 safety metrics 上不劣于基线，才能进入主线候选。

**OOD threshold policy（已知 test_ood 对阈值敏感，std=0.080）：**
- 阈值**只在 val 上选择**（F1-optimal），绝不在 test 上调
- 最终汇报**必须同时报告 test_id 和 test_ood**，不允许只报 ID
- 如果某模型靠更激进阈值换来 test_id 提升但 test_ood 明显崩溃（recall 下降 > 5pp 或 precision 下降 > 10pp），**不可晋升**，即使 ID 指标达标

### 规则 4：center-only loss（Route A）

在 Route A 中，**只对采样的中心窗口计算 loss**，邻居窗口只作为 context 输入。

理由：如果对 K 内所有位置算 loss，同一窗口会在多个 neighborhood 中重复出现。长 contig、dense BGC 区域会被过度加权，BalancedSampler 维护的 15/85 先验将实质性失效。这不是实现细节，是训练目标本身的失真。

### 规则 5：v1 不上 contrastive / DANN

理由：上下文模型中，相邻窗口在 Evo2 embedding 空间中天然高度相似（Stage 2 诊断：intra-genome distance 仅 0.013）。如果将 (B, K) 展平后做 contrastive，模型容易将"局部平滑"当作正信号。DANN 也可能因 contig 内窗口高度相关而产生过强约束。

v1 只用 BCE，待 v1 结果稳定后再考虑是否引入 contrastive 和 DANN。

### 规则 6：不用 BatchNorm

上下文模块中不使用 BatchNorm。标准 BN 无法区分 padding 位置和真实位置，大量零填充会污染 batch 统计量。不同 batch 中 contig 长度分布差异也会导致 BN 统计不稳定，尤其在 OOD 推理时。

替代：使用 **LayerNorm** 或 **GroupNorm**。

### 规则 8：checkpoint 选择以 contig-aware region metric 为主

`bgc_binary_auroc` 是 window-level ranking 指标。Phase 3 要修的是 short-BGC、boundary、OOD 问题——这些改善未必反映在 window-level AUROC 上（例如：边界窗口 score 从 0.80 涨到 0.87 不影响 AUROC 排序，但决定了 region calling 是否多召回一个 BGC）。继续用 window AUROC 当北极星会选出"AUROC 好看但部署效果一般"的模型。

**checkpoint 选择策略：**

- **主指标：val region-level F1**（contig-aware inference → score_maxtype → F1-optimal threshold → region calling → 计算 recall × precision 的调和平均）
- **运行频率：** 每 N=5 epoch 运行一次（val 集约 95K 窗口，contig-level inference + region calling 约需 30-60 秒，开销可接受）
- **Guardrail（辅助条件，不决定选择但可触发警告）：**
  - val short-bucket recall（<10kb）不应低于基线
  - val OOD-phylum precision（如果 val 中有 OOD-like 样本）不应崩溃
  - discovery@20 不应明显退化（仅在最终候选时检查）
- **快速筛除：** 逐 epoch 仍计算 window-level `bgc_binary_auroc`，若某 epoch 该指标骤降 > 5pp，标记为异常但不一定排除（可能是 region metric 仍改善的情况）

### 规则 9：discovery 不是最后补查，每个 serious candidate 都要报告

本项目的顶层目标包括 novel BGC discovery，不仅是 antiSMASH recovery。如果架构优化只朝"更像 antiSMASH"方向偏，太晚发现 discovery 退化代价很高。

- Route A：至少做轻量 discovery smoke check（sig precision@20）
- Route B：每个达标候选都必须同时报告 discovery 指标（sig precision@20/50 + diversity@50）
- 晋升决策同时考虑 recovery 和 discovery

### 规则 10：残差连接必须安全且可学习地初始化

~~残差 α 初始化为 0~~ **（已修正：α=0 会导致 ContextModule 拿不到梯度，是个 bug 而非安全设计）。**

当 `z_final = z_proj + α * z_context` 且 α=0 时，`d_loss/d_context_params = α * d_loss/d_z_final * d_z_context/d_params = 0`。ContextModule 完全收不到梯度信号，训练会停滞，导致假阴性（"上下文无效"实际上是"上下文从未被训练"）。

**修正方案：** 使用 sigmoid gate，`α = sigmoid(β)`，β 初始化为 -4.0（对应 α ≈ 0.018）。这保证：
- 初始时上下文贡献很小（≈1.8%），不打乱已有决策边界
- 梯度始终非零，ContextModule 从第一个 step 就能学习
- α 有自然上界 1.0（sigmoid 饱和），不会爆炸
- β 可以学成很大负数（α→0）如果上下文确实无益

Classifier 从当前主线 checkpoint warm start，保证初始输出与基线接近。

---

## 6. 架构设计

### 6.1 总体架构

```
输入: (B, K, D)
 ├── D = 256（Route A：冻结 ProjectionHead 预提取）
 └── D = 4096（Route B：端到端，含 ProjectionHead）

Route A:
  (B, K, 256)
    → ContextModule          → (B, K, 256)
    → GatedResidual(β=-4, α≈0.018)     → (B, K, 256)
    → L2Normalize            → (B, K, 256)
    → Classifier(256→8)      → (B, K, 8)    [只取中心位置算 loss]

Route B:
  (B, K, 4096) 或 contig chunk (B, L, 4096)
    → ProjectionHead(4096→256) [warm start, 可训练]
    → ContextModule            → (B, L, 256)
    → GatedResidual(β=-4, α≈0.018)       → (B, L, 256)
    → L2Normalize              → (B, L, 256)
    → Classifier(256→8)        → (B, L, 8)   [chunk 内所有非 ambiguous 位置算 loss]
```

### 6.2 ContextModule：膨胀 1D CNN

```python
ContextModule:
  Conv1d(257, 256, kernel=3, dilation=1, padding=1)  →  LayerNorm  →  GELU  →  Dropout(0.1)
  Conv1d(256, 256, kernel=3, dilation=2, padding=2)  →  LayerNorm  →  GELU  →  Dropout(0.1)
  Conv1d(256, 256, kernel=3, dilation=4, padding=4)  →  LayerNorm  →  GELU  →  Dropout(0.1)
```

注意：第一层 in_channels=257（256-dim 投影 + 1-dim valid_mask），后续层均为 256。

| 属性 | 值 |
|------|-----|
| 感受野 | 1 + 2×(1+2+4) = 15 窗口 = **120kb** |
| 参数量 | (257×256×3 + 256) + 2×(256×256×3 + 256) ≈ **590K** |
| 覆盖 BGC 比例 | 覆盖 97%+ 的 BGC（仅 run>15 的 <1% 超出） |

**Padding 语义区分与 mask 处理：**

系统中存在两种语义不同的"padding"，不能混为一种零向量：

| 类型 | 来源 | 含义 | 有 Evo2 embedding | 示例 |
|------|------|------|-------------------|------|
| **edge-padded window** | Phase 1 边缘填充 | 真实窗口，DNA 序列用 N 填充到 16kb | **有**（值非零） | 短 contig 末尾 |
| **chunk padding** | Route B chunk 右侧补齐 | 人工占位，无对应基因组位置 | **无**（全零） | chunk 末尾填充 |

edge-padded window 是 Evo2 提取过的真实窗口（metadata 中 `edge_padded=1`），有正常的非零 embedding。模型应正常处理它们。chunk padding 是纯人工零向量，模型需要忽略它们。

**显式 mask 设计：**

1. **valid_window_mask（0/1）：** 作为额外特征通道拼接到 ContextModule 输入。`(B, K, 256)` → `(B, K, 257)`，第一层 Conv1d 的 `in_channels=257`。
   - edge-padded window → **1**（真实窗口）
   - chunk padding → **0**（人工占位）
2. **LayerNorm 不受 padding 影响：** LayerNorm 在每个样本内独立计算统计量，不受 batch 级 padding 分布影响。
3. **padding 位置输出置零：** 残差连接后，对 `valid_window_mask=0` 的位置乘以 mask 置零（不参与 loss、不参与推理）。
4. **loss_mask（Route B 额外需要）：** 标记 chunk 内哪些位置参与 loss（= valid_window_mask AND label != ambiguous AND 处于 chunk 中心 loss 区域内）。

### 6.3 GatedResidual

```python
class GatedResidual(nn.Module):
    def __init__(self, dim, init_beta=-4.0):
        self.beta = nn.Parameter(torch.tensor(init_beta))  # sigmoid(-4) ≈ 0.018
    
    def forward(self, x_proj, x_context):
        # x_proj: 原始投影, x_context: 上下文增强
        alpha = torch.sigmoid(self.beta)
        return x_proj + alpha * x_context
```

sigmoid gate 意味着：
- 训练开始时，α ≈ 0.018，上下文贡献极小，不打乱已有决策边界
- 梯度始终非零，ContextModule 从 step 0 就能学习（修正了 α=0 的零梯度 bug）
- α 有上界 1.0（sigmoid 饱和），不会爆炸
- 如果上下文无益，β 会学成更大负数，α→0

### 6.4 Classifier warm start

Classifier (`Linear(256→8)`) 的权重和偏置从 H_hardneg_r10 checkpoint 直接加载。这确保在 α≈0 的初始阶段，模型输出与当前主线一致。

---

## 7. Route A：验证版（快速假设检验）

### 7.1 目的

以最低成本验证"上下文是否能帮助剩余失败模式"。这不是最终系统，只是一个 smoke test。

### 7.2 数据流

```
Step 0: 预提取
  H_hardneg_r10 的 ProjectionHead 对全部 1,064,568 个窗口提取 256-dim 特征
  保存为 projections_256.npy (~1GB)
  同时构建 contig 索引（pickle）

训练时:
  BalancedBatchSampler 采样中心窗口（与当前完全一致）
  ↓
  ContextDataset.__getitem__(center_idx):
    查找中心窗口所在 contig
    加载左右各 K//2=5 个邻居的 256-dim 向量
    不足部分零填充 + mask
    返回: (K=11, 256), center_label, center_phylum, mask(K,)
  ↓
  batch: (B=256, K=11, 256)

推理时:
  逐 contig 处理，输入 (1, L_contig, 256)
  输出 (1, L_contig, 8) 的逐窗口预测
```

### 7.3 模型

```
输入: (B, 11, 256) — 冻结 ProjectionHead 的输出
  → ContextModule (膨胀 CNN, 3层)
  → GatedResidual(β=-4, α≈0.018)
  → L2Normalize
  → Classifier(256→8) [warm start from H_hardneg_r10]

Loss: BCE，只对中心位置（index=5）计算
      使用当前 pos_weight（从训练集统计）
```

### 7.4 训练配置

| 参数 | 值 | 理由 |
|------|-----|------|
| 优化器 | AdamW | 新组件用 Adam 更稳定，不需要像 SGD 那样精调 lr |
| 学习率 | 1e-3 | ContextModule + Classifier 联合训练 |
| 权重衰减 | 1e-4 | 防止过拟合 |
| Epochs | 100 | 与基线一致 |
| LR schedule | Cosine annealing | 与基线一致 |
| Batch size | 256 | 与基线一致 |
| pos_ratio | 0.15 | 与基线一致 |
| hard_neg_ratio | 0.10 | 与基线一致 |
| 检查点指标 | **val region F1**（主），bgc_binary_auroc（辅助/快速筛除）| 见规则 8 |
| K | 11 | 上下文窗口大小 |

**注意：** Route A 中 ProjectionHead 完全冻结，不可训练。只训练 ContextModule + Classifier。

### 7.5 评估

**训练时 checkpoint 选择：** 使用新建的 `evaluate_context.py`，逐 contig 推理得到 window-level predictions，然后计算 `bgc_binary_auroc`（逐 epoch）。每 **5 epoch** 运行 val region-level 评估（val recall + val precision），最终模型以 val region-level F1 为准（频率与规则 8 一致）。

**测试时 region-level 评估：** 使用 `context_predict.py` 做 contig-level 推理，输出逐窗口 score，然后对接现有 `ws1_score_swap.py`：

1. 对每条 contig，按**基因组物理坐标（`window_start`）升序排列**其 256-dim 向量
2. 过 ContextModule → GatedResidual → Classifier
3. 取 `score_maxtype = max(sigmoid(logits[:7]))` 作为检测分数
4. 接入现有 region calling pipeline（阈值 → 合并 → 3kb 最小长度）

阈值在 val 上重新选择（F1-optimal），不沿用基线的 0.8635。

**discovery smoke check：** 对 test_id 结果运行 `ws_discovery_benchmark.py`，至少报告 sig precision@20。

### 7.6 成功标准

Route A 的目的是提供 cheap signal，判断上下文是否值得继续投入。**但 Route A 不能单独否决 Route B**——因为 Route A 冻结了 ProjectionHead，天然低估了上下文架构的上限（当前 256-dim 是为独立分类优化的，不是为上下文融合优化的）。

**Route A 结果的解读规则：**

| Route A 结果 | 对 Route B 的影响 | 理由 |
|-------------|------------------|------|
| 通过（全部条件满足） | **立即启动 Route B**，高优先级 | 冻结 projector 都能涨，端到端上限更高 |
| 部分通过（有正信号但未全满足） | **仍启动 Route B**，正常优先级 | 冻结 256-dim 可能限制了效果表达 |
| 完全失败 + α 始终 ≈ 0 | **降低 Route B 优先级**，但不否决 | 端到端可能激活不同的上下文信号 |
| 完全失败 + 明显退化 | 暂停，先诊断退化原因 | 架构设计可能有 bug |

**Route A 通过条件（全部 AND）：**

| 条件 | 阈值 | 理由 |
|------|------|------|
| short-bucket 改善 | <10kb 中至少一个 bucket recall 提升 > 3pp | 上下文的核心理论靶点 |
| test_id 不退化 | test_id recall ≥ 0.725 | 不能用 short-bucket 换 overall |
| test_ood 不明显变差 | test_ood recall ≥ 0.54 | OOD 不崩 |
| precision 安全 | test_id P(pred) ≥ 0.62 | 不能严重丢精度 |
| discovery 不退化 | sig precision@20 ≥ 80% | 轻量 discovery smoke check |
| 退化检查 | 单窗口 contig 上的性能不劣于基线 | 残差连接保底有效 |

**未通过时的诊断矩阵：**

| 结果模式 | 解读 | 下一步 |
|----------|------|--------|
| short-bucket 无改善 + α 有增长 | 上下文有信号但未命中短 BGC | 调 K、增加位置编码，仍进 Route B |
| short-bucket 无改善 + α ≈ 0 | 冻结 projector 下上下文无益 | 降低优先级但不否决，可直接试 Route B |
| overall 退化 + precision 下降 | 上下文引入噪声 | 检查 mask 处理、检查 sigmoid gate 行为 |
| discovery 退化 | 模型朝 antiSMASH-like 偏移 | 检查是否过度拟合 known BGC pattern |

### 7.7 服务器端输出 artifact

```
/data/muskliu/np_master/data/phase3/
  route_a/
    projections_256.npy          # 预提取的 256-dim 特征
    contig_index.pkl             # contig 索引
    best_route_a.pt              # 最佳 checkpoint
    train_log_route_a.csv        # 训练日志
    eval_test_id_route_a.json    # test_id 评估
    eval_test_ood_route_a.json   # test_ood 评估
    score_swap_results.json      # 完整 region-level 评估
    discovery_smoke.json         # sig precision@20 smoke check
```

---

## 8. Route B：主线版（效果导向）

### 8.1 目的

在 Route A 验证正信号后，构建面向最终性能的端到端系统。

### 8.2 与 Route A 的关键差异

| 维度 | Route A | Route B |
|------|---------|---------|
| 输入维度 | 256（冻结 ProjectionHead） | 4096（端到端） |
| ProjectionHead | 冻结 | 可训练（warm start） |
| 训练单元 | 中心窗口 neighborhood | **contig chunk** |
| Loss 范围 | 只对中心窗口 | chunk 内所有非 ambiguous 窗口 |
| Loss 类型 | BCE only | BCE（v1），后续视情况加 contrastive/DANN |

### 8.3 Contig Chunk 数据流

**为什么用 contig chunk 而不是 center-window neighborhood：**

在 Route A 中，center-only loss + neighborhood context 是安全的验证方案。但对于主线版，这种方式有效率问题——每个 batch 只有 B=256 个有效 loss 位置，而模型对 B×K=2816 个位置都做了前向传播。

Contig chunk 方案：将每条 contig 切成固定长度的 chunk，chunk 内所有非 ambiguous 窗口都参与 loss。

**重叠 chunk 设计（避免边界截断）：**

不重叠 chunk 会在 chunk 边界重新制造"看不到完整上下文"的问题——这和 Phase 1 修过的 short/boundary 问题冲突。因此采用重叠 chunk + 中心区域算 loss 的方案：

```
预处理:
  将每条 contig 切成长度 L_chunk 的 chunk
  chunk 之间有 overlap = L_chunk // 2（即 stride = L_chunk // 2）
  短 contig（< L_chunk）单独作为一个 chunk，右侧零填充
  每个 chunk 记录: (contig_key, start_pos, end_pos, n_valid, n_positive,
                    loss_start, loss_end)  # 只在中心区域算 loss

Loss 区域:
  每个 chunk 只在中心 L_chunk // 2 个窗口上算 loss
  首尾各 L_chunk // 4 个窗口仅作 context（有上下文但不算 loss）
  这保证每个窗口恰好出现在一个 chunk 的 loss 区域中，避免重复加权
  例外：contig 首尾 chunk 的边界侧 loss 区域扩展到 chunk 边缘

遍历:
  每个 epoch 遍历全部 chunk（不做加权采样）
  类别不平衡在 loss 级别处理（pos_weight + focal-style reweighting），不在采样级别
  这保证每个窗口每 epoch 恰好参与 loss 一次，class prior 由 loss weighting 控制

Batch:
  (B_chunks, L_chunk, 4096) + valid_mask(B_chunks, L_chunk) + loss_mask(B_chunks, L_chunk)
```

**采样与 class balance 策略：**

不在 chunk 采样层面做 class prior 操控。Route B 的 class balance 完全通过 loss-level 机制实现：
- `pos_weight`：与基线一致，按训练集负/正比例计算
- 可选 focal loss（γ=2）：自动聚焦难分类样本，与 hard-neg mining 互补
- 这比采样层面操控更干净：(1) 每个窗口贡献确定，(2) 接近真实部署分布，(3) calibration 更容易解释

### 8.4 Chunk 长度选择

| L_chunk | 基因组范围 | Batch 内存 | 有效 loss 位置/batch | 备注 |
|---------|-----------|-----------|-------------------|------|
| 16 | 128kb | B×16×4096×4 ≈ 64MB (B=256) | ~4096 | 太短，上下文不足 |
| 32 | 256kb | B×32×4096×4 ≈ 128MB (B=128) | ~4096 | 合理起点 |
| 64 | 512kb | B×64×4096×4 ≈ 128MB (B=32) | ~2048 | 更多上下文，batch 更小 |

**推荐 L_chunk = 32，B = 128。** 这给出 256kb 的基因组上下文，远超大多数 BGC，且内存可控。

### 8.5 模型

```
输入: (B, L_chunk=32, 4096)
  → ProjectionHead(4096→256) per-window [warm start from H_hardneg_r10]
  → ContextModule (膨胀 CNN, 3层, LayerNorm)
  → GatedResidual(β=-4, α≈0.018)
  → L2Normalize
  → Classifier(256→8) [warm start from H_hardneg_r10]

Loss: masked BCE
  loss_mask = valid_window_mask & (label != ambiguous) & in_center_loss_region
  （与 Section 6.2 和 8.3 中的 loss_mask 定义一致：三个条件缺一不可）
  对 loss_mask 为 True 的所有位置计算 BCE，取平均
  pos_weight 与基线一致
```

### 8.6 训练配置

| 参数 | 值 | 理由 |
|------|-----|------|
| 优化器 | AdamW | 端到端训练，Adam 对多组件更稳定 |
| 学习率 | ProjectionHead: 1e-4, ContextModule: 1e-3, Classifier: 1e-3 | 分层学习率：已训练组件低 lr，新组件高 lr |
| 权重衰减 | 1e-4 | |
| Epochs | 100-200 | 端到端可能需要更长训练 |
| LR schedule | Cosine annealing with warmup (5 epochs) | 保护 warm start 权重 |
| Batch size | 128 chunks | 每个 chunk 32 窗口 |
| L_chunk | 32 | |
| 检查点指标 | **val region F1**（主，每 5 epoch），bgc_binary_auroc（辅助） | 见规则 8 |

### 8.7 评估

**与 Route A 相同的基础流程**：contig-level 推理 → score_maxtype → 阈值（val F1-optimal）→ region calling。

**增强：每个 serious candidate（seed=0 达标后）都同时报告以下全部指标：**
- region-level recall / precision（test_id + test_ood）
- 分段 recall（<5kb, 5-10kb, 10-20kb, 20-50kb, >50kb）
- **失败模式对齐指标**（Phase 3 的核心靶点）：
  - 10-20kb recall（基线 0.584，边界窗口问题的直接度量）
  - single-window-run recall（run_length=1 的 BGC 的召回率，最需要上下文的群体）
  - isolated OOD FP count（test_ood 中 unmatched prediction 数量，基线 ~711）
- discovery benchmark（sig precision@20/50 + diversity@50）

discovery 不是最后补查，是与 recovery 同权重的并行指标。

### 8.8 成功标准

Route B 的标准更严格，目标是产出可晋升为新主线的候选：

**Recovery 条件（全部 AND）：**

| 条件 | 阈值 | 基线参考 |
|------|------|---------|
| test_id recall | > 0.75 | 基线 0.731 |
| test_id P(pred) | ≥ 0.63 | 基线 0.650 |
| <5kb recall | > 0.25 | 基线 0.222 |
| 5-10kb recall | > 0.38 | 基线 0.352 |
| 10-20kb recall | > 0.60 | 基线 0.584 |
| test_ood recall | > 0.57 | 基线 0.566 |
| test_ood P(pred) | ≥ 0.50 | 基线 0.531 |
| 3-seed 稳定性 | test_id recall std < 0.02 | 基线 0.012 |

**Discovery 条件（与 recovery 并行，全部 AND）：**

| 条件 | 阈值 |
|------|------|
| sig precision@20 | ≥ 85%（对比基线 90%，允许小幅退化） |
| sig precision@50 | ≥ 80%（对比基线 86%） |
| diversity@50 | ≥ 35（对比基线 38） |

**晋升条件：** Recovery 条件 AND Discovery 条件全部满足。

### 8.9 服务器端输出 artifact

```
/data/muskliu/np_master/data/phase3/
  route_b/
    chunk_index.pkl               # 重叠 chunk 索引（含 loss_mask）
    best_route_b_s{0,1,2}.pt     # 3-seed checkpoints
    train_log_route_b_s{0,1,2}.csv
    eval_test_id_route_b_s{0,1,2}.json
    eval_test_ood_route_b_s{0,1,2}.json
    score_swap_results_s{0,1,2}.json
    discovery_benchmark_s{0,1,2}.json  # 每个 seed 都跑 discovery
```

---

## 9. 需要新建的代码文件

### 9.1 Route A 所需

| 文件 | 位置 | 功能 |
|------|------|------|
| `extract_projections.py` | `scripts/phase3/` | 用 H_hardneg_r10 的 ProjectionHead 提取全部 256-dim 特征 + 构建 contig 索引 |
| `context_dataset.py` | `bgc-gcd/data/` | ContextDataset：加载 256-dim 特征 + contig 索引，返回 (K, 256) 邻域 |
| `context_model.py` | `bgc-gcd/model/` | ContextModule（膨胀 CNN）+ GatedResidual |
| `train_context.py` | `bgc-gcd/` | Route A 训练循环 |
| `context_predict.py` | `scripts/phase3/` | contig-level 推理，输出逐窗口 score，对接 ws1_score_swap.py |

### 9.2 Route B 额外所需

| 文件 | 位置 | 功能 |
|------|------|------|
| `chunk_dataset.py` | `bgc-gcd/data/` | ChunkDataset：加载 contig chunks，返回 (L_chunk, 4096) |
| `chunk_builder.py` | `scripts/phase3/` | 预处理：构建重叠 chunk 索引（含 loss_mask 区域标注） |
| `context_full_model.py` | `bgc-gcd/model/` | 端到端模型：ProjectionHead + ContextModule + Classifier |
| `train_context_e2e.py` | `bgc-gcd/` | Route B 端到端训练循环 |

### 9.3 需要新建的共用文件

| 文件 | 位置 | 功能 |
|------|------|------|
| `evaluate_context.py` | `bgc-gcd/` | **上下文模型专用评估路径**：contig-aware prediction → window-level metrics。当前 `evaluate.py` 是单窗口 `DataLoader → model(emb)` 路径，无法正确评估上下文模型（上下文模型需要 contig 邻域作为输入）。训练时的 checkpoint selection（val bgc_binary_auroc）和二次 region-level 验证都需要这个新评估路径。 |

### 9.4 不需要修改的现有文件

| 文件 | 原因 |
|------|------|
| `train.py` | 保留原始训练入口不变，新训练脚本独立 |
| `evaluate.py` | 保留用于基线对比，上下文模型使用 `evaluate_context.py` |
| `ws1_score_swap.py` | region-level 评估逻辑不变，预测输入由 `context_predict.py` 提供 |
| `losses/classification_loss.py` | BCE loss 实现不变 |
| `configs/mainline.yaml` | 基线配置不动 |

---

## 10. 执行顺序

### P3-T0: 预提取与索引构建

**目标：** 生成 Route A 所需的 256-dim 特征和 contig 索引。

**操作：**
1. 加载 H_hardneg_r10 checkpoint 的 ProjectionHead
2. 对全部 1,064,568 个窗口（包括 ambiguous）做前向传播
3. 保存 `projections_256.npy` (1,064,568 × 256, float32, ~1GB)
4. 从 metadata CSV 构建 contig 索引：`{(genome_id, contig_id): [window indices sorted by window_start ascending]}`（**必须按基因组物理坐标排序**，不能使用 CSV 行号或 window_id 字典序，否则上下文建模会失真）
5. 保存为 `contig_index.pkl`

**交付物：**
- `/data/muskliu/np_master/data/phase3/route_a/projections_256.npy`
- `/data/muskliu/np_master/data/phase3/route_a/contig_index.pkl`

**验证：**
- `projections_256.npy` shape 为 (1064568, 256)
- 随机抽查 10 个窗口，对比直接过 ProjectionHead 的输出，误差 < 1e-5
- contig 索引覆盖全部 127,863 条 contig

---

### P3-T1: Route A 模型实现与训练

**目标：** 实现 ContextDataset + ContextModule + 训练循环，跑出第一版结果。

**操作：**
1. 实现 `context_dataset.py`：加载 256-dim，支持 K 邻域查找 + padding mask
2. 实现 `context_model.py`：膨胀 CNN + GatedResidual + Classifier
3. 实现 `train_context.py`：center-only BCE loss，BalancedSampler
4. warm start classifier 从 H_hardneg_r10
5. 训练 seed=0，100 epochs
6. 实现 `context_predict.py`：contig-level 推理

**交付物：**
- 训练代码 + checkpoint
- `docs/phase3-route-a-results.md`：完整评估结果

**评估指标（必须报告）：**
- test_id / test_ood recall 和 P(pred)
- <5kb, 5-10kb, 10-20kb, 20-50kb, >50kb 分段 recall
- truncated BGC recall
- 预测区域数
- α 的训练结束值（判断上下文信号强度）
- 训练曲线（loss, val bgc_auroc vs epoch）

---

### P3-T2: Route A 结果分析与决策

**目标：** 判断上下文假设是否成立。

**分析内容：**
1. 与基线逐指标对比
2. 分析 α 的学习轨迹：是否从 0 增长？最终值多少？
3. 分析各长度 bucket 的 recall 变化：哪些 bucket 受益最大？
4. 分析 false negative 变化：上下文模型新召回了哪些基线遗漏的 BGC？
5. 分析 false positive 变化：上下文是否引入了新的 FP 模式？
6. 检查单窗口 contig 上的性能是否退化

**决策矩阵：**

| 结果 | 决策 |
|------|------|
| 主条件通过 + 安全条件通过 | → 启动 Route B |
| 主条件通过 + 安全条件未通过 | → 诊断原因，调整后重试 Route A |
| 主条件未通过 + α > 0.1 | → 上下文有信号但当前架构不够，考虑换 Transformer 或调 K |
| 主条件未通过 + α ≈ 0 | → 零假设成立，上下文无益，转向其他方向（LoRA / 小窗口） |

---

### P3-T3: Route B 实现与训练

**前置条件：** Route A 通过主条件。

**操作：**
1. 实现 contig chunk 数据流：`chunk_dataset.py` + `chunk_sampler.py`
2. 实现端到端模型：`context_full_model.py`
3. 实现端到端训练循环：`train_context_e2e.py`
4. 分层学习率 + warmup
5. 训练 seed=0，100-200 epochs
6. 如果 seed=0 结果达标，跑 seed=1,2 确认稳定性

**交付物：**
- 训练代码 + 3-seed checkpoints
- `docs/phase3-route-b-results.md`

---

### P3-T4: Route B 评估与晋升决策

**操作：**
1. 3-seed 完整评估（test_id + test_ood）
2. 分段 recall 对比
3. 如果候选达标，运行 discovery benchmark
4. 撰写 Phase 3 总结文档

**交付物：**
- `docs/phase3-summary.md`：phase 结论
- 晋升/不晋升决策 + 理由

---

### P3-T5（可选）: v2 改进

**仅当 Route B v1 达标但仍有改进空间时执行。**

可能的改进：
- 引入 contrastive loss（仅对中心位置）
- 引入 DANN（仅对中心位置）
- 尝试 Transformer 替代 CNN
- 调整 L_chunk 或 K
- 将 ambiguous 窗口以 soft label 纳入 loss

每项改进单独测试，遵循单轴变化原则。

---

## 11. 风险分析

| 风险 | 可能性 | 影响 | 对策 |
|------|--------|------|------|
| 上下文模型过拟合 contig 模式 | 中 | 模型记忆了特定 contig 的窗口排列 | dropout + 残差保底 + 较少参数 + 训练/测试 contig 无交叉 |
| 冻结 256-dim 低估上限 | 高 | Route A 结果偏保守 | Route A 只是 smoke test，Route B 端到端释放上限 |
| contig chunk 边界截断 | 中 | chunk 边界处丢失上下文 | 重叠 chunk + 中心区域算 loss |
| 短 contig 在完整遍历中被欠表示 | 低 | loss-level pos_weight 补偿 | pos_weight + 可选 focal loss |
| OOD 推理时上下文不可靠 | 中 | OOD contig 的邻域模式与训练不同 | α 初始化为 0 的残差连接提供安全退化 |
| 训练变慢 | 低 | Route A: K×更多数据读取 | 256-dim 数据仅 1GB，IO 不是瓶颈 |
| Classifier warm start 后被冲掉 | 中 | 上下文扰动导致 classifier 重新学习 | sigmoid gate (α≈0.018) 初始化 + Route B 分层学习率 |

---

## 12. 明确非目标

本 phase **不做**以下事项：

- 改变 Evo2 backbone（不做 LoRA）
- 重新提取 embedding（不改窗口大小或步长）
- 改变检测分数定义（仍然是 `score_maxtype`）
- 改变阈值策略（仍然是 val F1-optimal）
- 改变 region calling 逻辑（阈值 → 合并 → 3kb 最小长度）
- 改变训练/测试 split（仍然是 genome-level split）
- 在 v1 中引入 contrastive 或 DANN loss
- 用 test set 选择超参数

---

## 13. 最终决策问题

本 phase 结束时，项目需要回答：

> **在当前 Evo2 embedding 之上引入窗口级序列上下文，是否能显著缓解独立分类遗留的失败模式？如果能，最终系统应采用什么样的上下文建模方案？**

次要问题：

> 上下文改善最大的是哪个失败模式（短 BGC、边界窗口、OOD）？这是否指向了下一步的优化方向？

---

## 附录 A：与前 Phase 的关系

```
Phase 0: Score/threshold/prior 修复    → recall 0.271 → 0.668
Phase 1: 短 BGC 覆盖（edge padding）  → recall 0.668 → 0.737
Phase 2: OOD 鲁棒性（hard-neg mining） → recall 0.737 → 0.731 (precision ↑↑)
Phase 3: 上下文模型                    → recall 0.731 → ???
```

Phase 0-2 修复的是 specification/data 问题。Phase 3 是首次真正的模型架构升级。

---

## 附录 B：为什么先做 Route A 而不是直接 Route B

| 维度 | Route A | Route B |
|------|---------|---------|
| 实现复杂度 | 低（5 个新文件） | 中（8 个新文件 + chunk sampler） |
| 迭代速度 | 极快（256-dim，训练 ~2min） | 中（4096-dim，训练 ~10-15min） |
| 假设验证 | 充分（正信号/无信号判断） | 过度（投入大量工程 before 知道方向对不对） |
| 最终性能 | 低估上限 | 接近上限 |

Route A 的核心价值是**用最小代价获取信号**：上下文到底有没有用？但 Route A 冻结了 ProjectionHead，天然低估上限，因此**不能单独否决 Route B**（见 Section 7.6 决策规则）。Route A 失败只降低 Route B 的优先级，不否决。

---

## 附录 C：审查记录

| 日期 | 审查方 | 审查内容 | 修正数 |
|------|--------|---------|--------|
| 2026-04-10 | Codex review #1 | 初版方案审查 | 4 项（见下） |
| 2026-04-10 | Codex review #2 | 执行文档审查 | 8 项（5 项新修 + 3 项已在 #2 中解决） |
| 2026-04-10 | Codex review #4 | 执行细节收口 | 4 项（见下） |
| 2026-04-10 | Codex review #3 | 收敛审查 | 4 项必须改 + 4 项建议改（见下） |

**Codex 审查 #1 修正（方案设计阶段）：**

1. **Loss 范围修正：** 初版提出"对 K 内所有位置算 loss"，被指出会导致 effective class prior 失真。修正为：Route A center-only loss，Route B 改用 contig chunk 方案消除重复加权。
2. **Normalization 修正：** 初版使用 BatchNorm，被指出 padding 会污染 BN 统计。修正为 LayerNorm。
3. **Loss 组件修正：** 初版计划 v1 同时使用 BCE + contrastive + DANN，被指出相邻窗口高度相似会使 contrastive 学成局部平滑。修正为 v1 仅 BCE。
4. **两阶段定位修正：** 初版倾向于将两阶段作为长期方案，被指出冻结 256-dim 会低估上限。修正为：两阶段仅作 smoke test，正信号后尽快切端到端。

**Codex 审查 #2 修正（执行文档阶段）：**

1. **瓶颈定性修正：** Section 1 "已转移到模型表达能力问题"措辞过满。修正为承认剩余 gap 混合 representation / operating-point / geometry 三类因素，本 phase 选择先攻击因素 (1)。
2. **Checkpoint 指标修正：** bgc_binary_auroc 是 window-level 指标，可能无法捕捉 region-level 改善。新增规则 8：二次 region-level 验证机制。
3. **Route A 门槛收紧：** 原门槛使用 OR 条件，太松。修正为全部 AND：short-bucket 提升 + test_id 不退化 + test_ood 不崩 + precision 安全 + discovery 不退化。
4. **Route B chunk 边界修正：** 不重叠 chunk 会在边界重新制造截断问题。修正为重叠 chunk（stride = L_chunk/2）+ 中心区域算 loss，确保每个窗口在每个 epoch 恰好参与 loss 一次。
5. **Route B 采样一致性修正：** "按正样本密度加权采样"与"每窗口每 epoch 一次"矛盾。修正为：完整遍历所有 chunk，class balance 完全在 loss-level 处理（pos_weight / focal loss）。
6. **evaluate.py 修正：** 原文称 evaluate.py 不需要改，但上下文模型的 val eval 也需要 contig-aware prediction。新增 `evaluate_context.py` 到文件清单。
7. **valid_mask 显式输入：** 原文仅在后处理中使用 mask。修正为将 valid_mask 作为额外通道输入 ContextModule（第一层 in_channels=257），让模型显式区分真实窗口和 padding。
8. **Discovery 权重提升：** 原文将 discovery 仅放在晋升最后一步。修正为：Route A 就做 discovery smoke check，Route B 每个 candidate 同时报告 discovery 指标，晋升条件包含 discovery AND recovery。

**Codex 审查 #3 修正（收敛阶段）：**

审查提出 8 点（4 必须 + 4 建议）。其中 3 点（overlap chunk、采样一致性、discovery 提前）已在 #2 修正中解决。实际新改 4 点 + 1 点细化：

1. **Route A 不再 veto Route B：** 原文将 Route A 作为 Route B 的硬门槛。修正为：Route A 通过 → 高优先级启动 Route B；Route A 部分通过 → 仍启动 Route B；Route A 完全失败 → 仅降低优先级，不否决。理由：冻结 projector 天然低估上限。
2. **sigmoid gate 替代 α=0 初始化：** α=0 导致 ContextModule 拿不到梯度（`d_loss/d_params = α × ... = 0`），是 bug 而非安全设计。修正为 `α = sigmoid(β)`，β 初始化 -4.0（α ≈ 0.018），保证梯度始终非零。
3. **Checkpoint 选择升级为 region metric 主导：** 原文将 region-level 作为"二次验证"。修正为：val region F1 作为主指标（每 5 epoch 运行），bgc_binary_auroc 降为辅助/快速筛除。
4. **区分两种 padding 语义：** edge-padded window（Phase 1 生成，有真实 Evo2 embedding）和 chunk padding（人工零向量）语义不同。修正为显式区分 valid_window_mask，edge-padded window 标记为 valid=1。
5. **成功标准加入失败模式对齐指标：** Route B 成功标准新增 5-10kb recall、10-20kb recall 的具体阈值，评估指标新增 single-window-run recall 和 isolated OOD FP count。

**Codex 审查 #4 修正（执行细节收口）：**

1. **Route B loss_mask 定义统一：** 模型定义处 loss mask 缺少 `in_center_loss_region` 条件，与 Section 6.2/8.3 的三条件定义不一致。统一为 `loss_mask = valid_window_mask & (label != ambiguous) & in_center_loss_region`。
2. **Region-level eval 频率统一为每 5 epoch：** Route A 评估部分写"每 10 epoch"，与规则 8 的"每 5 epoch"矛盾。统一为每 5 epoch。
3. **Contig 内窗口排序显式定义：** 多处写"按窗口顺序排列"但未定义"顺序"。显式写明**按 `window_start` 升序排列**，不使用 CSV 行号或 window_id 字典序。
4. **OOD threshold policy 约束：** 新增规则：阈值只在 val 选，汇报必须同时含 test_id 和 test_ood，靠激进阈值换 ID 提升但 OOD 崩溃的模型不可晋升。
