# 学姐审查导览

> 写给准备帮我审查这份工作的学姐。这份文档是一个**导航**，帮助你用最少时间找到最需要审的东西。

---

## 0. 项目一句话

用冻结的 DNA 基础模型 (Evo2 7B) 在 ~1.06M 个基因组窗口上训练一个检测头，目标是：

1. 在已有的 antiSMASH 标注上达到可比或更好的 recovery
2. 识别 antiSMASH 遗漏的候选 BGC 区域，交湿实验验证（和 Isabella Z 合作）

当前阶段：模型部分已收敛（Phase 3 Route B），discovery 证据链在 Phase 4 刚刚跑完，**准备开始选湿实验候选**。

---

## 1. 我希望你重点审的三件事

### (A) 数据集构造的合理性

**数据核心事实：**

- 来源：GTDB r226 过滤后的 1996 个细菌 genome（按系统发育均衡抽样）
- 标注：antiSMASH 7 生成的 19,762 个 BGC 注释
- 窗口：16,384 bp 长度，8,192 bp 步长（50% overlap）
- 共 **1,064,568** 个窗口（含 edge-padded 短 contig）
- 分类：8 维多标签（7 种 BGC type + 1 个 is_negative）
- Split：**genome-level** 划分（train / val / test_id / test_ood），test_ood 是训练集未见的 phyla（主要是 Cyanobacteriota）
- 已做 GCF leakage check：0 泄漏

**我觉得你应该质疑的点：**

1. **我们用 antiSMASH 的输出作为训练 label，然后又用 antiSMASH 的输出作为 evaluation ground truth。这是循环论证的主要源头。**
   - 我已经试过用 MIBiG 做外部验证，但 Tier 1（assembly-resolved coordinate-exact matching）需要 MIBiG 的 metadata JSON，我本地没有。Tier 2（species-level matching）噪声太大。
   - → 请看 `docs/phase4-comparison-and-discovery-plan.md` §4.2 和 `docs/phase4-morning-report.md` 的 "限制" 部分
2. **"Negative" 的定义问题：** 一个 window "is_negative" 意味着它不与任何 antiSMASH BGC 重叠。但这不等于"这里没有 BGC"——只是"antiSMASH 没标注"。所以 false positive 的标签本身就有噪声。
   - → 这是我 Phase 2 做 hard-negative mining 的主因：用模型自己 flag 的高置信负样本作为"怀疑"的负样本继续训练
3. **Edge padding 可能改变分布：** 短 contig 末尾不足 16k 的部分用 N 填充生成"真实窗口"并过了 Evo2。Evo2 对 N 的处理可能和正常 DNA 不同。
   - → `docs/phase1-short-bgc-coverage.md` 有结果；`<5kb` recall 从 0 涨到 22%，所以看起来信号没被污染，但这一项还没做过仔细 ablation
4. **Test_ood 的代表性不够：** test_ood 名义上是"训练集未见 phyla"，但实际主要是 Cyanobacteriota，少量其他 phyla 样本太少不能单独报告。所以我的 "OOD generalization" claim 其实只覆盖到 "跨 Actino/Pseudo/Bacillota → Cyano"。
   - → `docs/mainline.md` §3.2 和 `docs/phase3-route-b-results.md` 里都有披露
5. **1996 个 genome 中很多是 Streptomyces：** BGC 本身在 Streptomyces 里非常稠密，负样本实际上经常"接近 BGC 边缘"。这是为什么我在 Phase 4 做了 matched null（同基因组 + GC/gene density matched）替代 random null。
   - → `docs/phase4-morning-report.md` §D2

**审查路径：**

→ 先读 [docs/mainline.md §Training data](docs/mainline.md)（总览）  
→ 然后 `docs/_archive/stage_reports/STAGE1_REPORT.md`（数据构造细节）  
→ 然后 `docs/_archive/stage_reports/STAGE2_REPORT.md`（Evo2 embedding 诊断）  
→ 如果怀疑 negative 定义：`docs/_archive/ws_execution_history/ws3_label_geometry_spec.md`

### (B) 训练流程的合理性

**当前 mainline 是 Route B（Phase 3），端到端 context model：**

```
Evo2 7B (frozen) 4096-dim embedding per window
  ↓
ProjectionHead (warm-started from H_hardneg_r10, trainable): 4096 → 2048 → 256, L2-norm
  ↓
Dilated 1D CNN (3 layers, dilations 1/2/4): 15-window receptive field = 120 kb
  ↓
GatedResidual: z_final = z_proj + sigmoid(β) * z_context, β init -4.0
  ↓
Classifier (warm-started, trainable): 256 → 8 labels
```

**训练数据流（Route B）：**

- 将每条 contig 切成长度 L_chunk=32 的 chunks
- **Overlap stride = 16**（50% 重叠）
- 每个 chunk 只在**中心区域**算 loss（`loss_mask = valid_window & non_ambiguous & in_center_loss_region`）
- 完整遍历所有 chunk（每个窗口每 epoch 恰好参与 loss 一次），**不**在 sampler 层面操控 class prior
- Class balance 完全通过 loss-level `pos_weight` 处理
- Optimizer: **AdamW with layered lr** (projector 1e-4, context 1e-3, classifier 1e-3)
- 100 epochs, cosine annealing with 5-epoch warmup

**我觉得你应该质疑的点：**

1. **Warm start 是作弊吗？** ProjectionHead 和 Classifier 都从之前的 MLP mainline (H_hardneg_r10) 热启动。理由是：不热启动的话 α=sigmoid(β) 很小（0.018），context module 需要时间 learn，classifier 会被干扰。热启动后模型从 "独立分类基线 + 微小扰动" 开始。但**这也意味着 Route B 的 recall 增益可能被 H_hardneg_r10 的 training set exposure 污染**，特别是如果 H_hardneg_r10 checkpoint 已经见过某些 test genome（不会——是 genome-level split，但值得验证）。
   - → 代码在 `bgc-gcd/train_context_e2e.py` L87-100
2. **Sigmoid gate 的初始化：** β=-4.0 意味着 α≈0.018，context module 初始贡献 ~2%。这是为了防止"α=0 会让 context module 拿不到梯度" 的 bug（早期版本踩过这个坑）。但 sigmoid gate 仍然是个 trick，可能让效果差的 context 也能通过 loss 慢慢"拉大 α" 而不回退到 0。
   - → 代码在 `bgc-gcd/model/context_model.py` `GatedResidual` 类
3. **训练/推理的 sequence length 不匹配：**
   - 训练时每个 sample 是 L=32 的 chunk
   - 推理时输入一整条 contig（1 到 1482 windows）
   - 我做过 audit（`scripts/phase4/analysis/04_audit/audit_route_b.py`），验证了 length invariance 和 batch independence，但这个 train/test gap 仍然是一个 suspicious point
4. **Checkpoint selection 用 val region F1**：不是 window-level AUROC。这是 Codex review 后升级的。每 5 epoch 跑一次 region-level eval，选最好的。
   - → 见 `docs/phase3-context-model-plan.md` §Rule 8 + `docs/phase3-context-model-plan.md` §审查记录 #3
5. **Chunk 边界处理：** Route B 的关键设计点。chunk 中心算 loss，边界只作为 context 输入。每个窗口恰好在一个 chunk 的中心区域内，避免重复加权。
   - → 详见 `docs/phase3-context-model-plan.md` §8.3

**审查路径：**

→ 先读 [docs/mainline.md](docs/mainline.md) §Current Mainline 和 §Training（2 分钟）  
→ 再读 [docs/phase3-context-model-plan.md](docs/phase3-context-model-plan.md)（20 分钟，含 4 轮 Codex 审查历史）  
→ 代码：`bgc-gcd/train_context_e2e.py` (300 行) + `bgc-gcd/model/context_full_model.py` (80 行) + `bgc-gcd/data/chunk_dataset.py` (60 行) — 加起来不到 500 行  
→ Audit 脚本：`scripts/phase4/analysis/04_audit/audit_route_b.py`

### (C) 实验结果的合理性

**当前核心 claim（按强度排列）：**

| Tier | Claim | 证据 |
|------|-------|------|
| 1 | Route B 在 antiSMASH proxy benchmark 上 test_id recall 0.820, precision 0.857, 3-seed std < 0.01 | `docs/phase3-route-b-results.md` |
| 1 | Route B **在短 BGC (<10kb) 上显著优于 GECCO**：`<5kb` 0.75 vs 0.06, `5-10kb` 0.71 vs 0.22 | `docs/phase4-morning-report.md` §1 |
| 1 | Route B 在 overall recall 上比 GECCO 高 34pp (0.82 vs 0.48) | 同上 |
| 2 | Top-50 candidates 的 signature density 在 matched null 下有 **1.79x enrichment**，Mann-Whitney U p = 1.8e-5 | `docs/phase4-morning-report.md` §D2 |
| 2 | 56% (28/50) top candidates 获得 cross-tool support from GECCO and/or DeepBGC | 同上 §D4 |
| 3 | 双轨 composite ranker 捕获了不同类型的候选：Conservative top-10 全部 canonical module complete；Novelty top-10 全部 atypical；两榜仅 1/10 重叠 | 同上 §D3 |
| 4 | Track 1 (保守) × 5 + Track 2 (探索) × 5 候选清单可以交湿实验 | 同上 §D5 |

**我觉得你应该质疑的点：**

1. **最严重的：我们的 comparison 对 GECCO 公平吗？**
   - GECCO 用默认 operating point，Route B 用 val F1-optimal threshold
   - GECCO 的训练集可能包含我们的 test genomes（未审计）
   - GECCO 的 region merging 规则与我们不同——我强制 min 3kb，GECCO 有自己的原生规则
   - `docs/phase4-comparison-and-discovery-plan.md` §4.4 明确说了"每个工具用原生 operating point，不互相调参"，但这个决定本身可以质疑
2. **Multi-tool concordance ≠ biological validation：** 我在 Phase 4 Codex 审查后已经把措辞从 "validated" 改成 "cross-tool support"，但在整个 discovery 叙事里，"多工具一致" 仍然是最硬的非实验证据，它的可信度取决于"工具间独立性"这个假设——**而这个假设是弱的**，因为 GECCO/DeepBGC 都基于 Pfam HMM 和 antiSMASH-curated 数据训练
3. **Matched null 每候选平均只有 3.5 个 control，不是 10：** GC 和 gene density 的约束太严，加上同 genome 内 BGC 稠密导致可采样区域少。这让 per-candidate z-score 只能算 18/50，统计力不足
4. **Top-50 候选只是按 max_score 选出来的，**然后在这 50 个上跑 Pfam。理论上全部 1006 个 clean-missed 都应该做 Pfam 然后再 rank。这是我知道但还没做的事（需要一次 batch hmmscan 全量跑）
5. **3-seed 稳定性是 `test_id recall std = 0.001`，看起来好到不真实：** 可能是因为 chunk 数据流 + overlap 导致 seed 之间的随机性被吸收掉。我自己都觉得这个数字需要更多 seed 验证
6. **Phase 4 的 GECCO batch 没检查 leakage：** GECCO 的 MIBiG 训练集可能覆盖了我们 test 中的一部分 genome。如果这是真的，Route B vs GECCO 的对比会偏向 Route B（因为 Route B 没见过这些 genome）

**审查路径：**

→ 先读 [docs/phase4-morning-report.md](docs/phase4-morning-report.md)（10 分钟，有一页总结）  
→ 再读 [docs/phase3-route-b-results.md](docs/phase3-route-b-results.md)（5 分钟）  
→ 然后看原始分析脚本：`scripts/phase4/analysis/README.md` 有完整导航

---

## 2. 最值得先看的 5 个文件

如果你只有 30 分钟：

1. **[docs/mainline.md](docs/mainline.md)**（5 分钟）— 当前系统的完整概要
2. **[docs/phase4-morning-report.md](docs/phase4-morning-report.md)**（10 分钟）— 最新的比较结果 + 候选清单
3. **[bgc-gcd/train_context_e2e.py](bgc-gcd/train_context_e2e.py)**（10 分钟）— 当前训练代码，不到 300 行
4. **[bgc-gcd/data/chunk_dataset.py](bgc-gcd/data/chunk_dataset.py)**（5 分钟）— 训练数据流关键
5. **[bgc-gcd/model/context_full_model.py](bgc-gcd/model/context_full_model.py)**（5 分钟）— 模型结构

---

## 3. 已知的脏东西

为了透明，我把我知道的**"如果你认真看会发现的问题"** 列出来：

### 代码层面

- `scripts/phase4/analysis/*` 里的脚本是一次性执行的，**硬编码了服务器绝对路径**。想在别的地方跑需要改路径。
- `bgc-gcd/evaluate_context.py` 的 `evaluate_region_level` 函数里复用了 `_eval_val_full_model` 的逻辑，有一些代码重复
- DeepBGC 因为网络问题我用了 antiSMASH 的 Pfam v35.0 symlink 成 Pfam v31.0——这会让 DeepBGC 的检测性能略偏（具体多少未测）
- Route B 的 `scripts/phase4/reproduce_mainline.sh` 只是 wrapper，没做参数校验

### 实验层面

- Phase 4 的 MIBiG Tier 1 external validation **没做**（Tier 2 做了但样本噪声大）
- GECCO / DeepBGC 的 training set leakage **没审计**
- Route B 3-seed 太稳定（std 0.001），**可能需要 5+ seeds 才能相信这个数字**
- Per-candidate z-score 只有 18/50 可计算
- Top-50 是 max_score 排序选的，不是 composite score 选的（composite ranker 只对这 50 个重排）

### 文档层面

- `docs/_archive/` 里的老文档和当前 mainline 的符号/术语不完全一致（比如老文档里用 H_hardneg_r10，新文档用 Route B）
- Phase 3 和 Phase 4 的"限制"部分可能有重复

---

## 4. 我希望你给的反馈

按重要性排序：

1. **数据集 + Split 的合理性**：我的 genome-level split + edge padding + hard-neg mining 有没有逻辑漏洞？
2. **训练流程**：warm start + overlap chunks + sigmoid gate 这套组合在你看来有没有明显错误或过度设计？
3. **Comparison 是否公平**：GECCO / DeepBGC 的对比有没有方法论层面可以被 reviewer 抓住的地方？
4. **Discovery 证据链能撑起论文吗**：matched null + 双轨 ranker + cross-tool concordance 这一套，对一个 bioinformatics-friendly 审稿人而言是不是够？还差什么？
5. **哪些数字你觉得 "不太真实"**：尤其是 3-seed std 和 short-BGC 的改善幅度

如果你发现了明显的 bug，**请直接骂我**。我宁愿现在被骂也不想投稿后被 reviewer 骂。

---

## 5. 一些上下文

- 我是 CS 背景，没有生物学背景。合作的生物老师是 Isabella Z（可以帮我做湿实验）和我的导师陈老师
- 这是毕业论文，目标是生物方向的发现类论文
- 时间不紧（导师没有强 deadline）
- 湿实验是计划内的，等你帮我审完我就打算拿 Track 1 + Track 2 各 2 个最强候选给 Isabella Z
- 我用的服务器是两台 node7/node8 A800（8 GPU 每台），数据在共享 `/data/muskliu/np_master/` 下。如果你要连服务器跑代码，直接找我要 SSH 配置

---

## 6. 谢谢

认真看一遍可能要 1-2 小时，有空就看，没空就挑一两条质疑的点也行。你的任何反馈都比我自己来得准。
