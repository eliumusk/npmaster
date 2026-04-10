# Phase 4 执行计划：比较基线 + 可交付湿实验的 novel discovery

**日期:** 2026-04-10  
**状态:** 待执行  
**前置依赖:** Phase 3 Route B 已锁定为主线  
**论文定位:** 生物学方向的发现类论文（非方法论 workshop）  
**最终目标:** 交付一组**高置信度的 novel BGC 候选**给湿实验团队（Isabella Z），并在生物论文中建立可辩护的发现叙事

---

## 1. 目的

Phase 1-3 已解决方法论层面的问题：Route B 相比基线在 recovery 和 short-BGC 两个维度上都有明显提升，3-seed 稳定性达到一个数量级改善。但这还不够做一篇**生物发现类论文**。

原因在于：**"recover antiSMASH annotations better" 本身不是生物发现。** 对细菌自然产物领域的读者而言，能 recover antiSMASH 自己的标注无法证明科学价值——antiSMASH 已经在了，它能找到的我们找到只是 parity。真正能支持生物论文的是这两件事：

1. **竞争基线层面的诚实对比**：在统一评估协议下，与 antiSMASH、GECCO、DeepBGC 做公平比较，证明 Route B 在某些维度上有**明确的、可辩护的优势**。这是论文 methodology 部分的基础，也是 reviewer 要求的标准对照。
2. **Novel discovery 叙事的扎实化**：识别出 antiSMASH 遗漏的、有**多重独立证据**支持的候选 BGC 区域，建立 wet-lab handoff 标准，交付 top-N 候选给湿实验。这是论文的 main story。

本 phase 的核心判断是：**模型部分已经够用，下一步的瓶颈在评估协议和证据链，不在模型架构。**

---

## 2. Phase 结构

本 phase 分两个并行但解耦的工作流：

```
WS4A: Comparative Baselines（比较基线）
  └── P0: 锁定 Route B 作为主线基础设施
  └── P1: MIBiG 外部验证
  └── P2: GECCO / DeepBGC 安装与运行
  └── P3: 统一评估协议 + 对比表
  └── P4: 分 phylum / 分 bucket 的诚实对比

WS4B: Discovery Narrative（发现叙事）
  └── D1: 诊断当前 discovery 证据为什么弱
  └── D2: 更严格的 null distribution 构造
  └── D3: Composite candidate ranking（非单一 score）
  └── D4: Multi-tool concordance 作为硬证据
  └── D5: Top-N candidate deep dive（gene-level 注释）
  └── D6: Wet-lab handoff 规范 + 交付包
```

**WS4A 是 WS4B 的前置条件**：D4（multi-tool concordance）需要 GECCO 和 DeepBGC 的输出。但 WS4A 内部的 P1-P4 和 WS4B 的 D1-D3 可以并行。

---

## 3. 主线基线锚定

所有 Phase 4 实验必须与以下固定主线对比。**本 phase 不改变模型**。

| 条目 | 值 |
|------|---|
| 模型 | Phase 3 Route B 3-seed |
| Checkpoints | `best_route_b_s{0,1,2}.pt` |
| 代码 | `bgc-gcd/train_context_e2e.py` + `model/context_full_model.py` |
| 评估脚本 | `scripts/phase3/eval_route_b_full.py`（新）|
| test_id recall | 0.820 ± 0.001 |
| test_id P(pred) | 0.857 ± 0.002 |
| test_ood recall | 0.594 ± 0.006 |
| test_ood P(pred) | 0.653 ± 0.009 |
| Sig@20 / Sig@50 | 90% / 92% |
| diversity@50 | 45 profiles |

**Rule**: 本 phase 不改 Route B 的权重、训练流程、推理路径。所有评估都在上面固定的 3-seed checkpoints 上进行。

---

## 4. WS4A：比较基线工作流

### 4.1 P0: 锁定 Route B 主线基础设施

**目标：** 让 Route B 成为一个"按一个键就能复现所有数字"的主线系统。这是后续一切对比的锚点。

**操作：**
1. 创建 `bgc-gcd/configs/mainline_route_b.yaml` 固定所有超参
2. 创建 `scripts/phase4/reproduce_mainline.sh` — 一条命令复现 3-seed 训练
3. 创建 `scripts/phase4/eval_mainline.sh` — 一条命令复现所有评估数字
4. **H_hardneg_r10 freeze but keep accessible**：保留在原路径，只做只读锁定（不改文件，不写入），**暂不归档**。Phase 4 可能需要频繁做"Route B vs H_hardneg_r10"的对比（discovery pool 对比、per-bucket 对比）。**归档推迟到 WS4A + WS4B 定稿之后。**
5. 在 `docs/mainline.md` 加上版本 tag（`v3.0 Route B`），但 `docs/_archive/` 里暂不移入 H_hardneg_r10

**交付物：**
- 可复现脚本
- `docs/mainline.md` 更新
- Route B 和 H_hardneg_r10 的并列可访问性

**验收标准：**
- 用 `reproduce_mainline.sh` 从头训练得到的指标与当前 3-seed 记录误差 < 0.005
- 新人拿到仓库能在 30 分钟内跑出一个评估结果
- H_hardneg_r10 的 checkpoint / discovery pool 仍可在 1 条命令内被重新加载做对比

---

### 4.2 P1: MIBiG 外部验证

**目标：** 在一个**非 antiSMASH 生成的 gold standard** 上评估 Route B，消除"自我验证"嫌疑。

**背景：** 当前所有 recall/precision 数字都是对比 antiSMASH 的标注。这对 ML 评估是合理的，但对生物论文的读者而言"antiSMASH 能找到的我们也能找到"并不 impressive。MIBiG 是 curated 的已知 BGC 数据库，里面每一条都有文献/实验支持——这是真正的 gold standard。

**操作：**
1. 拉取 MIBiG v3.0 数据库（~2500 条 curated BGCs + 每条的 source 信息）
2. **严格匹配协议**（这是本任务的核心，不能含糊）：
   - **Tier 1（assembly-resolved entries）**：MIBiG 条目直接对应到一个 GenBank accession，且该 accession 在我们的 1996 个 genome 中（strain 级别匹配）。这类条目有精确 coordinate。
   - **Tier 2（species-level approximate）**：MIBiG 条目只有 species 或 genus 信息，没有直接 accession。这些条目**不放入主评估**，只作为参考。
   - **Tier 3（no coordinate）**：MIBiG 条目缺少 start/end 坐标的，**直接剔除**。
3. 只用 Tier 1 entries 作为外部 benchmark。预期数量会比 MIBiG 总量小很多（可能 50-200 条，取决于 overlap）
4. 对落在 test_id + test_ood 中的 Tier 1 条目，计算 Route B 的 recall
5. **样本量披露**：如果 Tier 1 交集太小（< 30），在论文中显式说明"MIBiG external benchmark limited by sample size"，不做统计推断
6. 按 BGC 类型 / 长度 / phylum 分别报告
7. 做与 antiSMASH 的对比：MIBiG Tier 1 条目中，哪些是 antiSMASH 也找到的 / 只有我们找到的 / 只有 antiSMASH 找到的 / 都没找到的

**关键问题要回答：**
- Tier 1 MIBiG 条目数是多少？在 test split 里落了几个？
- Route B 在这些 Tier 1 条目上的 recall 是多少？
- 是否存在"MIBiG 有但 antiSMASH 没标注到"的条目？我们能找到这些吗？
- 短 BGC 在 Tier 1 中的分布是什么？（如果样本量允许）

**诚实披露（论文要写清的事）：**
- MIBiG 本身不是"对齐到我们 assembly 的 coordinate truth"，它是 cluster-level curated entry
- 我们只用了 Tier 1 (assembly-resolved) 子集，其余做参考
- 如果 Tier 1 样本量小（预期很可能），这是一个"support 证据"而非"决定性 benchmark"

**交付物：**
- `docs/phase4-mibig-external-validation.md`
- `/data/muskliu/np_master/data/phase4/mibig_eval/` 完整 artifact

**为什么重要：** 这是论文 Result 部分的一个独立小节——"我们在外部 labeled set 上的表现"。没有这个，reviewer 会说"你只是 recover 了你自己的训练标签"。

---

### 4.3 P2: GECCO / DeepBGC 安装与运行

**目标：** 跑出两个主流 DL BGC 预测工具的结果，作为对比基线。

**背景：** 老 archive 明确写过"GECCO/DeepBGC 未装"。这是论文 reviewer 一定会问的盲点。

**操作：**
1. **GECCO 安装**：`conda install -c bioconda gecco` 或直接 pip 安装。测试 1 个 genome 跑通
2. **DeepBGC 安装**：`pip install deepbgc`，下载模型权重，测试 1 个 genome 跑通
3. 对 test_id + test_ood 的全部 genome 批量运行（两者都支持直接输入 FASTA）
4. 解析输出为统一格式：`{genome_id, contig_id, start_bp, end_bp, predicted_type, score}`
5. 保存到 `/data/muskliu/np_master/data/phase4/baselines/{gecco,deepbgc}/`

**预期运行时间：**
- GECCO：~5-20 分钟/genome × 640 genomes（test_id+test_ood）= 几十小时，可并行
- DeepBGC：类似量级

**交付物：**
- 两个工具对所有 test genome 的预测结果（统一 JSON）
- 安装 + 运行脚本归档
- 跑通说明文档

**风险：**
- GECCO/DeepBGC 的训练数据可能与我们的 test genome 有交叉。需要检查并在论文里披露
- 两个工具的输出 BGC 定义可能与我们不完全一致（如最小长度、merge 策略）。需要在评估层面统一

---

### 4.4 P3: 统一评估协议 + 对比表

**目标：** 把 antiSMASH / GECCO / DeepBGC / Route B 放在同一把尺子下比较。这是论文的一张核心表。

**Benchmark 选择协议（不是"ground truth 选择"，因为我们没有真正的 ground truth）：**

1. **Proxy benchmark — antiSMASH annotations**
   - 这是一个 proxy，不是 gold standard
   - 用途：给 GECCO/DeepBGC/Ours 一个共同参照物，antiSMASH 自己不参与此榜
   - 局限：antiSMASH 可能漏标，所以"漏标 = FN"的推断需要小心
2. **External curated benchmark — MIBiG-mapped loci (Tier 1 only)**
   - 这是最接近 gold standard 的证据源
   - 用途：所有方法（含 antiSMASH）都在这里被评估
   - 局限：样本量可能小，需要披露
3. **Supporting analysis — cross-tool agreement set**
   - antiSMASH、GECCO、DeepBGC 三方一致同意的区域
   - **绝对不作为 ground truth**，只作为"high-confidence agreement set"
   - 用途：分析 Route B 在强一致区域上的行为（是否遗漏了大家都同意的东西），以及 Route B 的"only us" 调用的分布
   - **论文叙事里也绝对不能把这个叫 ground truth** —— 那会变成循环论证：把其他 baseline 的输出当作评判 baseline 的裁判

**Operating point 合同（对比公平性的核心）：**

这是对比表能不能过 reviewer 的关键。必须一次性锁定：

| 维度 | 决策 |
|------|------|
| **antiSMASH** | 使用 default parameters（`--taxon bacteria`）+ default BGC call。**不重新调阈值**，因为 antiSMASH 是 rule-based，没有可调 operating point |
| **GECCO** | 使用 default threshold（package 默认值）。记录默认值，报告时写清楚 |
| **DeepBGC** | 使用 default threshold（package 默认值）。记录默认值 |
| **Route B** | 使用 val F1-optimal threshold（与 Phase 3 一致），每个 seed 独立选阈值，**报告 3-seed mean ± std**。不做 ensemble |
| **Region merge** | 每个工具用**自己的原生 region output**，不做二次 merge。只有 Route B 用我们的 merge（相邻 window → region → min 3kb） |
| **Minimum region length** | 每个工具用自己的原生最小长度。**不统一到 3kb**，因为这样会砍掉 GECCO/DeepBGC 的合法短 call |
| **Coordinate normalization** | 所有工具输出统一为 `(genome_id, contig_id, start_bp, end_bp, predicted_type)` 格式，start/end 统一为 0-based half-open |

**关键决定：每个工具用自己的原生 operating point，不互相调参。** 理由：
1. 不同工具的 score 含义不同，强行统一阈值会系统性歧视某一方
2. 在 val 上 tune 每个工具的 operating point 可能是 fair comparison 的最严格做法，但这需要逐工具重 tune，超出本 phase 范围
3. 原生 operating point + 诚实披露 = 可辩护的 comparison

**Matching 规则（对所有方法统一，这一项必须一致）：**

- Overlap fraction ≥ 0.5，denominator = min(pred_len, gt_len)
- 与 ws1_score_swap.py 一致
- 每个 ground truth BGC 最多匹配一个预测（greedy by overlap）

**评估指标：** recall, prediction-level precision, per-bucket recall, per-type recall

**对比表设计（分两个 benchmark，每个一张表）：**

**Table 1 — On antiSMASH annotations (proxy benchmark)**

| 方法 | test_id recall | test_id P(pred) | test_ood recall | test_ood P(pred) | <5kb recall | 5-10kb recall | 10-20kb recall |
|------|---------------|-----------------|-----------------|------------------|-------------|---------------|----------------|
| GECCO | ? | ? | ? | ? | ? | ? | ? |
| DeepBGC | ? | ? | ? | ? | ? | ? | ? |
| **Route B (ours, 3-seed)** | 0.820 ± 0.001 | 0.857 ± 0.002 | 0.594 ± 0.006 | 0.653 ± 0.009 | 0.751 ± 0.004 | 0.706 ± 0.003 | 0.691 ± 0.005 |

*antiSMASH 不参与本表（自我参照）。*

**Table 2 — On MIBiG Tier 1 external benchmark**

| 方法 | Tier 1 recall | Tier 1 precision | overlap-only antiSMASH | 短 BGC sub-recall |
|------|--------------|------------------|------------------------|-------------------|
| antiSMASH | ? | ? | (self) | ? |
| GECCO | ? | ? | ? | ? |
| DeepBGC | ? | ? | ? | ? |
| **Route B (ours)** | ? | ? | ? | ? |

**Table 3 (supporting only) — Cross-tool agreement set breakdown**

- "3-way agreement set"：antiSMASH ∩ GECCO ∩ DeepBGC 三方都同意的区域数量
- Route B 在这个 set 上的 recall（作为参考，非 ground truth）
- "Only Route B" candidates 分布（这里是 discovery section 的材料）

**诚实披露：**
- GECCO/DeepBGC 训练数据是否与我们的 test 有重合（要查阅文档）
- Operating point 是 native，不是 re-tuned
- MIBiG Tier 1 的样本量限制

**交付物：**
- `docs/phase4-comparison-table.md` — 完整对比
- 对比表的 JSON 版本（方便后续做图）

**验收标准：**
- Route B 至少在 **一个维度** 上对其他所有方法有**统计显著**的优势（我的先验猜测：短 BGC 是最可能的）
- 如果没有，诚实报告，论文叙事调整为 "complementary" 而非 "superior"

---

### 4.5 P4: 分 phylum / 分 bucket 的诚实对比

**目标：** 避免 overall 数字掩盖 phylum-specific 或 bucket-specific 的差异。

**操作：**
1. test_ood 拆开：Cyanobacteriota 单独报告，其他 phylum 合并
2. 每个方法在每个 bucket × phylum 上的 recall 单独报告
3. 标注样本量小（<20 BGCs）的 cell，避免基于少量样本的结论

**为什么重要：** 论文的诚实性。如果 Route B 只在 Cyanobacteriota 上赢 GECCO，那就说"在 Cyanobacteriota OOD setting 下 Route B 有优势"，不要说"在 OOD 上普遍更好"。

---

## 5. WS4B：Discovery Narrative 工作流

### 5.1 D1: 为什么当前 discovery 证据弱

**问题陈述：**

Route B 的 discovery benchmark 数字如下：

| 指标 | Route B | Null distribution | Enrichment |
|------|---------|-------------------|------------|
| biosynthetic Pfam @20 | **100%** | 95% | 1.1x |
| biosynthetic Pfam @50 | 100% | 96% | 1.0x |
| sig precision @20 | 90% | 65% | 1.4x |
| sig precision @50 | 92% | 72% | 1.3x |
| sig diversity @50 | 45 profiles | 26 profiles | 1.7x |

**这里有一个严重问题：生物合成 Pfam 在细菌基因组里几乎无处不在（SDR、SAM methyltransferase、AMP-binding 都是"公用工具基因"）。** 所以 "biosynthetic Pfam @K = 100%" 几乎没有信息量——random negative 区域也有 96%。

即便是更严格的 "signature precision"（KS/AT/ACP/C/A/PCP/TE 这些核心结构域），null distribution 也有 65%——这说明我们当前的 null 太强（可能采样到了富含生物合成基因的 Streptomyces 非 BGC 区域）。

**核心诊断：当前 discovery 测试的 signal-to-background ratio 只有 1.1-1.4x，这不足以支持"我们找到了 novel BGC"的 claim。**

**为什么 null 太强？三个可能原因：**

1. **采样偏差**：当前 null 是 `sample_null_regions()` 从"训练集 negative window" 里随机取。但训练集的 "negative" 定义是 "这个 window 不与已知 BGC overlap"，不等于"这个 window 没有生物合成基因"。在 Streptomyces 这种 BGC 稠密的 genome 里，很多 "negative" window 就在 BGC 边缘或跨界区。
2. **长度匹配但未结构匹配**：null region 和 candidate 一样长，但不保证基因密度、GC 含量、物种背景一致
3. **Pfam 粒度太粗**：使用的是 "biosynthetic Pfam" 列表（数百个 family），不是只在 BGC 中富集的"签名 family"

### 5.2 D2: 更严格的 null distribution 构造

**目标：** 让 null distribution 真正代表 "BGC-like but not BGC" 的区域。

**新 null 构造方案：**

```
对每个 candidate region c（genome g, contig t, start s, end e, length L）:
  采样 10 个 null control region，每个满足：
    1. 来自同一 genome g
    2. 长度等于 L（严格）
    3. GC content 在 c 的 ±2% 范围内
    4. Gene density（基因数/kb）在 c 的 ±20% 范围内
    5. 与任何 antiSMASH BGC 距离 > 50kb（减少 antiSMASH 已标注区域的污染）
    6. 与任何 Route B high-score call 距离 > 20kb（避免"我们自己漏掉的漏标"污染）
    7. 与任何 GECCO / DeepBGC call 距离 > 20kb（避免其他工具漏标污染）
    8. 与 c 本身不重叠
  **保留全部 10 个 controls**，不做随机取 1
```

**与初版的关键差别：**
1. **Null 排除多工具的 call**，不只是 antiSMASH。否则 antiSMASH 漏标但 GECCO 或 Route B 已经识别出的真 BGC 会污染 null
2. **保留所有 10 个 controls**，后续做分布比较而非点比较

**分布比较而非点比较：**

每个 candidate 对应 10 个 matched controls，所以 50 个 candidates 对应 500 个 controls。比较方式：
- **per-candidate z-score**：`(candidate_sig_density - mean(its 10 controls)) / std(its 10 controls)`
- **aggregate distribution**：500 controls 的 signature density 分布 vs 50 candidates 的分布
- **enrichment metric**：`median(candidates) / median(controls)`，以及 Mann-Whitney U test 的 p-value
- **这样更稳健**：避免单个 control 随机性影响结论

**额外控制：**
- **Positive upper bound**：把已知 antiSMASH BGC 作为"已知 BGC"组。如果我们的 candidate 的 signature 接近或超过这个上界，这是最强的证据。
- **Candidate 自身的 Pfam density / kb**：不只看 "有没有"，看密度

**新 discovery 对比表：**

| Group | sig domain / kb | module completeness | sig diversity |
|-------|-----------------|--------------------|--------------| 
| Known antiSMASH BGCs (positive upper bound) | ? | ? | ? |
| Route B candidates (clean_missed) | ? | ? | ? |
| Matched null (same-genome, GC, density-matched) | ? | ? | ? |
| Random null (old definition) | ? | ? | ? |

**我们希望看到的**：candidates 的 signature density 在 positive upper bound 附近，明显高于 matched null。只要在 matched null 上 enrichment > 3x，discovery claim 就成立。

### 5.3 D3: Composite Candidate Ranking（双轨：保守 vs 新颖）

**问题：** 当前 ranking 只用 `max_score`，不反映区域整体证据。

**关键设计决定：双轨 ranker，不是单一 composite score。**

单一 composite 的问题：如果用 "known antiSMASH BGC rank 高于 matched null" 作为 tune 的目标，composite 会被推向 canonical BGC 的形态——这和 discovery 主目标（找**新颖**的东西）有张力。最有新意的候选往往是非典型架构，单一 composite 反而会把它们压下去。

所以拆成两条，分别服务不同目的：

---

#### D3-A: Conservative ranker（wet-lab safety 优先）

**目标：** 给湿实验的"保守轨"选候选——结构完整、证据充分、失败率低。

**Components:**

1. **Model confidence**：mean(top-3 window scores)
2. **Signature density**：signature domain 个数 / kb
3. **Module completeness (binary)**：
   - NRPS：≥1× Condensation + ≥1× Adenylation + ≥1× PCP
   - PKS (type I)：≥1× KS + ≥1× AT + ≥1× ACP
   - PKS (type II)：≥1× KS_α + ≥1× KS_β (CLF)
   - Terpene：≥1× Terpene_synth 或 1× SQS_PSY
   - RiPP：≥1× radical SAM + 1× precursor region
4. **Signature diversity**：不同 signature family 的个数
5. **Contig quality**：不在 contig 末端 20kb 内
6. **Type-signature coherence**：模型预测类型与 signature 是否一致

```
conservative = 0.2 × model_conf
             + 0.3 × log(sig_density + 1)
             + 0.3 × module_completeness_binary
             + 0.1 × log(sig_diversity + 1)
             + 0.05 × contig_quality_binary
             + 0.05 × type_coherence_binary
```

**初版权重偏向 canonical BGC architecture**，目的是给湿实验相对安全的选择。

---

#### D3-B: Novelty-supporting ranker（exploratory）

**目标：** 不押宝"像已知 BGC"，让非典型架构也能浮出来。作为湿实验的**探索轨道**。

**Components（故意与 conservative 拉开差异）：**

1. **Model confidence**：mean(top-3 window scores)
2. **Enzymatic density**：enzymatic Pfam 个数 / kb（**不限定** signature list，接受广义酶）
3. **Cluster compactness**：ORF 之间的间距呈"compact cluster"结构（基因密集 + 同向转录）
4. **Novelty distance**：距离最近 antiSMASH BGC（越大越加分，200kb 饱和）
5. **Low canonical similarity**：signature 分布与已知 NRPS/PKS/Terpene/RiPP 都不完全匹配反而加分：`1 - max(similarity_to_each_canonical_type)`
6. **Contig quality**：不在 contig 末端 20kb 内

```
novelty = 0.25 × model_conf
        + 0.2 × log(enz_density + 1)
        + 0.15 × cluster_compactness
        + 0.2 × sigmoid(novelty_distance / 50kb)
        + 0.15 × (1 - max_canonical_similarity)
        + 0.05 × contig_quality_binary
```

**关键：novelty ranker 故意不包含 `module_completeness`**。如果候选已经有完整 NRPS/PKS 模块，conservative ranker 会捕获它。novelty ranker 要找的是"看起来像酶 cluster 但不完全像已知类型"的候选。

---

#### Ranking 的验证方式（避免循环论证）

**不要用 "known antiSMASH BGCs rank 高于 null" 来 tune 权重。** 这会把 ranker 推向 canonical，与 discovery 目标冲突。

正确的验证方式：

1. **Conservative ranker 的 sanity check**：在已知 antiSMASH BGC 上排序，前 10% precision 应 ≥ 95%。**这只是确认 ranker 没 bug，不是权重调参目标**。权重由先验固定。
2. **Novelty ranker 的独立验证**：比较 "novelty top 10% of unmatched candidates" 与 "random selection of unmatched candidates" 在 multi-tool concordance 比例上的差异。如果 novelty ranker 确实给那些多工具 agree 的候选更高分，说明它捕获了"像 BGC 但不是已知 canonical" 的信号。
3. **两 ranker 的交集分析**：`top-K conservative ∩ top-K novelty` 的候选是**双重证据支持**，优先交湿实验。`top-K novelty only` 的候选是"可能的惊喜"，交湿实验的探索轨道。
4. **权重是预注册的**：在看到 test 结果之前，权重就固定为上面初版值。不做事后调参。

### 5.4 D4: Multi-tool Concordance 作为"Strong Cross-Tool Support"

**注意措辞：这是 cross-tool support，不是 hard biological evidence。**

antiSMASH、GECCO、DeepBGC 都是工具输出，**不是独立的生物真值源**：
- 它们的训练集可能有重叠（都基于 MIBiG / antiSMASH-curated 数据）
- 它们的 inductive bias 可能相似（都基于 biosynthetic Pfam / HMM / DL features）
- 多工具一致 ≠ 生物学意义上的真 BGC

所以 concordance 是**"同时代工具的一致性"**，不是"独立验证"。湿实验才是真正的生物验证。

**目标：** 用 cross-tool support 作为 candidate ranking 的一个附加证据层，帮助筛选 top candidates。**不作为论文里的"硬证据"**。

**方法：** 对每个 Route B 的 unmatched candidate，检查 GECCO 和 DeepBGC 在同一区域是否也预测了 BGC。

**Cross-tool support 等级定义：**

| Level | 定义 | 证据解读 |
|-------|------|---------|
| **Tier A** | antiSMASH 漏 + GECCO 和 DeepBGC 都 flag（overlap ≥ 0.5）| Strong cross-tool support：三个独立方法都认为是 BGC。**不意味着一定是真 BGC**，但失败率低 |
| **Tier B** | antiSMASH 漏 + GECCO 或 DeepBGC 其中一个 flag | Moderate cross-tool support：两个方法 agree |
| **Tier C** | antiSMASH 漏 + 只有我们 flag + 结构证据强（signature density、module completeness）| Structural-only support：单工具 + 结构证据。这里可能包含真正新颖的发现，也可能是 FP |
| **Tier D** | antiSMASH 漏 + 只有我们 flag + signature 稀疏 | Weak：可能是 FP，不交湿实验 |

**论文措辞的修正**（这决定 reviewer 能不能接受）：
- ❌ "M of these (Tier A) are **validated** by independent methods" — 错：Tier A 不是 validation
- ❌ "confirmed as BGCs by multiple tools" — 错：工具一致不是 confirmation
- ✅ "M of these (Tier A) receive **strong cross-tool support** (concordantly flagged by GECCO and DeepBGC)" — 对
- ✅ "We emphasize that cross-tool agreement is **not independent biological validation**; wet-lab experiments provide the ground truth" — 对

**Tier 在湿实验选择中的角色：**
- Tier A/B：高置信，作为湿实验"保守轨"
- Tier C：结构证据强但只有我们 flag，作为湿实验"novelty 探索轨" — 这些可能是真正"只有我们能看到"的候选，不应一律砍掉（见 5.6 D6）

### 5.5 D5: Top-N Candidate Deep Dive

**目标：** 论文的 memorable 部分。从 composite ranking + Tier A concordance 的交集中选 **top 5 candidates**，对每一个做 gene-level 深度注释。

**每个 top candidate 产出一个 page 的分析：**

1. **区域信息**：genome + 物种 + contig + 坐标 + 长度
2. **预测类型**：Route B 预测的 BGC 类型 + 置信度 + Composite score
3. **Gene prediction**：Prodigal 注释的 ORF + 方向 + 长度
4. **Signature hits**：每个 ORF 的 Pfam domains（尤其 signature 的）
5. **Module architecture**：NRPS A-C-PCP 或 PKS KS-AT-ACP 等的排列图
6. **Synteny 图**：基因布局的可视化（类似 antiSMASH 的 cluster layout）
7. **Multi-tool agreement**：antiSMASH（漏）、GECCO、DeepBGC 的 call 状态
8. **Novelty 分析**：距离最近已知 BGC 多远？是否在 BGC hotspot 区？
9. **推测产物类型**：基于 signature pattern 猜测可能的化学类别（polyketide / NRPS / hybrid / terpene / RiPP / other）
10. **提供给湿实验的具体建议**：哪个 ORF 是核心催化、用什么 primer 扩增、可能的生物合成产物 class

**这是论文的 Fig 3/4/5，也是答辩时展示"我们的工作有生物意义"的载体。**

### 5.6 D6: Wet-lab Handoff 规范 + 双轨交付包

**目标：** 给 Isabella Z 一个标准化的交付包，让她能直接启动湿实验。

**关键设计：双轨交付，不是单一"最保守的 top-N"。**

单轨保守筛选（要求 Tier A/B + module complete + top 10% conservative score）会系统性排除 "只有我们能看到" 的候选——而这些恰恰是 discovery 叙事中最有新意的部分。论文的核心叙事是 "Route B 找到了别人漏掉的东西"，如果我们只交那些 GECCO/DeepBGC 也 agree 的候选，论文就无法讲这个故事。

所以拆成两轨：

---

#### Track 1: Conservative wet-lab set（保守轨）

**目的：** 最大化湿实验成功率。这些候选湿实验如果成功，证明 Route B 找到的东西是真的 BGC。

**Handoff criteria（全部 AND）：**

| Criterion | 阈值 |
|-----------|------|
| Cross-tool support tier | 至少 Tier B（Route B + 至少一个其他 DL 工具 agree） |
| Conservative composite | top 10% of all unmatched candidates |
| Module completeness | 必须至少一个完整模块 |
| Novelty distance | 距离最近 antiSMASH BGC > 20kb |
| Contig quality | 不在 contig 末端 20kb 内 |
| Organism accessibility | 该 genome 的物种能被湿实验团队拿到 |
| Region length | 20-100 kb（可 PCR / 可克隆的范围） |

**数量目标：** top 5-8 candidates

---

#### Track 2: Novelty-first exploratory set（探索轨）

**目的：** 捕获"只有 Route B 能看到"的真正新颖候选。湿实验成功率预期低于 Track 1，但一旦成功，生物论文的影响力远大于 Track 1。

**Handoff criteria（全部 AND）：**

| Criterion | 阈值 |
|-----------|------|
| Cross-tool support tier | Tier C 也接受（只有 Route B flag，但结构证据强）|
| **Novelty composite** | top 5% of all unmatched candidates（按 novelty ranker）|
| **Structural evidence required** | signature density > matched null p95 **或** module completeness true |
| Signature density enrichment vs matched null | per-candidate z-score > 3 |
| Novelty distance | 距离最近 antiSMASH BGC > 50kb（比 Track 1 更严）|
| Contig quality | 不在 contig 末端 20kb 内 |
| Organism accessibility | 该 genome 的物种能被湿实验团队拿到 |
| Region length | 20-100 kb |

**数量目标：** top 3-5 candidates

**关键差别**：Track 2 不要求 module completeness（可能是非典型架构），但要求 per-candidate z-score > 3 和 novelty distance > 50kb。

---

#### 交付给 Isabella Z 的 package

给 Isabella Z 的不是一个列表，是**两个列表 + 明确的标注**：

- "**Track 1: Conservative candidates**" — 5-8 个，预期成功率高
- "**Track 2: Novelty-first exploratory**" — 3-5 个，预期成功率低但 discovery upside 大
- 每个候选独立评估生物可行性（物种可获得性、PCR 可行性等）
- 让 Isabella Z 从两轨中**各选 1-2 个**作为第一批湿实验对象，不是只从 Track 1 中选

这样即使 Track 2 全部失败，Track 1 的结果也能支撑论文的 Section "Route B recovers BGCs missed by other tools"。而如果 Track 2 有一个成功，论文的 Section "Route B discovers structurally novel BGCs" 就有了 wet-lab support——这是论文的 killer feature。

**交付包内容（每个 top candidate 一个 zip）：**

```
candidate_{rank}_{genome}_{start}-{end}/
  README.md              # 概要 + 推测产物
  region.fasta           # 区域 DNA 序列
  proteins.faa           # Prodigal 预测的蛋白
  domains.json           # Pfam hits with coordinates
  module_architecture.png  # 模块图
  synteny.png            # 基因布局
  multitool_status.json  # 各工具的 agreement
  evidence_report.pdf    # 一页综合分析（D5 产出）
  suggested_experiments.md  # 具体实验建议
```

**与 Isabella Z 的对接 milestone：**
1. **第一轮**：交付 top 5 候选，请她从生物可行性角度反馈（哪个物种好拿、哪个区域好 clone）
2. **筛选后**：确定 1-3 个优先候选，进入湿实验流程
3. **结果**：湿实验结果反馈后，作为论文 validation section 的核心

---

## 6. Phase 4 成功标准

### WS4A 成功标准

| 条件 | 阈值 |
|------|------|
| MIBiG Tier 1 external recall | Tier 1 样本量 + Route B recall + 所有方法的 recall 对比 |
| GECCO/DeepBGC 运行完成 | 所有 test genome（test_id + test_ood，共约 640 个）跑通 |
| 对比表完整 | 两个 benchmark (antiSMASH proxy + MIBiG Tier 1) × 四个方法 × 所有 bucket 的完整数据 |
| Operating point 合同公开 | 每个工具的 operating point（native vs tuned）在论文 methods 中清晰披露 |
| 至少一个维度上的明确优势 | Route B 在（短 BGC / OOD / 某 bucket）上对其他方法有**可辩护的**优势 |

**WS4A 不追求"赢所有"**。如果 Route B 只在短 BGC 上赢，那就诚实说"Route B has a clear advantage on short BGCs, is competitive elsewhere"。

### WS4B 成功标准

| 条件 | 阈值 |
|------|------|
| Matched null 构造完成 | 满足 same-genome + GC + density-matched + 排除所有工具 call + 每候选保留 10 个 controls |
| Candidate enrichment（分布级别）| Route B candidates 的 signature density 中位数 vs matched null 中位数：enrichment ≥ 2x，Mann-Whitney U p < 0.01 |
| Per-candidate z-score | ≥ 30% 的 unmatched candidates 有 z-score > 2 vs 自己的 10 个 controls |
| Conservative ranker sanity | 在已知 antiSMASH BGC 上排序，conservative ranker 前 10% precision ≥ 95% |
| Novelty ranker 独立验证 | novelty top 10% 的 multi-tool concordance 比例显著高于 random top 10%（p < 0.05）|
| Track 1 候选数量 | 至少 5 个 Tier A/B + module complete 的候选 |
| Track 2 候选数量 | 至少 3 个 Tier C + novelty z-score > 3 的候选 |
| Top deep dive | 8 个候选（5 Track 1 + 3 Track 2）的完整分析报告产出 |
| Wet-lab handoff 包 | 双轨交付包已产出，Isabella Z 可启动 |

---

## 7. 执行顺序与并行度

```
Week 1 (P0, D1):
  - P0 锁定 Route B 基础设施（0.5 day）
  - D1 诊断 discovery 当前证据弱的原因（0.5 day）
  - 启动 P2 的 GECCO/DeepBGC 安装（可能需要调 env, 1-2 day）

Week 2 (P1, D2 并行):
  - P1 MIBiG 外部验证（2 days）
  - D2 构造 matched null（2-3 days，需要解析 Prodigal 输出）
  - P2 GECCO/DeepBGC 在 test genome 上批量运行（后台跑，2-3 days wall clock）

Week 3 (P3, D3):
  - P3 统一评估协议 + 初版对比表（2 days）
  - D3 Composite ranking 实现 + ranking sanity check（2 days）

Week 4 (P4, D4):
  - P4 分 phylum / bucket 的诚实对比（1 day）
  - D4 Multi-tool concordance 分析（2 days）

Week 5 (D5, D6):
  - D5 Top-5 candidate deep dive（3-4 days，包括 synteny 图和 module 分析）
  - D6 Wet-lab handoff 包整理 + 与 Isabella Z 对接（1-2 days）
```

**并行建议：** WS4A（P0-P4）和 WS4B 的 D1-D3 可以并行推进。但 D4（multi-tool concordance）必须等 WS4A 的 P2 完成。D5-D6 必须等 D4 完成。

---

## 8. 明确非目标

本 phase **不做**以下事项：

- **不改 Route B 的权重或训练流程**。本 phase 只做评估和分析
- **不尝试 Route B v2 改进**（contrastive/DANN/focal loss）。那是 Phase 5 的事
- **不做 LoRA 微调 Evo2**。同上
- **不扩大数据集**（不加新 genome、不加新 phylum）
- **不做湿实验本身**（但要为湿实验做充分的交付准备）
- **不追求在所有指标上 beat 所有 baseline**。诚实对比优先于"赢所有"

---

## 9. 风险与对策

| 风险 | 可能性 | 影响 | 对策 |
|------|--------|------|------|
| GECCO/DeepBGC 在我们的 test genome 上跟 Route B 持平或更好 | 中 | Tier 2 claim 削弱 | 诚实报告，重点放在"complementary" 和 "短 BGC 优势"上 |
| GECCO/DeepBGC 训练集与我们的 test 有交叉 | 高 | 对比不公平 | 在论文中披露，做交叉过滤后再比 |
| Matched null 构造后 enrichment 仍然弱 | 中 | Tier 3 claim 削弱 | 诊断是 signature 选择问题还是模型问题，可能需要调 signature list |
| Tier A candidates 很少（< 5） | 中 | 可交付湿实验的候选不足 | 放宽到 Tier B + module complete，或扩大候选池 |
| MIBiG 条目与 test genome 匹配度低 | 低 | P1 结论样本量小 | 预先评估匹配度，必要时扩大到 train genome 上报告 |
| Isabella Z 那边 wet-lab 启动慢 | 高 | 论文最后一步卡住 | 提前对接，先给 top candidates 让她筛选可行性 |

---

## 10. 最终决策问题

本 phase 结束时，项目需要能回答三个问题：

1. **比较层面：** 在公开的 operating point 合同下，Route B 相对 antiSMASH / GECCO / DeepBGC 有什么**具体的、可辩护的**优势？（答案必须指向特定维度，如 "short BGC" 或 "某 phylum"，而不是 vague 的 "更好"）

2. **发现层面：** 我们识别的 unmatched candidates 中：
   - 相对 matched null（排除其他工具 call），signature density 和 module completeness 是否显著升高？
   - 有多少有 cross-tool support（Tier A/B）？
   - 有多少 novelty-first candidates（Tier C + z-score > 3）？
   - 这些证据能否在论文里以**"strong support but not independent validation"** 的语气讲清楚？

3. **交付层面：** 我们能给湿实验团队交付：
   - 多少 Conservative Track 候选（Tier A/B + module complete）？
   - 多少 Novelty-First Track 候选（Tier C + 结构证据强）？
   - 这两轨加起来能否覆盖"recover + discover" 双重叙事？

---

## 附录 A：生物论文的叙事骨架（预估）

| Section | 内容 | 对应 Phase 4 产出 |
|---------|------|-------------------|
| Introduction | BGC discovery 背景、antiSMASH 的局限、DNA 基础模型的新机会 | — |
| Methods | Route B 架构 + 训练 + **operating point 合同** + 评估协议 | P0 + P3 + mainline.md |
| Results 1: Recovery benchmark (proxy) | 在 antiSMASH annotations 下 GECCO/DeepBGC/Route B 的 recall/precision | P3 + P4 |
| Results 2: External benchmark (MIBiG Tier 1) | 所有方法在 curated external set 上的表现 | P1 |
| Results 3: Comparative baseline | 重点短 BGC 和 OOD | P2 + P3 + P4 |
| Results 4: Discovery statistics | Matched null (分布比较) 下 Route B 的 signature enrichment；cross-tool support breakdown | D1 + D2 + D4 |
| Results 5: Cross-tool agreement set (supporting) | 作为 supporting analysis，不是 ground truth | P3 |
| Results 6: Novel candidate deep dive | Top candidates 的 gene-level 分析（两轨）| D5 |
| Results 7: Wet-lab validation | Isabella Z 的湿实验结果 | 湿实验（本 phase 外）|
| Discussion | 方法的 complementary value、cross-tool support 的局限、短 BGC 机制、局限 | — |
| Conclusion | 多少 candidate 进入湿实验（两轨分别）、几个 validated | — |

**关键：Section 4-7 是论文的"心脏"，这是 Phase 4 必须交付的核心内容。**

---

## 附录 B：为什么不现在就做湿实验

湿实验是**验证步骤**，不是**探索步骤**。现在直接做湿实验的问题：

1. 我们的候选列表用 `max_score` ranking，不一定对应最好的生物候选
2. 我们没有 multi-tool 证据，选出的 top-5 可能全是假阳性
3. 没有 matched null 对照，不知道我们的候选相对背景有多强的信号
4. 没有 module-level 证据，不知道这些候选的生物合成功能完整性

**正确的顺序是：先把证据链做扎实，然后用严格的 handoff criteria 选出真正有希望的候选，再做湿实验。** 这样湿实验成功率高，也能在论文里讲清楚"为什么选这几个而不是别的"的方法论。

---

## 审查记录

| 日期 | 审查方 | 审查内容 | 修正数 |
|------|--------|---------|--------|
| 2026-04-10 | Codex review #1 | 初版 Phase 4 规划 | 8 项（见下） |

**Codex 审查 #1 修正：**

1. **Consensus BGCs 不作为 ground truth：** 原文把 "antiSMASH ∩ GECCO ∩ DeepBGC" 当作三套 ground truth 之一，是循环论证。修正为：只有两个 benchmark（antiSMASH proxy + MIBiG Tier 1），cross-tool agreement set 降级为 "supporting analysis"，论文叙事里明确不称为 ground truth。
2. **Operating point 合同明确化：** 原文只说了 overlap 和 min length，没定义 operating point / region merge / seed 策略。修正为：每个工具用 native operating point（不再 tune），Route B 报告 3-seed mean ± std 不做 ensemble，每个工具用自己的原生 region output 不做二次 merge。
3. **MIBiG 匹配协议严格化：** 原文说"按 genus/species/strain 匹配"过粗。修正为三级 tier：Tier 1 assembly-resolved（主评估）、Tier 2 species-level approximate（参考）、Tier 3 no coordinate（剔除）。只用 Tier 1 做主评估，样本量小时显式披露。
4. **Multi-tool concordance 从"硬证据"降为"strong cross-tool support"：** 原文说 "hard evidence"，过满。修正为 "cross-tool support"，明确工具间 agreement 不是独立生物验证，论文措辞调整避免 "validated"/"confirmed"/"independent" 等词。
5. **Composite ranking 拆成双轨：** 原文单一 composite + "用 known BGC 验证"，会导致 ranking 偏向 canonical。修正为 Conservative ranker（偏 canonical，给 wet-lab safety）+ Novelty ranker（偏非典型，给 discovery upside），权重预注册，不做事后调参，验证方式分别独立。
6. **Matched null 反污染加强 + 分布比较：** 原文只排除 antiSMASH regions、"10 选 1 随机"。修正为：同时排除 Route B 高分 call、GECCO call、DeepBGC call（避免漏标污染）；保留全部 10 个 controls 做分布比较（per-candidate z-score + Mann-Whitney U test）。
7. **Wet-lab handoff 双轨：** 原文要求 Tier B 才能进湿实验，会系统性排除"只有我们能看到的 novel"。修正为双轨交付：Conservative Track（Tier A/B + module complete，5-8 个）+ Novelty-First Track（Tier C + novelty z-score > 3，3-5 个），Isabella Z 从两轨各选 1-2 个。
8. **H_hardneg_r10 推迟归档：** 原文 P0 立即归档旧主线，过早。修正为：P0 只做 freeze + keep accessible，不移入 `_archive/`。Phase 4 需要频繁做"新旧 discovery pool 对比"，归档推迟到 WS4A + WS4B 定稿之后。
