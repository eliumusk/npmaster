# OER004256 BGC 预测交付包

**日期**: 2026-04-23
**数据集**: OER004256（7 个宏基因组样本，过滤 contig ≥ 10 kb 后约 1.48 Gbp）
**工具**: 两套 BGC 检测系统，同一批 FASTA 输入

## 📦 文件清单

| 文件 | 内容 | 行数 | 主要用于 |
|---|---|---:|---|
| **`v41_regions_fully_typed.csv`** ⭐ | **以 v4.1 regions 为主，每行带 v4.1 type + defrag_v2 type 双标注** | 254 | **老师主要看这个——一个文件搞定** |
| `v41_regions_typed.csv` | 仅 v4.1 type 的版本 | 254 | 备份 |
| `defrag_v2_regions_typed.csv` | defrag_v2 自己检测的 regions（只 93 个，偏长 BGC） | 93 | 参考——老师大概率不需要 |
| `OER004256_wrap_up_report.json` | 全套统计摘要 | — | 快速 overview |

**核心建议**：**只要 `v41_regions_fully_typed.csv` 一个文件** 就够老师做湿实验筛选。里面 254 个候选都有两个独立工具给的 type 标签，老师筛"两个工具 agree 是 Terpene/RiPP"的子集（18 个）做 pilot 湿实验即可。

## 📝 CSV 列说明

### 公共列（两个文件都有）

| 列名 | 含义 |
|---|---|
| `genome` | shard ID（`OER004256_v41_shard00` … `shard15`，是对 7 个样本切分的 shard） |
| `contig` | contig 在原 FASTA 中的 ID，前缀说明来自哪个 sample（`OER00425668` 至 `OER00425673`） |
| `start`, `end` | BGC 区域在 contig 上的起止坐标（0-based 半开区间） |
| `max_score` | 该区域内所有窗口中的最高 BGC 置信度分数（0-1） |

### v4.1 特有列（主文件 `v41_regions_fully_typed.csv` 都有）

| 列名 | 含义 |
|---|---|
| `source_scale` | `primary`（w=8192 主档）或 `short_rescue`（w=4096 短 BGC 补档） |
| **`v41_type`** | **v4.1 预测的 BGC 类型**（Stage 2 per-type LR，7 类之一）：Alkaloid / Terpene / NRP / Polyketide / RiPP / Saccharide / Other |
| `v41_type_score` | 该 type 的置信度（0-1） |
| `v41_type_top2` | 第二可能的 type |
| `v41_type_scores` | 7 类全部分数（分号分隔） |

### defrag_v2 特有列（仅 `v41_regions_fully_typed.csv` 和 `defrag_v2_regions_typed.csv` 有）

| 列名 | 含义 |
|---|---|
| **`defrag_v2_type`** | defrag_v2 CNN 预测的 BGC 类型（**分类准确率在 9g 上更高**，82.7% vs v4.1 的 70.5%） |
| `defrag_v2_type_score`, `defrag_v2_type_top2`, `defrag_v2_type_scores` | 同上 |

### 老师怎么看两个 type 列

- **两个都 = Terpene/RiPP** → 最可靠的候选，进湿实验 pilot
- **v41_type = Terpene，defrag_v2_type = Polyketide** → 有分歧，先放入 "second tier"，优先级降低
- **两个都 = Polyketide 或 NRP** → 长 BGC，不是湿实验目标但可以做 secondary 验证

在本数据集上两工具 type **一致率 87.4%（222/254 regions）**，大部分候选双方判断是一致的。

## 🎯 两个工具的区别 & 选谁

| 维度 | v4.1 | defrag_v2 |
|---|---|---|
| 架构 | Evo2-7B mean-pool + 逻辑回归 + 多尺度 ensemble | Evo2-7B + ContextFullModel (CNN) |
| 最小检测 BGC 长度 | **~5 kb**（有 w=4096 短档） | ~20 kb（硬下限，结构上无法检测更短的） |
| 9-genome benchmark（held-out）dedup_F1 | 0.488 | 0.439 (9g-dev 调参后) / 0.270 (honest 不调参) |
| 分类准确率（9-genome 验证，2026-04-23 修正后） | top-1 70.5%, top-any 82.7% | **top-1 82.7%, top-any 97.1%** ★ |
| Terpene F1 (9-genome) | 0.94 | **0.97** |
| RiPP F1 (9-genome) | 0.81 | **0.83** |
| Polyketide F1 (9-genome) | 0.60 | **0.84** |
| 适合场景 | 短 BGC 检测（唯一能覆盖 <20 kb）、更多候选 | 长 BGC + 更准的分类（type 可信度更高） |

**Detection 和 Classification 是两回事**：
- **Detection（找 BGC 在哪）**: v4.1 更强，尤其短 BGC（5-10 kb 专有覆盖）
- **Classification（BGC 是哪类）**: defrag_v2 更强，**9-genome 实测 top-1 82.7% vs v4.1 的 70.5%**

推荐用法：**v4.1 给候选池，defrag_v2 给 type 标签**（交叉使用；或至少参考 defrag_v2 的 type，当作 v4.1 type 的校准）。两套 type 在 OER004256 上的分布修正后其实**高度一致**（都 PKS-dominant，Alkaloid 都近 0）。

**推荐老师**:
- **Detection（BGC 在哪）**：以 `v41_regions_typed.csv` 为主，coverage 更全，短 BGC 能看到
- **Classification（BGC 是哪类）**：参考 `defrag_v2_regions_typed.csv` 的 type 列——top-1 82.7% 比 v4.1 的 70.5% 更准
- **交叉验证**：在 v4.1 给出的候选中，去 defrag_v2 的同区域查 type，两个工具都指向 Terpene/RiPP 的候选最可靠

## 📊 数据分布

### v4.1 (254 regions)

**按长度**：
| bucket | 数量 |
|---|---:|
| <5 kb | 3 |
| 5-10 kb | 85 |
| 10-20 kb | 147 |
| 20-50 kb | 19 |
| ≥50 kb | 0 |

**按类型**：
| Type | 数量 | 占比 |
|---|---:|---:|
| Polyketide | 150 | 59% |
| NRP | 78 | 31% |
| Terpene | 17 | 7% |
| RiPP | 4 | 1.6% |
| Other | 4 | 1.6% |
| Saccharide | 1 | — |
| Alkaloid | 0 | — |

### 重叠情况（IoU ≥ 0.3）

- **两个工具都检测到**: 58 个区域
- **仅 v4.1 检测到**: 196 个（多数是 5-20 kb，defrag_v2 结构上覆盖不到）
- **仅 defrag_v2 检测到**: 36 个（多数是 20-50 kb 长 BGC）

## 🧪 湿实验候选推荐（短 Terpene + RiPP）

在 `v41_regions_fully_typed.csv` 中筛选 **v4.1 和 defrag_v2 都判断为 Terpene 或 RiPP** 的候选，共 **18 个高置信候选**：
- Terpene: 14 个（全部 score 0.99+）
- RiPP: 4 个（全部 score 0.98+）

下面展示的是老师应该重点关注的候选清单：

**Top 10 Terpene 候选**（按 score 降序，score ≥0.99）：

| genome | contig | start | end | len (bp) | score |
|---|---|---:|---:|---:|---:|
| shard05 | OER00425672\|k141_7824444 | 0 | 10,794 | 10,794 | 1.000 |
| shard11 | OER00425677\|k141_7499930 | 10,240 | 15,649 | 5,409 | 1.000 |
| shard03 | OER00425668\|k141_12279675 | 14,336 | 19,468 | 5,132 | 1.000 |
| shard15 | OER00425677\|k141_2104096 | 4,096 | 10,771 | 6,675 | 1.000 |
| shard08 | OER00425668\|k141_6355510 | 3,072 | 10,424 | 7,352 | 0.9998 |
| shard03 | OER00425668\|k141_6166673 | 6,144 | 11,228 | 5,084 | 0.9997 |
| shard15 | OER00425670\|k141_8003444 | 10,240 | 21,363 | 11,123 | 0.9997 |
| shard02 | OER00425672\|k141_9874813 | 9,216 | 16,415 | 7,199 | 0.9996 |
| shard09 | OER00425672\|k141_4462833 | 4,096 | 16,384 | 12,288 | 0.9994 |
| shard02 | OER00425670\|k141_9213366 | 4,096 | 18,432 | 14,336 | 0.9993 |

**全部 4 个 RiPP 候选**（全部 score ≥0.98）：

| genome | contig | start | end | len (bp) | score |
|---|---|---:|---:|---:|---:|
| shard02 | OER00425671\|k141_8264069 | 8,192 | 13,924 | 5,732 | 0.9991 |
| shard01 | OER00425670\|k141_8138120 | 4,096 | 10,033 | 5,937 | 0.9989 |
| shard07 | OER00425671\|k141_4883135 | 9,216 | 14,516 | 5,300 | 0.9942 |
| shard08 | OER00425670\|k141_6055645 | 3,072 | 10,090 | 7,018 | 0.9819 |

## ⚠️ 诚实说明的已知局限

1. **训练标签来源**：v4.1 和 defrag_v2 的训练标签都 **完全来自 antiSMASH**（在 1188 个其他基因组上跑 antiSMASH 得到的 region + type 标注）。这意味着：
   - 模型本质上学的是"antiSMASH 的规则"，在新数据上是"老师（antiSMASH）的学生"
   - 能找出 antiSMASH 也能找到的 BGC 是预期的
   - "找出 antiSMASH 漏掉的新 BGC" 理论上可能，但**不保证**——需要湿实验验证

2. **9-genome 验证结果（held-out，两模型都没调参）**：
   - v4.1: dedup_F1 **0.488**（CI95 0.345-0.609），分类 top-1 **70.5%**
   - defrag_v2: honest dedup_F1 0.270；调参后 0.439
   - 这些是 benchmark 级别的参考值，实际 OER004256 上准确率**未知**

3. **defrag_v2 的分类列（`defrag_v2_type`）2026-04-23 新做了验证**：9-genome vs antiSMASH top-1 82.7%。**比 v4.1 的分类头（top-1 70.5%）更准**。老师如果要用 type 信息，defrag_v2 的 type 列可信度更高。

4. **两个工具 type 分布实际一致性好**（2026-04-23 修正 class-order bug 后）：
   - 两个工具都说 Polyketide 是主导类，NRP 次之，Terpene 第三
   - 两个工具都说 Alkaloid 近 0、Saccharide 近 0
   - v4.1 叫 150 Polyketide；defrag_v2 叫 49 Polyketide（defrag_v2 region 数少，比例上差不多）
   - **NRP 与 Polyketide 之间有相互混淆**——两个工具都有——这是 PKS-NRPS hybrid 生物学模糊带来的真实问题，不是 bug

5. **建议 wet-lab pilot 用法**：
   - 先看 v4.1 的 Terpene + RiPP 候选（因 v4.1 对短 BGC 敏感，且两工具都对 Terpene、RiPP 预测可靠）
   - 交叉验证用 defrag_v2 type：在 `v41_regions_typed.csv` 中标记为 Terpene 的候选里，去 `defrag_v2_regions_typed.csv` 查同区域 defrag_v2 type 是否也 = Terpene → 两工具一致的才进 top-N

5. **坐标约定**：start-end 是 **0-based half-open**（Python/BED 风格），不是 1-based full-closed（GenBank 风格）。用 antiSMASH 或其他工具交叉验证时注意转换。

6. **没有 antiSMASH 对比数据**：OER004256 上的 antiSMASH **还没跑全**（只有 OER00425669 的 part00+part01，另外 6 个样本没动）。如果老师需要 antiSMASH vs v4.1 的对照表，需要先跑 antiSMASH（估计 10-20 小时/样本 × 7 样本）。

## 🔬 给老师的操作建议

1. **先看 `v41_regions_typed.csv`**，按 `v41_type_score` 降序排 Terpene + RiPP 子集，选 top 5-10 做初步分析
2. **关注 `contig` 列前缀**，可以推断候选来自哪个样本（OER00425668, 70, 71, 72, 73, 77 等）
3. **回到原 FASTA** `/data/syh/OER0042/OER004256/{sample}.final.contigs.fa` 按 contig ID 提取原序列（或请我们提取）
4. **交叉验证工具**：手动跑 antiSMASH（`--hmmdetection-strictness loose` 多抓点）、BLAST 核心酶（terpene cyclase / RiPP precursor + modifying enzymes）、查 MiBIG 里同类 cluster 的保守域

## 📂 对应的原始数据路径（服务器）

| 路径 | 内容 |
|---|---|
| `/data/syh/OER0042/OER004256/` | 原 FASTA（7 个样本） |
| `/data/muskliu/npmaster/experiments/OER004256_compare/v41_shards_gt10kb_16/` | 过滤后切好的 16 个 shard FASTA |
| `.../v41_primary_w8192_s2048_embeddings/` | Evo2 embedding 缓存（主档，5.89 GB） |
| `.../v41_short_w4096_s1024_embeddings/` | Evo2 embedding 缓存（短档，12 GB） |
| `.../v41_primary_preds/`, `.../v41_short_preds/` | v4.1 per-window BGC 分数 |
| `.../defrag_v2_preds/` | defrag_v2 per-window 预测 |
| `.../v41_regions/`, `.../defrag_v2_regions/` | region 级 CSV（带和不带 type 两份） |

如老师需要其他切片（比如单独某个 type 的 top-N、某个 sample 的所有候选、按置信度过滤），随时再提。

---

## 🔧 变更历史

**2026-04-23**: 修复了一个 class-ordering bug。之前 defrag_v2 的 per-window CSV 里 `pred_class` 字符串和推理脚本里 `DEFAULT_CLASS_NAMES` 的 8 类顺序是**错的**（假设成 `[Alkaloid, Terpene, NRP, Polyketide, RiPP, Saccharide, Other, non-BGC]`，实际是 `[Polyketide, NRP, Terpene, RiPP, Saccharide, Alkaloid, Other, non-BGC]`）。影响：
- `defrag_v2_regions_typed.csv` 里 `defrag_v2_type` 列（在 2026-04-23 之前的版本）**全是错的**——Polyketide 被标成 Alkaloid，NRP 被标成 Terpene 等
- `defrag_v2_preds/*.csv` 里 `pred_class` 列也是错的（但 `probs` 数值本身是正确的，只是列名对不上）
- `v41_regions_typed.csv` **不受影响**（v4.1 type 走的是 Stage 2 per-type LR，每个 LR 文件有显式命名，不依赖 column order）

**当前（交付）版本已经全部修正**——新的 `defrag_v2_type` 列用正确的 class order，与 antiSMASH 9-genome 验证匹配（top-1 82.7%）。原始 probs 数值一直正确，这次只是重新打了标签。
