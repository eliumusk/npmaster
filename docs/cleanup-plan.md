# npmaster 清理重构方案（交接文档）

**创建**：2026-04-18
**状态**：阶段 0 完成，阶段 1 待开始
**说明**：这是一次性重构交接文档，完成后和本清理过程一起删除。任何新 Claude 会话读完本文即可无缝接手。

---

## 背景：为什么要做这次重构

项目演进了多个阶段（stage1/2/3/phase1-4），代码和文档堆积。当前状态：
- 后处理参数优化已到边际收益：`stride8192_tuned_v2` R=0.5311, P=0.2513, F1=0.3412，已超过 DeepBGC/antiSMASH，追不上 BGC-Prophet
- 下阶段主攻：**hard-negative mining + 数据侧改造 + 窗口/模型参数 grid**
- 当前代码臃肿（`bgc-gcd/` 下一堆死代码，`scripts/phase4/` 18 个脚本，`docs/` 18 份阶段报告）
- 服务器上 73GB 数据，其中 ~37GB 是历史产物可删

## 已完成（阶段 0）

- ✅ GitHub 仓库改名 `np_master → npmaster`（https://github.com/eliumusk/npmaster）
- ✅ 本地目录改名 `/Users/musk/Documents/code/np-master → npmaster`
- ✅ git remote URL 已更新为 `https://github.com/eliumusk/npmaster.git`
- ✅ Claude memory 已迁移到 `~/.claude/projects/-Users-musk-Documents-code-npmaster/`
- ✅ 工作树干净，无未推送 commit（最新 `b58d3e8 调整后处理参数让precision上升一下`）
- ⚠️ 服务器还**完全没动**：`/data/muskliu/np_master/` 还是旧名字、散码、不是 git 仓库

---

## 目标结构

```
npmaster/
├── README.md                         # 重写：当前状态 + 如何跑
├── configs/
│   ├── mainline.yaml                 # 从 bgc-gcd/configs 提到根
│   ├── route_b_hardneg.yaml          # 原 mainline_route_b.yaml
│   └── operating_points.yaml         # ★ 新：冻结的两个 operating point
├── src/                              # ★ 重命名自 bgc-gcd/（去掉 GCD 遗留）
│   ├── model/                        # 只保留 context_full_model + context_model + projector
│   ├── data/                         # chunk_dataset + context_dataset
│   ├── train.py                      # = 当前 train_context_e2e.py
│   ├── evaluate.py                   # = 当前 evaluate_context.py
│   └── utils.py
├── scripts/
│   ├── build_chunks.py               # 来自 scripts/phase3/
│   ├── inference_stride.py           # = run_nine_genome_stride_inference.py
│   ├── postprocess_v2.py             # 冻结，读 operating_points.yaml
│   ├── evaluate_nine_genome.py       # = eval_nine_genome_operating_point.py
│   ├── mine_hard_negatives.py        # = mine_route_b_hardneg.py
│   └── baselines/
│       ├── parse_gecco.py
│       ├── parse_deepbgc.py
│       └── compare_results.py        # = p3_compare.py
├── tests/
│   └── test_nine_genome_tools.py
└── docs/
    ├── evaluation-protocol.md        # 合并 threshold-contract + nine-genome-evaluation-contract
    ├── operating-points.md           # 两个冻结点的说明（参数在 configs/operating_points.yaml）
    ├── mainline.md                   # 保留并更新为当前状态
    ├── route-b-hardneg-v1.md         # 下阶段进行中
    └── next-phase.md                 # 新：hardneg + grid 计划
```

---

## 工作流（重构后）

1. **本地 = 代码唯一源**，在 `/Users/musk/Documents/code/npmaster/` 开发
2. **服务器 git clone**：`/data/muskliu/npmaster/` 是这个仓库的 checkout，只 `git pull` + 跑实验，**严禁 SSH 上去手改代码**
3. **数据留服务器**：`data/` `checkpoints/` `wandb/` `results/` 全部 gitignore，本地仓库永远不含这些
4. **结果回流**：需要分析时 `rsync pull` 小文件（JSON、log、metric）回本地 `analysis/` 目录（gitignore）

---

## 阶段 1：参数固化（下一步，**删 sweep 脚本前必做**）

**问题**：两个冻结 operating point 的精确参数**目前只存在于 docs md 里**，没进任何代码。sweep 脚本一删，复现链就断。

**动作**：

1. 新建 `configs/operating_points.yaml`：

```yaml
# 九基因组 region-level 评估（50% IoU overlap）冻结 operating points
# 所有点都在 H_hardneg_r10_s1 checkpoint 上、stride=8192 下产出

val_frozen_fused:
  # 正式汇报点：在 internal val 上选参数，9-genome 做冻结评测
  description: "internal val tuned, generalization-safe"
  stride: 8192
  score: fused
  start_threshold: 0.98
  extend_threshold: 0.95
  max_gap: 0
  length_cap: 50000
  min_support_windows: 1
  metrics:
    recall: 0.5934
    precision: 0.1649
    f1: 0.2580

tuned_v2:
  # 诊断上限点：直接在 9-genome 上 sweep 出的最强参数，仅用于表征模型上限
  description: "9-genome tuned, diagnostic upper bound only — NOT for formal claims"
  stride: 8192
  score: max_type
  start_threshold: 0.95
  extend_threshold: 0.95
  max_gap: 0
  length_cap: 30000
  min_support_windows: 3
  metrics:
    recall: 0.5311
    precision: 0.2513
    f1: 0.3412
```

2. 修改 `scripts/phase4/postprocess_v2.py`：让它支持从 `configs/operating_points.yaml` 读配置（加一个 `--operating-point {val_frozen_fused,tuned_v2}` 参数，也保留当前的命令行覆盖）。

3. 验证：用新的 yaml-driven 入口跑一次九基因组评估，产出数字和上表 `metrics` 一致。

4. 参数来源（若 yaml 里的数字和 doc 对不上，以 doc 为准）：
   - `docs/nine-genome-postprocess-v2.md`（tuned_v2）
   - `docs/score-family-diagnosis.md`（val_frozen_fused 的 score=fused 选择）
   - 如果 doc 不清楚，去服务器 `/data/muskliu/np_master/data/phase4/postprocess/sweep_results.json` 和 `sweep_val*.json` 查实际最优参数

5. 完成后 commit："feat(configs): freeze operating points into yaml, prep for sweep cleanup"

---

## 阶段 2：本地代码重构（分 3 个原子 commit）

### Commit A：死代码删除

**bgc-gcd/ 下可删**（零活跃引用或被更新版本取代，证据见下方"审计详情"）：

```
bgc-gcd/train.py                    # Phase 1/2 老版本，被 train_context_e2e 取代
bgc-gcd/train_context.py            # Route A 旧实现，已废弃
bgc-gcd/evaluate.py                 # Phase 1/2 评估脚本
bgc-gcd/model/full_model.py         # BGCModel (Phase 1) 依赖 DANN，废弃
bgc-gcd/model/classifier.py         # MultiLabelClassifier 仅被 full_model 用
bgc-gcd/model/dann.py               # DomainClassifier 仅被 Phase 1/2 用
bgc-gcd/losses/classification_loss.py  # MultiLabelBCE 只被 train.py 用，route_b 用内置 BCEWithLogitsLoss
bgc-gcd/losses/contrastive_loss.py  # MultiLabelSupCon 仅 train.py
bgc-gcd/losses/dann_loss.py         # DANNLoss 仅 train.py
bgc-gcd/losses/__init__.py          # 整个 losses/ 全废弃
bgc-gcd/losses/                     # 空目录删掉
bgc-gcd/data/dataset.py             # BGCDataset 被 ChunkDataset 取代
bgc-gcd/data/sampler.py             # BalancedBatchSampler 仅老版本用
bgc-gcd/data/augmentation.py        # EmbeddingAugmentation 仅老版本用
bgc-gcd/configs/mainline.yaml       # 老版本，mainline_route_b.yaml 才是当前主线
bgc-gcd/configs/_archive/           # 历史 yaml 归档
```

**保留并搬到 src/**：

```
bgc-gcd/train_context_e2e.py        → src/train.py
bgc-gcd/evaluate_context.py         → src/evaluate.py
bgc-gcd/utils.py                    → src/utils.py
bgc-gcd/model/context_full_model.py → src/model/context_full_model.py
bgc-gcd/model/context_model.py      → src/model/context_model.py
bgc-gcd/model/projector.py          → src/model/projector.py  (ProjectionHead 还在用)
bgc-gcd/model/__init__.py           → src/model/__init__.py  (改只导出仍保留的)
bgc-gcd/data/chunk_dataset.py       → src/data/chunk_dataset.py
bgc-gcd/data/context_dataset.py     → src/data/context_dataset.py
bgc-gcd/data/__init__.py            → src/data/__init__.py  (同上改)
bgc-gcd/configs/mainline_route_b.yaml → configs/mainline.yaml  (重命名：这才是真正的 mainline)
bgc-gcd/README.md                   → 删除（内容合并进根 README）
```

**其他根目录删除**：

```
REVIEW.md                           # 历史 review，价值已被新 audit 覆盖
BGC-Prophet.pdf                     # 38MB 参考论文，移出仓库（可留在本地 Papers/ 目录但不进 git）
conversation-2026-04-13-115139.txt  # 临时聊天记录
.DS_Store                           # macOS 垃圾
_archive/                           # 整个 _archive 目录删（17G 在服务器，本地应该也有一份）
```

### Commit B：scripts/ 精简

**保留（CORE + BASELINE，7 个）**：

```
scripts/phase3/build_chunks.py                                   → scripts/build_chunks.py
scripts/phase4/run_nine_genome_stride_inference.py              → scripts/inference_stride.py
scripts/phase4/postprocess_v2.py                                → scripts/postprocess_v2.py
scripts/phase4/eval_nine_genome_operating_point.py              → scripts/evaluate_nine_genome.py
scripts/phase4/mine_route_b_hardneg.py                          → scripts/mine_hard_negatives.py
scripts/phase4/analysis/01_baselines/parse_gecco.py             → scripts/baselines/parse_gecco.py
scripts/phase4/analysis/01_baselines/parse_deepbgc.py           → scripts/baselines/parse_deepbgc.py
scripts/phase4/analysis/01_baselines/p3_compare.py              → scripts/baselines/compare_results.py
```

**删除（FROZEN，产出已固化进 docs/yaml）**：

```
scripts/eval/                                # 整个 p1/p2/ws 早期脚本目录
scripts/phase3/context_predict.py            # 被 inference_stride 取代
scripts/phase3/extract_projections.py        # route_a 遗物
scripts/phase4/sweep_internal_val_postprocess_v2.py
scripts/phase4/sweep_nine_genome_postprocess_v2.py
scripts/phase4/nine_genome_sweep_s1.py
scripts/phase4/postprocess_sweep.py
scripts/phase4/diagnose_score_families.py
scripts/phase4/plot_nine_genome_case.py
scripts/phase4/eval_mainline.sh
scripts/phase4/reproduce_mainline.sh
scripts/phase4/analysis/02_discovery/        # d1-d5 整个目录，结果已在 docs
scripts/phase4/analysis/03_helpers/
scripts/phase4/analysis/04_audit/
scripts/phase4/analysis/README.md
scripts/_archive/                            # 整个归档目录
```

### Commit C：docs/ 合并 + 根 README 重写

**新建 3 份**：

1. `docs/evaluation-protocol.md`（合并 `threshold-contract.md` + `nine-genome-evaluation-contract.md`）
   - window score 定义（score_maxtype vs fused）
   - 阈值选择策略
   - 9-genome 4-set 评估设计
   - region-level 50% IoU 协议

2. `docs/operating-points.md`
   - 两个冻结点的说明、使用场景、复现命令
   - 精确参数指向 `configs/operating_points.yaml`

3. `docs/next-phase.md`
   - hard-negative mining 路线
   - 数据构造侧 grid 计划（窗口长度、stride、negative ratio）
   - 评估协议保持不变

**保留并更新 2 份**：

- `mainline.md`：更新为当前状态（H_hardneg_r10_s1 + tuned_v2 数字 + 下阶段指向）
- `route-b-hardneg-v1.md`：保持原样，进行中

**删除**：

```
docs/_archive/                                        # 整个归档目录
docs/phase1-short-bgc-coverage.md
docs/phase2-ood-robustness.md
docs/phase3-context-model-plan.md
docs/phase3-route-a-results.md
docs/phase3-route-b-results.md
docs/phase4-morning-report.md
docs/phase4-comparison-and-discovery-plan.md
docs/nine-genome-execution-checklist.md
docs/nine-genome-evaluation-audit.md
docs/nine-genome-baseline-factor-analysis.md
docs/nine-genome-postprocess-v2.md                    # 参数搬到 operating_points.yaml
docs/score-family-diagnosis.md                        # 结论已合并
docs/discovery-benchmark.md                           # 下阶段不做 novel discovery
docs/stability-3seed.md                               # 关键数字合并进 mainline.md
docs/threshold-contract.md                            # 合并进 evaluation-protocol.md
docs/nine-genome-evaluation-contract.md               # 同上
docs/route-b-hardneg-v1.md                           # 保留
docs/nine_genome_case_GCA_000568915.1.json           # 一次性诊断数据
docs/figures/nine_genome_case_GCA_000568915.1.png    # 一次性图
```

**根 README.md 重写要点**：
- 项目目的：BGC detection + discovery，Evo2 7B + context model，antiSMASH 训练，9-genome 评估
- Quick start：三行命令（训练、评估、推理）
- 关键数字：当前 tuned_v2 指标 + baseline 对比表
- 目录结构说明
- 去掉所有 phase/stage 历史叙事

### .gitignore 补充

```
# 数据和产物（服务器独有）
data/
checkpoints/
wandb/
results/
*.pt
*.npy
*.npz

# 临时文件
.DS_Store
__pycache__/
*.pyc
.ipynb_checkpoints/
analysis/                    # 本地拉回的分析结果
```

---

## 阶段 3：服务器切换 git（需要 SSH）

SSH 配置：`Voc_gpu_node7_a800` (port 50000) 和 `Voc_gpu_node8_a800` (port 50010)，两节点共享 NFS 挂载 `/data/muskliu`，改一次等于两台都改。

步骤：

1. **备份**（给自己留退路）：
```bash
ssh Voc_gpu_node7_a800 "cd /data/muskliu && tar czf _backup/np-master-pre-cleanup-20260418.tar.gz --exclude='data/' np_master/"
```
（只压缩代码，不压数据——数据 73G 不打包）

2. **保留数据目录，单独搬**：
```bash
ssh Voc_gpu_node7_a800 "mv /data/muskliu/np_master/data /data/muskliu/_np_master_data_tmp"
```

3. **删旧代码 + git clone 新仓库**：
```bash
ssh Voc_gpu_node7_a800 "rm -rf /data/muskliu/np_master && cd /data/muskliu && git clone https://github.com/eliumusk/npmaster.git"
```

4. **把数据搬回来**：
```bash
ssh Voc_gpu_node7_a800 "mv /data/muskliu/_np_master_data_tmp /data/muskliu/npmaster/data"
```

5. **验证能跑**：
```bash
ssh Voc_gpu_node7_a800 "cd /data/muskliu/npmaster && python scripts/evaluate_nine_genome.py --operating-point val_frozen_fused"
```
产出数字应与 `configs/operating_points.yaml` 里的 metrics 一致。

6. **确认 OK 后删备份**（数周后）：
```bash
ssh Voc_gpu_node7_a800 "rm /data/muskliu/_backup/np-master-pre-cleanup-20260418.tar.gz"
```

---

## 阶段 4：服务器数据瘦身（73G → 28G）

**每项执行前先 `du -sh` 确认大小、md5 抽查关键文件，分批 rm。**

| 路径 | 释放 | 验证前提 |
|---|---|---|
| `data/_archive/` | 17.0G | 列 README 或 ls 确认是历史归档，无下阶段引用 |
| `data/embeddings/embeddings.npy` + `nll_features.npy`（根旧版） | 14.5G | mtime 04-08 早于 edge_padding/embeddings_merged.npy 04-09；extraction_summary.json 确认是 pre-edgepad 版本 |
| `data/embeddings/edge_padding/padded_{embeddings,nll_features}.npy` | 1.8G | 已合入 merged 版本 |
| `data/phase3/route_a/` | 1.2G | route_a 路线已放弃 |
| `data/expanded_strep/` | 0.92G | 早期 Streptomyces 窗口实验，已被全基因组训练取代 |
| `data/{antismash_pilot, genomes_pilot, genome_scan, multigenus, diagnostic_day1}/` | ~0.56G | 早期 pilot，stage1 前产物 |
| `data/windows/{windows_metadata.csv, _cap065.csv, _cap08.csv, edgepad.csv}` | ~485M | edgepad_filtered.csv 已取代 |
| `data/stage4_results/p2_t2_hard_neg/` | 129M | 首代 hardneg，被 phase5 route_b_hardneg_v1 继承 |

**必须保留**（下阶段核心依赖）：
- `data/embeddings/edge_padding/embeddings_merged.npy`（16.3G）+ `nll_features_merged.npy`
- `data/antismash_results/`（13G，原始注释，hardneg 回溯用）
- `data/genomes/`（2.5G，FASTA 源）
- `data/windows/edge_padding/padded_sequences.fasta` + `_edgepad_filtered.csv` + `*_windows.txt`
- `data/stage3_results/H_hardneg_r10{,_s1,_s2}/`（主线 checkpoint）
- `data/phase3/route_b/`（hardneg v1 直接延伸）
- `data/phase4/{baselines,postprocess,nine_genome_sweep_*,mibig_eval,d*}`
- `data/phase5/route_b_hardneg_v1/`
- `data/stage4_results/{pfam_validation,p2_t2_hardneg_r10_*,p1_t1_audit,p2_stability_*}`
- `data/bac120_metadata.tsv.gz`

**需先核对再删**：
- `nine_genome_sweep_muskliu_s1/` 是否和 `nine_genome_sweep_s1/` 完全重复（md5 核）
- `_archive/stage3_old/{G_edgepad,H_hardneg}/` 的 train_log CSV 建议抽出保留（< 20K each）再删 ckpt

---

## 审计详情（供追溯）

### 代码依赖审计结论

主线 `train_context_e2e.py` + `evaluate_context.py` 的最小生存集：
- `model/context_full_model.py`（主模型）
- `model/context_model.py`（DilatedContextCNN, GatedResidual；ContextBGCModel 已废弃但文件共享）
- `model/projector.py`（ProjectionHead）
- `data/chunk_dataset.py`（Route B 训练数据）
- `data/context_dataset.py`（Route B val eval）
- `utils.py`

`configs/mainline_route_b.yaml` 是当前唯一活跃主线配置。

### Phase4 脚本审计结论

CORE（下阶段复用）：`postprocess_v2.py`, `mine_route_b_hardneg.py`, `run_nine_genome_stride_inference.py`, `eval_nine_genome_operating_point.py`。

FROZEN（产出已冻结，可删）：所有 sweep_*, nine_genome_sweep_s1, postprocess_sweep, diagnose_score_families, plot_nine_genome_case, analysis/02_discovery/ d1-d5, analysis/03_helpers/, analysis/04_audit/。

BASELINE（下阶段继续刷 baseline 对比要用）：analysis/01_baselines/ 三个 parse/compare 脚本。

**硬编码路径需改 config-driven**：
- `run_nine_genome_stride_inference.py`: `/data/syh/benchmark/nine_genome/fna`
- `nine_genome_sweep_s1.py`: 多处 `/data/zhanglin/...`（此脚本本身删了，不用管）
- `diagnose_score_families.py`: `/data/muskliu/...`（删了）
- 所有 analysis/ 脚本: `/data/muskliu/np_master/...`（多数删了，保留的 3 个 baseline 脚本需检查）

### 服务器状态审计结论

- 不是 git 仓库，rsync 散码
- 两节点 NFS 共享同一份 `/data/muskliu`
- **本地 `train_context_e2e.py` MD5 与服务器不一致**（服务器 mtime 4-10，本地有未提交改动时检查过）——git commit 已推送，新服务器 clone 会用本地版本，**建议 clone 后 diff 一下确认没有服务器 4-10 hotfix 遗失**
- 本地独有脚本服务器上不存在（postprocess_v2.py, mine_route_b_hardneg.py 等），服务器那份散码没东西值得救
- GPU 全空闲（两台 8×A800 都 0% 占用），无活跃训练，重构安全
- 没有 wandb 目录（未使用 wandb）

---

## 新会话如何接手

读完本文件后，你应该知道：
1. 当前处于阶段 0 完成、阶段 1 待开始的状态
2. 下一步是新建 `configs/operating_points.yaml` 并让 `scripts/phase4/postprocess_v2.py` 支持从 yaml 读参数
3. 阶段 2-4 的完整执行脚本在本文件里，按顺序做

用户的预期节奏：一阶段一报告，每阶段结束前 diff 给用户看，用户点头再 commit + 进下一阶段。

**注意**：
- 不要在服务器执行任何破坏性操作前没跟用户确认
- rm -rf 之前先 du -sh 确认大小
- 每个 phase 末尾记得 commit + push 到 GitHub（本地是唯一源）
- 本文件 `docs/cleanup-plan.md` 在阶段 4 结束时一并删除
