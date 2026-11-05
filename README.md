# NP-Master

NP-Master 是面向细菌基因组和宏基因组的 BGC 区域检测与发现流程。它不是生成式模型：给定 DNA 序列后，系统预测看起来像生物合成基因簇的基因组区间，并可选地添加 BGC 类型标签与新颖性初筛元数据。

## 系统输入输出

输入：一个或多个 DNA contig/基因组，通常为 FASTA。服务端流程会清洗序列并提取冻结的 Evo2 7B `blocks.20.mlp.l3` per-token 特征。

输出：BGC 区域调用结果，包含 `genome`、`contig`、`start`、`end`、分数/来源字段，以及可选的类型与新颖性注释。

## 当前模型结构

当前流程分成三个层次：

1. **Evo2 特征层**：冻结 Evo2 7B backbone，抽取 `blocks.20.mlp.l3` hidden states。原始 4096 维 base-level 表征先投影到 128 维，再按 8 bp pooling 成 latent tokens。
2. **区域检测层**：`evo2_per_token_unet` 使用轻量 1D U-Net 分割头，把 Evo2 latent token 序列映射成 BGC probability track。训练时使用 16,384 bp windows；推理时使用 8,192 bp windows 和 2,048 bp stride，对重叠窗口概率做 genome-coordinate averaging，再从连续高置信区间解码出候选 BGC regions。
3. **注释与 triage 层**：`evo2_mibig_type` 是独立的 product-type annotation head。它对候选 region 预测 ORFs，扫描 Pfam domains，并用 MIBiG 训练的 RandomForest 多标签分类器输出 `Alkaloid`、`Terpene`、`NRP`、`Polyketide`、`RiPP`、`Saccharide`、`Other` 类型概率。类型注释和 HMM/MIBiG evidence 用于 wet-lab triage，不参与区域检测本身。

| 模块 | 作用 | 关键入口 | 主要结果 |
| --- | --- | --- | --- |
| `evo2_per_token_unet` | 从 DNA contigs 检测 BGC 区域坐标 | `scripts/train/evo2_per_token_unet/`, `scripts/inference/evo2_per_token/`, `scripts/decode/evo2_per_token_regions.py` | `experiments/evo2_per_token_unet/` |
| `evo2_mibig_type` | 对已检测 region 添加 BGC product-type 标签 | `scripts/prepare/mibig_domains.py`, `scripts/train/type_head/train.py`, `scripts/classify/antismash_regions_type.py` | `experiments/evo2_mibig_type/` |

其他历史史文件已移动到 `_archive/2026-04-26-mainline-reset/`，并且有意不被 git 跟踪。

## 标准结果数字

当前主张与注意事项见 `docs/REGRESSION_LEDGER.md`。当前 U-Net 检测器的核心数字如下：

- `evo2_per_token_unet` weak-negative BCE 在 OER004256 上：`thr0.50/min2kb` 下 AS-recall 为 `0.5703`，579 个 antiSMASH 漏检候选，经过 HMM/MIBiG triage 后有 171 个 first-pass wet-lab PASS。
- `evo2_per_token_unet` weak-negative BCE 在 9-genome 上：`thr0.70/min8kb` 下诊断性 dedup F1 为 `0.5094`，raw bench F1 为 `0.521`。
- `evo2_mibig_type` RandomForest type head 在 MIBiG family-disjoint test split 上：macro-F1 `0.6715`，micro-F1 `0.8789`，macro-AUROC `0.9442`。

注意：当前 U-Net 的 9-genome 数字是 diagnostic/exploratory，因为该模型族有 9-genome-exposed selection history。正式 9-genome claim 需要预注册 checkpoint 和 threshold rule。

## 定性边界

当前 weak-negative U-Net 的 9-genome 诊断点（`threshold=0.70`，`min_len=8kb`）在 `figures/evo2_per_token_unet_boundaries_weakneg_thr070_min8000/` 中包含逐基因组的边界图。

## 目录结构

- `configs/`：仅包含活跃 YAML 配置。
- `src/`：活跃主线的可导入库代码。
- `scripts/`：按动词分组的 CLI 入口。
- `experiments/`：仅包含已提升的活跃输出。
- `docs/`：当前约定、评估协议和回归台账。
- `_archive/`：已退休的本地溯源文件；不被跟踪。

## 快速命令

```bash
python scripts/inference/evo2_per_token/infer_shards.py \
  --features-dir <per-token-features-dir> \
  --ckpt experiments/evo2_per_token_unet/train_runs/full_weakneg_bce_w05_ddp4_gb64_seed0/best.pt \
  --out-dir <per-token-probs-dir>
```

```bash
python scripts/decode/evo2_per_token_regions.py \
  --probs-dir <per-token-probs-dir> \
  --out-regions-csv <regions.csv> \
  --threshold 0.50 \
  --min-len-bp 2000
```

```bash
python scripts/train/type_head/train.py \
  --bundle experiments/evo2_mibig_type/mibig_domains.json \
  --mmseqs-tsv experiments/evo2_mibig_type/families_cluster.tsv \
  --out-dir experiments/evo2_mibig_type/run_full
```
