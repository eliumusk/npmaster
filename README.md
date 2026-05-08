# NP-Master

NP-Master 是面向细菌基因组和宏基因组的 BGC 区域检测与发现流程。它不是生成式模型：给定 DNA 序列后，它会预测看起来像生物合成基因簇的基因组区间，并可选地添加 BGC 类型标签与新颖性初筛元数据。

## 系统输入输出

输入：一个或多个 DNA contig/基因组，通常为 FASTA。服务端流程会提取冻结的 Evo2 7B `blocks.20.mlp.l3` 特征。

输出：BGC 区域调用结果，包含 `genome`、`contig`、`start`、`end`、分数/来源字段，以及可选的类型与新颖性注释。

## 活跃主线


| 主线                    | 用途                                            | 状态                                              | 关键入口                                                                                                                          | 结果                                 |
| --------------------- | --------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `evo2_lr_multiscale`  | Evo2 mean-pool embedding + LR 评分器 + 多尺度解码器。   | OER learned-decoder 变体面向宏基因组。                   | `scripts/decode/evo2_lr_multiscale.py`, `scripts/evaluate/evo2_lr_multiscale.py`, `scripts/decode/evo2_lr_learned_decoder.py` | `experiments/evo2_lr_multiscale/`  |
| `evo2_per_token_unet` | per-token Evo2 特征 + U-Net 解码器，用于短宏基因组 contig。 | 当前主线。当前 wet-lab handoff 使用 weak-negative BCE 模型 | `scripts/train/evo2_per_token_unet/`, `scripts/inference/evo2_per_token/`, `scripts/decode/evo2_per_token_regions.py`         | `experiments/evo2_per_token_unet/` |


其他历史史文件已移动到 `_archive/2026-04-26-mainline-reset/`，并且有意不被 git 跟踪。

## 标准结果数字

当前主张与注意事项见 `docs/REGRESSION_LEDGER.md`。活跃主线的核心数字如下：

- `evo2_per_token_unet` weak-negative BCE 在 OER004256 上：`thr0.50/min2kb` 下 AS-recall 为 `0.5703`，579 个 antiSMASH 漏检候选，经过 HMM/MIBiG triage 后有 171 个 first-pass wet-lab PASS。
- `evo2_per_token_unet` weak-negative BCE 在 9-genome 上：`thr0.70/min8kb` 下诊断性 dedup F1 为 `0.5094`
- `evo2_lr_multiscale` 在 9-genome 上：dedup F1 `0.4878`，bench precision `0.5976`，bench recall `0.4590`。
- `evo2_lr_learned_decoder` 在 OER004256 上：已提交的 handoff artifact 中包含 1,230 个区域，其中 429 个为 antiSMASH 漏检候选。

## 定性边界

当前 weak-negative U-Net 的 9-genome 诊断点（`threshold=0.70`，`min_len=8kb`）在 `figures/evo2_per_token_unet_boundaries_weakneg_thr070_min8000/` 中包含逐基因组的边界图。

Weak-negative Evo2 per-token U-Net 9-genome 边界

## 目录结构

- `configs/`：仅包含活跃 YAML 配置。
- `src/`：活跃主线的可导入库代码。
- `scripts/`：按动词分组的 CLI 入口。
- `experiments/`：仅包含已提升的活跃输出。
- `docs/`：当前约定、评估协议和回归台账。
- `_archive/`：已退休的本地溯源文件；不被跟踪。

## 快速命令

```bash
python scripts/evaluate/evo2_lr_multiscale.py \
  --primary-preds <server-primary-preds-dir> \
  --short-preds <server-short-preds-dir> \
  --gt-json /data/syh/benchmark/nine_genome/ground_truth.json \
  --out-json experiments/evo2_lr_multiscale/nine_genome_readout.json
```

```bash
python scripts/decode/evo2_per_token_regions.py \
  --probs-dir <per-token-probs-dir> \
  --out-regions-csv <regions.csv> \
  --threshold 0.3
```

