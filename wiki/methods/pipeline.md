---
type: method
updated: 2026-04-05
status: active
related: [[embedding提取]], [[分类器]], [[评估方法]]
---

# 完整 Pipeline

## 端到端流程

```
基因组 FASTA
    │
    ├─ MIBiG BGC (正样本来源)
    │      │
    │      ↓ 按物种过滤 (只要Streptomyces)
    │      ↓ 滑动窗口切片 (w=8192, stride=4096)
    │      ↓
    │     正样本 pkl
    │
    └─ Streptomyces ref genomes
           │
           ↓ antiSMASH 标注 (排除BGC区域)
           ↓ 从非BGC segment随机采样窗口
           ↓ N含量<5%过滤
           ↓
          负样本 pkl

合并 → train/val/test 按BGC/genome分组split
    ↓
    ↓ Evo2 7B 推理 (blocks.20.mlp.l3)
    ↓ Mean pooling
    ↓
{train,val,test}_emb.pkl  (每个样本4096维)
    ↓
    ↓ RandomForestClassifier(100, max_depth=20)
    ↓
BGC 概率 + 指标 + cluster bootstrap CI
```

## 主要阶段

### 1. 数据构建

脚本：`scripts/14_build_expanded_strep.py`

- 输入：`data/mibig_gbk_3.1/`, `data/antismash_result/`, `data/ref_genomes_batch/`
- 输出：`data/expanded_strep/w{4096,8192,16384}/{train,val,test}.pkl`
- 时间：约3分钟

详见 [[../data/expanded-strep]]。

### 2. Embedding 提取（GPU）

脚本：`scripts/15_extract_expanded_embeddings.py`

- 输入：`{train,val,test}.pkl`
- 输出：`{train,val,test}_emb.pkl`（在GPU服务器上）
- 硬件：1×A800-80GB per task
- 速度：
  - w4096: ~2.2 samples/s
  - w8192: ~1.3 samples/s  
  - w16384: ~0.6 samples/s
- 时间：w8192全套约4小时（两台8卡并行）

详见 [[embedding提取]]。

### 3. 分类器训练 + 评估

脚本：`scripts/16_expanded_eval.py`, `scripts/07_train_classifier.py`

- 模型：`RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)`
- CI：BGC级 cluster bootstrap (n_boot=5000)
- 输出：AUROC, F1, Precision, Recall + 95% CI

详见 [[分类器]], [[评估方法]]。

## 代码清单

| 脚本 | 用途 | 阶段 |
|---|---|---|
| `01_build_dataset.py` | 早期v1数据集 | 废弃 |
| `02_baseline_kmer.py` | k-mer baseline | 参考 |
| `03-05_*` | 数据集迭代 | 历史 |
| **`14_build_expanded_strep.py`** | **构建主数据集** | 当前 |
| **`15_extract_expanded_embeddings.py`** | **提取embedding (GPU)** | 当前 |
| **`16_expanded_eval.py`** | **评估 (同属+跨属+窗口消融)** | 当前 |
| `19_leave_one_class_out.py` | LOCO诊断 | 当前 |
| `21_source_bias_diagnosis.py` | 来源偏差诊断 | 当前 |
| `run_antismash_*.sh` | antiSMASH批处理 | 辅助 |

## 计算环境

### 本地 Mac
- 数据构建、antiSMASH标注、部分分析
- Python 3.x + Biopython + sklearn

### GPU 服务器
- 两台 8×A800-80GB
- SSH: `Voc_gpu_node7_a800` (port 50000), `Voc_gpu_node8_a800` (port 50010)
- 路径: `/data/muskliu/np_master/`
- conda env: `bgc` (含sklearn 1.8.0, matplotlib, scipy, pytorch + evo2)
- 其他env: `evo2_bgc`, `antismash-env`, `bgc-prophet`

## 典型运行命令

```bash
# 本地：构建数据集
python3 14_build_expanded_strep.py --all-windows

# 服务器：提取embedding
python3 -u 15_extract_expanded_embeddings.py --shard 0   # node7
python3 -u 15_extract_expanded_embeddings.py --shard 1   # node8

# 服务器：评估
conda run -n bgc python3 -u 16_expanded_eval.py
```
