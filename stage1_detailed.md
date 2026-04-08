# 阶段1：数据基座构建 — 详细执行手册

> 本文档供Claude Code直接执行。每个步骤有明确的输入、输出、执行命令和验收标准。

---

## 全局更新记录

- 分类任务：8维多标签分类（7种BGC类型 + 1维非BGC），使用sigmoid而非softmax
- M=0：不做新类型发现，只做已知类型检测
- MIBiG中一个BGC可以同时属于多个类型（如PKS+NRP hybrid）

---

## 步骤1.1 基因组Panel获取

### 输入
- GTDB RS226 数据库（https://data.gtdb.ecogenomic.org/）
- 目标：约2000个细菌基因组

### 执行

1. 从GTDB下载细菌代表基因组的metadata文件（`bac120_metadata.tsv`）
2. 按以下标准筛选：
   - 基因组质量：checkm_completeness ≥ 95%, checkm_contamination ≤ 5%
   - 组装级别：complete 或 scaffold（排除 contig 级别）
   - 每个species只保留1个代表（取quality最高的）
3. 按门分层采样：
   - Actinomycetota：~600个（BGC最密集的门）
   - Pseudomonadota：~500个
   - Bacillota：~500个
   - Bacteroidota：~200个（作为test_ood候选）
   - Cyanobacteriota：~100个（作为test_ood候选）
   - 其他门：~100个（增加多样性）
4. 下载所有筛选通过的基因组fasta文件

### 输出

```
data/genomes/                          # 所有fasta文件
data/genome_metadata.csv               # 元数据表
```

`genome_metadata.csv` schema：
```csv
genome_id,species,phylum,class,order,family,genus,
genome_size_bp,n_contigs,checkm_completeness,checkm_contamination,
assembly_level,ncbi_accession,fasta_path
```

### 验收标准

1. `genome_metadata.csv` 行数 ≥ 1500
2. 门的分布：
   - Actinomycetota ≥ 400
   - Pseudomonadota ≥ 300
   - Bacillota ≥ 300
   - Bacteroidota ≥ 100
   - 其他 ≥ 50
3. 每个species最多出现1次（去冗余验证）
4. 所有fasta文件可正常读取（无空文件、无格式错误）

### 反馈给我的信息

- 最终选中的基因组总数和门分布表
- 筛选过程中被排除的基因组数量及原因分布
- 基因组大小的分布统计（min, median, max, 按门分组）
- 如果某个门的基因组数量不足预期，说明原因

---

## 步骤1.2 antiSMASH标注

### 输入
- `data/genomes/` 目录
- antiSMASH 8.0（需要预先安装或通过conda安装）

### 执行

1. 确认antiSMASH可用（`antismash --version`）
2. 对每个基因组运行antiSMASH：
   ```bash
   antismash --genefinding-tool prodigal \
             --cb-general --cb-subclusters --cb-knownclusters \
             --output-dir antismash_results/{genome_id} \
             data/genomes/{genome_id}.fasta
   ```
3. 解析每个基因组的antiSMASH输出JSON，提取BGC信息
4. 汇总为统一的标注文件

### 输出

```
data/antismash_results/                  # 原始antiSMASH输出
data/bgc_annotations.csv                 # 汇总的BGC标注表
data/antismash_summary.json              # 运行统计
```

`bgc_annotations.csv` schema：
```csv
bgc_id,genome_id,contig_id,start_bp,end_bp,bgc_length,
bgc_types,bgc_product,similarity_to_known,phylum
```

关键字段说明：
- `bgc_types`：多标签，用分号分隔（如 "Polyketide;NRP"）
- 类型映射规则（antiSMASH的细分类型 → 7大类）：
  - T1PKS, T2PKS, T3PKS, transAT-PKS, hglE-KS → **Polyketide**
  - NRPS, NRPS-like → **NRP**
  - terpene → **Terpene**
  - lanthipeptide, thiopeptide, RiPP-like, lasso peptide, sactipeptide, bottromycin, cyanobactin, LAP, microcin, proteusin, ranthipeptide, head_to_tail, ... → **RiPP**
  - saccharide, oligosaccharide → **Saccharide**
  - alkaloid, indole → **Alkaloid**
  - 其他所有 → **Other**
- 如果一个BGC的antiSMASH产物类型跨多个大类，`bgc_types`记录所有匹配的大类

### 验收标准

1. antiSMASH成功运行的基因组比例 ≥ 95%（记录失败的基因组及原因）
2. `bgc_annotations.csv` 总BGC数 ≥ 20000
3. 7种BGC类型都有代表（每种 ≥ 100个）
4. 多标签BGC（bgc_types含分号的行）的比例统计
5. BGC长度分布合理（median 在 20-40kb之间）

### 反馈给我的信息

- 总BGC数、每种类型的数量分布表
- 多标签BGC的数量和最常见的组合（如 PKS+NRP 有多少个）
- 每个门的BGC密度（平均每个基因组有多少个BGC）
- BGC长度分布直方图（按类型分面）
- antiSMASH运行失败的基因组列表及原因
- 总CPU时间消耗

---

## 步骤1.3 窗口切片 + 标签生成

### 输入
- `data/genomes/`
- `data/bgc_annotations.csv`
- `data/genome_metadata.csv`

### 执行

#### 1.3a 切窗口

对每个基因组的每条contig，用固定窗口滑动切片：
- 窗口大小 W = 16384 bp
- 步长 S = 8192 bp（50%重叠）
- 如果contig末尾剩余不足W，丢弃该窗口（不padding）

每个窗口记录：
```
window_id, genome_id, contig_id, window_start_bp, window_end_bp
```

#### 1.3b 标注窗口

对每个窗口，计算它与所有BGC的重叠：

```python
for each window:
    overlapping_bgcs = [bgc for bgc in same_contig_bgcs 
                        if overlap(window, bgc) > 0]
    
    if not overlapping_bgcs:
        # 计算到最近BGC边界的距离
        min_distance = min_distance_to_any_bgc(window, same_contig_bgcs)
        if min_distance >= 10000:  # 10kb buffer
            label = "negative"
        else:
            label = "ambiguous"
    else:
        # 计算最大重叠比例
        max_overlap_ratio = max(overlap_ratio(window, bgc) for bgc in overlapping_bgcs)
        if max_overlap_ratio >= 0.5:
            label = "positive"
            # 收集所有重叠BGC的类型（合并多标签）
            types = union(bgc.bgc_types for bgc in overlapping_bgcs if overlap_ratio >= 0.3)
        else:
            label = "ambiguous"
```

标签编码为8维二值向量：
```
[Polyketide, NRP, Terpene, RiPP, Saccharide, Alkaloid, Other, is_negative]
```
- positive窗口：对应类型位=1，is_negative=0
- negative窗口：前7位全0，is_negative=1
- ambiguous窗口：标记但不参与训练

#### 1.3c 数据划分

划分流程：

1. **BGC聚类**：对所有检测到的BGC，用BiG-SCAPE或antiSMASH内置的GCF聚类，得到基因簇家族(GCF)分组。如果BiG-SCAPE安装困难，备选方案：按antiSMASH的knownclusterblast结果做相似度聚类，或直接用BGC类型+物种作为分组依据。

2. **门级hold-out**：Bacteroidota + Cyanobacteriota 整体作为 test_ood。这些门的所有基因组和窗口只出现在test_ood中。

3. **基因组级划分**（在剩余门中）：
   - 随机选20%基因组作为 test_id
   - 剩余80%基因组中，随机选12.5%（=总体10%）作为 val
   - 剩余作为 train

4. **GCF泄漏检查**：检查是否有同一个GCF的BGC同时出现在train和test中。如果有，将test中的这些BGC移到train（保守做法，宁可train有冗余也不要test被污染）。

每个窗口的最终标签：
```
split: train / val / test_id / test_ood
```

### 输出

```
data/windows/
├── windows_metadata.csv        # 所有窗口的元数据和标签
├── train_windows.txt           # train集的window_id列表
├── val_windows.txt             # val集
├── test_id_windows.txt         # test_id集
├── test_ood_windows.txt        # test_ood集
├── split_summary.json          # 划分统计
└── sequences/                  # 窗口序列（可选，也可以运行时从基因组动态切）
    ├── {genome_id}_{contig}_{start}.seq
    └── ...
```

`windows_metadata.csv` schema：
```csv
window_id,genome_id,contig_id,window_start,window_end,
label(positive/negative/ambiguous),
Polyketide,NRP,Terpene,RiPP,Saccharide,Alkaloid,Other,is_negative,
max_bgc_overlap_ratio,nearest_bgc_distance,
phylum,split,sequence_length
```

`split_summary.json` 内容：
```json
{
  "train": {
    "n_windows": ...,
    "n_positive": ...,
    "n_negative": ...,
    "n_ambiguous": ...,
    "n_genomes": ...,
    "type_distribution": {"Polyketide": ..., "NRP": ..., ...},
    "phylum_distribution": {...}
  },
  "val": { ... },
  "test_id": { ... },
  "test_ood": { ... },
  "gcf_leakage_check": {
    "n_shared_gcf_before_fix": ...,
    "n_shared_gcf_after_fix": 0,
    "moved_samples": ...
  }
}
```

### 验收标准

1. **数量**：
   - train集 ≥ 40000个窗口（positive + negative，不含ambiguous）
   - val集 ≥ 5000个窗口
   - test_id ≥ 10000个窗口
   - test_ood ≥ 3000个窗口

2. **正负比例**：
   - 每个split中positive比例在5%-25%之间（符合自然分布）
   - 如果positive比例过低（<5%），说明buffer zone太大或BGC标注太少

3. **多标签分布**：
   - 至少有500个多标签正样本窗口（bgc_types含多个类型）
   - 7种类型都至少有200个正样本窗口

4. **无泄漏**：
   - GCF泄漏检查后，train和test之间无共享GCF
   - train和test之间无共享基因组

5. **ambiguous窗口**：
   - ambiguous占总窗口的5%-15%（太多说明buffer太大，太少说明标注边界太精确不真实）

### 反馈给我的信息

- `split_summary.json` 的完整内容
- 每种BGC类型在train/val/test中的窗口数分布表
- 多标签窗口的类型组合分布（top 10最常见的组合）
- 正负样本比例（按split和按phylum分别统计）
- ambiguous窗口数量及其位置特征（是集中在BGC边界附近还是分散的）
- GCF泄漏检查的结果（修复前有多少泄漏、修复后确认为0）
- 整个阶段1的磁盘占用量

---

## 阶段1完成后的检查清单

在进入阶段2之前，确认以下所有项：

- [ ] 基因组panel ≥ 1500个，门分布合理
- [ ] antiSMASH标注完成，≥ 95%成功率
- [ ] BGC标注 ≥ 20000个，7类都有代表
- [ ] 窗口切片完成，train ≥ 40000个
- [ ] 多标签标注正确（hybrid BGC有多个类型位=1）
- [ ] 数据划分无泄漏
- [ ] 所有输出文件可正常读取
- [ ] `split_summary.json` 已生成并人工审核

全部通过后，进入阶段2（Evo2 embedding提取）。
