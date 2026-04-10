# Phase 4 Analysis Scripts

执行 Phase 4 (比较基线 + discovery 证据链) 产出的所有分析脚本。**按工作流分组，不是按时间。**

这些脚本原本在服务器 `/tmp/` 下一次性写成，现在整理进仓库方便复现和审查。

---

## 01_baselines/ — 竞争基线工具

跑 GECCO 和 DeepBGC，并和 Route B 做统一比较表。

| 脚本 | 功能 | 对应 Phase 4 任务 |
|------|------|-------------------|
| `run_gecco_batch.sh` | 在全部 639 test genomes 上跑 GECCO（16 workers 并行）| P2 |
| `parse_gecco.py` | 把 GECCO 输出解析成统一 JSON 格式 | P2 |
| `run_deepbgc_small.sh` | 在 46 个含候选的 genomes 上跑 DeepBGC（8 workers 并行）| P2 |
| `run_deepbgc_batch.sh` | 废弃：在全部 639 genomes 上跑 DeepBGC（~16 小时，未采用）| — |
| `parse_deepbgc.py` | 把 DeepBGC 输出解析成统一 JSON 格式 | P2 |
| `p3_compare.py` | 生成 Route B vs GECCO 对比表（含 per-bucket recall）| P3 |

**运行顺序：** `run_gecco_batch.sh` → `parse_gecco.py` → `run_deepbgc_small.sh` → `parse_deepbgc.py` → `p3_compare.py`

---

## 02_discovery/ — Discovery 证据链升级

升级 discovery benchmark：matched null → 双轨 ranker → multi-tool concordance → top candidates。

| 脚本 | 功能 | 对应 Phase 4 任务 |
|------|------|-------------------|
| `d1_diagnosis.py` | 量化诊断当前 discovery 证据弱的原因 | D1 |
| `d2_matched_null.py` | 构造 matched null（同 genome + GC ±3% + 排除所有工具 call）| D2 |
| `d2_run_hmmscan.sh` | 对 null 区域运行 Prodigal + biosyn_pfam hmmscan | D2 |
| `d2_analyze.py` | 比较 candidates vs matched null 分布（Mann-Whitney U + z-score）| D2 |
| `d3_rankers.py` | 双轨 Composite Ranker（Conservative + Novelty）| D3 |
| `d4_concordance.py` | Multi-tool concordance tier 分级（A/B/C/D）| D4 |
| `d5_deep_dive.py` | 选出 Track 1 + Track 2 top 候选，打印 evidence report | D5 |

**运行顺序：** D1 → D2 三步 → D3 → D4（需要先跑完 baselines）→ D5

---

## 03_helpers/ — 辅助脚本

| 脚本 | 功能 |
|------|------|
| `gen_genome_list.py` | 从 windows_metadata CSV 提取 test_id / test_ood genome list |
| `get_cand_genomes.py` | 提取 top-50 candidates 涉及的 46 个 genome（用于 DeepBGC subset）|
| `mibig_match.py` | P1: MIBiG species-level matching 到 test genomes（Tier 2 only）|
| `eval_route_b.py` | Route B full-contig eval（早期 debug 版本）|
| `eval_route_b_full.py` | Route B eval，产出 ws1_score_swap 兼容 JSON（供 discovery benchmark 用）|
| `eval_3seed.py` | Route B 3-seed 完整评估（per-bucket mean±std）|
| `bucket_eval.py` | Per-bucket recall 分析（早期版本）|
| `run_route_b.sh` | Route B seed=0 训练启动脚本 |
| `run_b_s1.sh` / `run_b_s2.sh` | Route B seed=1/2 并行训练脚本 |
| `run_discovery.sh` | Discovery benchmark 启动 wrapper |

---

## 04_audit/ — Route B 评估路径审计

| 脚本 | 功能 |
|------|------|
| `audit_route_b.py` | 核心审计：determinism, length invariance, batch independence, padding sensitivity |
| `audit2.py` | 补充 sanity check：正样本 score 分布 |

---

## 数据产出位置

所有脚本产出的 JSON / PKL 在服务器上：

```
/data/muskliu/np_master/data/phase4/
  d1_diagnosis.json
  d2_matched_null/
    candidates_with_null.json
    null_index.json
    null_hits/                # 173 Pfam hit 目录
    d2_results.json           # matched null 分布比较
  d3_ranking/d3_results.json  # 双轨 ranker 输出
  d4_concordance.json         # Tier 分级 + 候选列表
  d5_deep_dive/top_candidates.json  # 双轨交付清单
  baselines/
    gecco/                    # 639 GECCO 输出目录
    deepbgc/                  # 46 DeepBGC 输出目录
    gecco_predictions.json    # 统一格式
    deepbgc_predictions.json  # 统一格式
  p3_comparison.json          # Route B vs GECCO 对比表
  mibig_eval/mibig_matches.json  # P1 MIBiG Tier 2 matching
```

---

## 复现须知

这些脚本有**硬编码**的服务器路径（`/data/muskliu/np_master/...`）。要在其他环境复现，需要：

1. 改 `/data/muskliu/np_master/` 为你的数据根路径
2. 确保 conda env 包含 `pandas`, `numpy`, `scipy`, `scikit-learn`, `torch`, `biopython`
3. GECCO: `conda create -n gecco python=3.9 && pip install gecco-tool`
4. DeepBGC: `conda create -n deepbgc python=3.7 && pip install deepbgc biopython`
   - Pfam-A.31.0 数据库：若没有，可用更新版（antismash 的 Pfam 35.0）symlink 为 Pfam-A.31.0
   - Prodigal 必须在 PATH 中

这些脚本**不是**生产级代码，是一次性执行脚本。如果要做 v2 清理，应该把通用函数抽到 shared utils。
