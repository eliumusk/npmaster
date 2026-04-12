# 项目重构方案

**日期：** 2026-04-09  
**项目：** `np-master`  
**用途：** 在继续做实验之前，先把项目的文档、代码、结果和服务器产物整理清楚，避免主线继续发散。

---

## 1. 现在为什么要重构

现在项目最大的问题，不是“实验不够多”，而是“东西太多、太散”。

具体表现是：

- 文档很多，但权威性不清楚
- 有些文档是主线规则，有些只是临时诊断，混在一起
- 代码里有主线代码，也有一次性实验脚本，边界不清楚
- 服务器上结果越来越多，但哪些是主线、哪些是历史结果，没有统一管理

这样继续下去会有几个风险：

- agent 每次都要重新理解上下文，成本很高
- 很容易拿旧结果和新结果混着讲
- 很容易在错误 baseline 上继续优化
- 很容易把“诊断性结果”误当成“正式结论”

所以现在必须先做一轮**项目控制面重构**。

---

## 2. 这次重构的目标

这次重构不是改模型结构。  
这次重构的目标很简单：

1. 明确当前主线是什么
2. 明确哪些文档是“项目规则”
3. 明确哪些结果是“当前主线结果”
4. 明确哪些东西应该归档
5. 明确哪些东西以后可以删

一句话：

**先把项目整理成一条清晰主线，再继续做实验。**

---

## 3. 当前主线怎么定义

截至现在，项目主线可以先定义为：

- detection score 用 `score_maxtype`
- 评测口径用 `full-window`
- 训练采样用 `prior-aligned`，也就是 `pos_ratio=0.15`
- checkpoint selection 用 `bgc_binary_auroc`
- `WS3 cap=0.8` 目前是**暂定主线候选**，还不是最终冻结版本

这意味着：

- 后面所有实验都应该默认相对这条主线汇报
- 旧的 `is_negative` 默认 Stage 4 不再作为主线 baseline
- 旧 threshold、旧 score、旧结果不能再混着讲

---

## 4. 文档怎么分层

以后文档只分四类。

### 4.1 第一类：总控文档

这类文档定义项目方向和执行规则。

保留：

- [2026-04-08-bgc-system-redesign.md](/Users/musk/Documents/code/np-master/docs/2026-04-08-bgc-system-redesign.md)
- [2026-04-09-bgc-execution-roadmap.md](/Users/musk/Documents/code/np-master/docs/2026-04-09-bgc-execution-roadmap.md)
- [2026-04-09-project-refactor-plan.md](/Users/musk/Documents/code/np-master/docs/2026-04-09-project-refactor-plan.md)

### 4.2 第二类：合同文档

这类文档定义当前主线的训练/评测规则。

保留并提升为正式文档：

- [ws_checkpoint_selection_contract.md](/Users/musk/Documents/code/np-master/docs/execution/ws_checkpoint_selection_contract.md)
- [ws1_t2_threshold_contract.md](/Users/musk/Documents/code/np-master/docs/execution/ws1_t2_threshold_contract.md)
- [ws4_t1_fullwindow_eval_spec.md](/Users/musk/Documents/code/np-master/docs/execution/ws4_t1_fullwindow_eval_spec.md)
- [ws4_t2_precision_spec.md](/Users/musk/Documents/code/np-master/docs/execution/ws4_t2_precision_spec.md)
- [ws3_label_geometry_spec.md](/Users/musk/Documents/code/np-master/docs/execution/ws3_label_geometry_spec.md)

### 4.3 第三类：主线结果文档

这类文档只保留当前最重要的主线结果。

保留：

- [ws2_t2_prior_aligned_results.md](/Users/musk/Documents/code/np-master/docs/execution/ws2_t2_prior_aligned_results.md)
- [ws3_capture_sweep_results.md](/Users/musk/Documents/code/np-master/docs/execution/ws3_capture_sweep_results.md)
- [ws_discovery_benchmark.md](/Users/musk/Documents/code/np-master/docs/execution/ws_discovery_benchmark.md)

### 4.4 第四类：历史诊断文档

这类文档曾经有用，但现在不应该继续放在执行主界面里。

应该归档：

- [ws0_t1_artifact_inventory.md](/Users/musk/Documents/code/np-master/docs/execution/ws0_t1_artifact_inventory.md)
- [ws0_t2_metric_contract.md](/Users/musk/Documents/code/np-master/docs/execution/ws0_t2_metric_contract.md)
- [ws1_t1_score_swap_results.md](/Users/musk/Documents/code/np-master/docs/execution/ws1_t1_score_swap_results.md)
- [ws2_t1_calibration_results.md](/Users/musk/Documents/code/np-master/docs/execution/ws2_t1_calibration_results.md)
- [ws3_mvp_results.md](/Users/musk/Documents/code/np-master/docs/execution/ws3_mvp_results.md)

这些文档不是没用。  
它们是**历史诊断证据**。  
但它们已经不是“当前主线的最好表达”了。

---

## 5. 目标目录结构

项目文档最终应该整理成这样：

```text
docs/
  2026-04-08-bgc-system-redesign.md
  2026-04-09-bgc-execution-roadmap.md
  2026-04-09-project-refactor-plan.md
  contracts/
    checkpoint_selection.md
    threshold_contract.md
    fullwindow_eval_spec.md
    precision_spec.md
    label_geometry_spec.md
  mainline/
    current_status.md
    prior_aligned_results.md
    capture_sweep_results.md
    discovery_benchmark.md
  archive/
    execution/
      ...
```

现在不要求一步到位。  
但方向要朝这个结构走。

---

## 6. 代码怎么分层

代码也要分清楚。

### 6.1 主线代码

这些代码是当前主线的一部分，要保持干净。

保留并重点维护：

- [train.py](/Users/musk/Documents/code/np-master/bgc-gcd/train.py)
- [evaluate.py](/Users/musk/Documents/code/np-master/bgc-gcd/evaluate.py)
- [sampler.py](/Users/musk/Documents/code/np-master/bgc-gcd/data/sampler.py)
- [05_windows_and_labels.py](/Users/musk/Documents/code/np-master/scripts/stage1/05_windows_and_labels.py)
- [run_stage4.py](/Users/musk/Documents/code/np-master/scripts/stage4/run_stage4.py)

### 6.2 阶段工具代码

这些脚本还有用，但不是每天都要改的主线代码。

例如：

- Stage 1 预处理脚本
- [error_analysis.py](/Users/musk/Documents/code/np-master/scripts/stage4/error_analysis.py)

### 6.3 执行脚本

这些是为了跑某一轮实验写的。

例如：

- [ws1_score_swap.py](/Users/musk/Documents/code/np-master/scripts/execution/ws1_score_swap.py)
- [ws2_calibration.py](/Users/musk/Documents/code/np-master/scripts/execution/ws2_calibration.py)
- [ws3_mvp_relabel.py](/Users/musk/Documents/code/np-master/scripts/execution/ws3_mvp_relabel.py)
- [ws_discovery_benchmark.py](/Users/musk/Documents/code/np-master/scripts/execution/ws_discovery_benchmark.py)

规则是：

- 真正还会继续复用的，保留
- 已经被主线吸收掉的，后续归档或删除

### 6.4 生成垃圾

这些东西不应该继续留着：

- `__pycache__`
- 临时图
- 重复日志
- 不再被引用的中间文件

这些可以删，但要放到最后做。

---

## 7. 服务器结果怎么管理

服务器现在是主结果来源，所以一定要更严格。

### 7.1 必须保留的

必须保留：

- 当前主线 checkpoint
- 当前主线 eval JSON
- 被主线文档引用的 artifact

至少包括：

- `Config E` 的 checkpoint 和 eval
- `WS3 cap=0.8` 的 checkpoint 和 eval
- 当前 discovery benchmark artifact

### 7.2 先归档的

这些先别删，先归档：

- `WS1-T1` 的早期诊断结果
- `WS2-T1` calibration-only 结果
- `WS3-MVP` 被 `capture sweep` 取代的那版结果

### 7.3 以后可以删的

只有在满足下面两个条件时才允许删：

1. 结果已经被主线或历史文档总结过
2. 没有任何保留文档再引用它的路径

一句话：

**先归类，再归档，最后才删除。**

---

## 8. 重构分三步走

### 第一步：整理控制面

目标：

- 先把项目的“说法”统一

要做的事：

- 建一个当前主线状态文档
- 把 `docs/execution/` 里的文档分成：
  - 合同
  - 主线结果
  - 历史诊断

### 第二步：整理代码面

目标：

- 让主线代码和实验代码分开

要做的事：

- 标记哪些配置是主线
- 标记哪些脚本只是实验脚本
- 清理本地生成垃圾

### 第三步：整理服务器

目标：

- 让服务器结果可以追溯

要做的事：

- 先生成 artifact 清单
- 再决定哪些归档
- 最后再删重复或无引用的

---

## 9. 接下来先做什么

现在最合理的下一步只有 4 件事：

1. 建一个 `current mainline status` 文档
2. 重组 `docs/execution/`
3. 标记当前 canonical code/config
4. 生成服务器 artifact manifest

**在这 4 件事完成之前，不建议开始大规模删除。**

---

## 10. 删除原则

删除顺序必须是：

1. **先重新分类**
2. **再归档**
3. **最后删除**

不要一步到位直接删。  
否则很容易把后面还要解释结果的证据链删掉。

---

## 11. 这次重构真正保护的是什么

这次重构不是为了好看。

它真正保护的是项目最核心的问题：

- 我们现在到底还在修“系统合同问题”
- 还是已经触到了“模型结构瓶颈”

如果项目一直很乱，这两个问题很快又会混在一起。  
那后面的实验就会再次失去解释力。

所以这次重构，实际上是在保护项目的科学主线。
