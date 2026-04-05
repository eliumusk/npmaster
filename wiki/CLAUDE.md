# Wiki 维护规范

这是BGC预测研究项目的持续维护的知识库。本文档告诉LLM如何读取、更新和维护这个wiki。

## 目的

把 `project_docs/` 里零散的笔记、`scripts/` 里的代码逻辑、`gkaf305.md` 里的参考论文，以及每次实验的新发现，增量地整合成一个结构化、互相引用的知识库。

**核心原则**：
- 用户负责提出问题、决定方向、跑实验
- LLM负责读取信息源、更新wiki、维护交叉引用、保持一致性
- 每次聊天都要把新知识沉淀进wiki，不要只留在对话历史里

## 目录结构

```
wiki/
├── CLAUDE.md                 # 本文件（维护规范）
├── index.md                  # 内容目录（每页一行，含简短hook）
├── log.md                    # 时间线（只追加，每次ingest/实验/整理加一条）
├── 项目总览.md                # 项目目标、当前状态、核心发现
├── concepts/                 # 概念页面（BGC、Evo2、GC偏差等）
├── data/                     # 数据相关（MIBiG、参考基因组、数据集演进）
├── methods/                  # 方法相关（pipeline、embedding、分类器）
├── experiments/              # 实验记录（每个实验一页）
└── refs/                     # 外部参考（论文总结等）
```

## 页面规范

### 语言
- **中文为主**，生物/技术术语保留英文（如 BGC、Evo2、RF、AUROC）

### Frontmatter
每个页面开头用YAML frontmatter标注元数据：

```markdown
---
type: concept | data | method | experiment | reference | overview
updated: 2026-04-05
status: active | archived | draft
related: [[另一个页面]], [[xxx]]
---
```

### 文件名
- 用中文或小写英文（避免空格）
- 例子：`项目总览.md`, `MIBiG数据库.md`, `leave-one-class-out实验.md`

### 交叉引用
- 用 Obsidian 风格的 `[[页面名]]` 做内部链接
- 引用实验结果时指向 `[[experiments/实验名]]`
- 引用代码时用 `scripts/XX_name.py:line` 格式
- 引用外部资源用标准 markdown 链接

### 页面长度
- 单页面控制在 ~100-300 行
- 太长的话拆分成多个相关页面
- 太短的话考虑合并或补充

## 工作流

### 1. Ingest（新信息入库）

当用户给出新的实验结果、文档、想法时：
1. 读取相关源材料
2. 跟用户讨论关键信息
3. **创建或更新相应的wiki页面**
4. 更新 `index.md`（添加新页面条目）
5. 在 `log.md` 追加一条记录：`## [2026-04-05] ingest | 新实验结果 | 更新了实验/XXX.md, 总览.md`
6. **同时更新所有相关页面的交叉引用**（一次ingest可能影响5-10个页面）

### 2. Query（查询）

当用户问问题时：
1. 先读 `index.md` 找相关页面
2. 读相关页面的内容
3. 综合回答，附上引用（页面名）
4. **如果回答本身有价值（不是一次性的），创建新页面或更新已有页面把答案沉淀下来**

### 3. Lint（定期体检）

用户可以要求"检查wiki健康状况"：
- 找出 orphan 页面（没有任何页面链接到它）
- 找出 过时 的内容（被新实验推翻的结论）
- 找出 矛盾（不同页面说不一样的话）
- 找出 缺失的交叉引用
- 提出 应该补充的新页面

## 重要约定

### 实验页面必须包含
- 目的（为什么做这个实验）
- 方法（数据+模型+参数）
- 结果（表格/数字，带CI）
- 分析（关键发现、局限性）
- 对项目主线的影响（是否修正结论，是否需要新实验）

### 被推翻的结论
不要删除，改为用 `~~删除线~~` + "已被XXX实验推翻，见[[新实验]]" 的方式保留历史。

### 数字和CI
只要引用了AUROC/recall等数字，必须附上95% CI或者sample size。不要只说"0.94"。

### 日期
所有日期用绝对日期格式 `YYYY-MM-DD`，不要"上周"、"最近"这种相对说法。

## 与 project_docs 的关系

`project_docs/` 是wiki建立前的原始笔记，作为raw source保留。wiki建好后，新内容**只进wiki不进project_docs**。

`gkaf305.md` 是BGC-Prophet论文原文，保留为raw source。其总结在 `wiki/refs/BGC-Prophet论文.md`。

## 快速命令

- **添加新实验**：创建 `experiments/XXX.md` + 更新 `index.md` + 追加 `log.md`
- **修订结论**：改相关页面 + 更新 `log.md` + 检查交叉引用
- **健康检查**：`grep -l "待补充\|TODO\|FIXME" wiki/` 找待完成项
