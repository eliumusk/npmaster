---
type: reference
updated: 2026-04-05
status: active
citation: Lai Q, Yao S, Zha Y, et al. Nucleic Acids Research, 2025, 53, gkaf305
related: [[../项目总览]], [[../concepts/BGC]], [[../data/antiSMASH]]
source: gkaf305.md
---

# BGC-Prophet 论文总结

**原文**：Deciphering the biosynthetic potential of microbial genomes using a BGC language processing neural network model

**作者**：Lai Q. et al. (华中科技大学 Kang Ning 实验室)

**期刊**：Nucleic Acids Research, 2025

**本地原文**：`gkaf305.md`

## 一句话总结

用**"语言模型"思路**做BGC预测：把基因当token，基因序列当句子，用Transformer判断哪些基因属于BGC、属于哪类。

## 核心创新

| 方法 | Token | 编码 | 缺点 |
|---|---|---|---|
| antiSMASH | 蛋白结构域 | pHMM匹配 | 慢，只找已知类型 |
| DeepBGC | Pfam结构域 | one-hot | 依赖Pfam，丢信息 |
| **BGC-Prophet** | **整个基因** | **ESM-2** | 仍在蛋白层面 |

### 相对DeepBGC的改进
1. 用基因而非结构域做token → 更自然
2. 用 ESM-2 嵌入 → 不需要 Pfam 比对，快100倍
3. Transformer捕捉长距离位置关系

## 数据构造（最关键部分）

### 正样本
- 来源：MIBiG (2502个BGC)
- 每个BGC的基因 + 从6KG基因组取 non-BGC 基因做padding → 128基因窗口
- 重复5次（不同基因组padding）→ 12510个正样本

### 负样本
- 从 non-BGC 基因库取128连续基因
- 数量：20000个

### 关键设计
- **固定长度128基因**（容纳最长BGC）
- **同一基因组取正负**（避免物种偏差）
- **标注每个基因**：0/1 binary（检测） + 7类multi-label（分类）

## 模型架构

### 编码层 (ESM-2)
```
每个基因的氨基酸序列 → ESM-2 t6_8M_UR50D → 320维向量
（不训练，只做推理）
```

### 检测模型
```
输入: (batch, 128, 320)
  → 正弦位置编码
  → 2层 Pre-LN Transformer Encoder (5 heads, ff=1280)
  → MLP (320→128→32→1) + Sigmoid
输出: (batch, 128)  每个基因的BGC概率
损失: BCE
```

### 分类模型
```
同样Transformer Encoder
  → 序列维度平均 → (batch, 320)
  → Linear(320→7) + Sigmoid
损失: Focal Loss (α=0.25, γ=2)
```

## 主要结果

### 检测（NG数据集9基因组）
| 指标 | BGC-Prophet | DeepBGC |
|---|---|---|
| AUROC | 0.92 | 0.93 |
| F1 | 0.61 | 0.35 |
| Precision | 0.59 | 0.22 |

### 分类
| 指标 | BGC-Prophet | DeepBGC |
|---|---|---|
| AUROC | 0.988 | 0.913 |
| F1 | 0.909 | 0.809 |

### 速度
- 比DeepBGC快 **~100倍**（ESM-2推理 vs Pfam HMM）

### 大规模应用
- 85,203基因组 → 119,305 BGC
- 9,428宏基因组 → 248,229 BGC
- Aspergillus属上比antiSMASH多3倍

## 论文局限性（我们的机会）

1. **检测AUROC略低于DeepBGC** (0.92 vs 0.93)
2. **对RiPP表现差** (训练数据不平衡)
3. **依赖ESM-2蛋白表示** — 无法用非编码区、调控信息
4. **无法识别非典型BGC** (与已知模式差异大的)
5. **无法预测具体产物** (只分7类)

## 本项目与BGC-Prophet的差异

| 维度 | BGC-Prophet | 本项目 |
|---|---|---|
| 层面 | 蛋白质 | **DNA** |
| 编码器 | ESM-2 (8M) | **Evo2 (7B)** |
| 需要基因预测? | 是 | **否** |
| 输入 | 128基因窗口 | DNA窗口 (4k/8k/16k bp) |
| 任务 | 基因级标注 | 窗口级分类 |

**方向差异**：BGC-Prophet 做精细的基因级定位，我们做粗粒度的窗口级检测但利用了完整DNA信息（包括非编码区、调控区、基因间区）。

## 可借鉴的设计

1. **同基因组取正负样本** → 我们采纳（[[../data/expanded-strep]]）
2. **去掉antiSMASH标注区域做负样本池** → 我们采纳
3. **类别不平衡用 focal loss** → 将来可尝试
4. **Transformer 做下游** → 将来可替代 RF

## 引用格式

```
Lai Q, Yao S, Zha Y, Zhang H, Zhang H, Ye Y, Zhang Y, Bai H, Ning K.
Deciphering the biosynthetic potential of microbial genomes using a BGC language processing neural network model.
Nucleic Acids Research. 2025;53:gkaf305.
doi:10.1093/nar/gkaf305
```

## 相关页面

- [[../concepts/BGC]]：BGC 定义与7类标签
- [[../data/antiSMASH]]：BGC-Prophet 也用antiSMASH做标注
- [[../项目总览]]：本项目的差异化定位
