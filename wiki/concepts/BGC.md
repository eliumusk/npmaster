---
type: concept
updated: 2026-04-05
status: active
related: [[../项目总览]], [[../refs/BGC-Prophet论文]], [[../data/MIBiG数据库]]
---

# BGC（生物合成基因簇）

## 定义

BGC (Biosynthetic Gene Cluster) 是细菌/真菌基因组中一组**空间相邻、协同工作**的基因，共同编码一个或多个次级代谢产物的合成通路。

典型结构：
- **核心生物合成基因**（如PKS、NRPS）：催化产物骨架的合成
- **修饰酶基因**（氧化还原酶、甲基转移酶等）：修饰骨架
- **转运、调控、抗性基因**：辅助功能

一个BGC通常跨越 5-100 kb，包含 10-100 个基因。

## 7大类（MIBiG标签体系）

| 类别 | 英文 | 代表产物 |
|---|---|---|
| 聚酮 | Polyketide (PKS) | 红霉素、放线紫红素 |
| 非核糖体肽 | NRP (NRPS) | 万古霉素、青霉素前体 |
| 核糖体肽 | RiPP | 乳酸菌素、套索肽 |
| 萜类 | Terpene | 胆固醇、青蒿素 |
| 糖类 | Saccharide | 氨基糖苷类抗生素 |
| 生物碱 | Alkaloid | 罂粟碱、麦角碱 |
| 其他 | Other | 杂项 |

一个BGC可以同时属于多类（multi-label）。

## 为什么BGC预测难

1. **BGC的边界不明确**：基因簇两端的"附属基因"是否算在BGC内没有统一标准
2. **非典型BGC难识别**：antiSMASH的规则基于已知模式，新类型的BGC会漏检
3. **序列多样性大**：同类BGC的不同成员序列差异可能>60%
4. **基因组中BGC占比小**：Strep基因组中BGC约占7-10%，大部分DNA是"背景"
5. **边缘情况**：退化的BGC、horizontal transfer的BGC难以定位

## 检测任务的两种层次

- **基因级别**（BGC-Prophet方法）：给每个基因打标签，1=BGC基因, 0=非BGC基因
- **DNA窗口级别**（本项目方法）：给每个固定长度的DNA窗口打标签，1=含BGC, 0=不含BGC

两种方法的可比性有限，参见 [[../experiments/首次验证实验]] 的讨论。

## 本项目中的使用

- **正样本**：来自 [[../data/MIBiG数据库]] 的2502个实验验证BGC
- **类别标签**：用 `category_vector` 表示（7维binary），详见 [[../data/数据集演进]]
- **窗口切分**：每个BGC按 window_size 切成多个窗口，stride=window_size/2
- **Leave-one-class-out 诊断**：[[../experiments/leave-one-class-out]] 测试模型对每类的泛化能力

## 相关概念

- [[Evo2]]：用来提取DNA embedding的基础模型
- [[GC偏差]]：负样本构造中最常见的陷阱
- [[数据泄露]]：train/test分割时要按BGC和基因组分组
