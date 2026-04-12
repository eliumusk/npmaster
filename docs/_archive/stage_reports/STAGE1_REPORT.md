# Stage 1 Report: 数据基座构建
生成时间: Mon Apr  6 03:15:55 UTC 2026
数据目录磁盘占用: **18G**

---
## 1.1 基因组 Panel
- GTDB 版本: GTDB v226
- 原始记录: 715,230
- 筛选流程:
  - bad_qc_numeric: -0
  - fail_completeness(<95): -215,778
  - fail_contamination(>5): -5,518
  - fail_assembly_level(Contig): -275,611
  - no_species: -0
- 通过QC且去冗余(每species一个): **31,643** 种
- 分层采样后选中: **2000** 个基因组
- 实际下载成功: **1996** (4个withdrawn跳过)

### 门分布

| 门 | 池子大小 | 采样目标 | 实际 |
|---|---|---|---|
| p__Pseudomonadota | 10,152 | 500 | 500 |
| p__Bacillota | 6,184 | 500 | 500 |
| p__Actinomycetota | 6,011 | 600 | 596 |
| p__Bacteroidota | 4,123 | 200 | 200 |
| p__Cyanobacteriota | 647 | 100 | 100 |
| p__Desulfobacterota | 495 | - | 9 |
| p__Campylobacterota | 475 | - | 10 |
| p__Planctomycetota | 455 | - | 6 |
| p__Verrucomicrobiota | 329 | - | 5 |
| p__Spirochaetota | 324 | - | 7 |
| p__Chloroflexota | 318 | - | 10 |
| p__Acidobacteriota | 299 | - | 6 |
| p__Bacteroidota_A | 261 | - | 12 |
| p__Desulfobacterota_I | 153 | - | 2 |
| p__Nitrospirota | 142 | - | 4 |

### 基因组大小 (按门)

| 门 | N | min | median | max | mean |
|---|---|---|---|---|---|
| p__Actinomycetota | 596 | 1,314,730 | 5,117,018 | 14,076,249 | 5,699,117 |
| p__Pseudomonadota | 500 | 451,592 | 4,332,792 | 10,840,732 | 4,434,071 |
| p__Bacillota | 500 | 511,268 | 2,875,961 | 8,511,810 | 3,231,889 |
| p__Bacteroidota | 200 | 621,526 | 3,904,854 | 8,898,242 | 4,078,315 |
| p__Cyanobacteriota | 100 | 1,631,014 | 5,483,988 | 11,584,393 | 5,570,818 |
| p__Bacteroidota_A | 12 | 2,264,616 | 3,367,012 | 4,633,311 | 3,348,716 |
| p__Chloroflexota | 10 | 2,512,416 | 3,803,471 | 6,165,587 | 3,835,080 |
| p__Campylobacterota | 10 | 1,484,544 | 1,842,967 | 2,783,018 | 1,974,499 |
| p__Desulfobacterota | 9 | 2,681,094 | 3,073,811 | 6,794,304 | 3,630,869 |
| p__Spirochaetota | 7 | 2,396,298 | 3,201,144 | 4,325,376 | 3,187,858 |
| p__Planctomycetota | 6 | 3,524,469 | 6,631,356 | 7,704,373 | 6,323,377 |
| p__Acidobacteriota | 6 | 3,139,755 | 5,451,881 | 8,459,073 | 5,875,363 |
| p__Verrucomicrobiota | 5 | 3,131,565 | 4,169,785 | 6,172,906 | 4,262,628 |
| p__Nitrospirota | 4 | 1,988,952 | 2,583,010 | 3,687,670 | 2,710,660 |
| p__Fusobacteriota | 3 | 2,218,931 | 3,505,533 | 3,539,325 | 3,087,929 |
| p__Gemmatimonadota | 3 | 4,212,451 | 4,348,178 | 4,511,341 | 4,357,323 |
| p__Poribacteria | 3 | 5,039,918 | 5,923,603 | 6,349,617 | 5,771,046 |
| p__Cloacimonadota | 2 | 1,570,356 | 1,729,973 | 1,889,590 | 1,729,973 |
| p__Deinococcota | 2 | 2,263,488 | 2,287,350 | 2,311,212 | 2,287,350 |
| p__Thermotogota | 2 | 1,890,074 | 1,960,998 | 2,031,923 | 1,960,998 |
| p__Elusimicrobiota | 2 | 1,529,947 | 1,792,595 | 2,055,243 | 1,792,595 |
| p__Aquificota | 2 | 1,693,068 | 1,749,492 | 1,805,917 | 1,749,492 |
| p__Desulfobacterota_I | 2 | 3,000,979 | 3,474,031 | 3,947,083 | 3,474,031 |
| p__Atribacterota | 2 | 1,786,877 | 2,028,310 | 2,269,744 | 2,028,310 |
| p__Myxococcota | 1 | 9,016,985 | 9,016,985 | 9,016,985 | 9,016,985 |
| p__FCPU426 | 1 | 2,915,893 | 2,915,893 | 2,915,893 | 2,915,893 |
| p__Bdellovibrionota | 1 | 2,311,718 | 2,311,718 | 2,311,718 | 2,311,718 |
| p__Edwardsbacteria | 1 | 2,637,412 | 2,637,412 | 2,637,412 | 2,637,412 |
| p__CLD3 | 1 | 3,278,090 | 3,278,090 | 3,278,090 | 3,278,090 |
| p__Armatimonadota | 1 | 5,630,841 | 5,630,841 | 5,630,841 | 5,630,841 |
| p__Vulcanimicrobiota | 1 | 2,645,631 | 2,645,631 | 2,645,631 | 2,645,631 |
| p__Muiribacteriota | 1 | 5,678,741 | 5,678,741 | 5,678,741 | 5,678,741 |

---
## 1.2 antiSMASH 标注

- 处理基因组: 1997
- 失败: 1
- **总BGC数: 19,762**
- truncated (contig edge): 6,270 (31.73%)
- 多标签 BGC: 2,340 (11.84%)

### 7类BGC分布

| 类 | 数量 |
|---|---|
| Terpene | 5,296 |
| NRP | 4,389 |
| Other | 4,286 |
| RiPP | 4,222 |
| Polyketide | 4,122 |
| Alkaloid | 94 |
| Saccharide | 84 |

### 多标签组合 Top 10

| 组合 | 数量 |
|---|---|
| NRP;Polyketide | 857 |
| NRP;Other | 179 |
| Other;Terpene | 166 |
| NRP;RiPP | 131 |
| Other;Polyketide | 126 |
| NRP;Terpene | 106 |
| NRP;Other;Polyketide | 105 |
| Polyketide;Terpene | 98 |
| Other;RiPP | 95 |
| RiPP;Terpene | 92 |

### BGC密度 (每个基因组平均BGC数, 按门)

| 门 | 基因组数 | 总BGC | BGC/genome |
|---|---|---|---|
| p__Myxococcota | 1 | 30 | 30.0 |
| p__Actinomycetota | 596 | 10,916 | 18.32 |
| p__Acidobacteriota | 6 | 94 | 15.67 |
| p__Cyanobacteriota | 100 | 1,225 | 12.25 |
| p__Poribacteria | 3 | 28 | 9.33 |
| p__Bdellovibrionota | 1 | 8 | 8.0 |
| p__Pseudomonadota | 500 | 3,759 | 7.52 |
| p__Planctomycetota | 6 | 45 | 7.5 |
| p__Gemmatimonadota | 3 | 21 | 7.0 |
| p__Desulfobacterota_I | 2 | 13 | 6.5 |
| p__Muiribacteriota | 1 | 6 | 6.0 |
| p__Chloroflexota | 10 | 57 | 5.7 |
| p__Desulfobacterota | 9 | 51 | 5.67 |
| p__FCPU426 | 1 | 5 | 5.0 |
| p__Bacillota | 500 | 2,447 | 4.89 |
| p__Verrucomicrobiota | 5 | 23 | 4.6 |
| p__Bacteroidota | 200 | 880 | 4.4 |
| p__Bacteroidota_A | 12 | 49 | 4.08 |
| p__Elusimicrobiota | 2 | 7 | 3.5 |
| p__Aquificota | 2 | 7 | 3.5 |
| p__Spirochaetota | 7 | 23 | 3.29 |
| p__Nitrospirota | 4 | 13 | 3.25 |
| p__Fusobacteriota | 3 | 9 | 3.0 |
| p__Deinococcota | 2 | 6 | 3.0 |
| p__Thermotogota | 2 | 6 | 3.0 |
| p__CLD3 | 1 | 3 | 3.0 |
| p__Armatimonadota | 1 | 3 | 3.0 |
| p__Vulcanimicrobiota | 1 | 3 | 3.0 |
| p__Cloacimonadota | 2 | 5 | 2.5 |
| p__Edwardsbacteria | 1 | 2 | 2.0 |
| p__Campylobacterota | 10 | 15 | 1.5 |
| p__Atribacterota | 2 | 3 | 1.5 |

### BGC长度分布 (总体)

- min=1,004 bp, median=22,214 bp, max=357,437 bp, mean=29,917 bp

### BGC长度分布 (按类)

| 类 | N | min | median | max |
|---|---|---|---|---|
| Terpene | 5,296 | 1,004 | 21,017 | 264,332 |
| NRP | 4,389 | 1,011 | 44,170 | 357,437 |
| Other | 4,286 | 1,006 | 22,172 | 312,276 |
| RiPP | 4,222 | 1,072 | 20,667 | 312,276 |
| Polyketide | 4,122 | 1,031 | 43,996 | 357,437 |
| Alkaloid | 94 | 3,686 | 21,216 | 135,382 |
| Saccharide | 84 | 5,628 | 52,867 | 246,074 |

### 原始 antiSMASH product 类型 (Top 30)

| product | 数量 |
|---|---|
| terpene | 3,039 |
| NRPS | 2,927 |
| terpene-precursor | 2,261 |
| T1PKS | 2,221 |
| RiPP-like | 1,549 |
| NRPS-like | 1,393 |
| T3PKS | 870 |
| NI-siderophore | 714 |
| betalactone | 612 |
| RRE-containing | 523 |
| NAPAA | 437 |
| arylpolyene | 430 |
| ectoine | 427 |
| NRP-metallophore | 424 |
| hydrogen-cyanide | 388 |
| azole-containing-RiPP | 373 |
| ranthipeptide | 345 |
| butyrolactone | 330 |
| redox-cofactor | 329 |
| lassopeptide | 303 |
| T2PKS | 303 |
| cyclic-lactone-autoinducer | 277 |
| lanthipeptide-class-i | 259 |
| hserlactone | 257 |
| hglE-KS | 235 |
| lanthipeptide-class-iii | 227 |
| PKS-like | 218 |
| lanthipeptide-class-ii | 177 |
| melanin | 165 |
| phosphonate | 145 |

---
## 1.3 窗口切片 + 数据划分

- 窗口: W=16384 bp, S=8192 bp (50% 重叠)
- 总窗口数: **948,362**
- ambiguous: 6.31%

### Split 规模 (基因组级)

| split | 初始 | GCF泄漏修复后 |
|---|---|---|
| train | 1190 | 1190 |
| val | 170 | 170 |
| test_id | 340 | 340 |
| test_ood | 300 | 300 |

### Windows 分布

| split | windows | positive | negative | ambiguous |
|---|---|---|---|---|
| train | 562,331 | 41,467 | 483,462 | 37,402 |
| val | 85,831 | 7,064 | 72,831 | 5,936 |
| test_id | 158,189 | 11,331 | 136,565 | 10,293 |
| test_ood | 142,011 | 6,958 | 128,799 | 6,254 |

### 每个split的类型分布 (positive windows)

| split | Polyketide | NRP | Terpene | RiPP | Saccharide | Alkaloid | Other |
|---|---|---|---|---|---|---|---|
| train | 14665 | 15019 | 9038 | 6688 | 418 | 239 | 8796 |
| val | 2666 | 2867 | 1399 | 1024 | 69 | 35 | 1466 |
| test_id | 3854 | 4233 | 2463 | 1917 | 84 | 41 | 2480 |
| test_ood | 2450 | 2325 | 1870 | 1202 | 21 | 17 | 629 |

### GCF 泄漏检查 (GCF proxy = primary_class|genus|species)

- 泄漏的 GCF 数 (修复前): 0
- 泄漏的 GCF 数 (修复后): **0**
- 从 val/test 移动到 train 的基因组数: 0

---

## 进入 Stage 2 前的检查清单

- [x] 基因组 panel ≥ 1500
- [x] antiSMASH 成功率 99.9%
- [ ] BGC ≥ 20000 (实际 19,762)
- [x] train windows ≥ 40000
- [x] GCF 无泄漏
