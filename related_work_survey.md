# Related Work 文献调研

> 为论文 Related Work 部分整理的参考文献，分三个方向。

---

## 一、BGC 检测工具

| 工具 | 论文 | 年份/刊物 | 方法 | 数据集 | 核心指标 |
|------|------|-----------|------|--------|----------|
| **antiSMASH 7/8** | Blin et al. | 2023/2025, *NAR* | 人工策定规则 + pHMM (PFAM/TIGRFAM/SMART + custom)；v7=81类, v8=**101类**；也用 dynamic profiles (Python code) | MIBiG; 无标准 ML benchmark | 无统一 P/R 报告 (作为 reference standard)；NRP metallophore P=97% R=78% |
| **ClusterFinder** | Cimermancic et al. | 2014, *Cell* | Two-state HMM, Pfam domain 频率建模 + Viterbi decode | 732 BGCs + 100 负例基因组；测试: 10 个 annotated genomes | AUROC **0.84**；检出 95% held-out BGCs；但 **>42% 非 BGC domain** 超阈值，FPR 高 |
| **DeepBGC** | Hannigan et al. | 2019, *NAR* | BiLSTM (128 units) + pfam2vec (100d skip-gram, 15,686 Pfam IDs from 3,376 genomes) + 2 binary flags；逐 gene sigmoid | 617 MIBiG BGCs + 10,128 负例；测试: 9 genomes (291 BGCs) | Domain-level AUROC **0.946** (vs CF 0.837)；Avg Precision **0.75** (vs 0.28)；10-fold CV on MIBiG: 0.984；leave-class-out: 0.946；**注意: 这是 domain-level，cluster-level precision 很低** |
| **GECCO** | Carroll et al. | 2021, *bioRxiv* | CRF + Fisher Exact Test 特征选择 (Pfam v33.1 + TIGRFAM v15.0)；threshold T=0.35 | 1,137 MIBiG v2.0 BGCs；应用于 12,000+ genomes | AUPR 显著高于 DeepBGC；比 DeepBGC 快 3x+；在 12k genomes 上检出 ~2x antiSMASH 的 BGCs |
| **BiGCARP** | Rios-Martinez et al. | 2023, *PLOS Comp. Bio.* | **自监督** masked LM (dilated 1D-CNN, ByteNet, 32层) + ESM-1b (1280d) Pfam embeddings；预训练 127k antiSMASH BGCs | 预训练: 127,294 BGCs；fine-tune: DeepBGC 的 617/10,128 集 | 6-genome AUROC **0.941**；9-genome AUROC **0.950**；Avg Precision **0.447** (vs DeepBGC 0.398) |
| **SanntiS** | Sanchez et al. | 2023, *bioRxiv* | 1D-CNN + BiLSTM, InterPro signature 输入；支持宏基因组 | MIBiG + MGnify metagenomic assemblies | 1.9M BGC predictions on MGnify；实验验证了一个 SanntiS-only 发现的 novel 抗菌肽 |
| **BGC-Prophet** | Lai et al., "Deciphering the biosynthetic potential of microbial genomes using a BGC language processing neural network model" | **2025, *NAR*** | Transformer encoder (2层, 5-head) + **ESM-2** (8M, 320d) gene embedding + linear-chain CRF；2.5M 参数 | 12,510 MIBiG v3.1 样本 + 20,000 GTDB 负例 | Gene-level AUROC 0.919；**Precision 0.592** (vs DeepBGC 0.220)；**F1 0.57** (9-genome avg, 所有工具最高)；7-class product classification AUROC **0.988** |

### 核心对比表 (标准 9-genome benchmark)

| 工具 | 评估粒度 | AUROC | Precision | F1 | 关键特点 |
|------|----------|-------|-----------|-----|----------|
| antiSMASH | cluster-level | N/A (reference) | 高 | N/A | 规则高精度，不能发现新类型 |
| ClusterFinder | domain-level | 0.84 | 低 (FPR >42%) | - | 首个 ML 方法 |
| DeepBGC | domain-level | **0.946** | 0.220 | ~0.33 | Recall 高但 precision 低 |
| GECCO | cluster-level | - | 高于 DeepBGC | - | CRF, 快, 低 FP |
| BiGCARP | domain-level | **0.950** | - | - | 自监督预训练 |
| BGC-Prophet | gene-level | 0.919 | **0.592** | **0.57** | Transformer + ESM-2, 目前最高 F1 |

### 关键观察

1. **所有现有方法都依赖 Pfam/InterPro domain annotation 作为输入**——包括最新的 BGC-Prophet 也用 ESM-2 的 protein-level embedding，仍需 gene calling。(a) pipeline 复杂；(b) 对 novel BGC 中 unknown domains 盲区。
2. **我们的方法直接从 raw DNA sequence → Evo2 embedding，跳过了 gene calling + Pfam annotation 步骤**，这是核心差异化。
3. 现有方法都是 **closed-set classification**（BGC vs non-BGC，或已知 BGC 类型分类），没有 GCD 范式下的 novel class discovery 能力。
4. **重要 caveat: domain/gene-level AUROC 会膨胀性能**（大多数 gene 是非 BGC），cluster-level F1 很少超过 0.5-0.6。我们在论文中应使用 cluster-level 或 window-level 指标。

---

## 二、Generalized Category Discovery (GCD)

### 2.1 GCD 基础方法论文

| 论文 | 年份/刊物 | 方法 | 数据集 | 核心指标 |
|------|-----------|------|--------|----------|
| **Vaze et al.**, "Generalized Category Discovery" | 2022, *CVPR* | 提出 GCD 问题定义：labeled set 含 known classes，unlabeled set 含 known + novel classes；方法：contrastive learning + semi-supervised k-means | CIFAR-10/100, ImageNet-100, CUB, Stanford Cars, FGVC-Aircraft | All/Old/New accuracy；CUB: All 51.3, Old 56.6, New 48.7 |
| **Wen et al.**, "Parametric Classification for Generalized Category Discovery: A Baseline Study" (SimGCD) | 2023, *ICCV* | 用 parametric classifier 代替 k-means；self-distillation + entropy regularization | 同上 + Herbarium19, ImageNet-1K | CUB: All 60.3, Old 65.6, New 57.7；显著优于 Vaze et al. |
| **Zhang et al.**, "PromptCAL: Contrastive Affinity Learning via Auxiliary Prompts for GCD" | 2023, *CVPR* | Visual prompt tuning + contrastive affinity learning；用 auxiliary prompts 桥接 known/novel classes | CIFAR-10/100, ImageNet-100, CUB, Stanford Cars | CUB: All 62.9, Old 64.4, New 62.1 |
| **Zhao et al.**, "Learning Semi-supervised Gaussian Mixture Models for Generalized Category Discovery" (μGCD) | 2023, *ICCV* | GMM-based，用 EM 算法在 representation space 中同时建模 known 和 novel clusters | CIFAR-10/100, ImageNet-100, CUB | 在多个 benchmark 上接近 SOTA |
| **Chiaroni et al.**, "Parametric Information Maximization for Generalized Category Discovery" | 2023, *ICCV* | 信息最大化目标，直接优化 mutual information between input and cluster assignment | 同上 | 与 SimGCD 竞争性能 |

### 2.2 GCD 在非 CV 领域的应用

| 论文 | 年份/刊物 | 领域 | 方法 | 数据集 | 核心指标 |
|------|-----------|------|------|--------|----------|
| **Ceccarelli et al.**, "AnnoGCD: a generalized category discovery framework for automatic cell type annotation" | 2024, *NAR Genomics and Bioinformatics* | scRNA-seq 细胞类型 | Semi-supervised block (encoder-decoder + One-Class SVM) + unsupervised block (k-NN + Deep Graph Infomax + DPMM) | BM-CITE (30k cells), LUNG-CITE, PBMC-Multiome 等 5 个数据集 | BM-CITE: All=0.79, Known=0.82, Novel=0.67；优于 OpenNCD 和 PBN |
| **Zhai et al.**, "scGAD: Generalized Cell Type Annotation and Discovery for scRNA-Seq Data" | 2023, *AAAI* | scRNA-seq 细胞类型 | Anchor-based self-supervised learning：几何+语义 mutual nearest neighbors 作为 anchor pairs + connectivity-aware attention | 多个 real scRNA-seq datasets | 同时分类已知 + 聚类新细胞类型，优于 assign-as-unassigned 的传统方法 |
| **Das et al.**, "MedGCD: Generalized Category Discovery in Medical Imaging" | 2025, *MICCAI* | 医学图像 | Weak-to-strong framework + confidence-aware pairwise objective (CAPO) | 3 个 MedMNIST benchmarks | Novel categories +4-15%, overall +2-8% vs ORCA |
| **An et al.** (LOOP), "Generalized Category Discovery with Large Language Models in the Loop" | 2024, *ACL Findings* | NLP 意图发现 | LLM-in-the-loop: Local Inconsistent Sampling + Scalable Query + Refined Neighborhood Contrastive Learning | BANKING (58/19), StackOverflow (15/5), CLINC (113/37) | BANKING H-score **74.60**；avg +7.67% over prior SOTA, novel +11.52% |
| **GLEAN** (Amazon Science) | 2025, *arXiv* | NLP 意图发现 | LLM 三种反馈 (similar instance + category description + pseudo assignment) + neighborhood contrastive + pseudo alignment | BANKING, StackOverflow, CLINC | 全面优于 LOOP |
| **Tian et al.** (STAR), "A Generic Method for Fine-grained Category Discovery in Natural Language Texts" | 2024, *EMNLP* | NLP 细粒度文本分类 | 对数空间语义相似度引导欧氏空间样本分布 + centroid inference | 3 个 benchmark tasks | SOTA on Accuracy, ARI, NMI |
| **Troisemaine et al.** (TabularNCD), "A Method for Discovering Novel Classes in Tabular Data" | 2022, *ICKG* → 2024, *DMKD* | 表格数据 | VIME 自监督预训练 + classification/clustering 联合训练 + pairwise pseudo labels | UCI tabular benchmarks | +36.5% balanced accuracy over baselines；注: 仅 NCD，非完整 GCD |

### 2.3 生物学中的 Novel Class Discovery (相关但非 GCD)

| 论文 | 年份/刊物 | 领域 | 方法 | 核心结果 |
|------|-----------|------|------|----------|
| **scNovel** | 2024, *Briefings in Bioinformatics* | scRNA-seq | 深度学习检测未见过的稀有细胞类型 | 首个自动检测 novel rare cell type 的模型 |
| **DPCfam** | 2022, *PLOS Computational Biology* | 蛋白质家族 | All-vs-all pairwise alignment + Density Peak Clustering | 恢复 81% 的中大型 Pfam 家族；发现 **14,000+ novel candidate families** |
| **BiG-SLiCE / BiG-SCAPE** | 2020-2021, *Nature Chemical Biology / NAR* | BGC 聚类 | 基于规则的 Gene Cluster Family 聚类 | BGC → GCF 聚类，但非学习方法，无 GCD 框架 |

### 关键观察

1. **AnnoGCD + scGAD 是目前将 GCD/NCD 应用于生物学的仅有工作**，且都在 scRNA-seq 细胞类型发现上，不是基因组序列。
2. **GCD 在基因组学/BGC 领域完全空白**——我们的工作是首次将 GCD 框架应用于 biosynthetic gene cluster 检测和 novel BGC type discovery。
3. NLP 中的 LOOP/GLEAN (意图发现) 与我们的 problem setting 有相似性（已知类别 + 发现新类别），可以作为 cross-domain related work 提及。
4. **表格数据上甚至还没有完整的 GCD 方法**（只有 NCD），说明 GCD 向非 CV 领域的扩展仍处于早期阶段。
5. DPCfam 的 14k novel protein family 发现证明了生物序列中存在大量未知类别等待发现——这为我们在 BGC 领域应用 GCD 提供了动机支撑。

---

## 三、DNA 基础模型 (Foundation Models for Genomics)

| 模型 | 论文 | 年份/刊物 | 架构 | 参数量 | 预训练数据 | 上下文长度 | 下游任务 & 指标 |
|------|------|-----------|------|--------|------------|------------|-----------------|
| **Evo2** | Nguyen, Poli et al., "Genome modeling and design across all domains of life with Evo 2" | 2025 preprint → **2026, *Nature*** | StripedHyena2：三种卷积算子 (Hyena-SE L=7 局部 + Hyena-MR L=128 中程 + Hyena-LI 隐式长程) + rotary self-attention 交替 | 1B / 7B / 40B | OpenGenome2: 9.3T tokens (40B), 128k+ genomes，覆盖 bacteria/archaea/eukarya/phage | 原生 8,192 → midtraining 扩展至 **1M bp** | ClinVar coding SNV AUROC **0.96**；noncoding SNV 无监督 #1；indel 所有模型 #1；ProteinGym Spearman 0.360 (7B)；gene essentiality (lncRNA)；线粒体基因组生成 (~16kb, 基因数正确)；合成基因 70% 命中 Pfam domain |
| **Evo (v1)** | Nguyen et al., "Sequence modeling and design from molecular to genome scale with Evo" | 2024, *Science* | StripedHyena: 29 hyena layers + 3 rotary attention across 32 blocks, width 4096 | 7B | OpenGenome: 2.7M prokaryotic + phage genomes, ~300B tokens | 131,072 bp | Gene essentiality: Lambda phage AUROC **0.90**, *P. aeruginosa* AUROC **0.84**；promoter-RBS expression Spearman 0.61 (vs RBS Calculator 0.39)；CRISPR-Cas: EvoCas9-1 可比 SpCas9 (79.9% AA identity)；whole-genome generation ~1Mb |
| **HyenaDNA** | Nguyen et al., "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution" | 2023, *NeurIPS* (Spotlight) | Hyena operator stack, sub-quadratic, 比 Transformer 快 160x | 参数量极小 (数百万级) | Human reference genome (hg38) | 最长 **1M bp** | NT benchmarks: **12/18** datasets SOTA；GenomicBenchmarks: **7/8** SOTA, avg +10 accuracy points；首次实现基因组 in-context learning |
| **Nucleotide Transformer** | Dalla-Torre et al., "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics" | 2023 preprint → 2024, *Nature Methods* | Standard Transformer (BERT-style MLM), 6-mer tokenization | 50M / 100M / 250M / 500M / 2.5B | 3,202 human genomes + 850 diverse species | ~6,000 bp | 18 genomic tasks: promoter/splice site **全部最优** (vs DNABERT-2, HyenaDNA-32k, Enformer)；建立标准化评测 leaderboard |
| **DNABERT** | Ji et al., "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome" | 2021, *Bioinformatics* | BERT-base (12层, k=6 mer tokenization) | ~110M | Human reference genome | 512 bp | Promoter: +0.335 acc, +0.554 MCC over DeePromoter；TFBS 690 ENCODE datasets: mean acc **0.918**, F1 0.919 (p < 4.5e-100 vs DeepSEA)；Splice site acc 0.923 |
| **DNABERT-2** | Zhou et al., "DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome" | 2024, *ICLR* | BERT + **BPE tokenization** (替代 k-mer) + ALiBi 位置编码 | ~117M (NT-2.5B 的 1/21) | Multi-species genomes | 最长 ~10,000 bp | GUE benchmark (36 datasets, 9 tasks)；**92x 更少 GPU 预训练时间**，性能可比 NT-2.5B |
| **Caduceus** | Schiff et al., "Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling" | 2024, *ICML* | BiMamba (双向 SSM) + **reverse complement equivariance**；Caduceus-PS (参数共享) / Caduceus-Ph (后处理) | ~数千万级 (d=256, 16层) | Human reference genome | 长程 | 在 variant effect prediction 上超越 NT v2 (500M)；>100kb 距离 TSS 的 variant 超越 Enformer；**首个 RC-equivariant DNA LM** |
| **GenSLMs** | Zvyagin et al., "GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics" | 2023, *IJHPCA* (**Gordon Bell Prize**) | GPT decoder, **codon-level tokenization** (64词表), 64 heads × 64 layers | 25M / 250M / 2.5B / 25B | 110M prokaryotic gene sequences (BV-BRC), fine-tune on 1.5M SARS-CoV-2 genomes | ~2k codons | COVID variant clustering (无标签分离 VOC)；生成序列回溯匹配 Eris/Pirola subvariants；1.63 Zettaflops 训练 |
| **scGPT** | Cui et al., "scGPT: toward building a foundation model for single-cell multi-omics using generative AI" | 2024, *Nature Methods* | GPT + specialized attention masks, gene token + expression value 输入 | ~50M | 33M human single cells (CELLxGENE) | Gene set (非 DNA 序列) | Cell type annotation, batch correction, perturbation prediction, gene network inference: 5 tasks 均 SOTA (注: 2025 benchmark 显示 perturbation 任务上被 mean-baseline 超越，需谨慎引用) |

### 关键观察

1. **Evo2 是我们使用的 backbone**——它是目前最大的 DNA 基础模型 (40B)，且是唯一覆盖所有生命域预训练数据的模型。已发表于 Nature (2026)。
2. **没有任何 DNA 基础模型被用于 BGC 检测任务**——这是我们工作的核心 novelty 之一。
3. 现有 DNA FM 下游任务集中在：fitness prediction、regulatory element、variant effect——都是短序列 (< 10kb) 任务。**我们的 BGC 检测涉及 16kb windows，是 DNA FM 在较长 genomic context 上的新应用场景。**
4. 与 Nucleotide Transformer/DNABERT (最长 ~10kb) 不同，Evo2 的 StripedHyena2 天然支持 1M context，适合我们的 16kb 窗口甚至更长的完整 BGC (~20-100kb)。
5. Evo2 的 gene essentiality 和 variant effect 结果表明模型已经捕获了 functional genomic signals，为 BGC 检测提供了合理的 representation 基础。

---

## 四、论文 Related Work 写作建议

建议 Related Work 分三个小节：

### §2.1 Computational BGC Detection
- antiSMASH (rule-based 代表) → ClusterFinder (probabilistic) → DeepBGC/GECCO/SanntiS (ML-based)
- 指出共同局限：依赖 Pfam annotation，closed-set，无 novel class discovery

### §2.2 DNA Foundation Models
- 从 DNABERT → Nucleotide Transformer → HyenaDNA → Evo/Evo2 的演进
- 强调 Evo2 的优势：长上下文、全生命域预训练、single-nucleotide resolution
- 指出 gap：未被应用于 BGC 检测

### §2.3 Generalized Category Discovery
- 从 Vaze et al. (2022) 问题定义 → SimGCD/PromptCAL 方法发展
- 跨领域扩展：NLP (LOOP/GLEAN, 意图发现)、医学影像 (MedGCD)、表格数据 (TabularNCD, 仅 NCD)
- 生物学：AnnoGCD + scGAD (scRNA-seq)，DPCfam (蛋白质家族, unsupervised)
- 指出 gap：GCD 未被应用于基因组序列/BGC；我们是首次

---

## 五、BibTeX 条目 (部分关键文献)

```bibtex
@article{blin2023antismash,
  title={antiSMASH 7.0: new and improved predictions for detection, regulation, chemical structures and visualisation},
  author={Blin, Kai and Shaw, Simon and Augustijn, Hannah E and Reitz, Zachary L and Biber
man, Marnix H and Mohimani, Hosein and Weber, Tilmann and others},
  journal={Nucleic Acids Research},
  volume={51},
  number={W1},
  pages={W46--W50},
  year={2023}
}

@article{hannigan2019deepbgc,
  title={A deep learning genome-mining strategy for biosynthetic gene cluster prediction},
  author={Hannigan, Geoffrey D and Prihoda, David and Palicka, Andrej and others},
  journal={Nucleic Acids Research},
  volume={47},
  number={18},
  pages={e110},
  year={2019}
}

@article{carroll2021gecco,
  title={Accurate de novo identification of biosynthetic gene clusters with GECCO},
  author={Carroll, Laura M and Larralde, Martin and Flor, Jonas Simon and others},
  journal={bioRxiv},
  year={2021}
}

@article{cimermancic2014clusterfinder,
  title={Insights into secondary metabolism from a global analysis of prokaryotic biosynthetic gene clusters},
  author={Cimermancic, Peter and Medema, Marnix H and Claesen, Jan and others},
  journal={Cell},
  volume={158},
  number={2},
  pages={412--421},
  year={2014}
}

@inproceedings{vaze2022generalized,
  title={Generalized Category Discovery},
  author={Vaze, Sagar and Han, Kai and Vedaldi, Andrea and Zisserman, Andrew},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{wen2023simgcd,
  title={Parametric Classification for Generalized Category Discovery: A Baseline Study},
  author={Wen, Xin and Zhao, Bingchen and Qi, Xiaojuan},
  booktitle={ICCV},
  year={2023}
}

@article{ceccarelli2024annogcd,
  title={AnnoGCD: a generalized category discovery framework for automatic cell type annotation},
  author={Ceccarelli, Federico and others},
  journal={NAR Genomics and Bioinformatics},
  volume={6},
  number={4},
  year={2024}
}

@article{nguyen2024evo,
  title={Sequence modeling and design from molecular to genome scale with Evo},
  author={Nguyen, Eric and Poli, Michael and Durrant, Matthew G and others},
  journal={Science},
  volume={386},
  number={6723},
  year={2024}
}

@article{nguyen2025evo2,
  title={Genome modeling and design across all domains of life with Evo 2},
  author={Nguyen, Eric and Poli, Michael and others},
  journal={Nature},
  year={2026},
  note={Preprint: bioRxiv 2025.02.18.638918}
}

@inproceedings{nguyen2023hyenadna,
  title={HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution},
  author={Nguyen, Eric and Poli, Michael and Faez, Marjan and others},
  booktitle={NeurIPS},
  year={2023}
}

@article{dallatorre2023nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza-Revilla, Javier and others},
  journal={Nature Methods},
  year={2023}
}

@article{ji2021dnabert,
  title={DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome},
  author={Ji, Yanrong and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V},
  journal={Bioinformatics},
  volume={37},
  number={15},
  pages={2112--2120},
  year={2021}
}

@inproceedings{zhou2024dnabert2,
  title={DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome},
  author={Zhou, Zhihan and Ji, Yanrong and Li, Weijian and others},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{schiff2024caduceus,
  title={Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling},
  author={Schiff, Yair and Kao, Chia-Hsiang and Gokaslan, Aaron and others},
  booktitle={ICML},
  year={2024}
}

@article{zvyagin2023genslms,
  title={GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics},
  author={Zvyagin, Maxim T and Brace, Alexander and others},
  journal={IJHPCA},
  year={2023}
}

@article{sanchez2023sanntis,
  title={Expansion of novel biosynthetic gene clusters from diverse environments using SanntiS},
  author={Sanchez, Santiago and Rogers, Joel D and Rogers, Alexander B and others},
  journal={bioRxiv},
  year={2023}
}

@article{lai2025bgcprophet,
  title={Deciphering the biosynthetic potential of microbial genomes using a BGC language processing neural network model},
  author={Lai, Qilong and Yao, Shuai and Zha, Yuguo and others},
  journal={Nucleic Acids Research},
  volume={53},
  number={7},
  pages={gkaf305},
  year={2025}
}

@article{riosmartinez2023bigcarp,
  title={Deep self-supervised learning for biosynthetic gene cluster detection and product classification},
  author={Rios-Martinez, Carolina and Bhattacharya, Nicholas and Amini, Ava P and Crawford, Lorin and Yang, Kevin K},
  journal={PLOS Computational Biology},
  volume={19},
  number={5},
  pages={e1011162},
  year={2023}
}

@inproceedings{zhai2023scgad,
  title={Generalized Cell Type Annotation and Discovery for Single-Cell RNA-Seq Data},
  author={Zhai, Yuyao and Chen, Liang and Deng, Minghua},
  booktitle={AAAI},
  year={2023}
}

@inproceedings{an2024loop,
  title={Generalized Category Discovery with Large Language Models in the Loop},
  author={An, Wenbin and Shi, Wenkai and Tian, Feng and others},
  booktitle={ACL Findings},
  year={2024}
}

@article{glean2025,
  title={GLEAN: Generalized Category Discovery with Diverse and Quality-Enhanced LLM Feedback},
  author={{Amazon Science}},
  journal={arXiv:2502.18414},
  year={2025}
}

@inproceedings{tian2024star,
  title={A Generic Method for Fine-grained Category Discovery in Natural Language Texts},
  author={Tian, Chang and Blaschko, Matthew B and Yin, Wenpeng and others},
  booktitle={EMNLP},
  year={2024}
}

@article{troisemaine2022tabularncd,
  title={A Method for Discovering Novel Classes in Tabular Data},
  author={Troisemaine, Colin and others},
  journal={ICKG / Data Mining and Knowledge Discovery},
  year={2022/2024}
}

@inproceedings{das2025medgcd,
  title={MedGCD: Generalized Category Discovery in Medical Imaging},
  author={Das, Ankit and Gautam, Chandan and Agrawal, Pritee and Yang, Feng and Liu, Yong and Savitha, Ramasamy},
  booktitle={MICCAI},
  year={2025}
}
```

---

*Generated: 2025-04-07. 部分 BibTeX 条目的 author list 已简化，正式使用时请核对完整作者列表。*
