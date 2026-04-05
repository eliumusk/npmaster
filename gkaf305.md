<!-- Meanless: Nucleic Acids Research, 2025, 53, gkaf305<br>20<br>OXFORD<br>https://doi.org/10.1093/nar/gkaf305<br>Methods<br>anniversary<br>してもあるときっきっぱっぱりをすっちゃっています。そのような気持ちを手にござってもうと思っています。そのようなくなっても -->

# Deciphering the biosynthetic potential of microbial genomes using a BGC language processing neural network model

Qilong Lai \( {}^{1, \dagger  } \) ,Shuai Yao \( {}^{1, \dagger  } \) ,Yuguo Zha \( {}^{{11}, \dagger  } \) ,Haohong Zhang \( {}^{1,1} \) ,Haobo Zhang \( {}^{1,1} \) Ying Ye \( {}^{2} \) , Yonghui Zhang \( {}^{\oplus 2} \) ,Hong Bai \( {}^{\oplus 1, * } \) ,Kang Ning \( {}^{\oplus 1, * } \)

\( {}^{1} \) MOE Key Laboratory of Molecular Biophysics of the Ministry of Education,Hubei Key Laboratory of Bioinformatics and Molecular-imaging, Center of AI Biology, Department of Bioinformatics and Systems Biology, College of Life Science and Technology, Huazhong University of Science and Technology, Wuhan 430074, Hubei, China

\( {}^{2} \) Hubei Key Laboratory of Natural Medicinal Chemistry and Resource Evaluation,School of Pharmacy,Tongji Medical College,Huazhong University of Science and Technology, Wuhan 430030, Hubei, China

*To whom correspondence should be addressed. Email: baihong@hust.edu.cn

Correspondence may also be addressed to Kang Ning. Email: ningkang@hust.edu.cn

\( {}^{ \dagger  } \) The first three authors should be regarded as Joint First Authors.

## Abstract

Biosynthetic gene clusters (BGCs), key in synthesizing microbial secondary metabolites, are mostly hidden in microbial genomes and metagenomes. To unearth this vast potential, we present BGC-Prophet, a transformer-based language model for BGC prediction and classification. Leveraging the transformer encoder, BGC-Prophet captures location-dependent relationships between genes. As one of the pioneering ultrahigh-throughput tools, BGC-Prophet significantly surrasses existing methods in efficiency and fidelity, enabling comprehensive pan-phylogenetic and whole-metagenome BGC screening. Through the analysis of 85 203 genomes and 9428 metagenomes, BGC-Prophet has profiled an extensive array of sub-million BGCs. It highlights notable enrichment in phyla like Actionmycetota and the widespread distribution of polyketide, NRP and RiPP BGCs across diverse lineages. It reveals enrichment patterns of BGCs following important geological events, suggesting environmental influences on BGC evolution. BGC-Prophet's capabilities in detection of BGCs and evolutionary patterns offer contributions to deeper understanding of microbial secondary metabolites and application in synthetic biology.

<!-- Media -->

## Graphical abstract

<!-- figureText: BGC-Prophet<br>Biosynthetic gene cluster prediction<br>85,203 genomes<br>Transformer encoder<br>Comprehensive profile of BGCs<br>- -<br>HOT<br><br><br>Pantholes<br><br>Pan-phylogenetic screening and whole-metagenome screening of BGCs<br>Relative abundance<br>Highlights<br>ESM2 encoding<br>>Captures the distant location-dependent relationships among genes<br>>Achieves Ultrahigh-throughput and accuracy<br>>Builds the comprehensive profile of BGCs<br>BGC profile analyzing after important geological events<br>>Reveals the profound enrichment<br>illèo<br>0.005<br>9,428 Metagenomes<br>Ultrahigh throughput & accuracy<br>BGC enrichment pattern -->

<img src="https://cdn.noedgeai.com/bo_d78vubc91nqc73epp0u0_0.jpg?x=155&y=1278&w=1459&h=594&r=0"/>

<!-- Media -->

---

<!-- Footnote -->

Received: October 9, 2024. Revised: March 21, 2025. Editorial Decision: March 28, 2025. Accepted: April 3, 2025

© The Author(s) 2025. Published by Oxford University Press on behalf of Nucleic Acids Research.

This is an Open Access article distributed under the terms of the Creative Commons Attribution-NonCommercial License

(https://creativecommons.org/licenses/by-nc/4.0/), which permits non-commercial re-use, distribution, and reproduction in any medium, provided the

permissions can be obtained through our RightsLink service via the Permissions link on the article page on our site—for further information please contact

journals.permissions@oup.com.

<!-- Footnote -->

---

<!-- Meanless: original work is properly cited. For commercial re-use, please contact reprints@oup.com for reprints and translation rights for reprints. All other -->


<!-- Meanless: 2<br>Lai et al. -->

## Introduction

Microbial secondary metabolites, one of the important sources of natural products, are generated through the coordinated action of numerous genes organized into biosynthetic gene clusters (BGCs) [1, 2]. Across all of the species on the tree of life, these natural products have thousands of different chemical structures, including nonribosomal peptides (NRPs), polyketides, saccharides, terpenes, and alkaloids, that facilitate an organism's ability to thrive in a particular environment [3, 4]. These secondary metabolites also demonstrate efficacy across multiple therapeutic areas, including antimicrobial and cancer chemotherapy [5, 6]. The biosynthesis of these compounds involves multigene loci called BGCs, which encode the biosynthetic pathways for one or more specific compounds [7, 8]. With the exponential growth of genomic data, identifying and classifying BGCs from microbial genomes or metage-nomic assembled genomes (MAGs) has become a promising venue for exploring and exploiting natural product diversity [9, 10]. Developments in computational omics technologies have provided new means to assess the hidden diversity of natural products, revealing new potential for drug discovery [11, 12].

A BGC encodes a series of genes involved in biosynthetic or metabolic pathways, which are arranged in sequential order within a genome. These genes and proteins work together to produce one or more small molecular compounds, such as penicillin [13, 14]. BGC comprises a cluster of spatially adjacent colocalization genes, including biosynthetic genes and auxiliary genes (e.g. transport-related genes and regulatory genes) [15, 16]. Biosynthetic genes catalyze the initial step in a biosynthetic pathway that uses some primary metabolites to synthesize the skeleton structure of a natural product. Common biosynthetic genes include terpene cyclase, polyketide synthase, and NRP synthases. There are also biosynthetic genes encoding tailoring enzymes, such as ox-idoreductases, dehydrogenases, decarboxylases, and methyl-transferases. These biosynthetic genes play key catalytic roles in the formation of microbial secondary metabolites. In addition to biosynthetic genes, many BGCs also harbor genes that synthesize specialized moieties for a pathway. For example, the erythromycin gene cluster encodes a set of enzymes for the biosynthesis of two deoxy sugars that are appended to the polyketide aglycone [17]. In many cases, transporters, regulatory elements, and genes that mediate host resistance are also contained within the BGC [18]. Although some BGCs are so well understood that the biosynthesis of their small molecule products has been reconstituted in heterologous hosts, little is known about the vast majority of BGCs, including those that have been linked to a small molecule product.

Identifying BGCs directly from genomic sequences is essential for navigating the natural product space and nominating new natural products. The explosion of microbial genomic data, including complete and partial genome sequences, has led to a transformative change in how computational methods are employed in natural product drug candidate discovery. Computational approaches are being developed to predict BGCs based on genome sequences alone, fuelled by data on known biosynthetic pathways and their chemical products, which are currently standardized with predicted BGCs stored in public databases [19]. Current methods for mining microbial genomes for BGC identification can be roughly divided into two main types: rule-based and machine learning approaches. Identifying natural product BGCs still largely relies on rule-based methods such as those used in antiSMASH [15, 16] and PRISM [20]. Although these approaches are successful at detecting known BGC categories, including widely accepted categories such as alkaloids, NRPs, polyketides, ri-bosomally synthesized and post-translationally modified peptides (RiPPs), saccharides and terpenes, they are less proficient at identifying novel categories of BGC [21, 22]. In these more complex cases of identifying novel BGCs, machine learning algorithms have been shown to offer significant advantages over rule-based methods. For example, ClusterFinder [23], NeuRiPP [24] and DeepRiPP [25] each use machine learning to identify BGCs. These methods often involve a tradeoff in terms of efficiency and accuracy, have a higher false positive rate (FPR) than rule-based approaches and suffer from false negatives for known categories of BGC. Recently, deep learning approaches, including DeepBGC [26], e-DeepBGC [27], Deep-BGCPred [28], BiGCARP [29], GECCO [30], and San-ntiS [31], have been developed for BGC annotation. All of these deep learning approaches call protein families (Pfam) by means of using curated profile-Hidden Markov Models (pH-MMs) together with employing neural network models such as the bidirectional long short-term memory (BiLSTM) recurrent neural network [26-28, 31]. Although these deep learning approaches have improved the ability to detect BGCs from bacterial genomes and harness great potential to detect novel categories of BGCs, they have common drawbacks: (i) supervised machine learning approaches suffer from small number of training data (ii) BiLSTM usually loses long memories and is unable to capture distant location-dependent relationships between biosynthetic genes, (iii) while Pfam heavily relies on manual determination by experts to define the scope of each domain, and (iv) the utilization of pHMMs is computationally intensive.

The prediction and classification of BGCs remain challenging due to several factors inherent in existing methods. Traditional rule-based methods, such as antiSMASH, perform well for known biosynthetic pathways but are limited by scalability issues when analyzing large-scale genomic datasets. Additionally, these methods struggle to detect novel BGCs, especially those that do not fit into predefined biosynthetic classes. Furthermore, while deep learning methods like DeepBGC show promise, they often rely on BiLSTM architectures, which may not effectively capture the location-dependent relationships between genes in complex BGCs. These challenges highlight the need for a more efficient and flexible approach, such as BGC-Prophet, which leverages transformer-based models to overcome these limitations.

To address these limitations, we proposed BGC-Prophet, a deep learning approach that leverages a language model to accurately and efficiently identify known BGCs and extrapolate novel BGCs among the microbial universe. Previous studies have shown the success of language models [29, 32]. Inspired by this, our BGC-Prophet employs the powerful transformer-based language model [33, 34], which captures the location-dependent relationships among biosynthetic genes for improved BGC detection and classification.

Our experiments showed that BGC-Prophet achieved a >90% area under the receiver operating characteristic curve (AUROC) in the validation datasets and that the ability of BGC identification was comparable to that of existing tools such as DeepBGC. BGC-Prophet is one of the pioneering ultrahigh-throughput (UHT) methods that is several orders of magnitude faster than existing tools such as DeepBGC, enabling pan-phylogenetic screening and whole-metagenome screening of BGCs. By analyzing 85, 203 genomes and 9428 metagenomes, we constructed a comprehensive profile of sub-million BGCs from the majority of bacterial and archaeal lineages and found that BGCs were enriched in phyla such as Actinomycetota, while BGCs for polyketide, NRP, and RiPP were widely distributed among lineages. Importantly, the profound enrichment of BGCs in microbes after important geological events has been revealed: both the Great Oxidation and Cambrian Explosion events led to a surge in BGC abundance and diversity, particularly in polyketides. These findings suggest that microorganisms could adapt to changing environments by evolving BGC to produce specific secondary metabolites. In summary, BGC-Prophet enables accurate and fast detection of BGCs on a large scale, holds great promise for expanding BGC knowledge, and sheds light on the evolutionary patterns of BGCs for possible applications in synthetic biology.

<!-- Meanless: にしますのです。できてきますができるようなこともありますのですができるか？です。そうなく、そのようなくなっても -->


<!-- Meanless: BGC-Prophet for BGC mining using AI<br>3<br>GZOZ !!duty gl. uo Jasn Kuedy! / (60/oU/98_1 % @usag/SELUQUEL /Q 02/LELUE/SELUEL/JUL/QLUEL/QUIR/QUIRO/QUOJ/QUO! WOW ререгруг!SH/HW WOJ рарегимод. -->

## Materials and methods

## Datasets used in this study

We manually curated several datasets in this study, including MIBiG v3.1 (Minimum Information about a BGC [3]), 6KG (5886 genomes from the GTDB RS214 database [35]), NG (nine genomes used in ClusterFinder and DeepBGC [23, 26]), AG (982 genomes from the genus Aspergillus), 85KG (85203 available species/genomes in GTDB RS214 [35]), and MG (metagenomes from 47 metagenomic studies [36]). These datasets are used for a variety of purposes, with MIBiG and 6KG being used to construct training and testing sets, NG and AG being used to validate and compare the performance of various methods, and 85KG and MG being used for large-scale genome mining of BGCs (Supplementary Tables S1-S3).

## The MIBiG dataset

The MIBiG dataset specification provides a robust community standard for annotating and metadata on BGCs and their molecular products, which contains 2502 experimentally validated BGCs.

## The 6KG dataset

The 6KG dataset comprises a set of phylogenetically diverse genomes that were manually curated in GTDB RS214, and it contains 5886 species/genomes that spread across the bacterial evolutionary tree.

## The NG dataset

The NG dataset comprises nine bacterial genomes that were examined in previous studies, including those of ClusterFinder and DeepBGC [23, 26]. These genomes included a total of 291 BGCs, none of which were used for training.

## The AG dataset

The AG dataset contains a total of 982 genomes from the genus Aspergillus in the NCBI genome database. We utilized BGC-Prophet and antiSMASH to mine BGCs in these genomes and generated a comparison map between the BGCs identified by antiSMASH and BGC-Prophet on the Aspergillus genomes.

## The 85KG dataset

The 85KG dataset contains 85203 available species/genomes (one genome corresponds to one species) in GTDB RS214. We utilized BGC-Prophet to mine BGCs in those genomes and constructed a comprehensive profile of BGCs in the genomes of the majority of the bacterial and archaeal lineages.

## The MG dataset

The MG dataset contains metagenomes involved in 47 studies (Supplementary Table S3). These metagenomic data included 1792 406 629 contigs from 9428 metagenomic samples, of which 6238 438 contigs with nucleotide sequence lengths \( > {20000} \) were retained.

## Taxonomic classifications for metagenomes

We used 9428 metagenomic assemblies corresponding to 47 studies from the human microbial environment. These metagenomic assemblies were binned using MetaBAT2 (version 2.12.1), and a total of 160 814 bins (or MAGs) were obtained. Taxonomic annotation was subsequently performed on the resulting bins using the Genome Taxonomy Database Toolkit (GTDB-Tk, version 2.3.2) with reference to the GTDB release 214.0. A total of 160 814 bins were generated from 9428 metagenomic samples. Among them, 132 809 bins were successfully assigned to species, while 28 005 bins remained unclassified and were designated Unclassified (5875 bins), Unclassified Archaea (316 bins), or Unclassified Bacteria (21 814 bins), representing unknown species.

## Positive and negative sample generation

To train the language model of BGC-Prophet, we manually curated a training dataset of positive and negative samples. The MIBiG and 6KG datasets were used to build the positive and negative samples. Before generating positive and negative samples, we used antiSMASH (v6) to identify BGCs in a public reference set of 5886 microbial genomes (6KG dataset). For each reference genome, regions predicted to be part of a BGC were removed, and these pruned genomes without BGC-like regions served as the non-BGC gene library.

## Positive sample generation

The positive samples are derived from the 2502 BGCs in the MIBiG dataset. For each BGC in the MIBiG dataset, we applied two-sided padding with non-BGC genes (as described in the previous paragraph) until the gene sequence length equaled 128. Considering that the longest BGC in MIBiG consists of 115 genes and the gap (i.e. the average number of non-BGC genes) between BGCs in the genomes from the 6KG dataset (Supplementary Fig. S1), we set the maximum gene sequence length of a positive sample to 128. We repeated the generation procedure five times for each BGC in the MIBiG dataset, resulting in 12 510 positive samples.

## Negative sample generation

In the generation of a negative sample (non-BGC), a major challenge is to ensure that non-BGC have a certain degree of similarity with genes in BGCs but lack the semantic information preserved in BGCs (i.e. the order of genes in BGCs). To generate a single negative sample, a random region from the non-BGC gene library was selected, and a subregion containing 128 continuous genes was randomly selected from the selected region. In total, 20 000 negative samples were generated.


<!-- Meanless: 4 Lai et al.<br>(5)考えることができます。コーチガスが多いのではないでしょうか？こういうことで、コーチガスが -->

## Labeling the samples

According to the MIBiG database, there are seven categories of BGCs, including alkaloids, NRPs, polyketides, ribosomally synthesized and RiPPs, saccharides, terpenes, and others. Notably, each BGC may have more than one category, so the prediction of BGC categories is a multilabel seven-category problem. For example, a positive sample derived from BGC with the MIBiG accession BGC0000356 was labeled with both the alkaloid and NRP categories. For all the negative samples, they are not labeled into any of the seven categories.

## BGC-Prophet implementation

## Token of the BGC-Prophet model

In the field of natural language processing (NLP), the minimal semantic unit is called a "token", which makes up sentences. BGC-Prophet is a language processing neural network that takes genes as tokens to represent a BGC or non-BGC (sentence). Previously described methods, ClusterFinder and DeepBGC use Pfam domains as tokens that effectively balance genetic information loss and computational complexity. However, Pfam relies on manual determination by experts to define the scope of each domain, and the utilization of pHMMs for identifying conserved Pfam domains in sequences is computationally intensive. Therefore, a trade-off between the number of Pfam alignments and computational speed must be considered. The situation of multiple Pfam domains originating from the same gene requires the model to learn such relationships separately. Here, we choose genes as tokens because they are more natural and do not require additional operations.

## Vector representation of token

Each gene present in the training and testing samples needs to be represented as a word embedding vector to serve as input for subsequent language models. We used the ESM-2 8M model [evolutionary scale modeling (ESM): pretrained language models for proteins, version 2 with 8 million parameters] to generate vector representations of genes. ESM is the SOTA general-purpose protein language model, which can be used to predict structure, function and other protein properties directly from individual sequences [37]. For every positive and negative sample, we applied the ESM-2 8M model to generate a vector representation of genes (embedding dimension of 320). The ESM-2 8M model generates embedding of genes and removes the dependence between acquiring vector representations of tokens and training language models. The vector representation of tokens generated by the ESM-2 8M model directly from individual sequences is more concise and breaks the limitations inherent in the training samples, thus providing a higher possibility of predicting unknown BGCs. All genes in the training data are inferred using the ESM-2 8M model, and the mean of the model's last layer output is selected as the final word embedding for the sequence. This implies that our word vectors tend to represent higher-level information and can more effectively leverage GPU acceleration for computational processes.

## Model architecture and configuration

Many language models, such as long short-term memory (LSTM) and bidirectional encoder representations from transformers, have been proposed and used to solve the problem of text classification [33, 34]. Here, we proposed a BGC language processing neural network model, BGC-Prophet, to detect known and predict potentially novel BGCs from genome sequences. BGC-Prophet employs a language model (i.e. transformer encoder) [33] for BGC identification and classification. The transformer encoder is a neural network model of a specific architecture that uses a multi-head self-attention mechanism to speed up training. The self-attention mechanism introduced in the transformer encoder makes it suitable for parallel computation and better than the RNN or LSTM in terms of accuracy. In this study, PyTorch v2.0.0 was used to implement the transformer encoder structure of BGC-Prophet, which learns the representation of gene sequences for different downstream tasks.

In this study, the parameters of the transformer encoder are set as follows (Supplementary Fig. S2): The input dimension is set to 320, which is equivalent to the dimensionality of the embedding generated by the ESM-2 8M model. Then, pre-layer normalization is used to accelerate the convergence of the model [38]. The positional encoding adopts classical sine-cosine position coding, which does not require additional training and captures relative positional relationships between genes effectively. The transformer encoder is configured with two five-head self-attention layers and a dropout rate of \( {10}\% \) . The model was trained using the AdamW optimizer with a learning rate of 1e-2 and a batch size of 64. Given that the number of training epochs is not fixed, an early stopping strategy is employed, where the loss value of the verification set stops improving after 20 epochs without decreasing, and the model obtained from the epoch with the lowest loss value on the verification set is chosen as the final model.

## BGC gene detection and product classification

We assigned two downstream tasks to BGC-Prophet. The first task is predicting the BGC gene loci of a given BGC sequence, and the second task is predicting the BGC category of a given region on a genome.

The first task for BGC-Prophet is to predict the BGC loci of a given gene sequence. Specifically, given the sequence to be predicted to be composed of multiple genes, we determined whether each gene was part of a BGC according to the positional relationships of all the genes. There may be no correlation between the genes that make up the sequence to be predicted, while the gene tag sequence of each gene that makes up the BGC is related in order. Therefore, the task can be statistically modeled using a linear-chain conditional random field (linear-CRF) [39]. According to the linear-chain CRF algorithm, the input is the sequence to be predicted, and the output is the gene tag sequence. In this paper, the downstream neural network is set as a fully connected layer with 128 timesteps, and the weight of each timestep is shared, which is different from DeepBGC. After passing through the fully connected layer, the hidden state vector dimension of the transformer encoder is reduced from 320 to 128, 128 to 32, and 1, which represents the probability that a given gene is part of a BGC. The Gaussian Error Linear Unit (GELU) [40] is used as the activation function for each fully connected layer; finally, the sigmoid activation function is applied. The final fully connected layer outputs a scalar between 0 and 1 that measures how confident the model is that the gene belongs to the BGC. The loss function of the model in this paper is binary cross entropy, and the AdamW [41] optimization algorithm is used to make the loss functions converge.


<!-- Meanless: BGC-Prophet for BGC mining using AI<br>5<br>SZOZ !!duty gl. uo Jasn Kuenqi!7 (160/044)8.4 S 8048/05 (180/118/90%)/4/69/9/0111/9/9/10/11/10/10/10/10/10/10/10/10/10/10/11 -->

The second task for BGC-Prophet is to predict the BGC category of a given region in a genome. According to the MIBiG database, there are seven categories of BGCs, including alkaloids, NRPs, polyketides, RiPPs, saccharides, terpenes, and others. We encoded these categories using one-hot encoding and considered an all-zero vector to represent the non-BGC category. Notably, each BGC may have more than one category, so the prediction of the BGC category is a multilabel seven-category problem. The problem can be described as follows: Extract the sequence of hidden state variables from the transformer encoder model \( H = \left( {{h}_{1},{h}_{2},\ldots ,{h}_{n}}\right) ,{h}_{i}\varepsilon {\mathbb{R}}^{k} \) and calculate the average hidden state \( \bar{b} = \frac{1}{n} \cdot  \mathop{\sum }\limits_{i}^{n}{b}_{i} \) . Transformer encoders allow key padding masks to be input to masks given specific timesteps; therefore, this study used gene tags as masks to prevent non-BGC genes from influencing the classification of BGC. The hidden states of the sequence are output as 7D vectors through a simple fully connected layer, and the sigmoid function is applied to output the confidence score of each label.

## Hyperparameter tuning and performance evaluation

To evaluate the necessity of incorporating a Transformer-based architecture in BGC-Prophet, we compared its performance with simpler models, specifically bidirectional LSTM networks. The core Transformer architecture of BGC-Prophet was replaced with single-layer and two-layer bidirectional LSTMs, and their performance was assessed using the same dataset and evaluation metrics. The single-layer bidirectional LSTM, with 1.7 M parameters, achieved an AUROC of 0.88 and an F1 score of 0.65 on the test set. The two-layer version, despite having a higher parameter count of \( {4.2}\mathrm{M} \) ,exhibited lower performance, with an AUROC of 0.87 and an F1 score of 0.61 . During training, both LSTM models showed severe overfitting, with accuracy exceeding 95% on the training and validation sets, likely contributing to their suboptimal generalization. In contrast, BGC-Prophet, with a parameter count of 2.5 M, achieved superior performance, effectively balancing parameter efficiency and generalization. These results highlight the strength of the Transformer architecture in capturing sequence-specific information while mitigating overfitting.

An ablation study was also performed to explore the impact of key hyperparameters, including the number of encoder layers, attention heads, and embedding size, on the model's performance. Increasing the number of encoder layers beyond two resulted in slightly reduced test set performance, indicating that additional layers did not enhance the representation of sequence-specific features. Variations in the number of attention heads showed minimal impact, with five attention heads providing a suitable balance between computational complexity and performance. For the embedding size, larger dimensions improved the model's capacity to represent sequence-specific information but came at the cost of higher computational demands. An embedding size of 320 dimensions, corresponding to 64 dimensions per attention head, was identified as the optimal trade-off between accuracy and efficiency. Based on these findings, the final model was configured with two encoder layers, five attention heads, and an embedding size of 320, achieving the best overall performance while maintaining computational efficiency.

## Comparison methods

## DeepBGC

DeepBGC is a novel deep learning and NLP strategy for improved identification of BGCs in bacterial genomes. DeepBGC employs a BiLSTM recurrent neural network. DeepBGC improves the detection of BGCs of known classes from bacterial genomes and harnesses great potential to detect novel classes of BGCs. In this study, we used DeepBGC for BGC gene detection and BGC product classification tasks and compared its performance to that of BGC-Prophet.

## AntiSMASH

The antiSMASH (antibiotics & Secondary Metabolite Analysis Shell) is a comprehensive pipeline capable of identifying biosynthetic loci covering the whole range of known secondary metabolite compound classes. It employs a set of cu-rated pHMMs to call biosynthesis-related gene families and a set of heuristics to designate a portion of a genome as a BGC. Since antiSMASH [16] has been the most widely used pH-MMs rule-based tool for BGC mining and has the most comprehensive ecosystem, including tools (such as ARTS 2 [42], Pep2Path [43]) and databases (such as IMG-ABC [44], MicroScope [45], MIBiG [4]), and considering the high FPRs of existing machine learning tools, we used antiSMASH for performance comparison. In this study, we applied antiSMASH and BGC-Prophet to 982 genomes from Aspergillus and evaluated the ability of BGC-Prophet to identify BGCs.

## Benchmark measures

The evaluation was based on five measuring metrics, namely, accuracy, precision, recall, F1-score, and AUROC. First, four parameters of the confusion matrix must be clarified: TP (true positive, actually BGC, judged by the model as BGC); FN (false negative, actually BGC but judged by the model as non-BGC); TN (true negative, the actual value is non-BGC and judged by the model as non-BGC); and FP (false positive, the actual value is non-BGC but judged by the model as BGC). We introduced several measures, including precision, recall, F1, true positive rate (TPR), and FPR. The definitions of these measures and formulas are as follows:

\[\text{ Accuracy } = \frac{{TP} + {TN}}{{TP} + {TN} + {FP} + {FN}} \times  {100}\%\]

\[\text{ Precision } = \frac{TP}{{TP} + {FP}} \times  {100}\%\]

\[\text{ Recall } = \frac{TP}{{TP} + {FN}} \times  {100}\%\]

\[{F1} = \frac{2 \times  \text{ Precision } \times  \text{ Recall }}{\text{ Precision } + \text{ Recall }} \times  {100}\%\]

\[{TPR} = \frac{TP}{{TP} + {FN}}\]

\[{FPR} = \frac{FP}{{FP} + {TN}}\]

The AUROC is the area under the receiver operating characteristic (ROC) curve, and the ROC is the curve of TPR-FPR traversing different thresholds, which is also based on the confusion matrix.

## Statistical methods

Uniform manifold approximation and projection and t-distributed stochastic neighbor embedding (t-SNE) dimensionality reduction techniques were applied to visualize and explore the high-dimensional gene vectors. To evaluate the differences between two BGC number groups,a \( t \) -test was performed. The \( t \) -test is a parametric statistical test that determines whether the means of two groups are significantly different from each other. It was used to compare the means of specific variables or features between the groups of interest. The Pearson correlation coefficient was used to examine the linear relationship between the prediction results of the anti-SMASH and BGC-Prophet models. The Pearson correlation coefficient provides a measure of the strength and direction of the linear association between variables. It was employed to assess the correlation between different features or variables within the dataset.


<!-- Meanless: 6<br>Lai et al. -->

## Results

## BGC-Prophet model establishment and assessment strategy

A BGC is a group of functionally related, colocalized genes that can be conceptualized as a sentence, with BGC prediction being analogous to a text classification task in the field of NLP. Following the mentality and paradigm of NLP, we developed a BGC language processing neural network model called BGC-Prophet. This model captures location-dependent relationships between biosynthetic genes by training on thousands of BGCs and assigns gene types or product classes using two classifiers (Fig. 1A-C). Training on thousands of BGCs allows the model to learn the general syntax of genes, that is, gene location dependencies, which helps to improve generalizability and avoid overfitting.

The BGC-Prophet has innovative designs for unleashing its power in the BGC prediction task. First, BGC-Prophet uses genes as tokens to represent sentences, each representing a chain of genes in a BGC (Fig. 1A). Previous methods, such as DeepBGC, use Pfam domains as tokens that effectively balance genetic information retention and computational complexity. However, Pfam relies on expert-driven manual annotation to define domain boundaries, and using Pfam domains as tokens may lose global information of genes. In contrast, we choose genes as tokens because they are more intuitive and flexible, requiring no additional processing. Second, BGC-Prophet uses the ESM, a pretrained protein language model, to convert protein sequences into embeddings that describe the homology between multiple protein sequences [37] (Fig. 1B and C). The resulting numerical vectors of genes encapsulate evolutionary signals and functional properties based on their sequences, allowing us to leverage similarities among genes.

To establish the language model, we curated a training dataset of 12510 positive and 20000 negative samples, each of which was a gene cluster containing 128 genes (Fig. 1A and Supplementary Fig. S1). Considering that the longest BGC in MIBiG (Minimum Information about a BGC) consists of 115 genes and that the number of non-BGC genes between BGCs in genomes, we set the maximum number of genes in a sample to 128 (Supplementary Fig. S1). Details of the generation of positive and negative samples are provided in the 'Materials and methods' section. The BGC-Prophet model was trained on thousands of BGCs, utilizing sequence-specific representations of genes as input features. The model architecture is centered around a transformer encoder, which is designed to capture location dependencies between genes within a BGC. Specifically, the model comprises multiple layers of self-attention mechanisms, with each layer consisting of multiple attention heads that focus on different aspects of the gene sequences. Hy-perparameters such as the number of layers, attention heads, and embedding dimensions were carefully optimized to balance model complexity and performance. The training process involved extensive hyperparameter tuning and regularization techniques to prevent overfitting, especially given the relatively small magnitude of available training data.

BGC-prophet could accept a set of genes as input and predict BGC location and category (Supplementary Fig. S2). The input of the BGC-Prophet model is a sequence of embeddings represented by 320D vectors generated by the ESM method [37]. The output of the BGC-Prophet model consists of two parts. The first part is a sequence of values ranging from 0 to 1 representing the prediction scores of individual genes to be part of a BGC. The second part is which of the seven categories (see the 'Materials and methods') the input gene clusters belong to.

We clarified several experiments to evaluate and apply BGC-Prophet in this study (Fig. 1D). First, we evaluated the performance of BGC-Prophet on the NG dataset (see the 'Materials and methods'), which comprises a representative set of nine genomes used in pervious works on ClusterFinder and DeepBGC (Supplementary Tables S1 and S2) [23, 26]. Second, we compared the BGCs predicted by BGC-Prophet and anti-SMASH on the AG dataset (see the 'Materials and methods'), which comprises 982 genomes from Aspergillus, a genus with great biosynthetic potential. Third, we attempted to discover new insights into the diversity and novelty of BGCs on the 85KG dataset (see the 'Materials and methods'), which comprises 85 203 available bacterial and archaeal genomes from the genome taxonomy database (GTDB), as well as on MG dataset (see the 'Materials and methods') that contain 9428 metagenomic samples from 47 studies (Supplementary Table S3). We finally examined the relationships between microbial enrichment patterns of BGCs and important geological events (Fig. 1E).

## Evaluation of sequence-specific representations of genes

The ESM method generated sequence-specific (which enabled awareness of the gene position in the whole chain of genes in a BGC) representations of genes, thereby serving as meaningful input features for the BGC prediction model. Here, we evaluated the effectiveness of using vector representations generated by the ESM method. To this end, we first used the ESM-2 8M model (version 2 with 8 million parameters) to generate vectors for a set of genes. Then, we consolidated the numerous genes within each BGC into a single representative BGC vector by averaging the vectors. We evaluated the representative vectors of all BGCs from the MIBiG database via t-SNE analysis. Subsequently, we reduced the dimensionality of the representative BGC vectors from 320 dimensions to 2 dimensions by the t-SNE method for visualization.

The different categories of BGCs demonstrated distinct patterns within the t-SNE dimensionality reduction plot (Fig. 2). The seven distinct categories of BGCs exhibited a concentrated distribution into three clusters (top right, bottom left, and bottom right). For instance, terpenes predominantly cluster on the bottom right, saccharides and RiPPs primarily cluster on the top right, and polyketides primarily cluster on the bottom left and bottom right. The remaining categories exhibited a widespread distribution across all three clusters. This suggests that the genes constituting different BGCs have a certain specificity, possessing distinguishable foundations. The partial overlap in the scatter plot indicates that the order information was lost in computing average gene vectors, requires additional modeling in subsequent transformer encoder. The boxplot showed that the points of any two categories of BGCs, such as polyketides and terpenes ( \( t \) -test, \( P < \) .001), exhibited clear separation along both dimensions of the scatter plot (Fig. 2A). We also analyzed the 2D distributions of BGCs (positive samples) and non-BGCs (negative samples) in the training set. The scatter plot shows a different between the distribution patterns of BGCs and non-BGCs. Despite the fact that there are areas in the graph that are exclusively occupied by BGCs (bottom right), there is substantial overlap between BGCs and non-BGCs on the scatter plot (Fig. 2B), although their distributions are significantly different on both axes ( \( t \) - test, \( P < {.001} \) ),which indicates that determining whether a gene belongs to a BGC cannot rely solely on the features provided by the embedding vectors, as it also requires consideration of contextualized features learned by transformer encoder model. Our findings demonstrated that the use of the ESM model could generate sequence-specific representations of genes and therefore help the language model learn the location-dependent relationships between genes that distinguish between BGCs and non-BGCs.

<!-- Meanless: のようなことで、ここで、デーディアをしています。そのようなのですが、そので、そうです。スマースを -->


<!-- Meanless: BGC-Prophet for BGC mining using AI<br>Downloaded from https://academic.oup.com/nar/article/53/7/gkaf305/8113170 by National Science & Technology Library user on 16 April 2025<br>7 -->

<!-- Media -->

<!-- figureText: A<br>B<br>C<br>Minimum Information about a Biosynthetic Gene cluster<br>A phylogenetic diverse set of 5,886 genomes from GTDB<br>BGC and nonBGC samples<br>The hyperparameters of the transformer encoder used in the BGC-Prophet model:<br>- Number of attention layers: 2<br>- Number of heads: 5<br>MIBiG<br>6KG<br>↓<br>- Embedding dimensions: 1,280<br>ESM2-8M<br>Obtain BGC<br>Predict BGC using AntiSMASH<br>Embeddings of genes<br>Positional encoding<br>↘<br>Layer normalization<br>BGC<br>non-BGC genes<br>Predicted BGC (abandon)<br>5-heads attention<br>Two side padding<br>Add<br>×2<br>Extract by genomic order<br>BGC language model (Transformer encoder)<br>Feed forward (dim=1280)<br>Positive sample of BGC Sequence length = 128<br>Biosynthetic-aware embeddings<br>↓<br>Add & Layer normalization<br>Fully-connected layers<br>Negative sample of nonBGC Sequence length \( = {128} \)<br>↓<br>Gene detection and product classification<br>D<br>E<br>9 bacterial genomes mentioned in DeepBGC<br>982 genomes from the genus of Aspergillus<br>85,203 available genomes in GTDB RS214<br>Metagenomes involved in 47 metagenomic studies<br>Polyketide<br>NRP<br>Post<br>Pre<br>Relative abundance<br>NG<br>AG<br>85KG<br>MG<br>Validation on NG<br>Profiles of BGCs on AG<br>Large-scale BGC mining on 85KG and MG<br>Cambrian Explosion<br>True Positive Rate<br>PHATTERS<br>ALAHA<br>Great Oxidation<br>DeepBGC (AUC = 0.93)<br>BROC-Prophet (AUC = 0.92)<br>False Positive Rate -->

<img src="https://cdn.noedgeai.com/bo_d78vubc91nqc73epp0u0_6.jpg?x=149&y=140&w=1462&h=1393&r=0"/>

Figure 1. The workflow of this study. (A) Generation of positive and negative samples. To train the BGC-Prophet language model, we curated a training dataset of 12 510 positive and 20 000 negative samples, and each sample was a cluster of 128 genes. MIBiG, minimum information about a BGC; GKG, a phylogenetically diverse set of 5886 genomes from the GTDB database. (B) BGC-Prophet pipeline for BGC gene detection and product classification tasks. BGC-Prophet uses the ESM method to generate the sequence-specific embedding of gene tokens and employs a language model (i.e. transformer encoder) with two classifiers (i.e. fully connected layers) for BGC gene detection and product classification. (C) The architecture of the Transformer encoder used in BGC-Prophet. Prelayer normalization is used to accelerate the convergence of the model. The positional encoding adopts classical sine-cosine position coding, which does not require additional training and captures relative positional relationships between genes effectively. (D) Several datasets used in this study for various purposes. NG, nine genomes that were examined in previous studies, such as ClusterFinder and DeepBGC; AG, 982 genomes from the genus Aspergillus; 85KG, 85 203 available genomes in GTDB RS214; MG, 9428 metagenomics samples involved in 47 studies. Details of these datasets are available in the 'Materials and methods' section. (E) The bar diagram shows the enrichment of BGCs in microbes after important geological events.


<!-- Meanless: Downloaded from https://academic.oup.com/nar/article/53/7/gkaf305/8113170 by National Science & Technology Library user on 16 April 2025<br>8<br>Lai et al. -->

<!-- figureText: A<br>B<br>Polyketide<br>100<br>100<br>Negative<br>Other<br>40<br>ESMembed mean: tSNE_2<br>40<br>Saccharide<br>ESMembed mean: tSNE_2<br>BGC<br>50<br>0<br>-50<br>NRP<br>Alkaloid<br>Terpene<br>50<br>20<br>0<br>0<br>-20<br>-50<br>20<br>0<br>-20<br>-40<br>-40<br>-100<br>-100<br>-50<br>-25<br>0<br>25<br>50<br>-100<br>-50<br>0<br>50<br>100<br>ESMembed mean: tSNE_1<br>ESMembed mean: tSNE_1<br>-<br>-100<br>-50<br>0<br>50<br>100<br>-60<br>-40<br>-20<br>0<br>20<br>40<br>60 -->

<img src="https://cdn.noedgeai.com/bo_d78vubc91nqc73epp0u0_7.jpg?x=191&y=140&w=1447&h=675&r=0"/>

Figure 2. The distribution of ESM embeddings for genes. (A) Average ESM embedding distribution for BGCs in the MIBiG database. The representative vectors of all genes within each BGC were averaged, and subsequently, a dimensionality reduction technique called t-SNE was applied to project the BGCs of all categories into a 2D space. The resulting tSNE1 and tSNE2 values were subsequently utilized to generate scatter and boxplot visualizations. The scatter plot depicts the spatial distribution of the average vectors representing the BGCs in the 2D plane. Each BGC exhibits distinct distribution characteristics, and their separability is achieved through nonlinear means. Conversely, the boxplot graph displays the distribution patterns of the seven BGC categories along the tSNE1 and tSNE2 dimensions. The observed differences in distribution among the groups were statistically significant at the predetermentingle level of significance (*P < .05; **P < .01; ***P < .001; *****P < .0001; ns, not significant). NRP, non-ribosomal peptide; RiPP, ribosomally synthesized and RiPP (B) Average ESM embedding distribution for BGCs and non-BGCs. Using the t-SNE analysis as before, both the BGCs and non-BGCs were subjected to dimensionality reduction, resulting in scatter and boxplot visualizations. We plotted separate boxplots for the horizontal and vertical axes and applied a t-test to indicate that there were significant differences between pairwise comparisons of the samples at the given level of significance (*P<.05; **P<.01; ***P<.001; *****P<.0001).

<!-- Media -->

## Accurate and UHT BGC prediction

We assessed the performance of BGC-Prophet by evaluating its ability to (i) accurately locate BGCs throughout the bacterial genome (BGC gene detection) and (ii) categorize them into their respective categories according to the type of chemical structure as their products (BGC product classification). We initially evaluated the performance of BGC-Prophet and DeepBGC for BGC gene detection, and the results showed that the AUROC of BGC-Prophet (91.9%) and DeepBGC (93.1%) were comparable on the NG dataset (Fig. 3A and B). We also evaluated the model's performance using precision and recall metrics, which are crucial for assessing the utility of BGC-Prophet in practical applications. Using a default threshold of 0.5, the BGC-Prophet model achieved higher precision (59.2% versus 22.0%) compared to DeepBGC, with a recall that still yields a balanced performance (Supplementary Fig. S3). It is worth noting that the F1 of BGC-Prophet is higher than DeepBGC under the default threshold of 0.5, the F1 of BGC-Prophet strikes a balance between precision and minimizing missed positives, shows a marginal improvement of around 50% (Fig. 3A). Subsequently, we evaluated the performance of BGC-Prophet and DeepBGC in BGC product classification. In this task, BGC-Prophet achieved an AUROC of 98.8% with regard to differentiating among the seven BGC categories, while DeepBGC achieved an AUROC of 91.3% (Fig. 3C). In addition to AU-ROC, the model demonstrated a precision of 92.8% and a recall of 89.0% across all BGC categories, compared to 90.2% and 76.4% of DeepBGC (Fig. 3C). This indicates that BGC-Prophet is better at accomplishing the BGC product classification task. We evaluated BGC-Prophet not only on the standard NG dataset comprising nine genomes but also on Fun-BGC, an additional dataset containing experimentally validated BGCs [46]. The results demonstrate the following: on the FunBGC dataset, BGC-Prophet achieved a recall of 0.89 at default threshold (Supplementary Fig. S4). Although BGC-Prophet achieves good performance in overall accuracy for product classification, it should be emphasized that there are some differences in its performance for different categories of BGCs, which is largely caused by the imbalance of various categories of BGCs in the training data. For example, BGC-Prophet struggles with RiPP predictions, likely because only around 300 RiPP BGCs are contained in MIBiG, and different RiPP classes do not really share much of their biosynthetic approaches, apart from the precursor having been generated by a ribosome.

We have also evaluated the performance of antiSMASH, GECCO, and BiGCARP on the same NG dataset and compared their results with BGC-Prophet. The comparative results reveal that BGC-Prophet outperforms these methods, achieving the highest F1 score across all genomes, compared to the other four tools (average F1 = 0.57; Supplementary Fig. S5 and Supplementary Table S4). This superior performance can be attributed to BGC-Prophet's ability to effectively integrate contextual genomic features with deep learning-based predictive modeling, enabling enhanced detection and classification of BGCs.


<!-- Meanless: BGC-Prophet for BGC mining using AI<br>Downloaded from https://academic.oup.com/nar/article/53/7/gkaf305/8113170 by National Science & Technology Library user on 16 April 2025<br>9 -->

<!-- Media -->

<!-- figureText: AUROC F1<br>ROC curve<br>A<br>B<br>1.0<br>C<br>0.75<br>BGC-Prophet<br>0.930.61<br>DeepBGC BGC-Prophet<br>0.95<br>0.988<br>0.970<br>0.909<br>0.928<br>0.890<br>0.90<br>0.85<br>0.913<br>0.713<br>0.809<br>0.902<br>0.764<br>0.80<br>GCA_000154945<br>0.8<br>DeepBGC<br>0.960.35<br>BGC-Prophet<br>0.910.49<br>-0.8<br>True Positive Rate<br>0.4<br>AUROC<br>Accuracy<br>F1 Score<br>Precision<br>Recall<br>E<br>GCA_000156435<br>DeepBGC<br>0.940.32<br>BGC-Prophet<br>0.900.61<br>GCA_000156475<br>0.2<br>0<br>0.16<br>DeepBGC<br>0.910.35<br>BGC-Prophet (AUROC = 0.92)<br>8<br>74<br>DeepBGC (AUROC = 0.93)<br>16<br>75<br>0.14<br>BGC-Prophet<br>960.67<br>0.6<br>0.0<br>0.2<br>0.6<br>0.8<br>1.0<br>24<br>76<br>77<br>GCA_000158815<br>DeepBGC<br>0.950.51<br>D<br>40<br>78<br>0.12<br>48<br>79<br>80<br>6<br>BGC-Prophet<br>(estimated)<br>0.08<br>BGC-Prophet<br>0.840.45<br>DeepBGC<br>Query<br>GCA_000158875<br>0.06<br>DeepBGC<br>0.920.22<br>Log10 Time (min)<br>80<br>88<br>3<br>96<br>0.04<br>2<br>104<br>SGC-Prophet<br>0.950.63<br>0.4<br>GCA_000158895<br>DeepBGC<br>0.940.35<br>112<br>0.02<br>BGC-Prophet<br>0.940<br>0.64<br>1<br>120<br>GCA_000158915<br>0<br>43<br>8<br>3<br>吕<br>1.0<br>DeepBGC<br>0.930.35<br>10<br>100<br>1,000<br>Number of genomes<br>Key<br>0.120<br>BGC-Prophet<br>0.910.52<br>0.2<br>0.116<br>GCA_000568255<br>F<br>-0.151<br>DeepBGC<br>0.920.26<br>-0.167-<br>-0.165-<br>BGC-Prophet<br>0.89<br>山<br>GCA_000568915<br>KUTG_02123<br>KUTG_02124<br>KUTG_02125<br>KUTG_02126<br>KUTG_02127<br>KUTG 02128<br>KUTG_02127<br>DeepBGC<br>0.930.46<br>0.105 -->

<img src="https://cdn.noedgeai.com/bo_d78vubc91nqc73epp0u0_8.jpg?x=147&y=140&w=1459&h=997&r=0"/>

Figure 3. Evaluation of BGC-Prophet in different settings. (A) Evaluation metrics reflecting the performance of BGC-Prophet and DeepBGC for BGC gene detection. Evaluation of nine bacterial genomes that were used in previous studies was performed. All the metrics except for the AUROC were evaluated under the default threshold of 0.5. (B) The evaluation metrics reflecting the performance of BGC-Prophet and DeepBGC for BGC product classification. Note that the random forest classifier of DeepBGC was retrained using the MIBiG database (version 3.0). (C) The ROC curve reflecting the performance of BGC-Prophet compared with DeepBGC. (D) The running times of BGC-Prophet and DeepBGC on datasets with different numbers of genomes. Note that the results for 10 and 100 genomes were actual times, while the results for 1000 genomes were estimated based on linear extrapolation. (E) The gene heatmap for a gene cluster (128 timesteps) during a single prediction process on the NG dataset. This heatmap shows the average attention map of the first layer of the five heads of the detection model (see the 'Materials and methods') during a single prediction process by BGC-Prophet on the NG dataset. The vertical axis represents Query in the self-attention mechanism, corresponding to the input gene embedding vectors, while the horizontal axis represents Key, corresponding to the BGC search space. Horizontally, the large heatmap implies that determining whether a gene participates in the formation of a BGC requires considering information from multiple positions. Vertically, the vertical dark purple lines in the whole-gene heatmap represent genes influencing the formation of a BGC by multiple genes. Darker colors along the diagonal in the heatmap suggest that determining whether a BGC is formed relies primarily on the information embedded in its own vector, indicating the critical role played by the embedding vector's information. A zoomed-in heatmap was constructed to demonstrate the relationships between the BGCs and surrounding genes. The 75th-79th genes are annotated as BGC genes. (F) Schematic diagram of the mechanism of interest applied to BGC genes and genes at both ends; only the colored genes belong to this predicted BGC. Panel (F) provides a schematic explanation of the magnified section in panel (E), and only attention scores that exceeded 0.08 are shown as arrows. Gene 76 (KUTG_02 125), which encodes a NRP synthetase, received the highest attention scores from other BGC genes. This finding suggested that information from this gene needs to be paid higher attention during annotation, possibly implying its conservativeness and centrality in this BGC.

<!-- Media -->

BGC-Prophet uses a more efficient ESM method to generate vector representations of genes, avoiding time-consuming sequence alignment (Pfams alignment) and improving the throughput of genomic data processing. For instance, when we used 10 genomes (randomly selected and unduplicated genomes) for efficiency evaluation, DeepBGC needed an average of four hours per genome, whereas BGC-Prophet could process each genome in just 1 min (Fig. 3D). When we extrapolated the number of genomes to 100 , the differences in time consumption were also two orders of magnitude greater (Fig. 3D). Therefore, BGC-Prophet could process hundreds of thousands of genomes within tens of hours. Thus, we emphasize that BGC-Prophet is one of the pioneering UHT methods that enables pan-phylogenetic screening and whole-metagenome screening of BGCs.

BGC-Prophet captures location-dependent relationships among biosynthetic genes. The location-dependent relationships among biosynthetic genes were elucidated using attention maps generated by the model (Fig. 3E). One notable example is gene 76 (KUTG_02125), which encodes an NRP synthetase and received the highest attention scores from other biosynthetic genes. This indicates its potential conservativeness and central role within the BGC, suggesting it may be a key component in the biosynthetic pathway and providing insights into the structural and functional organization of BGCs (Fig. 3F). Such examples are plentiful (Supplementary Fig. S6), and these attention maps clearly show that the language model can capture location-dependent relationships among biosynthetic genes.


<!-- Meanless: 10<br>Lai et al. -->

## Comprehensive profiling of BGCs in 982 genomes from Aspergillus

BGC-Prophet predicts BGCs in a comprehensive manner and can predict more previously unannotated BGCs. Here, we utilized BGC-Prophet and antiSMASH to predict BGCs in the AG dataset, which contains genomes from Aspergillus, a genus with great biosynthetic potential and hundreds of genomes of this lineage. The results showed that BGC-Prophet predicted a greater number of potential BGCs compared to anti-SMASH, particularly in the terpene category (52 004 versus 7748, with 7260 intersection BGCs). The predictions of BGCs in the NRP category by the two tools were nearly identical (27603 versus 27100, with 26278 intersection BGCs). BGC-Prophet predicted a greater number of BGCs in the category of polyketide (35 606 versus 18 225, with 16 607 intersection BGCs). Moreover, the prediction of BGCs in the RiPPs category by both tools exhibited complementarity (27 155 versus 8082, with 1401 intersection BGCs), enhancing the coverage of predicted BGCs. Furthermore, BGC-Prophet predicted additional BGCs in the categories of alkaloids and saccharides compared to antiSMASH. These results showed a notable discrepancy between the BGCs predicted by the two tools, suggesting that BGC-Prophet can predict potentially novel BGCs beyond those detected by antiSMASH. We then studied the distribution spectrum of the predicted BGCs by both BGC-Prophet and antiSMASH. The results showed that BGC-Prophet predicted BGCs almost three times as many as anti-SMASH (167 375 versus 59 037, Fig. 4A). Considering that MIBiG includes only 2502 experimentally validated BGCs, we believe that most of them are new BGCs that have not yet been studied in experiments. (Fig. 4B and C). The prediction results of the two tools showed a clear linear correlation (Supplementary Fig. S7, r = 0.91, P < .001), indicating that the BGCs predicted by BGC-Prophet have no preference for a specific species within Aspergillus. To validate the accuracy of BGC-Prophet's predictions for Alkaloids and Saccharides, BiG-SCAPE [47] was used to compare all predicted Alkaloid and Saccharide BGCs from Aspergillus against all BGCs in MIBiG, generating distance and Jaccard Index data. These metrics were then compared between "same category" and "different category" BGCs within MIBiG using a t-test. The results demonstrate that the BGCs predicted by BGC-Prophet exhibit significantly higher similarity to their corresponding "same category" BGCs in MIBiG compared to "different category" BGCs, thus supporting the reliability of the predictions for these two BGC classes (Supplementary Fig. S8). Overall, we demonstrate that BGC-Prophet predicts BGC in a more comprehensive manner and can predict more previously unannotated BGCs.

## Comprehensive profiling of BGCs on 85 203 microbial genomes from the majority of bacterial and archaeal lineages

With BGC-Prophet, new insights have been gained about the abundance and diversity of BGCs in genomes from the majority of bacterial and archaeal lineages. We used BGC-Prophet to generate BGC profiles from the 85KG dataset, which comprises 85 203 microbial genomes from the majority of bacterial and archaeal lineages as cataloged in the GTDB database. Among these genomes, 41599 were found to contain BGCs, resulting in the identification of a total of 119305 BGCs. We first conducted an analysis to determine the proportions of different categories of BGCs in these genomes (Fig. 5A) by assigning BGCs to the species' genomes (Fig. 5B) and found that the three most widely distributed BGC categories were polyketide (in 34% of the total species), NRP (33%), and RiPP (24%), and the three most abundant categories were NRP (33%), polyketide (28%), and RiPP (27%). The alkaloid category exhibited the narrowest distribution (in 2% of the total species; Fig. 5B). In comparison, the three most abundant categories in the MIBiG database were polyketide (41%), NRP (34%), and RiPP (13%) [4]. Moreover, BGC-Prophet identified a significantly greater number of BGCs classified in the "other" category (increasing from 324 to 32 233 and from 13% to 24%), indicating its ability to mine potentially novel BGC categories (Supplementary Table S5).

The host distribution of the BGC genes exhibited species-specific characteristics, exemplified by the phylum Actino-mycetota having the highest predicted number of BGCs (39 252 in total) and the phylum Pseudomonadota exhibiting the widest genomic coverage, with 12 637 genomes containing at least one BGC, encompassing a total of 29 675 BGCs (Fig. 5A and C, and Supplementary Table S6). At the order level, the 27 orders with the highest average number of predicted BGCs (>7.0) were distributed across 15 phyla, such as Actinobacteria and Acidobacteriota (Fig. 5C), which were reported to have relatively high biosynthetic potential (Supplementary Text S1). We proceeded to analyze BGCs for archaea and bacteria separately and identified 1762 and 117 543 BGCs from 1079 archaeal genomes and 40 520 bacterial genomes, respectively. On average, archaeal genomes contained 1.63 BGCs per genome, while bacterial genomes contained 2.90 BGCs per genome. These results indicate a significantly lower abundance of BGCs in archaeal genomes than in bacterial genomes ( \( t \) -test, \( P = {6.1}\mathrm{e} - {29} \) ). The predominant BGC categories in archaea were saccharides (30%) and RiPP (24%), whereas in bacteria, they were 10% and 11%, respectively. The predominant BGC categories in bacteria were NRP (33%) and polyketide (28%), whereas in archaea, they were 8% and 1%, respectively. This difference may be attributed to the more ancient nature of archaea than bacteria, particularly in terms of energy acquisition and metabolism. Bacteria live in relatively complex environments and rely on strategies such as aerobic respiration for resources, while archaea survive by using alternative strategies such as sulfur reduction, denitrification, and nitrate reduction [48, 49]. These findings suggested that newly evolved species might have possessed higher frequencies of BGCs because of competition in environments with limited resources (Supplementary Text S2).

## Comprehensive profiling of BGCs in 9428 metagenomic samples

BGC-Prophet is one of the pioneering UHT methods that enables whole-metagenome screening of BGCs. We performed species annotation and BGC prediction on the MG dataset, which comprises 9428 metagenomic samples from 47 human microbiome studies (details in the 'Materials and methods'). A total of 160 814 bins (each corresponding to a partial genome with fragmented contigs) were generated from these metage-nomic samples, of which 132 809 bins were successfully assigned to species, while 28 005 bins remained unclassified. Of the 9428 metagenomic samples analyzed, a total of 8255 were predicted to contain at least one BGC. There were 248 229 predicted BGCs distributed among 2922 species. The distribution of predicted BGCs from these human metagenomes is shown in Fig. 6. Consistent with the findings from the GTDB dataset, BGCs identified from the metagenome dataset were significantly enriched in species with Actinomycetota compared to those of other species (average of 8.30 BGCs per genome versus 4.24 BGCs per genome, \( P = {1.06}\mathrm{e} - {105}). \)

<!-- Meanless: にしますのです。できてきますができるようなこともありますのですができるか？です。そうなく、そのようなくなっても -->


<!-- Meanless: BGC-Prophet for BGC mining using AI<br>Downloaded from https://academic.oup.com/nar/article/53/7/gkaf305/8113170 by National Science & Technology Library user on 16 April 2025<br>11 -->

<!-- Media -->

<!-- figureText: A<br>Other<br>PKS<br>NRP<br>RiPP<br>2,262 4,759<br>27,902<br>1,618<br>16,607<br>18,999<br>822<br>26,278<br>1,325<br>6,681 1,401<br>25,754<br>Terpene<br>Alkaloid<br>Saccharide<br>Number of all BGCs predicted by 2 tools<br>167,375<br>150,00<br>7,260<br>44,744<br>5,812<br>4,489<br>100,00<br>59,037<br>50,00<br>antiSMASH<br>BGC-Prophet<br>0<br>B<br>Categories<br>NRP<br>Polyketide<br>Other<br>Alkaloid<br>RiPP<br>Saccharide<br>Terpene<br>Number of BGCs<br>30<br>55<br>Aspergillus<br>85<br>110<br>140<br>170<br>200<br>225<br>250<br>280<br>310<br>C<br>antiSMASH<br>BGC-Prophet<br>50,000<br>52,004<br>Number of BGCs<br>40,000<br>35.606<br>32,661<br>30,000<br>27,100<br>27,603<br>27,155<br>20,000<br>18,225<br>10,000<br>8.082<br>7,748<br>7,021<br>5,812<br>4.489<br>0<br>NRP<br>Polyketide<br>RiPP<br>Terpene<br>Other<br>Terpene<br>Polyketide<br>Other<br>NRP<br>RiPP<br>Alkaloid Saccharide -->

<img src="https://cdn.noedgeai.com/bo_d78vubc91nqc73epp0u0_10.jpg?x=348&y=148&w=1066&h=1676&r=0"/>

Figure 4. The predicted BGCs in the Aspergillus Genome dataset generated by BGC-Prophet and antiSMASH. (A) The number of BGCs predicted by BGC-Prophet (green) and antiSMASH (red), categorized according to the seven BGC categories. The bar plot shows the total number of BGCs predicted by by the two tools. In case that two prediction tools can identify BGCs with identical genes, they are considered to have predicted the same BGC, and the prediction results for all seven categories of BGCs were visualized using a Venn diagram. The reason why the number of all BGCs is smaller than the total number of different categories of BGCs is that one BGC may belong to more than one category. Notably, antiSMASH does not predict any BGC that belong to the alkaloid and saccharide categories. The BGC-Prophet predictions were based on the default threshold of 0.5. (B) The distribution of BGCs in the genomes of the Aspergillus genus. Within the central core, the encompassed area represents the entirety of Aspergillus species (a total of 76 species). The meaning of each circle from the inside out is as follows: first circle, the total number of BGC predicted by antiSMASH; second circle, the total number of BGC predicted by BGC-Prophet; third circle, the number of each category of BGC predicted by antiSMASH; fourth circle, the number of each category of BGC predicted by BGC-Prophet. Considering the presence of multiple subspecies within a species, the number of predicted BGCs per species was averaged. (C) A bar chart depicting the total number of different types of BGCs predicted by antiSMASH and BGC-Prophet, revealing that BGC-Prophet predicts a profoundly higher number of BGCs across various categories than antiSMASH.


<!-- Meanless: 12<br>Lai et al. -->

<!-- figureText: A<br>Categories<br>NRP<br>Polyketide<br>Other<br>Alkaloid<br>RiPP<br>Saccharide<br>⑪<br>Terpene<br>Archaea<br>BGC Number<br>1<br>2<br>4<br>6<br>8<br>Bacteria<br>10<br>12<br>®<br>14<br>16<br>18<br>20<br>Acidobacteriota<br>Chloroflexota<br>Actinomycetota<br>Cyanobacteriota<br>Verrucomicrobiota<br>Armatimonadota<br>JADJOY01<br>(C) Bacillota<br>Myxococcota<br>BGC enriched Orders(Avg. No. BGCs ≥7.0)<br>Bacillota_C<br>Planctomycetota<br>C<br>Bacillota_F<br>Pseudomonadota<br>20<br>③ Bdellovibrionota<br>Tectomicrobia<br>Number of BGCs<br>15<br>10<br>B<br>Counts<br>40,000<br>39,463<br>Number of BGCs<br>33,120<br>32,233<br>5<br>30,000<br>0<br>20,000<br>akd<br>imal<br>12,823<br>Terpene<br>-<br>-<br>10,000<br>4,370<br>RiPP<br>1,157<br>NRP<br>Polyketide<br>Other<br>RiPP<br>Saccharide Terpene<br>Alkaloid<br>Alkaloid<br>●<br>Prevalence<br>0.5<br>Prevalence ratio<br>0.4<br>0.34<br>0.33<br>Saccharide<br>0.3<br>0.24<br>0.2<br>0.21<br>Other<br>0.1<br>0.09<br>Polyketide<br>0.02<br>Avg. No. BGCs01<br>⓯5<br>10<br>15<br>20<br>0.0<br>Other<br>Polyketide<br>NRP<br>RiPP<br>Saccharide<br>erpene<br>Alkaloid -->

<img src="https://cdn.noedgeai.com/bo_d78vubc91nqc73epp0u0_11.jpg?x=389&y=154&w=1057&h=1592&r=0"/>

Figure 5. Novelty and phylogenomic distribution of microbial biosynthetic potential in different branches of the evolutionary tree of life. (A) The distribution of predicted BGCs on an evolutionary tree. The evolutionary tree consists of 85 203 genomes from the GTDBR214 database. For visualization purposes, the tree is displayed at the level of 1772 orders. The number of BGCs was averaged per genome within each order. From the innermost to the outermost layers, the central core represents the evolutionary tree structure, consisting of 148 Archaea and 1624 Bacteria orders. The first circle depicts a heatmap of the total number of BGCs, with most genomes having two or fewer BGCs, while a few genomes exhibiting up to 20 BGCs, indicating high biosynthetic potential. The second to eighth rings display the number of BGCs in the seven categories. Among the orders, the 27 orders with the highest average number (>7.0) of predicted BGCs were distributed across 15 different phylla (as annotated in panel (A)]. These 27 orders represent a diverse range of phylal and showcase varying levels of BGC diversity and distribution across different BGC categories. (B) The number and prevalence of predicted BGCs according to the seven BGC categories. The prevalence ratio refers to the proportion of genomes that contain a specific category of BGC cut of all the genomes analyzed. (C) The average number of BGCs and their distributions according to the different BGC categories within the 27 selected orders described in panel (A). These 27 orders were: o_DY22613 (f), o_Streptomycetales (b), o_Acanthopleuribacterales (n, o. o___Entotheonellales (n), o___UBA2199 (a), o___Streptosporangiales (b), o___Myxococcacles (k), o___JAJFKO01 (m), o___UBA7656 (a), o___UBA1444 (e), o _Mycobacteriales (b), o _Ktedonobacterales (h), o _SYMT01 (c), o _JABSQK01 (o), o _CAINFC01 (a), o _JADLHZ01 (k), o _Taubeabillales (d), o___J058 (I), o_Cyanobacteriales (i), o_JABHGC01 (m), o_UBA5704 (a), o_UBA7976 (k), o_VXMN01 (a), o_JACPZX01 (a), o_YA12-FULL-61-11 (g), o_JADJOY01 (j), o_JAKLKC01 (k). The names of these orders follow the GTDB taxonomy terminology.

<!-- Media -->

<!-- Meanless: Downloaded from https://academic.oup.com/nar/article/53/7/gkaf305/8113170 by National Science & Technology Library user on 16 April 2025 -->


<!-- Meanless: BGC-Prophet for BGC mining using AI<br>13<br>920Z !!duty gl. uo Jasn Aveja?17 (1601)040-8 800.805 (1810/181908)/61/69/1911.81/29.0812/1/11.0812 -->

## The profound enrichment of BGC in microbes after important geological events

Large differences were observed in the distribution of BGCs among the different species, particularly in light of the evolution over billions of years. To understand this phenomenon, we searched TimeTree [50] and identified two time points for the rapid growth of lineages, which corresponded to the Great Oxidation [51] and Cambrian Explosion [52] events (Supplementary Fig. S9). After both of these events, we observed a surge in BGC abundance and diversity, possibly indicating that microbial adaptation to environmental changes may involve the production of specialized secondary metabolites.

The Great Oxidation event occurred \( \sim  {2.5} - {2.3} \) billion years ago [53, 54]. Prior to this time point, there were three microbial genera, Mesoaciditoga [55], Vampirovibrio [56], and Synechococcus [56], which were categorized as the "pre" group. Among these three genera, 56 out of the 41 599 genomes analyzed were predicted to contain BGCs. The remaining 2215 genera comprised 41 543 out of the 41 599 genomes that evolved after this time point and were categorized as the "post" group. A statistical analysis revealed a significant increase in the average number of BGCs per genome from 2.5 to 4.5 between these two groups ( \( t \) -test, \( P = {0.024} \) ; Supplementary Fig. S10). The abundance of polyketide BGCs also significantly increased after the Great Oxidation event, with the average number of polyketides per genome increasing from 1.09 to 2.81 (t-test, \( P = {0.057} \) ). While this could imply a role for polyketides in microbial adaptation to increased oxygen levels, it is also possible that the presence of oxygen facilitated the production of polyketides [57]. On the other hand, there were no significant differences in the average abundance of RiPPs per genome (decrease from 1.29 to 1.25, \( t \) -test, \( P = {0.807} \) ) or in the number of NRPs per genome (increase from 1.0 to \( {3.16},t \) -test, \( P = {0.242} \) ). Although the number of NRPs per genome increased, the difference was not statistically significant due to the limited data available before this event, which consists of only three genera [58].

The Cambrian Explosion event occurred \( \sim  {542} - {520} \) million years ago and was marked by the rapid diversification of multicellular organisms [59]. Prior to this time point, 1529 genera composed 9212 out of the 41599 genomes analyzed, which were categorized as the "pre" group. The remaining 589 genera comprised 32387 out of the 41599 genomes that evolved after this time point and were categorized as the "post" group. At this time point, there was a significant increase in the average number of BGCs per genome, with the "post" group having double the number compared to the "pre" group (6.07 versus 2.95, \( t \) -test, \( P = {4.89}\mathrm{e} - {305} \) ; Supplementary Fig. S10). Further analysis of different categories of BGCs revealed significant differences in their average abundance before and after this time point. All the BGC categories exhibited an increase in average abundance per genome, with significant increases observed for polyketides (1.77 versus 3.52, \( t \) -test, \( P = {2.53}\mathrm{e} - {157} \) ) and NRPs (2.14 versus 3.77, \( t \) -test, \( P = {2.12}\mathrm{e} - {132} \) ). Polyke-tides and NRPs are small compounds that play crucial roles in helping the host defend against other bacteria and enhancing fitness in diverse environments [60, 61]. One possible explanation for this finding is that during the Ediacaran period, <635-541 million years ago, Cyanobacteria began to appear, leading to a significant increase in oxygen production through photosynthesis, which resulted in heightened ocean oxygenation [62]. This amplified ocean and atmospheric oxygenation may have accelerated the process of life evolution [63, 64]. The diversity of microbes increases exponentially [65]. During this time, multicellular organisms started to emerge [66]. On the one hand, multicellular organisms are hosts of microorganisms, and there is evidence suggesting that the genetic evolution of multicellular organisms occurred five times faster during the early Cambrian [67], leading to rapid life evolution in the oceans. On the other hand, the Earth's ecological environment underwent alterations due to the activities of various species, generating numerous microenvironments [68]. These microenvironments provide a variety of environmental pressures for microbial selection, resulting in a surge in the biosynthetic potential of microorganisms and leading to the synthesis of diverse secondary metabolites that enable microorganisms to better adapt to different environments and compete for resources.

These findings highlight the dynamics of BGCs on a large temporal scale and shed light on the impact of environmental changes on the abundance and diversity of specialized metabolites generated by microbes. However, it is important to note that these BGCs may have evolved well after these geological events, and thus the direct influence of such events on BGC evolution requires further investigation, including the functional roles and ecological significance of these BGCs in the context of bacterial evolution, as well as their potential applications in various fields, including synthetic biology and conservation of species diversity.

## Discussion

BGCs are a valuable source of natural products, but their discovery, expression, and characterization present significant challenges. In this study, we developed BGC-Prophet, a supervised language processing neural network model, to systematically identify known and predict potentially novel BGCs and their products from microbial genomes and metagenomes. BGC-Prophet captures location-dependent relationships between genes and learns biosynthetic-aware representations of BGCs based on their gene evolutionary patterns. These features provide advantages in terms of profiling BGCs across a wide range of microbial lineages.

The novelty of this work is demonstrated in three contexts. First, BGC-Prophet integrates the powerful language model of the transformer encoder with sequence-specific representations of genes through ESM embeddings. This combination effectively captures the location-dependent relationships among biosynthetic genes, offering robust and interpretable results. The use of ESM embeddings enhances the model's ability to distinguish gene categories and noncoding regions, providing a unique advantage in detecting and classifying BGCs compared to existing tools. Specifically, BGC-Prophet achieved an AUROC of 91.9% with regard to locating BGCs throughout the genome and 98.8% with regard to differentiating among the seven BGC categories (Fig. 3A-C). While these results highlight the model's strong performance, variations across different BGC categories were observed, particularly for less well-represented classes such as RiPPs. This suggests that further refinement, particularly in data augmentation and model training, is necessary to improve the model's robustness. Nevertheless, the exceptional processing speed of BGC-Prophet enables it to quickly analyze vast amounts of genomes with high efficiency (Fig. 3D), allowing for extensive profiling of BGCs in large-scale genomic and metagenomic data. The location-dependent relationships among biosynthetic genes were elucidated using attention maps generated by the model (Fig. 3E). One notable example is gene 76 (KUTG_02 125), which encodes an NRP synthetase and received the highest attention scores from other biosynthetic genes. This indicates its potential conservativeness and central role within the BGC, suggesting it may be a key component in the biosynthetic pathway and providing insights into the structural and functional organization of BGCs (Fig. 3F). Moreover, while BGC-Prophet can identify potential novel BGCs, the risk of false positives remains, particularly for BGCs without well-defined biosynthetic signatures. Experimental validation is necessary to confirm the true nature of these predictions. We acknowledge that BGC-Prophet's performance is slightly lower than DeepBGC's in both BGC detection and classification, as indicated by the AUC values in Fig. 3. BGC-Prophet employs gene-level tokenization and a Transformer-based architecture, which allows for capturing complex, location-dependent relationships between genes. However, this complexity comes with its own trade-offs, especially in the early stages of model training. In contrast, Deep-BGC relies on Pfam domain tokens, which are well-established and focus on domain-level features. While this gives Deep-BGC a slight edge in detecting known BGCs, BGC-Prophet's architecture is advantageous for detecting novel BGCs and understanding gene relationships within larger genomic contexts. Thus, while BGC-Prophet slightly underperforms in detection and classification for some datasets, it excels in exploring unexplored or novel BGC categories, offering complementary value to DeepBGC for large-scale, high-throughput BGC mining.


<!-- Meanless: 14<br>Lai et al. -->

<!-- Media -->

<!-- figureText: A<br>Tree scale: 1<br>BGC<br>Categories<br>NRP<br>RiPP<br>Polyhetide<br>Other<br>Alkaloid<br>RiPP<br>Saccharide<br>Terpene<br>BGC Number<br>1<br>4<br>Bacillata A<br>7<br>10<br>13<br>16<br>Actinomycetota<br>19<br>Bacteroidota<br>22<br>25<br>28<br>31<br>B<br>Counts<br>GCs mined by BGC-Prophet in 47 metagenome datasets<br>5,000<br>AsnicarF<br>2017 -<br>13<br>16<br>89<br>37<br>12,000<br>Number of BGCs<br>4,000<br>Bengtsson-PalmeJ<br>2015-<br>274<br>255<br>2.522<br>1.893<br>1.214<br>216<br>BritoIL<br>2016 -<br>2.807<br>1.734<br>1.328<br>3,365<br>Castro-NallarE<br>3,000<br>2,568<br>CM cf -<br>2,000<br>1,206<br>1.045<br>966<br>CosteaPI<br>2017 -<br>10,000<br>1,000<br>2015 -<br>0<br>FerrettiP<br>2018 -<br>GeversD<br>2014 -<br>HanniganGD<br>2017 -<br>HeQ<br>2017 -<br>Prevalence<br>HMP<br>2012-<br>8,000<br>2017 -<br>33<br>0.82<br>KarlssonFH<br>0.8<br>0.73<br>KosticAD<br>2015 -<br>833<br>6,000<br>0.64<br>Dataset<br>2014 -<br>LiJ<br>2017 -<br>LiSS<br>2016 -<br>LiuW<br>2016 -<br>112<br>0.6<br>0.4<br>0.37<br>LomanNJ<br>2013 -<br>0.33<br>0.31<br>LoombaR<br>2017 -<br>50<br>NielsenHB<br>2014 -<br>0.2<br>Obreaon-TitoAJ<br>2015 -<br>552<br>229<br>2014 -<br>136<br>0.02<br>OImMR<br>2017 -<br>4,000<br>0.0<br>OinJ<br>2012 -<br>1.98<br>Other<br>RiPP<br>Saccharide<br>NRP<br>Polyketide<br>Terpene<br>Alkaloid<br>OinN<br>2014 -<br>Rampellis<br>2015 -<br>Counts<br>RaymondF<br>2016 -<br>2.257<br>1.298<br>SchirmerM<br>2016 -<br>278<br>2017 -<br>mber of BGCs per genom<br>2.12<br>Vatanen1<br>2016 -<br>293<br>2,000<br>2.0<br>VincentC<br>2016 -<br>181<br>96<br>929<br>489<br>385<br>78<br>1.58<br>WenC<br>180<br>333<br>2.557<br>1.494<br>1.092<br>164<br>XieH<br>2016 -<br>９４２１<br>Ｂ Ｅ ＥＺ<br>1.5<br>1.36<br>YuJ<br>1.12<br>1.1<br>1.07<br>Zeević<br>1.576<br>1.02<br>ZellerG<br>2014 -<br>1.0<br>0.5<br>BGC categories<br>0.0<br>Other<br>RiPP<br>Saccharide<br>NRP<br>Polyketide<br>Terpene<br>Alkaloid -->

<img src="https://cdn.noedgeai.com/bo_d78vubc91nqc73epp0u0_13.jpg?x=413&y=144&w=1023&h=1655&r=0"/>

Figure 6. BGC-Prophet reveals the biosynthetic potential from the human microbiome. (A) Distribution of predicted BGCs from the human microbiome metagenomic dataset (MG) on the evolutionary tree. BGCs were predicted using BGC-Prophet in the MG dataset, followed by species annotation. For the same species, the number of BGCs was averaged. From the innermost to the outermost region, the central core represents the weighted evolutionary tree structure, which is composed of 13 archaeal species and 2909 bacterial species, with different sectors representing several major phyla, such as Actinomycetota, Bacillota_A, and Bacteroidota. The first ring depicts the heatmap of the total number of BGCs, showing a clear enrichment in phyla such as Actinomycetota. The second to eighth rings display the distribution of BGCs belonging to the alkaloid, terpene, polyketid NRP, saccharide, RiPP, and other categories, respectively. (B) The bar plot from top to bottom shows the total count, prevalence ratio (# species with BGC/# species examined), and average number of BGCs per genome, for each type of BGC predicted. The order from largest to smallest was Other, RiPP, Saccharide, NRP, Polykteide, Terpene, and Allkaloid. The greater number of other BGCs is likely due to shorter contig lengths, which results in fragmented BGC predictions that cannot be further classified. (C) Heatmap showing the number of different types of predicted BGCs according to BGC-Prophet for the 47 metagenomic datasets (see the 'Materials and methods' for details). The predicted numbers vary depending on the number of genomes, contig lengths, ecological niches, and biosynthetic capabilities of the respective datasets. For detailed information, please refer to Supplementary Table S7.

<!-- Media -->

<!-- Meanless: Downloaded from https://academic.oup.com/nar/article/53/7/gkaf305/8113170 by National Science & Technology Library user on 16 April 2025 -->


<!-- Meanless: BGC-Prophet for BGC mining using AI<br>15<br>のコミテミュニューテーテーディアーテーテーテーショーショーテーズを考える方法であって -->

Second, BGC-Prophet is one of the pioneering UHT methods enabling pan-phylogenetic screening and whole-metagenome screening of BGCs, providing a comprehensive BGC profile across 85 203 genomes and 9428 metagenomes spanning bacterial and archaeal diversity. We first investigated the biosynthetic potential of the Aspergillus genomes and revealed numerous potentially novel BGCs missed by an-tiSMASH (Fig. 4). Then, we generated a comprehensive profile of BGCs on 85 203 genomes and 9428 metagenomes representing the majority of bacterial and archaeal lineages and revealed that BGC-Prophet allows for the detection of previously undiscovered BGCs, as well as the reconstruction of a comprehensive picture of BGCs on genomes from the majority of bacterial and archaeal lineages (Fig. 5). While this represents a significant advancement, it is important to note that existing methods such as antiSMASH have also been successfully applied to large-scale genomic analyses, including studies involving over 170 000 genomes [69]. BGC-Prophet offers significant speed advantages, making it particularly useful for ultrahigh-throughput applications where computational resources may be limited or when analyzing even larger datasets. This efficiency does not come at the cost of accuracy, as BGC-Prophet maintains high precision in BGC detection, which is critical for large-scale genomic studies.

Third, BGC-Prophet reveals enrichment patterns of BGCs that correspond with important geological events, suggesting potential environmental influences on BGC evolution. For instance, following the Great Oxidation event, there was a noticeable increase in the average number of BGCs per genome, particularly for polyketides. While this could imply a role for polyketides in microbial adaptation to increased oxygen levels, it is also possible that the presence of oxygen facilitated the production of polyketides. Similarly, the Cambrian Explosion event coincides with increased BGC diversity, particularly for polyketides and NRPs, indicating that microbial adaptation to environmental changes may involve the production of specialized secondary metabolites.

BGC-Prophet is not without limitations. First, it cannot determine the specific small molecules generated by these clusters. This presents an opportunity for future work, which should focus on predicting BGCs in microbial genomes associated with known small molecules, followed by computational chemistry to screen and validate these predictions [70]. Additionally, substantial and accurately annotated training data are crucial for the model's performance. Although BGC-Prophet demonstrates high accuracy among the existing methods, overfitting remains a concern due to the relatively small training dataset. Further efforts are needed to construct more diverse and comprehensive BGC databases to enhance model training and validation. While the Transformer architecture mitigates potential biases from the dataset's long-tail distribution by effectively handling varying sequence lengths, incorporating debiasing techniques in future iterations could further enhance robustness and fairness. Other possible improvements might include the discovery of new categories of BGCs, as well as the examination of the gain or loss of BGCs on a dynamic scale.

The BGC-Prophet language model introduced here shows the promise of extending BGC science and engineering, which can also be extended to other functional genes, such as antibiotic resistance genes and anti-CRISPR proteins [71, 72]. In NLP and protein engineering, language models are often fine-tuned on downstream tasks of interest. For BGCs, these downstream tasks could include predicting their expression conditions or the chemical structures of their products. For antimicrobial resistance genes and anti-CRISPR proteins, the downstream tasks could include predicting their type and mechanism. The language model proposed in this study for BGC predictions could shed light on other functional gene discoveries based on modeling approach.

In summary, our work demonstrates a novel approach to BGC discovery using a language modeling framework. As one of the pioneering UHT methods for pan-phylogenetic and whole-metagenome BGC screening, BGC-Prophet offers valuable insights into BGC patterns, mechanisms, and their ecological implications. While complementary to existing methods, BGC-Prophet distinguishes itself with its efficiency and broad applicability, making it an indispensable tool for synthetic biology, species diversity conservation, and beyond, promising a deeper understanding of microbial secondary metabolism and its broader ramifications.


<!-- Meanless: 16<br>Lai et al. -->

## Acknowledgements

Numerical computations were performed on the Hefei Advanced Computing Center.

Author contributions: Qilong Lai (Data curation [equal], Formal analysis [equal], Software [equal], Visualization [equal], Writing-original draft [equal], Writing-review & editing [equal]), Shuai Yao (Data curation [equal], Formal analysis [equal], Software [equal], Visualization [equal], Writing-original draft [equal]), Yuguo Zha (Formal analysis [equal], Visualization [equal], Writing-original draft [equal], Writing-review & editing [equal]), Haohong Zhang (Formal analysis [supporting], Visualization [supporting], Writing-review & editing [supporting]), Haobo Zhang (Formal analysis [equal], Writing-review & editing [supporting]), Ying Ye (Conceptualization [supporting], Writing-review & editing [supporting]), Yonghui Zhang (Funding acquisition [supporting], Writing—review & editing [supporting]), Hong Bai (Funding acquisition [equal], Writing—review & editing [supporting]), and Kang Ning (Conceptualization [lead], Funding acquisition [equal], Writing-review & editing [lead])

## Supplementary data

Supplementary data is available at NAR online.

## Conflict of interest

None declared.

## Funding

The National Key R&D Program of China (Grant Nos. 2021YFA0910500, SQ2023YFA1800082, and 2018YFC0910502) and the National Natural Science Foundation of China (Grant Nos. 32071465, 31871334, and 31671374).

## Data availability

All the resources, tools, and concepts used in this study are shown in Supplementary Tables S1-S3. All source codes have been uploaded to https://github.com/HUST-NingKang-Lab/BGC-Prophet and https://figshare.com/articles/software/ BGC-Prophet/28703642/1. All the data have been uploaded to FigShare (https://figshare.com/articles/dataset/ _b_Predicted_biosynthetic_gene_cluster_resources_from_ microbiome_data_b_/25974280). All the datasets and codes used in this study are publicly available.

## References

1. Bauman KD, Butler KS, Moore BS et al. Genome mining methods to discover bioactive natural products. Nat Prod Rep 2021;38:2100-29.

2. Ma J, Gu Y, Xu P. A roadmap to engineering antiviral natural products synthesis in microbes. Curr Opin Biotechnol 2020;66:140-9.

3. Kautsar SA, Blin K, Shaw S et al. MIBiG 2.0: a repository for biosynthetic gene clusters of known function. Nucleic Acids Res 2020;48:D454-8.

4. Terlouw BR, Blin K, Navarro-Muñoz JC et al. MIBiG 3.0: a community-driven effort to annotate experimentally validated biosynthetic gene clusters. Nucleic Acids Res 2023;51:D603-10. https://doi.org/10.1093/nar/gkac1049

5. Newman DJ, Cragg GM. Natural products as sources of new drugs over the nearly four decades from 01/1981 to 09/2019. J Nat Prod 2020;83:770-803.

https://doi.org/10.1021/acs.jnatprod.9b01285

6. Brown ED, Wright GD. Antibacterial drug discovery in the resistance era. Nature 2016;529:336-43. https://doi.org/10.1038/nature17042

7. Martin JF. Clusters of genes for the biosynthesis of antibiotics: regulatory genes and overproduction of pharmaceuticals. J Ind Microbiol 1992;9:73-90. https://doi.org/10.1007/BF01569737

8. Martin JF, Liras P. Organization and expression of genes involved in the biosynthesis of antibiotics and other secondary metabolites. Annu Rev Microbiol 1989;43:173-206.

9. Negri T, Mantri S, Angelov A et al. A rapid and efficient strategy to identify and recover biosynthetic gene clusters from soil metagenomes. Appl Microbiol Biotechnol 2022;106:3293-306. https://doi.org/10.1007/s00253-022-11917-y

10. Paoli L, Ruscheweyh H-J, Forneris CC et al. Biosynthetic potential of the global ocean microbiome. Nature 2022;607:111-8.

11. Thomford NE, Senthebane DA, Rowe A et al. Natural products for drug discovery in the 21st century: innovations for novel drug discovery. Int J Mol Sci 2018;19:1578.

12. Liu X, Ijzerman AP, van Westen GJP. In: Cartwright H (ed.), Artificial Neural Networks. New York, NY: Springer US, 2021, 139-65.

13. Martinet L, Naômé A, Deflandre B et al. A single biosynthetic gene cluster is responsible for the production of Bagremycin antibiotics and ferroverdin iron chelators. mBio 2019;10:e01230-19. https://doi.org/10.1128/mbio.01230-01219

14. Kwon Min J, Steiniger C, Cairns Timothy C et al. Beyond the biosynthetic gene cluster paradigm: genome-wide coexpression networks connect clustered and unclustered transcription factors to secondary metabolic pathways. Microbiol Spectr 2021;9:e00898-21.

15. Blin K, Shaw S, Kloosterman AM et al. antiSMASH 6.0: improving cluster detection and comparison capabilities. Nucleic Acids Res 2021;49:W29-35. https://doi.org/10.1093/nar/gkab335

16. Blin K, Shaw S, Augustijn HE et al. antiSMASH 7.0: new and improved predictions for detection, regulation, chemical structures and visualisation. Nucleic Acids Res 2023;51:W46-50. https://doi.org/10.1093/nar/gkad344

17. Oliynyk M, Samborskyy M, Lester JB et al. Complete genome sequence of the erythromycin-producing bacterium saccharopolyspora erythraea NRRL23338. Nat Biotechnol 2007;25:447-53.

18. Walsh CT, Fischbach MA. Natural Products Version 2.0: connecting genes to molecules. J Am Chem Soc 2010;132:2469-93.

19. van der Hooft JJJ, Mohimani H, Bauermeister A et al. Linking genomics and metabolomics to chart specialized metabolic diversity. Chem Soc Rev 2020;49:3297-314.

20. Skinnider MA, Merwin NJ, Johnston CW et al. PRISM 3: expanded prediction of natural product chemical structures from microbial genomes. Nucleic Acids Res 2017;45:W49-54.

21. Medema MH, Fischbach MA. Computational approaches to natural product discovery. Nat Chem Biol 2015;11:639-48.

22. Medema MH, de Rond T, Moore BS. Mining genomes to illuminate the specialized chemistry of life. Nat Rev Genet 2021;22:553-71.

23. Cimermancic P, Medema MH, Claesen J et al. Insights into secondary metabolism from a global analysis of prokaryotic biosynthetic gene clusters. Cell 2014;158:412-21.

24. de los Santos ELC. NeuRiPP: neural network identification of RiPP precursor peptides. Sci Rep 2019;9:13406.

25. Merwin NJ, Mousa WK, Dejong CA et al. DeepRiPP integrates multiomics data to automate discovery of novel ribosomally synthesized natural products. Proc Nat Acad Sci USA 2020;117:371-80.

<!-- Meanless: (5)今日の日の日には、三才三才三年の日の日がいいのですが、この日には、この日の日を迎えたので、この日の日がいいので、この日の日がいいので、この日は、日々の日から -->


<!-- Meanless: BGC-Prophet for BGC mining using AI<br>17<br>9707 11:dy 91, no tesn rieiql? 160104481041 8 201818 1 4 01:81:81/908.1976/21:89/01:01:41:30/01:01:10:01:11.001.0018:00 -->

26. Hannigan GD, Prihoda D, Palicka A et al. A deep learning genome-mining strategy for biosynthetic gene cluster prediction. Nucleic Acids Res 2019;47:e110.

27. Liu M, Li Y, Li H. Deep learning to predict the biosynthetic gene clusters in bacterial genomes. J Mol Biol 2022;434:167597.

28. Yang Z, Liao B, Hsieh C et al. Deep-BGCpred: a unified deep learning genome-mining framework for biosynthetic gene cluster prediction. bioRxiv, https://doi.org/10.1101/2021.11.15.468547, 16 November 2021, preprint: not peer reviewed.

29. Rios-Martinez C, Bhattacharya N, Amini AP et al. Deep self-supervised learning for biosynthetic gene cluster detection and product classification. PLoS Comput Biol 2023;19:e1011162. https://doi.org/10.1371/journal.pcbi.1011162

30. Laura MC, Martin L, Jonas Simon F et al. Accurate de novo identification of biosynthetic gene clusters with GECCO. bioRxiv, https://doi.org/10.1101/2021.05.03.442509, 4 May 2021, preprint: not peer reviewed.

31. Sanchez S, Rogers JD, Rogers AB et al. Expansion of novel biosynthetic gene clusters from diverse environments using SanntiS. bioRxiv, https://doi.org/10.1101/2023.05.23.540769, 2 June 2023, preprint: not peer reviewed.

32. Huang J, Gao Q, Tang Y et al. Protein language model-based end-to-end type II polyketide prediction without sequence alignment. bioRxiv, https://doi.org/10.1101/2023.04.18.537339, 20 April 2023, preprint: not peer reviewed.

33. Vaswani A, Shazeer N, Parmar N et al. Proceedings of the 31st International Conference on Neural Information Processing Systems. Long Beach, California, USA: Curran Associates Inc., 2017, 6000-10.

34. Devlin J, Chang M-W, Lee K et al. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Minneapolis, Minnesota: Association for Computational Linguistics, 2019, 4171-86.

35. Parks DH, Chuvochina M, Rinke C et al. GTDB: an ongoing census of bacterial and archaeal diversity through a phylogenetically consistent, rank normalized and complete genome-based taxonomy. Nucleic Acids Res 2022;50:D785-94. https://doi.org/10.1093/nar/gkab776

36. Pasolli E, Asnicar F, Manara S et al. Extensive unexplored human microbiome diversity revealed by over 150,000 genomes from metagenomes spanning age, geography, and lifestyle. Cell 2019;176:649-662.e20.

https://doi.org/10.1016/j.cell.2019.01.001

37. Lin Z, Akin H, Rao R et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 2023;379:1123-30. https://doi.org/10.1126/science.ade2574

38. Xiong R, Yang Y, He D et al. On layer normalization in the transformer architecture. In: Proceedings of the 37th International Conference on Machine Learning, Vol. 119, 2020, Article 975.

39. Wang F. Linear chain conditional random field for operating mode identification and multimode process monitoring. ACS Omega 2022;7:29483-94. https://doi.org/10.1021/acsomega.2c04005

40. Hendrycks D, Gimpel KJaL. Gaussian error linear units (GELUs). arXiv, https://arxiv.org/abs/1606.08415, 27 June 2026, preprint: not peer reviewed.

41. Zhuang Z, Liu M, Cutkosky A et al. Understanding AdamW through proximal methods and scale-freeness. arXiv, https://arxiv.org/abs/2202.00089, 31 January 2022, preprint: not peer reviewed.

42. Mungan MD, Alanjary M, Blin K et al. ARTS 2.0: feature updates and expansion of the Antibiotic Resistant Target Seeker for comparative genome mining. Nucleic Acids Res 2020;48:W546-52. https://doi.org/10.1093/nar/gkaa374

43. Medema MH, Paalvast Y, Nguyen DD et al. Pep2Path: automated mass spectrometry-guided genome mining of peptidic natural products. PLoS Comput Biol 2014;10:e1003822. https://doi.org/10.1371/journal.pcbi.1003822

44. Palaniappan K, Chen I-MA, Chu K et al. IMG-ABC v.5.0: an update to the IMG/Atlas of Biosynthetic Gene Clusters Knowledgebase. Nucleic Acids Res 2019;48:D422-30.

45. Vallenet D, Calteau A, Dubois M et al. MicroScope: an integrated platform for the annotation and exploration of microbial gene functions through genomic, pangenomic and metabolic comparative analysis. Nucleic Acids Res 2019;48:D579-89.

46. Tang J, Matsuda Y. Discovery of fungal onoceroid triterpenoids through domainless enzyme-targeted global genome mining. Nat Commun 2024;15:4312.

https://doi.org/10.1038/s41467-024-48771-7

47. Navarro-Muñoz JC, Selem-Mojica N, Mullowney MW et al. A computational framework to explore large-scale biosynthetic diversity. Nat Chem Biol 2020;16:60-8. https://doi.org/10.1038/s41589-019-0400-9

48. Delwiche CC, Bryan BA. Denitrification. Annu Rev Microbiol 1976;30:241-62.

https://doi.org/10.1146/annurev.mi.30.100176.001325

49. Baker BJ, De Anda V, Seitz KW et al. Diversity, ecology and evolution of Archaea. Nat Microbiol 2020;5:887-900. https://doi.org/10.1038/s41564-020-0715-z

50. Kumar S, Suleski M, Craig JM et al. TimeTree 5: an expanded resource for species divergence times. Mol Biol Evol 2022;39:msac174. https://doi.org/10.1093/molbev/msac174

51. Schopf JW. Geological evidence of oxygenic photosynthesis and the biotic response to the 2400-2200 ma "great oxidation event". Biochemistry Moscow 2014;79:165-77. https://doi.org/10.1134/S0006297914030018

52. Zhuravlev AY, Wood RA. The two phases of the Cambrian Explosion. Sci Rep 2018;8:16656. https://doi.org/10.1038/s41598-018-34962-y

53. Ostrander CM, Nielsen SG, Owens JD et al. Fully oxygenated water columns over continental shelves before the Great Oxidation Event. Nat Geosci 2019;12:186-91. https://doi.org/10.1038/s41561-019-0309-7

54. Fan L, Wu D, Goremykin V et al. Phylogenetic analyses with systematic taxon sampling show that mitochondria branch within Alphaproteobacteria. Nat Ecol Evol 2020;4:1213-9.

55. Zhou Z, St John E, Anantharaman K et al. Global patterns of diversity and metabolism of microbial communities in deep-sea hydrothermal vent deposits. Microbiome 2022;10:241. https://doi.org/10.1186/s40168-022-01424-7

56. Soo RM, Woodcroft BJ, Parks DH et al. Back from the dead; the curious tale of the predatory cyanobacterium vampirovibrio chlorellavorus. PeerJ 2015;3:e968. https://doi.org/10.7717/peerj.968

57. Gallimore AR. The biosynthesis of polyketide-derived polycyclic ethers. Nat Prod Rep 2009;26:266-80. https://doi.org/10.1039/B807902C

58. Marahiel MA. A structural model for multimodular NRPS assembly lines. Nat Prod Rep 2016;33:136-40. https://doi.org/10.1039/C5NP00082C

59. Wood R, Liu AG, Bowyer F et al. Integrated records of environmental change and evolution challenge the Cambrian Explosion. Nat Ecol Evol 2019;3:528-38.

60. Behnken S, Hertweck C. Anaerobic bacteria as producers of antibiotics. Appl Microbiol Biotechnol 2012;96:61-7. https://doi.org/10.1007/s00253-012-4285-8

61. Geller-McGrath D, Mara P, Taylor GT et al. Diverse secondary metabolites are expressed in particle-associated and free-living microorganisms of the permanently anoxic Cariaco Basin. Nat Commun 2023;14:656.

https://doi.org/10.1038/s41467-023-36026-w

62. Chen X, Ling H-F, Vance D et al. Rise to modern levels of ocean oxygenation coincided with the cambrian radiation of animals. Nat Commun 2015;6:7142. https://doi.org/10.1038/ncomms8142


<!-- Meanless: Downloaded from https://academic.oup.com/nar/article/53/7/gkaf305/8113170 by National Science & Technology Library user on 16 April 2025<br>18<br>Lai et al. -->

63. Fox D. What sparked the Cambrian explosion? Nature 2016;530:268-70. https://doi.org/10.1038/530268a

64. Yang D, Guo X, Xie T et al. Reactive oxygen species may play an essential role in driving biological evolution: the Cambrian Explosion as an example. J Environ Sci 2018;63:218-26. https://doi.org/10.1016/j.jes.2017.05.035

65. Benton MJ. Diversification and extinction in the history of life. Science 1995;268:52-8. https://doi.org/10.1126/science.7701342

66. Xiao S, Laflamme M. On the eve of animal radiation: phylogeny, ecology and evolution of the Ediacara biota. Trends Ecol Evol 2009;24:31-40.

67. Lee MSY, Soubrier J, Edgecombe GD. Rates of phenotypic and genomic evolution during the Cambrian explosion. Curr Biol 2013;23:1889-95. https://doi.org/10.1016/j.cub.2013.07.055

68. Briggs DEG. The Cambrian explosion. Curr Biol 2015;25:R864-8.

69. Gavriilidou A, Kautsar SA, Zaburannyi N et al. Compendium of specialized metabolite biosynthetic diversity encoded in bacterial genomes. Nat Microbiol 2022;7:726-35.

70. Hjörleifsson Eldjárn G, Ramsay A, van der Hooft JJJ et al. Ranking microbial metabolomic and genomic links in the NPLinker framework using complementary scoring functions. PLoS Comput Biol 2021;17:e1008920.

71. Wu J, Ouyang J, Qin H et al. PLM-ARG: antibiotic resistance gene identification using a pretrained protein language model. Bioinformatics 2023;39:btad690.

72. Li Y, Wei Y, Xu S et al. AcrNET: predicting anti-CRISPR with deep learning. Bioinformatics 2023;39:btad259.

<!-- Meanless: Received: October 9, 2024. Revised: March 21, 2025. Editorial Decision: March 28, 2025. Accepted: April 3, 2025<br>© The Author(s) 2025. Published by Oxford University Press on behalf of Nucleic Acids Research.<br>This is an Open Access article distributed under the terms of the Creative Commons Attribution-NonCommercial License (https://creativecommons.org/licenses/by-nc/4/.0/), which permits<br>non-commercial re-use, distribution, and reproduction in any medium, provided the original work is properly cited. For commercial re-use, please contact reprints@oup.com for reprints and<br>translation rights for reprints. All other permissions can be obtained through our RightsLink service via the Permissions link on the article page on our site—for further information please contact<br>journals.permissions@oup.com. -->