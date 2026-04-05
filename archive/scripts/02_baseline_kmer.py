"""
Baseline实验：用k-mer频率特征 + 逻辑回归/随机森林，验证BGC vs non-BGC是否可分

这个实验不需要GPU，在Mac上就能跑。
目的：验证我们构造的正负样本是否有区分度。
如果k-mer特征都能区分，Evo2 embedding一定能区分得更好。

k-mer是什么？
- 就是长度为k的DNA子串
- 比如 "ATCG" 的 2-mer(二联体)频率: AT=1/3, TC=1/3, CG=1/3
- k-mer频率是DNA序列最基础的统计特征
"""

import pickle
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)
from pathlib import Path
import time


def compute_kmer_features(dna_seq, k=4):
    """计算k-mer频率作为特征向量

    k=4 → 4^4 = 256维特征向量
    k=3 → 4^3 = 64维特征向量
    """
    # 只保留ATCG
    clean_seq = ''.join(c for c in dna_seq.upper() if c in 'ATCG')

    if len(clean_seq) < k:
        return np.zeros(4**k)

    # 计算k-mer频率
    kmer_counts = Counter()
    for i in range(len(clean_seq) - k + 1):
        kmer = clean_seq[i:i+k]
        kmer_counts[kmer] += 1

    # 转成固定维度向量
    bases = 'ATCG'
    all_kmers = []

    def generate_kmers(prefix, remaining):
        if remaining == 0:
            all_kmers.append(prefix)
            return
        for b in bases:
            generate_kmers(prefix + b, remaining - 1)

    generate_kmers('', k)

    total = sum(kmer_counts.values())
    feature_vector = np.array([kmer_counts.get(km, 0) / total for km in all_kmers])

    return feature_vector


def compute_additional_features(dna_seq):
    """额外的全局统计特征"""
    clean_seq = ''.join(c for c in dna_seq.upper() if c in 'ATCG')
    n = len(clean_seq)
    if n == 0:
        return np.zeros(5)

    gc_content = (clean_seq.count('G') + clean_seq.count('C')) / n
    at_content = 1 - gc_content

    # GC skew: (G-C)/(G+C)
    g = clean_seq.count('G')
    c = clean_seq.count('C')
    gc_skew = (g - c) / max(g + c, 1)

    # N比例(padding)
    n_ratio = dna_seq.upper().count('N') / len(dna_seq)

    # 序列复杂度(unique k-mers / possible k-mers)
    k = 4
    kmers = set()
    for i in range(len(clean_seq) - k + 1):
        kmers.add(clean_seq[i:i+k])
    complexity = len(kmers) / min(4**k, len(clean_seq) - k + 1) if len(clean_seq) > k else 0

    return np.array([gc_content, at_content, gc_skew, n_ratio, complexity])


def main():
    print("=" * 60)
    print("Baseline实验: k-mer特征 + 传统分类器")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    data_dir = Path("data/processed")

    with open(data_dir / "train_w8192.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(data_dir / "test_w8192.pkl", 'rb') as f:
        test_data = pickle.load(f)

    print(f"  训练集: {len(train_data)} 样本")
    print(f"  测试集: {len(test_data)} 样本")

    # 提取特征
    print("\n[2/4] 提取k-mer特征 (k=4, 256维 + 5维统计特征)...")
    t0 = time.time()

    X_train = []
    y_train = []
    for sample in train_data:
        kmer_feat = compute_kmer_features(sample['dna'], k=4)
        stat_feat = compute_additional_features(sample['dna'])
        X_train.append(np.concatenate([kmer_feat, stat_feat]))
        y_train.append(sample['is_bgc'])

    X_test = []
    y_test = []
    for sample in test_data:
        kmer_feat = compute_kmer_features(sample['dna'], k=4)
        stat_feat = compute_additional_features(sample['dna'])
        X_test.append(np.concatenate([kmer_feat, stat_feat]))
        y_test.append(sample['is_bgc'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  耗时: {time.time()-t0:.1f}s")
    print(f"  训练集正负比: {y_train.sum()}/{len(y_train)-y_train.sum()}")
    print(f"  测试集正负比: {y_test.sum()}/{len(y_test)-y_test.sum()}")

    # 训练分类器
    print("\n[3/4] 训练分类器...")

    results = {}

    # Logistic Regression
    print("\n  --- Logistic Regression ---")
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    print(f"  训练耗时: {time.time()-t0:.1f}s")

    auroc_lr = roc_auc_score(y_test, y_prob_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)

    print(f"  AUROC:     {auroc_lr:.4f}")
    print(f"  Accuracy:  {acc_lr:.4f}")
    print(f"  F1:        {f1_lr:.4f}")
    print(f"  Precision: {prec_lr:.4f}")
    print(f"  Recall:    {rec_lr:.4f}")
    results['LogisticRegression'] = {'auroc': auroc_lr, 'f1': f1_lr}

    # Random Forest
    print("\n  --- Random Forest ---")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    print(f"  训练耗时: {time.time()-t0:.1f}s")

    auroc_rf = roc_auc_score(y_test, y_prob_rf)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf)
    rec_rf = recall_score(y_test, y_pred_rf)

    print(f"  AUROC:     {auroc_rf:.4f}")
    print(f"  Accuracy:  {acc_rf:.4f}")
    print(f"  F1:        {f1_rf:.4f}")
    print(f"  Precision: {prec_rf:.4f}")
    print(f"  Recall:    {rec_rf:.4f}")
    results['RandomForest'] = {'auroc': auroc_rf, 'f1': f1_rf}

    # 分析
    print("\n[4/4] 结果分析...")
    print("\n  === 对比BGC-Prophet ===")
    print(f"  BGC-Prophet AUROC: 0.92 (检测任务)")
    print(f"  我们的Baseline:")
    print(f"    LR AUROC:  {auroc_lr:.4f}")
    print(f"    RF AUROC:  {auroc_rf:.4f}")

    if auroc_rf > 0.95:
        print("\n  ⚠ 警告：AUROC过高(>0.95)")
        print("  这可能说明负样本太简单了（打乱的DNA太容易区分）")
        print("  正式实验需要使用真实基因组的non-BGC区域作为负样本")
    elif auroc_rf > 0.80:
        print("\n  ✓ 结果合理")
        print("  k-mer特征已经能区分BGC和non-BGC")
        print("  Evo2的深度表示应该能做得更好")
    elif auroc_rf > 0.60:
        print("\n  △ 结果一般")
        print("  k-mer特征区分度有限，但这正是需要深度模型的原因")
    else:
        print("\n  ✗ 结果太差")
        print("  可能数据构造有问题，需要检查")

    # 保存特征重要性(RF)
    print("\n  === Random Forest 特征重要性 Top 10 ===")
    importances = rf.feature_importances_
    # 前256个是4-mer，后5个是统计特征
    stat_names = ['GC_content', 'AT_content', 'GC_skew', 'N_ratio', 'Complexity']

    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices[:10]):
        if idx < 256:
            # 恢复k-mer名称
            bases = 'ATCG'
            kmer = ''
            val = idx
            for _ in range(4):
                kmer = bases[val % 4] + kmer
                val //= 4
            print(f"    {rank+1}. {kmer}: {importances[idx]:.4f}")
        else:
            stat_idx = idx - 256
            print(f"    {rank+1}. {stat_names[stat_idx]}: {importances[idx]:.4f}")


if __name__ == '__main__':
    main()
