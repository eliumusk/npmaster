"""
用Evo2 embedding训练BGC分类器
输入: data/train_emb.pkl, data/val_emb.pkl, data/test_emb.pkl (由06脚本生成)
"""

import pickle
import numpy as np
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

DATA_DIR = "/data/muskliu/np_master/data"


def load_embeddings(split_name):
    path = os.path.join(DATA_DIR, f"{split_name}_emb.pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"  {split_name}: {data['num_samples']}个样本, dim={data['embedding_dim']}, errors={data['num_errors']}")
    return data


def kmer_features(data_dir, split_name):
    """提取k-mer特征作为baseline对比"""
    path = os.path.join(data_dir, f"{split_name}.pkl")
    with open(path, 'rb') as f:
        samples = pickle.load(f)

    features = []
    for s in samples:
        seq = ''.join(c for c in s['dna'].upper() if c in 'ATCG')
        if len(seq) < 4:
            features.append(np.zeros(260))
            continue
        counts = Counter(seq[i:i+4] for i in range(len(seq)-3))
        bases = 'ATCG'
        kmers = []
        def gen(p, r):
            if r == 0: kmers.append(p); return
            for b in bases: gen(p+b, r-1)
        gen('', 4)
        total = max(sum(counts.values()), 1)
        kf = [counts.get(km, 0)/total for km in kmers]
        gc = (seq.count('G')+seq.count('C'))/len(seq)
        features.append(np.array(kf + [gc, s['dna'].count('N')/len(s['dna']),
            len(set(seq[i:i+4] for i in range(len(seq)-3)))/min(256, max(len(seq)-3,1)),
            len(seq)/8192]))
    return np.array(features)


def main():
    print("=" * 60)
    print("BGC分类器训练 (Evo2 embedding)")
    print("=" * 60)

    # 加载embedding
    print("\n[1] 加载数据...")
    train = load_embeddings("train")
    val = load_embeddings("val")
    test = load_embeddings("test")

    X_train, y_train = train['embeddings'], train['labels']
    X_val, y_val = val['embeddings'], val['labels']
    X_test, y_test = test['embeddings'], test['labels']

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print(f"\n  训练: {X_train.shape}, 正={y_train.sum()}, 负={len(y_train)-y_train.sum()}")
    print(f"  测试: {X_test.shape}, 正={y_test.sum()}, 负={len(y_test)-y_test.sum()}")

    # k-mer baseline
    print("\n[2] k-mer baseline...")
    kmer_train = kmer_features(DATA_DIR, "train")
    kmer_test = kmer_features(DATA_DIR, "test")
    # 只取成功提取embedding的那些样本（按顺序应该一致，但以防万一）
    kmer_train = kmer_train[:len(y_train)]
    kmer_test = kmer_test[:len(y_test)]

    print("\n[3] 训练分类器...")
    print(f"\n{'方法':<30} {'AUROC':>8} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print("-" * 75)

    results = {}

    classifiers = [
        # k-mer baselines
        ("k-mer + LR", LogisticRegression(max_iter=1000, random_state=42), kmer_train, kmer_test, False),
        ("k-mer + RF", RandomForestClassifier(100, max_depth=20, random_state=42, n_jobs=-1), kmer_train, kmer_test, False),

        # Evo2 embedding
        ("Evo2 + LR", LogisticRegression(max_iter=1000, random_state=42), X_train_s, X_test_s, True),
        ("Evo2 + RF", RandomForestClassifier(200, max_depth=30, random_state=42, n_jobs=-1), X_train_s, X_test_s, True),
        ("Evo2 + GBT", GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42), X_train_s, X_test_s, True),
        ("Evo2 + MLP(256,128)", MLPClassifier((256, 128), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1), X_train_s, X_test_s, True),
        ("Evo2 + MLP(512,256,128)", MLPClassifier((512, 256, 128), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1), X_train_s, X_test_s, True),
    ]

    # 合并特征
    combined_train = np.concatenate([X_train_s, kmer_train], axis=1)
    combined_test = np.concatenate([X_test_s, kmer_test], axis=1)
    classifiers.extend([
        ("Evo2+kmer + LR", LogisticRegression(max_iter=1000, random_state=42), combined_train, combined_test, True),
        ("Evo2+kmer + RF", RandomForestClassifier(200, max_depth=30, random_state=42, n_jobs=-1), combined_train, combined_test, True),
        ("Evo2+kmer + MLP(512,256)", MLPClassifier((512, 256), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1), combined_train, combined_test, True),
    ])

    for name, clf, X_tr, X_te, is_evo2 in classifiers:
        t0 = time.time()
        try:
            clf.fit(X_tr, y_train)
            y_prob = clf.predict_proba(X_te)[:, 1]
            y_pred = clf.predict(X_te)

            auroc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            elapsed = time.time() - t0

            marker = " ***" if is_evo2 and auroc > 0.90 else ""
            print(f"{name:<30} {auroc:>8.4f} {acc:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}{marker}")
            results[name] = {'auroc': auroc, 'f1': f1, 'acc': acc}
        except Exception as e:
            print(f"{name:<30} ERROR: {e}")

    # 总结
    print(f"\n{'='*60}")
    print("结论:")
    best_kmer = max((v['auroc'], k) for k, v in results.items() if 'k-mer' in k and 'Evo2' not in k)
    best_evo2 = max((v['auroc'], k) for k, v in results.items() if 'Evo2' in k)

    print(f"  最佳k-mer: {best_kmer[1]} AUROC={best_kmer[0]:.4f}")
    print(f"  最佳Evo2:  {best_evo2[1]} AUROC={best_evo2[0]:.4f}")
    print(f"  提升: {best_evo2[0] - best_kmer[0]:+.4f}")

    if best_evo2[0] > best_kmer[0] + 0.02:
        print(f"\n  ✓ Evo2 embedding显著优于k-mer baseline")
    elif best_evo2[0] > best_kmer[0]:
        print(f"\n  △ Evo2略好于k-mer，但差距不大")
    else:
        print(f"\n  ✗ Evo2没有超过k-mer baseline")

    # 保存结果
    with open(os.path.join(DATA_DIR, "classification_results.pkl"), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
