"""
跨属评估实验
- 同属测试: train on A, test on A
- 跨属测试: train on Streptomyces, test on {Micro, Amyco, Pseudo}
- 混合训练: train on all, test on each
使用已有的Evo2 embedding + sklearn分类器
"""

import pickle, numpy as np, os, sys, json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

BASE = "/data/muskliu/np_master/data"
MULTIGENUS = os.path.join(BASE, "multigenus")
# 旧的Streptomyces embedding
STREP_OLD = BASE  # train_emb.pkl, val_emb.pkl, test_emb.pkl

GENERA = ["streptomyces", "micromonospora", "amycolatopsis", "pseudomonas"]
SEED = 42


def load_emb(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['embeddings'], d['labels']


def load_genus(genus, split):
    """加载某属某split的embedding"""
    if genus == "streptomyces":
        # 用旧的embedding
        path = os.path.join(STREP_OLD, f"{split}_emb.pkl")
    else:
        path = os.path.join(MULTIGENUS, genus, f"{split}_emb.pkl")
    return load_emb(path)


def evaluate(clf, X_test, y_test):
    """评估分类器"""
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    return {
        'auroc': roc_auc_score(y_test, y_prob),
        'acc': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'prec': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'n': len(y_test),
    }


def fmt(r):
    return f"AUROC={r['auroc']:.3f} Acc={r['acc']:.3f} F1={r['f1']:.3f} P={r['prec']:.3f} R={r['recall']:.3f} (n={r['n']})"


def main():
    print("=" * 70)
    print("跨属评估实验")
    print("=" * 70)

    results = {}

    # ===== 1. 同属测试 =====
    print("\n" + "=" * 70)
    print("实验1: 同属测试 (train on A, test on A)")
    print("=" * 70)

    for genus in GENERA:
        X_train, y_train = load_genus(genus, 'train')
        X_test, y_test = load_genus(genus, 'test')
        print(f"\n  {genus}: train={len(y_train)}, test={len(y_test)}")

        clf = RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)
        clf.fit(X_train, y_train)
        r = evaluate(clf, X_test, y_test)
        results[f"same_{genus}"] = r
        print(f"    RF: {fmt(r)}")

    # ===== 2. 跨属测试 (Strep → others) =====
    print("\n" + "=" * 70)
    print("实验2: 跨属测试 (train on Streptomyces, test on others)")
    print("=" * 70)

    X_train, y_train = load_genus('streptomyces', 'train')
    clf_strep = RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)
    clf_strep.fit(X_train, y_train)

    for genus in GENERA:
        X_test, y_test = load_genus(genus, 'test')
        r = evaluate(clf_strep, X_test, y_test)
        results[f"strep2{genus}"] = r
        tag = "(同属)" if genus == "streptomyces" else "(跨属)"
        print(f"  → {genus} {tag}: {fmt(r)}")

    # ===== 3. 混合训练 =====
    print("\n" + "=" * 70)
    print("实验3: 混合训练 (train on all 4 genera, test on each)")
    print("=" * 70)

    X_all, y_all = [], []
    for genus in GENERA:
        X, y = load_genus(genus, 'train')
        X_all.append(X)
        y_all.append(y)
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    print(f"  混合训练集: {len(y_all)} samples")

    clf_mixed = RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)
    clf_mixed.fit(X_all, y_all)

    for genus in GENERA:
        X_test, y_test = load_genus(genus, 'test')
        r = evaluate(clf_mixed, X_test, y_test)
        results[f"mixed2{genus}"] = r
        print(f"  → {genus}: {fmt(r)}")

    # ===== 4. 多分类器对比 (Strep→others) =====
    print("\n" + "=" * 70)
    print("实验4: 多分类器对比 (train Strep, test Pseudomonas)")
    print("=" * 70)

    X_train, y_train = load_genus('streptomyces', 'train')
    X_test, y_test = load_genus('pseudomonas', 'test')

    classifiers = [
        ('LR', LogisticRegression(max_iter=1000, random_state=SEED)),
        ('RF', RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)),
        ('GBT', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=SEED)),
        ('MLP', MLPClassifier((256, 128), max_iter=500, random_state=SEED)),
    ]

    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        r = evaluate(clf, X_test, y_test)
        results[f"clf_{name}_strep2pseudo"] = r
        print(f"  {name}: {fmt(r)}")

    # ===== 汇总表 =====
    print("\n" + "=" * 70)
    print("汇总: AUROC矩阵")
    print("=" * 70)

    print(f"\n  {'训练\\测试':<16}", end="")
    for g in GENERA:
        print(f"{g[:8]:>10}", end="")
    print()
    print(f"  {'-'*56}")

    # 同属
    print(f"  {'同属(RF)':<16}", end="")
    for g in GENERA:
        r = results.get(f"same_{g}", {})
        print(f"{r.get('auroc', 0):.3f}".rjust(10), end="")
    print()

    # Strep跨属
    print(f"  {'Strep→(RF)':<16}", end="")
    for g in GENERA:
        r = results.get(f"strep2{g}", {})
        print(f"{r.get('auroc', 0):.3f}".rjust(10), end="")
    print()

    # 混合
    print(f"  {'混合→(RF)':<16}", end="")
    for g in GENERA:
        r = results.get(f"mixed2{g}", {})
        print(f"{r.get('auroc', 0):.3f}".rjust(10), end="")
    print()

    # 保存结果
    out_path = os.path.join(MULTIGENUS, "cross_genus_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存到: {out_path}")


if __name__ == '__main__':
    main()
