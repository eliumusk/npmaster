"""
验证实验: 排查跨属结果是否有伪影
1. N-padding 检查: 正样本中含N的比例
2. Bootstrap CI: 每个AUROC的95%置信区间
3. 随机标签 baseline: 打乱标签后AUROC应该~0.5
"""

import pickle, numpy as np, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

BASE = "/data/muskliu/np_master/data"
MULTIGENUS = os.path.join(BASE, "multigenus")
STREP_OLD = BASE
SEED = 42
np.random.seed(SEED)

GENERA = ["streptomyces", "micromonospora", "amycolatopsis", "pseudomonas"]


def load_emb(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['embeddings'], d['labels']


def load_genus_emb(genus, split):
    if genus == "streptomyces":
        return load_emb(os.path.join(STREP_OLD, f"{split}_emb.pkl"))
    return load_emb(os.path.join(MULTIGENUS, genus, f"{split}_emb.pkl"))


def load_genus_raw(genus, split):
    if genus == "streptomyces":
        # 旧数据在 clean_strep 目录
        path = os.path.join(BASE, f"{split}.pkl")
    else:
        path = os.path.join(MULTIGENUS, genus, f"{split}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)


# ========== 检查1: N-padding ==========
print("=" * 60)
print("检查1: N-padding 伪影")
print("=" * 60)

for genus in GENERA:
    samples = load_genus_raw(genus, 'train')
    pos = [s for s in samples if s['is_bgc'] == 1]
    neg = [s for s in samples if s['is_bgc'] == 0]

    pos_n_frac = [s['dna'].count('N') / len(s['dna']) for s in pos]
    neg_n_frac = [s['dna'].count('N') / len(s['dna']) for s in neg]
    pos_has_n = sum(1 for f in pos_n_frac if f > 0.01)
    neg_has_n = sum(1 for f in neg_n_frac if f > 0.01)

    print(f"\n  {genus}:")
    print(f"    正样本: {len(pos)} total, {pos_has_n} 含>1%N ({pos_has_n/len(pos)*100:.0f}%)")
    print(f"    负样本: {len(neg)} total, {neg_has_n} 含>1%N ({neg_has_n/len(neg)*100:.0f}%)")
    print(f"    正N均值={np.mean(pos_n_frac):.4f}, 负N均值={np.mean(neg_n_frac):.4f}")

    if pos_has_n / max(len(pos), 1) > 0.2:
        print(f"    ⚠️  WARNING: {pos_has_n/len(pos)*100:.0f}%的正样本含N, 可能是shortcut!")


# ========== 检查2: Bootstrap CI ==========
print("\n" + "=" * 60)
print("检查2: Bootstrap 95% CI")
print("=" * 60)

def bootstrap_auroc(y_true, y_score, n_boot=2000):
    aurocs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, ys))
    aurocs = np.array(aurocs)
    return np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)


# Train Strep RF
X_train, y_train = load_genus_emb('streptomyces', 'train')
clf = RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)
clf.fit(X_train, y_train)

for genus in GENERA:
    X_test, y_test = load_genus_emb(genus, 'test')
    y_prob = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    lo, hi = bootstrap_auroc(y_test, y_prob)
    tag = "(同属)" if genus == "streptomyces" else "(跨属)"
    print(f"  Strep→{genus} {tag}: AUROC={auroc:.3f} [{lo:.3f}, {hi:.3f}] (n={len(y_test)})")


# ========== 检查3: 随机标签 baseline ==========
print("\n" + "=" * 60)
print("检查3: 随机标签 baseline (应该~0.5)")
print("=" * 60)

for genus in GENERA:
    X_test, y_test = load_genus_emb(genus, 'test')
    y_prob = clf.predict_proba(X_test)[:, 1]

    random_aurocs = []
    for _ in range(100):
        y_shuffled = np.random.permutation(y_test)
        if len(np.unique(y_shuffled)) < 2:
            continue
        random_aurocs.append(roc_auc_score(y_shuffled, y_prob))
    print(f"  {genus}: 随机AUROC={np.mean(random_aurocs):.3f}±{np.std(random_aurocs):.3f}")


# ========== 检查4: 只用N含量能区分多少 ==========
print("\n" + "=" * 60)
print("检查4: 仅用N含量的AUROC (N-shortcut检测)")
print("=" * 60)

for genus in GENERA:
    samples = load_genus_raw(genus, 'test')
    n_fracs = np.array([s['dna'].count('N') / len(s['dna']) for s in samples])
    labels = np.array([s['is_bgc'] for s in samples])
    if len(np.unique(labels)) < 2:
        continue
    auroc = roc_auc_score(labels, n_fracs)
    print(f"  {genus}: N-only AUROC={auroc:.3f} (n={len(labels)})")

print("\n完成!")
