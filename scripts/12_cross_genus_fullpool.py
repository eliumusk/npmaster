"""
跨属全量评估 (合并所有split)
- 训练: Streptomyces (固定)
- 测试: 每个属的全部数据 (train+val+test合并)
- Bootstrap: BGC级别重采样
"""

import pickle, numpy as np, os, json
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

BASE = "/data/muskliu/np_master/data"
MULTIGENUS = os.path.join(BASE, "multigenus")
SEED = 42
np.random.seed(SEED)

GENERA = ["micromonospora", "amycolatopsis", "pseudomonas"]


def load_emb(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['embeddings'], d['labels'], d.get('ids', list(range(d['embeddings'].shape[0])))


def load_raw(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pool_genus(genus):
    """合并一个属的全部 split 的 embedding 和 raw 数据"""
    if genus == "streptomyces":
        emb_base = BASE
        raw_base = BASE
    else:
        emb_base = os.path.join(MULTIGENUS, genus)
        raw_base = os.path.join(MULTIGENUS, genus)

    all_emb, all_labels, all_sources = [], [], []
    for split in ['train', 'val', 'test']:
        emb_path = os.path.join(emb_base, f"{split}_emb.pkl")
        raw_path = os.path.join(raw_base, f"{split}.pkl")

        X, y, ids = load_emb(emb_path)
        samples = load_raw(raw_path)

        all_emb.append(X)
        all_labels.append(y)
        for s in samples:
            all_sources.append(s['source'])

    return np.concatenate(all_emb), np.concatenate(all_labels), all_sources


def cluster_bootstrap(y_true, y_prob, sources, n_boot=5000):
    """BGC级别的cluster bootstrap"""
    source_to_idx = defaultdict(list)
    for i, src in enumerate(sources):
        source_to_idx[src].append(i)

    source_keys = list(source_to_idx.keys())
    n_sources = len(source_keys)

    aurocs = []
    for _ in range(n_boot):
        boot_keys = np.random.choice(source_keys, n_sources, replace=True)
        boot_idx = []
        for k in boot_keys:
            boot_idx.extend(source_to_idx[k])
        boot_idx = np.array(boot_idx)

        yt = y_true[boot_idx]
        yp = y_prob[boot_idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, yp))

    aurocs = np.array(aurocs)
    return np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5), len(aurocs)


def evaluate(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'auroc': roc_auc_score(y_true, y_prob),
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }


def main():
    print("=" * 65)
    print("跨属全量评估 (全部数据做测试, BGC级bootstrap)")
    print("=" * 65)

    # 1. 训练 Strep 模型
    print("\n[1] 训练 Streptomyces RF...")
    X_train, y_train, _ = pool_genus('streptomyces')
    clf_rf = RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)
    clf_rf.fit(X_train, y_train)
    clf_lr = LogisticRegression(max_iter=1000, random_state=SEED)
    clf_lr.fit(X_train, y_train)
    print(f"  训练集: {len(y_train)} windows")

    # 2. 在 Strep 自身测试 (用原来的test split, 不用全量避免train-on-test)
    print("\n[2] Streptomyces 同属 (test split only)...")
    X_test_s, y_test_s, ids_s = load_emb(os.path.join(BASE, "test_emb.pkl"))
    samples_s = load_raw(os.path.join(BASE, "test.pkl"))
    sources_s = [s['source'] for s in samples_s]

    y_prob_s = clf_rf.predict_proba(X_test_s)[:, 1]
    m = evaluate(y_test_s, y_prob_s)
    lo, hi, n_valid = cluster_bootstrap(y_test_s, y_prob_s, sources_s)
    n_bgc = len(set(s for s, l in zip(sources_s, y_test_s) if l == 1))
    n_neg = len(set(s for s, l in zip(sources_s, y_test_s) if l == 0))
    print(f"  RF: AUROC={m['auroc']:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"F1={m['f1']:.3f} P={m['prec']:.3f} R={m['recall']:.3f}")
    print(f"  独立单元: {n_bgc} BGC + {n_neg} genome")

    # 3. 跨属: 用全量 pool
    print("\n[3] 跨属评估 (全量pool)...")
    results = {}

    for genus in GENERA:
        X_pool, y_pool, sources_pool = pool_genus(genus)
        n_bgc = len(set(s for s, l in zip(sources_pool, y_pool) if l == 1))
        n_neg = len(set(s for s, l in zip(sources_pool, y_pool) if l == 0))

        print(f"\n  === {genus} ===")
        print(f"  全量: {len(y_pool)} windows, 独立: {n_bgc} BGC + {n_neg} genome")

        for clf_name, clf in [('RF', clf_rf), ('LR', clf_lr)]:
            y_prob = clf.predict_proba(X_pool)[:, 1]
            m = evaluate(y_pool, y_prob)
            lo, hi, n_valid = cluster_bootstrap(y_pool, y_prob, sources_pool)

            print(f"    {clf_name}: AUROC={m['auroc']:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"F1={m['f1']:.3f} P={m['prec']:.3f} R={m['recall']:.3f}  "
                  f"(boot={n_valid}/5000)")

            results[f"strep2{genus}_{clf_name}"] = {
                'auroc': m['auroc'], 'ci_lo': lo, 'ci_hi': hi,
                'f1': m['f1'], 'prec': m['prec'], 'recall': m['recall'],
                'n_windows': len(y_pool), 'n_bgc': n_bgc, 'n_neg_genome': n_neg,
            }

    # 4. 汇总表
    print("\n" + "=" * 65)
    print("汇总 (Strep训练 → 各属全量测试, RF)")
    print("=" * 65)
    print(f"  {'属':<18} {'AUROC':>7} {'[95% CI]':>16} {'F1':>6} {'BGCs':>5} {'neg':>5}")
    print(f"  {'-'*60}")

    for genus in GENERA:
        r = results.get(f"strep2{genus}_RF", {})
        print(f"  {genus:<18} {r['auroc']:>7.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
              f" {r['f1']:>6.3f} {r['n_bgc']:>5} {r['n_neg_genome']:>5}")

    # 保存
    out_path = os.path.join(MULTIGENUS, "fullpool_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存到: {out_path}")


if __name__ == '__main__':
    main()
