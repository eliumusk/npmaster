"""
扩充版评估: 对比新旧Strep数据集 + 跨属全量评估 + 窗口消融
- 新: 105个负样本基因组 (expanded_strep/w8192)
- 旧: 25个负样本基因组 (data/clean_strep -> data/)
- 跨属: 用新的Strep模型测试Micro/Amyco/Pseudo

用法: python3 -u 16_expanded_eval.py
"""

import pickle, numpy as np, os, json, time
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

BASE = "/data/muskliu/np_master/data"
EXPANDED = BASE + "/expanded_strep"
MULTIGENUS = BASE + "/multigenus"
SEED = 42
np.random.seed(SEED)


def load_emb(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['embeddings'], d['labels']


def load_raw(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def cluster_bootstrap(y_true, y_prob, sources, n_boot=5000):
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
        yt, yp = y_true[boot_idx], y_prob[boot_idx]
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


def fmt(m, lo=None, hi=None):
    ci = f" [{lo:.3f},{hi:.3f}]" if lo is not None else ""
    return f"AUROC={m['auroc']:.3f}{ci} F1={m['f1']:.3f} P={m['prec']:.3f} R={m['recall']:.3f}"


def main():
    results = {}

    # ============================================================
    # 实验A: 新旧Strep对比 (同属)
    # ============================================================
    print("=" * 70)
    print("实验A: 新旧Strep训练集对比 (8192bp)")
    print("=" * 70)

    configs = {
        "旧(25 genome)": {
            "train_emb": BASE + "/train_emb.pkl",
            "test_emb": BASE + "/test_emb.pkl",
            "test_raw": BASE + "/test.pkl",
        },
        "新(105 genome)": {
            "train_emb": EXPANDED + "/w8192/train_emb.pkl",
            "test_emb": EXPANDED + "/w8192/test_emb.pkl",
            "test_raw": EXPANDED + "/w8192/test.pkl",
        },
    }

    trained_models = {}
    for label, paths in configs.items():
        print(f"\n  --- {label} ---")
        X_train, y_train = load_emb(paths["train_emb"])
        X_test, y_test = load_emb(paths["test_emb"])
        samples = load_raw(paths["test_raw"])
        sources = [s['source'] for s in samples]

        n_train_bgc = len(set(s['source'] for s in load_raw(paths["train_emb"].replace("_emb.pkl", ".pkl").replace("train_emb", "train")) if s['is_bgc'] == 1)) if "expanded" in paths["train_emb"] else "?"
        n_test_bgc = len(set(s for s, l in zip(sources, y_test) if l == 1))
        n_test_neg = len(set(s for s, l in zip(sources, y_test) if l == 0))

        for clf_name, clf in [
            ('RF', RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)),
            ('LR', LogisticRegression(max_iter=1000, random_state=SEED)),
        ]:
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            m = evaluate(y_test, y_prob)
            lo, hi, _ = cluster_bootstrap(y_test, y_prob, sources)
            print(f"    {clf_name}: {fmt(m, lo, hi)}  (test: {n_test_bgc} BGC + {n_test_neg} genome)")
            key = f"strep_{label}_{clf_name}"
            results[key] = {**m, 'ci_lo': lo, 'ci_hi': hi, 'n_test_bgc': n_test_bgc, 'n_test_neg': n_test_neg}

            if clf_name == 'RF' and '新' in label:
                trained_models['new_rf'] = clf
            if clf_name == 'LR' and '新' in label:
                trained_models['new_lr'] = clf

    # ============================================================
    # 实验B: 新Strep模型 → 跨属全量测试
    # ============================================================
    print("\n" + "=" * 70)
    print("实验B: 新Strep模型(105 genome) → 跨属全量测试")
    print("=" * 70)

    for genus in ["micromonospora", "amycolatopsis", "pseudomonas"]:
        # 合并该属所有split
        all_emb, all_labels, all_sources = [], [], []
        for split in ['train', 'val', 'test']:
            X, y = load_emb(f"{MULTIGENUS}/{genus}/{split}_emb.pkl")
            samples = load_raw(f"{MULTIGENUS}/{genus}/{split}.pkl")
            all_emb.append(X)
            all_labels.append(y)
            for s in samples:
                all_sources.append(s['source'])

        X_pool = np.concatenate(all_emb)
        y_pool = np.concatenate(all_labels)
        n_bgc = len(set(s for s, l in zip(all_sources, y_pool) if l == 1))
        n_neg = len(set(s for s, l in zip(all_sources, y_pool) if l == 0))

        print(f"\n  === {genus} ({n_bgc} BGC, {n_neg} genome) ===")

        for clf_name, clf in [('RF', trained_models['new_rf']), ('LR', trained_models['new_lr'])]:
            y_prob = clf.predict_proba(X_pool)[:, 1]
            m = evaluate(y_pool, y_prob)
            lo, hi, _ = cluster_bootstrap(y_pool, y_prob, all_sources)
            print(f"    {clf_name}: {fmt(m, lo, hi)}")
            results[f"new_strep2{genus}_{clf_name}"] = {**m, 'ci_lo': lo, 'ci_hi': hi, 'n_bgc': n_bgc, 'n_neg': n_neg}

    # ============================================================
    # 实验C: 窗口消融 (如果4k和16k的embedding存在)
    # ============================================================
    print("\n" + "=" * 70)
    print("实验C: 窗口大小消融 (Strep同属, RF)")
    print("=" * 70)

    for ws in [4096, 8192, 16384]:
        train_path = f"{EXPANDED}/w{ws}/train_emb.pkl"
        test_path = f"{EXPANDED}/w{ws}/test_emb.pkl"
        test_raw_path = f"{EXPANDED}/w{ws}/test.pkl"

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"\n  w{ws}: 缺少embedding文件, 跳过")
            continue

        X_train, y_train = load_emb(train_path)
        X_test, y_test = load_emb(test_path)
        samples = load_raw(test_raw_path)
        sources = [s['source'] for s in samples]

        n_bgc = len(set(s for s, l in zip(sources, y_test) if l == 1))
        n_neg = len(set(s for s, l in zip(sources, y_test) if l == 0))

        clf = RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, y_prob)
        lo, hi, _ = cluster_bootstrap(y_test, y_prob, sources)
        print(f"\n  w{ws}: {fmt(m, lo, hi)}  ({n_bgc} BGC, {n_neg} genome, {len(y_test)} windows)")
        results[f"ablation_w{ws}_RF"] = {**m, 'ci_lo': lo, 'ci_hi': hi, 'window_size': ws}

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 70)
    print("汇总表")
    print("=" * 70)

    print("\n  A. Strep同属 (新 vs 旧):")
    for label in ["旧(25 genome)", "新(105 genome)"]:
        for clf in ["RF", "LR"]:
            r = results.get(f"strep_{label}_{clf}", {})
            if r:
                print(f"    {label} {clf}: AUROC={r['auroc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] F1={r['f1']:.3f}")

    print("\n  B. 新Strep→跨属 (RF):")
    for genus in ["micromonospora", "amycolatopsis", "pseudomonas"]:
        r = results.get(f"new_strep2{genus}_RF", {})
        if r:
            print(f"    {genus}: AUROC={r['auroc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] F1={r['f1']:.3f}")

    print("\n  C. 窗口消融 (RF):")
    for ws in [4096, 8192, 16384]:
        r = results.get(f"ablation_w{ws}_RF", {})
        if r:
            print(f"    w{ws}: AUROC={r['auroc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")

    # 保存
    out_path = EXPANDED + "/expanded_eval_results.json"
    # Convert numpy types for JSON
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()}
    with open(out_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n结果保存到: {out_path}")


if __name__ == '__main__':
    main()
