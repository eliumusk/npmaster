"""
Leave-one-class-out 实验：诊断模型是记忆已知BGC类型还是学到通用BGC特征

对MIBiG的7个BGC类型(Polyketide, NRP, RiPP, Terpene, Saccharide, Alkaloid, Other)：
  1. 从训练集中去掉该类型的所有BGC
  2. 在剩余数据上训练RF分类器
  3. 在测试集中单独统计被去掉那个类型的recall和AUROC

用法: python3 -u 18_leave_one_class_out.py
"""

import pickle, numpy as np, os, json
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

BASE = "/data/muskliu/np_master/data/expanded_strep"
WINDOW = "w8192"
SEED = 42
CATEGORY_NAMES = ['Polyketide', 'NRP', 'RiPP', 'Terpene', 'Saccharide', 'Alkaloid', 'Other']

np.random.seed(SEED)


def load_emb(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d


def load_raw(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    data_dir = os.path.join(BASE, WINDOW)

    print("=" * 70)
    print("Leave-one-class-out 实验")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载数据...")
    train_emb = load_emb(os.path.join(data_dir, "train_emb.pkl"))
    test_emb = load_emb(os.path.join(data_dir, "test_emb.pkl"))
    train_raw = load_raw(os.path.join(data_dir, "train.pkl"))
    test_raw = load_raw(os.path.join(data_dir, "test.pkl"))

    X_train_all = train_emb['embeddings']
    y_train_all = train_emb['labels']
    cat_train_all = train_emb['categories']  # (N, 7) binary vectors

    X_test = test_emb['embeddings']
    y_test = test_emb['labels']
    cat_test = test_emb['categories']  # (N, 7) binary vectors

    print(f"  Train: {len(y_train_all)} samples (pos={y_train_all.sum()}, neg={len(y_train_all)-y_train_all.sum()})")
    print(f"  Test:  {len(y_test)} samples (pos={y_test.sum()}, neg={len(y_test)-y_test.sum()})")

    # 验证: embedding和raw data的对齐（可能有错误样本被跳过）
    # embedding文件的ids应该和raw文件的顺序一致（跳过error的）
    n_train_emb = len(y_train_all)
    n_test_emb = len(y_test)
    print(f"  Train raw: {len(train_raw)}, emb: {n_train_emb}")
    print(f"  Test raw:  {len(test_raw)}, emb: {n_test_emb}")

    # 统计测试集中每个类型的样本数
    print("\n[2] 测试集BGC类型分布:")
    test_pos_mask = y_test == 1
    for ci, cname in enumerate(CATEGORY_NAMES):
        # 属于该类型的正样本（多标签：一个样本可能属于多个类型）
        type_mask = (cat_test[:, ci] == 1) & test_pos_mask
        n_windows = type_mask.sum()
        # 统计独立BGC数
        test_ids = test_emb.get('ids', [])
        if test_ids:
            unique_bgcs = set()
            for idx in np.where(type_mask)[0]:
                bgc_id = test_ids[idx].rsplit('_w', 1)[0] if '_w' in test_ids[idx] else test_ids[idx]
                unique_bgcs.add(bgc_id)
            print(f"  {cname:12s}: {n_windows:4d} windows, {len(unique_bgcs):3d} BGCs")
        else:
            print(f"  {cname:12s}: {n_windows:4d} windows")

    # Baseline: 全量训练
    print("\n[3] Baseline (全量训练)...")
    clf_base = RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)
    clf_base.fit(X_train_all, y_train_all)
    y_prob_base = clf_base.predict_proba(X_test)[:, 1]
    y_pred_base = (y_prob_base >= 0.5).astype(int)

    auroc_base = roc_auc_score(y_test, y_prob_base)
    recall_base = recall_score(y_test, y_pred_base)
    print(f"  Baseline AUROC={auroc_base:.4f}, Recall={recall_base:.4f}")

    # Leave-one-class-out
    print("\n[4] Leave-one-class-out 实验:")
    print(f"\n{'去掉的类型':12s} | {'训练正样本':>8s} | {'去掉数量':>8s} | {'整体AUROC':>10s} | {'该类型Recall':>12s} | {'该类型AUROC':>12s} | {'该类型N':>6s}")
    print("-" * 95)

    results = []

    for ci, cname in enumerate(CATEGORY_NAMES):
        # 找到训练集中属于该类型的正样本
        train_type_mask = (cat_train_all[:, ci] == 1) & (y_train_all == 1)
        n_removed = train_type_mask.sum()

        if n_removed == 0:
            print(f"{cname:12s} | {'N/A':>8s} | {0:>8d} | {'N/A':>10s} | {'N/A':>12s} | {'N/A':>12s} | {0:>6d}")
            continue

        # 去掉该类型的训练样本（保留所有负样本）
        keep_mask = ~train_type_mask
        X_train_sub = X_train_all[keep_mask]
        y_train_sub = y_train_all[keep_mask]
        n_pos_remaining = y_train_sub.sum()

        # 训练
        clf = RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)
        clf.fit(X_train_sub, y_train_sub)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # 整体指标
        overall_auroc = roc_auc_score(y_test, y_prob)

        # 该类型在测试集中的指标
        test_type_mask = (cat_test[:, ci] == 1) & test_pos_mask
        n_test_type = test_type_mask.sum()

        if n_test_type > 0:
            # Recall: 该类型正样本中被正确预测为正的比例
            type_recall = y_pred[test_type_mask].mean()

            # AUROC: 该类型正样本 vs 所有负样本
            test_neg_mask = y_test == 0
            type_eval_mask = test_type_mask | test_neg_mask
            if len(np.unique(y_test[type_eval_mask])) == 2:
                type_auroc = roc_auc_score(y_test[type_eval_mask], y_prob[type_eval_mask])
                type_auroc_str = f"{type_auroc:.4f}"
            else:
                type_auroc = None
                type_auroc_str = "N/A"
        else:
            type_recall = None
            type_auroc = None
            type_auroc_str = "N/A"

        recall_str = f"{type_recall:.4f}" if type_recall is not None else "N/A"
        print(f"{cname:12s} | {int(n_pos_remaining):>8d} | {int(n_removed):>8d} | {overall_auroc:>10.4f} | {recall_str:>12s} | {type_auroc_str:>12s} | {n_test_type:>6d}")

        results.append({
            'removed_class': cname,
            'n_removed_windows': int(n_removed),
            'n_train_pos_remaining': int(n_pos_remaining),
            'overall_auroc': float(overall_auroc),
            'type_recall': float(type_recall) if type_recall is not None else None,
            'type_auroc': float(type_auroc) if type_auroc is not None else None,
            'n_test_type_windows': int(n_test_type),
        })

    # Baseline行
    print(f"{'(Baseline)':12s} | {int(y_train_all.sum()):>8d} | {0:>8d} | {auroc_base:>10.4f} | {recall_base:>12.4f} | {'---':>12s} | {int(test_pos_mask.sum()):>6d}")

    # 分析
    print("\n" + "=" * 70)
    print("分析:")
    print("=" * 70)

    for r in results:
        cname = r['removed_class']
        if r['type_recall'] is not None:
            if r['type_recall'] < 0.3:
                verdict = "⚠️  严重依赖训练集 — 模型完全无法泛化到未见过的该类型"
            elif r['type_recall'] < 0.6:
                verdict = "△  部分依赖 — 模型能识别一些该类型BGC但有明显下降"
            elif r['type_recall'] < 0.8:
                verdict = "○  轻微下降 — 模型对该类型有一定泛化能力"
            else:
                verdict = "✓  强泛化 — 即使没见过该类型，模型仍能识别"
            print(f"  {cname:12s} (recall={r['type_recall']:.3f}): {verdict}")

    # 保存结果
    out_path = os.path.join(data_dir, "leave_one_class_out_results.json")
    output = {
        'baseline_auroc': float(auroc_base),
        'baseline_recall': float(recall_base),
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存到: {out_path}")


if __name__ == '__main__':
    main()
