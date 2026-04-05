"""
来源偏差诊断：量化MIBiG正样本 vs 基因组原位BGC vs 负样本的得分分布差异

Phase 1: MIBiG正样本 vs 负样本 (不需要GPU)
Phase 2: 加入基因组原位BGC对比 (需要GPU提取embedding后再跑)

用法:
  python3 -u 21_source_bias_diagnosis.py              # Phase 1
  python3 -u 21_source_bias_diagnosis.py --with-insitu # Phase 2 (需要insitu embedding)
"""

import pickle, numpy as np, os, json, argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "/data/muskliu/np_master/data/expanded_strep/w8192"
INSITU_EMB = "/data/muskliu/np_master/data/expanded_strep/insitu_bgc/insitu_bgc_emb.pkl"
OUT_DIR = "/data/muskliu/np_master/results/figures"
SEED = 42
np.random.seed(SEED)


def load_emb(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def score_stats(name, scores):
    return {
        'name': name,
        'n': len(scores),
        'mean': float(np.mean(scores)),
        'median': float(np.median(scores)),
        'q25': float(np.percentile(scores, 25)),
        'q75': float(np.percentile(scores, 75)),
        'std': float(np.std(scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-insitu', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("来源偏差���断：得分分布对比")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载数据...")
    train_emb = load_emb(f"{BASE}/train_emb.pkl")
    test_emb = load_emb(f"{BASE}/test_emb.pkl")

    X_train, y_train = train_emb['embeddings'], train_emb['labels']
    X_test, y_test = test_emb['embeddings'], test_emb['labels']

    print(f"  Train: {len(y_train)} ({int(y_train.sum())} pos, {int((1-y_train).sum())} neg)")
    print(f"  Test:  {len(y_test)} ({int(y_test.sum())} pos, {int((1-y_test).sum())} neg)")

    # 训练RF
    print("\n[2] 训练RF...")
    clf = RandomForestClassifier(100, max_depth=20, random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"  Test AUROC: {auroc:.4f}")

    # 对所有样本打分（合并train+test的MIBiG正样本）
    print("\n[3] 打分...")
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    scores_all = clf.predict_proba(X_all)[:, 1]

    scores_mibig = scores_all[y_all == 1]
    scores_neg_all = scores_all[y_all == 0]

    # 负样本随机取1000个
    rng = np.random.RandomState(SEED)
    neg_idx = rng.choice(len(scores_neg_all), min(1000, len(scores_neg_all)), replace=False)
    scores_neg = scores_neg_all[neg_idx]

    groups = [
        ("MIBiG Positives", scores_mibig, '#2196F3'),
        ("Negatives (1000)", scores_neg, '#757575'),
    ]

    # Phase 2: 加载in-situ BGC
    if args.with_insitu:
        if os.path.exists(INSITU_EMB):
            print("  加载基因组原位BGC embedding...")
            insitu = load_emb(INSITU_EMB)
            X_insitu = insitu['embeddings']
            scores_insitu = clf.predict_proba(X_insitu)[:, 1]
            groups.insert(1, ("In-situ BGC", scores_insitu, '#FF9800'))
            print(f"  基因组原位BGC: {len(scores_insitu)} windows")
        else:
            print(f"  ⚠️ {INSITU_EMB} 不存在，跳过in-situ")

    # 统计
    print(f"\n[4] 得分分布统计:")
    print(f"\n{'Group':20s} | {'N':>6s} | {'Mean':>6s} | {'Median':>6s} | {'Q25':>6s} | {'Q75':>6s} | {'Std':>6s}")
    print("-" * 78)

    all_stats = []
    for name, sc, _ in groups:
        s = score_stats(name, sc)
        all_stats.append(s)
        print(f"{name:20s} | {s['n']:>6d} | {s['mean']:>6.3f} | {s['median']:>6.3f} | {s['q25']:>6.3f} | {s['q75']:>6.3f} | {s['std']:>6.3f}")

    # 额外统计
    fn_rate = (scores_mibig < 0.5).mean()
    fp_rate = (scores_neg > 0.5).mean()
    print(f"\n  MIBiG positives with score<0.5: {fn_rate:.3f} ({int((scores_mibig<0.5).sum())}/{len(scores_mibig)})")
    print(f"  Negatives with score>0.5: {fp_rate:.3f} ({int((scores_neg>0.5).sum())}/{len(scores_neg)})")

    # t-test: MIBiG vs in-situ (if available)
    if len(groups) == 3:
        scores_insitu = groups[1][1]
        t_stat, p_val = stats.ttest_ind(scores_mibig, scores_insitu)
        mean_diff = np.mean(scores_mibig) - np.mean(scores_insitu)
        print(f"\n  MIBiG vs 原位BGC: 均值差={mean_diff:.4f}, t={t_stat:.2f}, p={p_val:.2e}")
        if abs(mean_diff) < 0.05 and p_val > 0.05:
            print("  → 来源偏差不严重，现有结果可信")
        elif abs(mean_diff) > 0.1 and p_val < 0.01:
            print("  → 来源偏差是真实问题，需要重新讨论数据构建方式")
        else:
            print("  → 存在一定偏差，需要在论文中讨论")

    # 画图
    print(f"\n[5] 画图...")
    os.makedirs(OUT_DIR, exist_ok=True)

    n_groups = len(groups)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Violin plot
    data_for_violin = [sc for _, sc, _ in groups]
    colors = [c for _, _, c in groups]
    labels = [n for n, _, _ in groups]

    parts = axes[0].violinplot(data_for_violin, positions=range(1, n_groups + 1),
                                showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('black')
    axes[0].set_xticks(range(1, n_groups + 1))
    axes[0].set_xticklabels([l.replace('(', '\n(') for l in labels], fontsize=9)
    axes[0].set_ylabel('RF Score (BGC probability)')
    axes[0].set_title('Score Distribution by Sample Source')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='threshold=0.5')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-0.05, 1.05)

    # Histogram
    for name, sc, color in groups:
        axes[1].hist(sc, bins=50, alpha=0.6, color=color, label=f'{name} (n={len(sc)})', density=True)
    axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('RF Score (BGC probability)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Score Histogram')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    suffix = "_with_insitu" if len(groups) == 3 else "_phase1"
    fig_path = os.path.join(OUT_DIR, f"source_bias_scores{suffix}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  保存到: {fig_path}")

    # 保���结果JSON
    result = {
        'test_auroc': float(auroc),
        'stats': all_stats,
        'mibig_fn_rate': float(fn_rate),
        'neg_fp_rate': float(fp_rate),
        'phase': 2 if len(groups) == 3 else 1,
    }
    json_path = os.path.join(OUT_DIR, f"source_bias_results{suffix}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if len(groups) == 2:
        print(f"\n{'='*70}")
        print("Phase 1 完成。待GPU空闲后，提取in-situ BGC embedding，��跑:")
        print("  python3 -u 21_source_bias_diagnosis.py --with-insitu")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
