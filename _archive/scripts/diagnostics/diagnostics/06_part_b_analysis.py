"""
Part B analysis:
- Compare NLL statistics between BGC vs non-BGC windows
- Check if NLL signal is orthogonal to embedding signal (correlation)
- Visualize distributions
"""
import pickle, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NLL_DIR = "/data/muskliu/np_master/data/diagnostic_day1/nll"
EMB_DIR = "/data/muskliu/np_master/data/diagnostic_day1/embeddings"
FIG_DIR = "/data/muskliu/np_master/data/diagnostic_day1/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load NLL results
files = sorted(glob.glob(f"{NLL_DIR}/nll_shard_*.pkl"))
all_results = []
for f in files:
    d = pickle.load(open(f,'rb'))
    all_results.extend(d['results'])
print(f"Loaded {len(all_results)} window NLL stats")

# Features per window
FEATURES = ['nll_mean','nll_std','nll_median','nll_q25','nll_q75','autocorr_100','autocorr_1000','diff_var']
is_bgc = np.array([r['is_bgc']==1 for r in all_results])
X = np.array([[r[f] for f in FEATURES] for r in all_results])
print(f"BGC windows: {is_bgc.sum()}, non-BGC: {(~is_bgc).sum()}")

# Per-feature distribution comparison
from scipy import stats
print("\n=== Per-feature BGC vs non-BGC (scipy ttest_ind) ===")
print(f"{'Feature':<18} {'BGC mean':<12} {'bg mean':<12} {'effect_size':<12} {'p-value':<12}")
feat_stats = {}
for i, f in enumerate(FEATURES):
    bgc_v = X[is_bgc, i]; bg_v = X[~is_bgc, i]
    if np.std(np.concatenate([bgc_v, bg_v])) > 0:
        d = (bgc_v.mean() - bg_v.mean()) / np.std(np.concatenate([bgc_v, bg_v]))
    else:
        d = 0.0
    t, p = stats.ttest_ind(bgc_v, bg_v)
    print(f"{f:<18} {bgc_v.mean():<12.4f} {bg_v.mean():<12.4f} {d:<+12.3f} {p:<12.2e}")
    feat_stats[f] = {'bgc_mean':float(bgc_v.mean()),'bg_mean':float(bg_v.mean()),
                     'effect_size':float(d),'pvalue':float(p)}

# Histograms: 2 key features
fig, axes = plt.subplots(2, 4, figsize=(16,8))
for i, f in enumerate(FEATURES):
    ax = axes[i//4, i%4]
    bgc_v = X[is_bgc, i]; bg_v = X[~is_bgc, i]
    bins = np.linspace(min(bgc_v.min(), bg_v.min()), max(bgc_v.max(), bg_v.max()), 40)
    ax.hist(bg_v, bins=bins, alpha=0.5, label=f'non-BGC μ={bg_v.mean():.3f}', color='gray', density=True)
    ax.hist(bgc_v, bins=bins, alpha=0.5, label=f'BGC μ={bgc_v.mean():.3f}', color='red', density=True)
    ax.set_title(f'{f} (d={feat_stats[f]["effect_size"]:+.2f})', fontsize=10)
    ax.legend(fontsize=8)
plt.suptitle('Part B: NLL feature distributions BGC vs non-BGC (Strep genome scan)')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/nll_features_distribution.png", dpi=140)
plt.close()
print(f"Saved nll_features_distribution.png")

# Simple classifier: Logistic Regression on NLL features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(X)
lr = LogisticRegression(max_iter=1000)
scores = cross_val_score(lr, Xs, is_bgc.astype(int), cv=5, scoring='roc_auc')
print(f"\n=== Logistic regression on NLL features only (5-fold CV) ===")
print(f"AUROC: {scores.mean():.3f} ± {scores.std():.3f}")
nll_auroc = float(scores.mean())

# Now: Orthogonality with embedding
# Need to match NLL windows to embeddings somehow
# NLL windows come from genome_scan (S_aquilus etc), embeddings come from MIBiG+bg diagnostic
# They don't share samples directly, so we can't do direct correlation
# Instead: test if genome-scan embeddings can ALSO discriminate BGC vs non-BGC

# Load genome scan windows, run inference? No — we don't have their embeddings
# Alternative: use the NLL-feature AUROC and compare vs what an embedding classifier would give on same task
# Since we don't have embeddings for these specific windows, skip orthogonality for now

# But we can check: are any NLL features strongly individually predictive?
print(f"\n=== Single-feature discrimination ===")
print(f"{'Feature':<18} {'|effect|':<10}")
for f, d in sorted(feat_stats.items(), key=lambda x: -abs(x[1]['effect_size'])):
    print(f"{f:<18} {abs(d['effect_size']):<10.3f}")

summary = {'n_bgc':int(is_bgc.sum()), 'n_bg':int((~is_bgc).sum()),
           'features':feat_stats, 'auroc_nll_only':nll_auroc}
json.dump(summary, open(f"{FIG_DIR}/../part_b_summary.json",'w'), indent=2)
print("\nSaved part_b_summary.json")

# Decision
print(f"\n=== Decision ===")
print(f"NLL features alone: AUROC={nll_auroc:.3f}")
if nll_auroc > 0.7:
    print(f"NLL signal is STRONG → keep as auxiliary channel for GCD")
elif nll_auroc > 0.55:
    print(f"NLL signal is MODERATE → marginal value, could be used as complement")
else:
    print(f"NLL signal is WEAK → embedding is enough, skip NLL")
