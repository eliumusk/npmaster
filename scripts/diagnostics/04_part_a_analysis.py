"""
Day-1 Part A analysis: UMAP + distance distributions + kNN leave-one-out.
Run after all 8 shards of 03_combine_and_extract.py are done.
"""
import os, pickle, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
import umap

ROOT = "/data/muskliu/np_master"
EMB_DIR = f"{ROOT}/data/diagnostic_day1/embeddings"
FIG_DIR = f"{ROOT}/data/diagnostic_day1/figures"
os.makedirs(FIG_DIR, exist_ok=True)

CLASSES = ["Alkaloid", "NRP", "Polyketide", "RiPP", "Saccharide", "Terpene", "Other"]
CLASS_COLORS = {
    "Alkaloid":"#e41a1c","NRP":"#377eb8","Polyketide":"#4daf4a",
    "RiPP":"#984ea3","Saccharide":"#ff7f00","Terpene":"#ffff33","Other":"#a65628",
}
BG_COLOR = "#999999"

def load_all_shards():
    files = sorted(glob.glob(f"{EMB_DIR}/shard_*.pkl"))
    embs, metas = [], []
    for f in files:
        d = pickle.load(open(f, 'rb'))
        embs.append(d['embeddings'])
        metas.extend(d['metas'])
    E = np.concatenate(embs, axis=0)
    print(f"Loaded {len(files)} shards: {E.shape}, {len(metas)} metas")
    return E, metas

def primary_type(categories):
    """Assign single primary type (first match in CLASSES order, else 'Multi' if 2+)."""
    if not categories:
        return 'bg'
    if len(categories) >= 2:
        return 'Multi'
    return categories[0]

def run_umap(E, metas):
    print("Running UMAP (this takes ~1 min) ...", flush=True)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42)
    Z = reducer.fit_transform(E)
    return Z

def plot_umap_by_class(Z, metas, fname):
    fig, ax = plt.subplots(figsize=(10,8))
    # Plot bg first (bottom)
    bg_mask = np.array([m['is_bgc']==0 for m in metas])
    ax.scatter(Z[bg_mask,0], Z[bg_mask,1], c=BG_COLOR, s=4, alpha=0.3, label=f'bg (n={bg_mask.sum()})')
    # Plot Multi-label first (as one color)
    multi_mask = np.array([m['is_bgc']==1 and len(m['categories'])>=2 for m in metas])
    if multi_mask.any():
        ax.scatter(Z[multi_mask,0], Z[multi_mask,1], c='black', s=6, alpha=0.7,
                   label=f'Multi (n={multi_mask.sum()})', marker='x')
    # Single-label classes
    for cls in CLASSES:
        mask = np.array([m['is_bgc']==1 and m['categories']==[cls] for m in metas])
        if mask.sum() > 0:
            ax.scatter(Z[mask,0], Z[mask,1], c=CLASS_COLORS[cls], s=8, alpha=0.7,
                       label=f'{cls} (n={mask.sum()})')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_title('Evo2 blocks.20.mlp.l3 mean-pool: BGC types vs background')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(fname, dpi=140); plt.close()
    print(f"Saved {fname}")

def plot_umap_by_length(Z, metas, fname):
    """Color by sequence length (continuous)."""
    fig, ax = plt.subplots(figsize=(10,8))
    lens = np.array([m.get('used_length', m.get('full_length', 16384)) for m in metas])
    is_bgc = np.array([m['is_bgc']==1 for m in metas])
    # BGCs colored by length, bg in gray
    ax.scatter(Z[~is_bgc,0], Z[~is_bgc,1], c=BG_COLOR, s=4, alpha=0.3, label='bg')
    sc = ax.scatter(Z[is_bgc,0], Z[is_bgc,1], c=lens[is_bgc], s=8, alpha=0.8,
                    cmap='viridis', vmin=200, vmax=16384)
    cbar = plt.colorbar(sc, ax=ax, label='used length (bp)')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP colored by BGC used length (bg gray)')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=140); plt.close()
    print(f"Saved {fname}")

def plot_umap_by_phylum(Z, metas, fname):
    """Color by background phylum + BGC as one color."""
    fig, ax = plt.subplots(figsize=(10,8))
    is_bgc = np.array([m['is_bgc']==1 for m in metas])
    ax.scatter(Z[is_bgc,0], Z[is_bgc,1], c='red', s=4, alpha=0.4, label='BGC (all)')
    phyla = set()
    for m in metas:
        if m['is_bgc']==0:
            phyla.add(m.get('phylum','unknown'))
    phyla_colors = {'Actinobacteria':'#1f77b4','Proteobacteria':'#2ca02c','Cyanobacteria':'#9467bd','Firmicutes':'#8c564b'}
    for ph in phyla:
        mask = np.array([m['is_bgc']==0 and m.get('phylum','')==ph for m in metas])
        c = phyla_colors.get(ph,'gray')
        ax.scatter(Z[mask,0], Z[mask,1], c=c, s=5, alpha=0.5, label=f'bg {ph} (n={mask.sum()})')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP: BGC vs background-by-phylum')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=140); plt.close()
    print(f"Saved {fname}")

def distance_analysis(E, metas, fname):
    """Compute intra-class, inter-class, and BGC-vs-bg cosine distance distributions."""
    is_bgc = np.array([m['is_bgc']==1 for m in metas])
    primary = np.array([primary_type(m['categories']) for m in metas])
    # Sample max 500 per group to make computation feasible
    np.random.seed(42)
    # Focus on single-label BGCs only for clean type analysis
    single_label = np.array([m['is_bgc']==1 and len(m['categories'])==1 for m in metas])
    bgc_idx = np.where(single_label)[0]
    bg_idx = np.where(~is_bgc)[0]
    # subsample
    n_each = 300
    bgc_sub = np.random.choice(bgc_idx, min(n_each, len(bgc_idx)), replace=False)
    bg_sub = np.random.choice(bg_idx, min(n_each, len(bg_idx)), replace=False)
    # distances
    print("Computing cosine distances ...", flush=True)
    d_bgc_bgc = cosine_distances(E[bgc_sub], E[bgc_sub])
    d_bgc_bg = cosine_distances(E[bgc_sub], E[bg_sub])
    d_bg_bg = cosine_distances(E[bg_sub], E[bg_sub])
    # get upper triangle without diagonal
    def upper(D):
        return D[np.triu_indices_from(D, k=1)]
    same_type = []
    diff_type = []
    for i in range(len(bgc_sub)):
        for j in range(i+1, len(bgc_sub)):
            if primary[bgc_sub[i]] == primary[bgc_sub[j]]:
                same_type.append(d_bgc_bgc[i,j])
            else:
                diff_type.append(d_bgc_bgc[i,j])
    same_type = np.array(same_type); diff_type = np.array(diff_type)
    bgc_bg = d_bgc_bg.flatten()
    bg_bg = upper(d_bg_bg)
    # plot
    fig, ax = plt.subplots(figsize=(10,6))
    bins = np.linspace(0, max(bgc_bg.max(), bg_bg.max(), diff_type.max()), 60)
    ax.hist(same_type, bins=bins, alpha=0.5, label=f'same-type BGC-BGC (n={len(same_type)}, μ={same_type.mean():.3f})', color='green')
    ax.hist(diff_type, bins=bins, alpha=0.5, label=f'diff-type BGC-BGC (n={len(diff_type)}, μ={diff_type.mean():.3f})', color='orange')
    ax.hist(bgc_bg, bins=bins, alpha=0.5, label=f'BGC-bg (n={len(bgc_bg)}, μ={bgc_bg.mean():.3f})', color='red')
    ax.hist(bg_bg, bins=bins, alpha=0.5, label=f'bg-bg (n={len(bg_bg)}, μ={bg_bg.mean():.3f})', color='gray')
    ax.set_xlabel('cosine distance'); ax.set_ylabel('count')
    ax.set_title('Distance distribution: same-type / diff-type / BGC-bg / bg-bg')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=140); plt.close()
    print(f"Saved {fname}")
    return dict(same_type=float(same_type.mean()), diff_type=float(diff_type.mean()),
                bgc_bg=float(bgc_bg.mean()), bg_bg=float(bg_bg.mean()))

def knn_leave_one_out(E, metas):
    """For each BGC, find 5 nearest neighbors; report hit rate (neighbors that are BGC)."""
    is_bgc = np.array([m['is_bgc']==1 for m in metas])
    # cosine kNN
    print("Fitting kNN ...", flush=True)
    nn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(E)
    dists, idx = nn.kneighbors(E)
    # idx[:,0] is self, use [1:6]
    hit_rates = []
    for i in range(len(metas)):
        if not is_bgc[i]:
            continue
        neighbors = idx[i, 1:6]
        hits = sum(1 for j in neighbors if is_bgc[j])
        hit_rates.append(hits / 5.0)
    hit = np.array(hit_rates)
    prior = is_bgc.mean()
    print(f"\nkNN leave-one-out (for BGC samples):")
    print(f"  samples: {len(hit)} BGCs")
    print(f"  prior P(neighbor is BGC): {prior:.3f}")
    print(f"  mean hit rate: {hit.mean():.3f}")
    print(f"  enrichment: {hit.mean()/prior:.2f}x")
    print(f"  100% hit (all 5 are BGC): {(hit==1.0).sum()}/{len(hit)}")
    print(f"  0% hit (no BGC neighbors): {(hit==0.0).sum()}/{len(hit)}")
    return {'mean_hit': float(hit.mean()), 'prior': float(prior),
            'enrichment': float(hit.mean()/prior) if prior > 0 else 0,
            'n_samples': len(hit)}

def per_type_analysis(E, metas):
    """For each of the 7 BGC types: mean distance to same-type / to other-type / to bg."""
    is_bgc = np.array([m['is_bgc']==1 for m in metas])
    primary = np.array([primary_type(m['categories']) for m in metas])
    bg_idx = np.where(~is_bgc)[0]
    np.random.seed(42)
    bg_sub = np.random.choice(bg_idx, min(300, len(bg_idx)), replace=False)
    E_bg = E[bg_sub]
    print(f"\nPer-type distance analysis (vs {len(E_bg)} bg samples):")
    print(f"{'Type':<12} {'N':<5} {'d(self)':<10} {'d(bg)':<10} {'bg/self':<10}")
    results = {}
    for cls in CLASSES:
        mask = np.where(primary == cls)[0]
        if len(mask) < 5:
            continue
        E_cls = E[mask]
        # same-type distance (mean upper triangle)
        d_self = cosine_distances(E_cls, E_cls)
        d_self_mean = d_self[np.triu_indices_from(d_self, k=1)].mean()
        d_bg_mean = cosine_distances(E_cls, E_bg).mean()
        results[cls] = {'n':int(len(mask)), 'd_self':float(d_self_mean), 'd_bg':float(d_bg_mean),
                        'ratio':float(d_bg_mean/d_self_mean)}
        print(f"{cls:<12} {len(mask):<5} {d_self_mean:<10.4f} {d_bg_mean:<10.4f} {d_bg_mean/d_self_mean:<10.3f}")
    return results

def main():
    E, metas = load_all_shards()
    # Normalize embeddings for cosine
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    print(f"Embedding shape: {E.shape}")
    print(f"BGC samples: {sum(1 for m in metas if m['is_bgc']==1)}")
    print(f"BG samples: {sum(1 for m in metas if m['is_bgc']==0)}")

    # UMAP
    Z = run_umap(E, metas)
    plot_umap_by_class(Z, metas, f"{FIG_DIR}/umap_by_class.png")
    plot_umap_by_length(Z, metas, f"{FIG_DIR}/umap_by_length.png")
    plot_umap_by_phylum(Z, metas, f"{FIG_DIR}/umap_by_phylum.png")
    # Distance distributions
    dist_means = distance_analysis(E, metas, f"{FIG_DIR}/distance_distributions.png")
    # kNN
    knn_res = knn_leave_one_out(E, metas)
    # Per-type
    per_type_res = per_type_analysis(E, metas)
    # Save summary
    import json
    summary = {
        'n_samples': len(metas),
        'n_bgc': sum(1 for m in metas if m['is_bgc']==1),
        'n_bg': sum(1 for m in metas if m['is_bgc']==0),
        'dist_means': dist_means,
        'knn': knn_res,
        'per_type': per_type_res,
    }
    with open(f"{FIG_DIR}/../part_a_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to part_a_summary.json")
    print("\n=== Decision signal ===")
    print(f"BGC-bg distance ({dist_means['bgc_bg']:.4f}) vs same-type ({dist_means['same_type']:.4f})")
    gap = dist_means['bgc_bg'] - dist_means['same_type']
    print(f"Gap: {gap:.4f}  ({'positive=good' if gap>0 else 'NEGATIVE=bad'})")
    print(f"kNN enrichment: {knn_res['enrichment']:.2f}x  ({'>1.5x=good' if knn_res['enrichment']>1.5 else 'weak'})")

if __name__ == '__main__':
    main()
