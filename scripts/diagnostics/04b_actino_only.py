"""
Supplementary: restrict to Actinobacteria phylum only.
Check if BGC vs bg signal holds WITHIN a single phylum.
"""
import pickle, glob, numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances

EMB_DIR = "/data/muskliu/np_master/data/diagnostic_day1/embeddings"

# Load
files = sorted(glob.glob(f"{EMB_DIR}/shard_*.pkl"))
embs, metas = [], []
for f in files:
    d = pickle.load(open(f, 'rb'))
    embs.append(d['embeddings']); metas.extend(d['metas'])
E = np.concatenate(embs, axis=0)
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
print(f"Total: {E.shape}")

# MIBiG BGCs are mostly Actinobacteria (Streptomyces, Micromonospora, etc.)
# Identify Actinobacteria BGCs by organism name
def is_actino(m):
    if m['is_bgc']==0:
        return m.get('phylum','')=='Actinobacteria'
    # BGC: check organism
    org = m.get('organism','').lower()
    actino_keywords = ['streptomyces','micromonospora','amycolatopsis','actinomad','nocardia',
                       'mycobacterium','saccharopolyspora','actinoplanes','verrucosispora',
                       'actinosynnema','kutzneria','rhodococcus','corynebacterium','actinom']
    return any(k in org for k in actino_keywords)

is_bgc = np.array([m['is_bgc']==1 for m in metas])
is_act = np.array([is_actino(m) for m in metas])

# Within Actinobacteria only
actino_mask = is_act
bgc_act = actino_mask & is_bgc
bg_act = actino_mask & ~is_bgc
print(f"\nWithin Actinobacteria:")
print(f"  BGC (Actino): {bgc_act.sum()}")
print(f"  bg (Actino=Strep): {bg_act.sum()}")

# kNN within Actinobacteria
idx_actino = np.where(actino_mask)[0]
E_actino = E[idx_actino]
metas_actino = [metas[i] for i in idx_actino]
is_bgc_actino = np.array([m['is_bgc']==1 for m in metas_actino])
prior_actino = is_bgc_actino.mean()
print(f"  prior (Actino P(BGC)): {prior_actino:.3f}")

nn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(E_actino)
_, nbr_idx = nn.kneighbors(E_actino)
# for BGC samples, compute hit rate
hits = []
for i in range(len(metas_actino)):
    if not is_bgc_actino[i]:
        continue
    neighbors = nbr_idx[i,1:6]
    hits.append(sum(1 for j in neighbors if is_bgc_actino[j]) / 5.0)
hits = np.array(hits)
print(f"  mean kNN hit (Actino-only): {hits.mean():.3f}")
print(f"  enrichment over Actino-prior: {hits.mean()/prior_actino:.2f}x")
print(f"  100% hits: {(hits==1.0).sum()}/{len(hits)}")
print(f"  0% hits: {(hits==0.0).sum()}/{len(hits)}")

# Distance analysis within Actinobacteria
np.random.seed(42)
bgc_act_idx = np.where(bgc_act)[0]
bg_act_idx = np.where(bg_act)[0]
n_each = 300
bgc_sub = np.random.choice(bgc_act_idx, min(n_each, len(bgc_act_idx)), replace=False)
bg_sub = np.random.choice(bg_act_idx, min(n_each, len(bg_act_idx)), replace=False)

d_bgc_bgc = cosine_distances(E[bgc_sub], E[bgc_sub])
d_bgc_bg = cosine_distances(E[bgc_sub], E[bg_sub])
d_bg_bg = cosine_distances(E[bg_sub], E[bg_sub])
print(f"\nActinobacteria-only distance means:")
print(f"  BGC-BGC (all pairs): {d_bgc_bgc[np.triu_indices_from(d_bgc_bgc,k=1)].mean():.4f}")
print(f"  BGC-bg:              {d_bgc_bg.mean():.4f}")
print(f"  bg-bg:               {d_bg_bg[np.triu_indices_from(d_bg_bg,k=1)].mean():.4f}")
print(f"  Gap (BGC-bg) - (BGC-BGC): {d_bgc_bg.mean() - d_bgc_bgc[np.triu_indices_from(d_bgc_bgc,k=1)].mean():.4f}")
print(f"  (positive=BGC tighter than BGC-vs-bg=GOOD, negative=BAD)")

# Length control analysis
lens = np.array([m.get('used_length', 16384) for m in metas_actino])
print(f"\nLength confound check (Actino only):")
print(f"  BGC length distribution: median={np.median(lens[is_bgc_actino]):.0f}, std={lens[is_bgc_actino].std():.0f}")
print(f"  bg length (should be 16384): median={np.median(lens[~is_bgc_actino]):.0f}, std={lens[~is_bgc_actino].std():.0f}")
# Filter BGC to length >=14000 (comparable to bg 16384)
long_bgc = is_bgc_actino & (lens >= 14000)
print(f"  Long BGCs (>=14000bp): {long_bgc.sum()}")

# Distance analysis with matched-length BGCs
if long_bgc.sum() >= 100:
    long_bgc_idx_in_actino = np.where(long_bgc)[0]
    bg_in_actino = np.where(~is_bgc_actino)[0]
    # get global indices
    long_bgc_global = idx_actino[long_bgc_idx_in_actino]
    bg_global = idx_actino[bg_in_actino]

    sub_lbgc = np.random.choice(long_bgc_global, min(300, len(long_bgc_global)), replace=False)
    sub_bg = np.random.choice(bg_global, min(300, len(bg_global)), replace=False)
    d_lbgc_lbgc = cosine_distances(E[sub_lbgc], E[sub_lbgc])
    d_lbgc_bg = cosine_distances(E[sub_lbgc], E[sub_bg])
    print(f"\nLength-matched (BGC>=14kb vs bg=16kb):")
    print(f"  BGC-BGC (long): {d_lbgc_lbgc[np.triu_indices_from(d_lbgc_lbgc,k=1)].mean():.4f}")
    print(f"  BGC-bg:         {d_lbgc_bg.mean():.4f}")
    print(f"  Gap: {d_lbgc_bg.mean() - d_lbgc_lbgc[np.triu_indices_from(d_lbgc_lbgc,k=1)].mean():.4f}")
