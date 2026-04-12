"""Supplementary: UMAP restricted to Actinobacteria to see BGC type structure clearly."""
import pickle, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from sklearn.neighbors import NearestNeighbors

EMB_DIR = "/data/muskliu/np_master/data/diagnostic_day1/embeddings"
FIG_DIR = "/data/muskliu/np_master/data/diagnostic_day1/figures"
CLASSES = ["Alkaloid","NRP","Polyketide","RiPP","Saccharide","Terpene","Other"]
COLORS = {"Alkaloid":"#e41a1c","NRP":"#377eb8","Polyketide":"#4daf4a",
          "RiPP":"#984ea3","Saccharide":"#ff7f00","Terpene":"#ffff33","Other":"#a65628"}

files = sorted(glob.glob(f"{EMB_DIR}/shard_*.pkl"))
embs, metas = [], []
for f in files:
    d = pickle.load(open(f,'rb'))
    embs.append(d['embeddings']); metas.extend(d['metas'])
E = np.concatenate(embs, axis=0); E = E / (np.linalg.norm(E,axis=1,keepdims=True)+1e-8)

def is_actino(m):
    if m['is_bgc']==0: return m.get('phylum','')=='Actinobacteria'
    org = m.get('organism','').lower()
    kws = ['streptomyces','micromonospora','amycolatopsis','actinomad','nocardia','mycobacterium',
           'saccharopolyspora','actinoplanes','verrucosispora','actinosynnema','kutzneria',
           'rhodococcus','corynebacterium','actinom']
    return any(k in org for k in kws)

actino_idx = np.where([is_actino(m) for m in metas])[0]
E_act = E[actino_idx]; metas_act = [metas[i] for i in actino_idx]
print(f"Actino: {len(metas_act)} samples")

# UMAP
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42)
Z = reducer.fit_transform(E_act)

# Plot
fig, ax = plt.subplots(figsize=(10,8))
bg_mask = np.array([m['is_bgc']==0 for m in metas_act])
ax.scatter(Z[bg_mask,0], Z[bg_mask,1], c="#999999", s=5, alpha=0.4, label=f'bg-Strep (n={bg_mask.sum()})')
multi_mask = np.array([m['is_bgc']==1 and len(m['categories'])>=2 for m in metas_act])
if multi_mask.any():
    ax.scatter(Z[multi_mask,0], Z[multi_mask,1], c='black', s=8, alpha=0.7, marker='x', label=f'Multi (n={multi_mask.sum()})')
for cls in CLASSES:
    mask = np.array([m['is_bgc']==1 and m['categories']==[cls] for m in metas_act])
    if mask.sum() > 0:
        ax.scatter(Z[mask,0], Z[mask,1], c=COLORS[cls], s=10, alpha=0.75, label=f'{cls} (n={mask.sum()})')
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.set_title('Evo2 embedding UMAP - Actinobacteria only (phylum confound removed)')
ax.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/umap_actino_only.png", dpi=140); plt.close()
print(f"Saved umap_actino_only.png")

# Also: length-matched (long BGCs) UMAP
lens = np.array([m.get('used_length',16384) for m in metas_act])
length_mask = (lens >= 14000)  # long BGC or bg
Z_len = Z[length_mask]
metas_len = [metas_act[i] for i in range(len(metas_act)) if length_mask[i]]

fig, ax = plt.subplots(figsize=(10,8))
bg_mask2 = np.array([m['is_bgc']==0 for m in metas_len])
ax.scatter(Z_len[bg_mask2,0], Z_len[bg_mask2,1], c="#999999", s=5, alpha=0.4, label=f'bg-Strep (n={bg_mask2.sum()})')
for cls in CLASSES:
    mask = np.array([m['is_bgc']==1 and m['categories']==[cls] for m in metas_len])
    if mask.sum() > 0:
        ax.scatter(Z_len[mask,0], Z_len[mask,1], c=COLORS[cls], s=10, alpha=0.75, label=f'{cls} (n={mask.sum()})')
multi2 = np.array([m['is_bgc']==1 and len(m['categories'])>=2 for m in metas_len])
if multi2.any():
    ax.scatter(Z_len[multi2,0], Z_len[multi2,1], c='black', s=8, alpha=0.7, marker='x', label=f'Multi (n={multi2.sum()})')
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.set_title('UMAP - Actino + length>=14kb only (length confound removed)')
ax.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/umap_actino_lengthmatched.png", dpi=140); plt.close()
print(f"Saved umap_actino_lengthmatched.png")

# Length-matched kNN enrichment
is_bgc_len = np.array([m['is_bgc']==1 for m in metas_len])
prior_len = is_bgc_len.mean()
E_len = E_act[length_mask]
nn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(E_len)
_, nbr = nn.kneighbors(E_len)
hits = []
for i in range(len(metas_len)):
    if not is_bgc_len[i]: continue
    hits.append(sum(1 for j in nbr[i,1:6] if is_bgc_len[j])/5.0)
hits = np.array(hits)
print(f"\nLength-matched Actino-only kNN:")
print(f"  n BGC={is_bgc_len.sum()}, n bg={(~is_bgc_len).sum()}")
print(f"  prior P(BGC): {prior_len:.3f}")
print(f"  mean hit: {hits.mean():.3f}")
print(f"  enrichment: {hits.mean()/prior_len:.2f}x")
