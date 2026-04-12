"""
Leave-One-Type-Out kNN (on Actinobacteria, length-matched):
For each BGC type T:
  Remove all type-T BGCs from 'positive' library
  For each type-T BGC: find 5-NN in (remaining 6 types + bg)
  Report: hit rate (proportion of neighbors that are ANY BGC of the remaining types)
  vs control: kNN hit rate WITH type T included (baseline)

This tests: can GCD "discover" novel types in frozen Evo2 space?
"""
import pickle, glob, numpy as np
from sklearn.neighbors import NearestNeighbors
import json

EMB_DIR = "/data/muskliu/np_master/data/diagnostic_day1/embeddings"
CLASSES = ["Alkaloid","NRP","Polyketide","RiPP","Saccharide","Terpene","Other"]

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

actino = np.array([is_actino(m) for m in metas])
lens = np.array([m.get('used_length', 16384) for m in metas])
is_bgc = np.array([m['is_bgc']==1 for m in metas])

# length-matched subset: BGCs >= 14kb + bg (16kb)
keep = actino & ((is_bgc & (lens >= 14000)) | (~is_bgc))
print(f"Actino + length-matched: {keep.sum()} samples")
print(f"  BGC: {(keep & is_bgc).sum()}, bg: {(keep & ~is_bgc).sum()}")

E_k = E[keep]
metas_k = [metas[i] for i in range(len(metas)) if keep[i]]
is_bgc_k = np.array([m['is_bgc']==1 for m in metas_k])

def primary_type(m):
    if m['is_bgc']==0: return 'bg'
    if len(m['categories'])>=2: return 'Multi'
    if m['categories']: return m['categories'][0]
    return 'bg'

types_k = np.array([primary_type(m) for m in metas_k])
print("Type counts in kept set:")
from collections import Counter
for t, c in Counter(types_k).most_common():
    print(f"  {t}: {c}")

# Baseline: regular kNN in this pool (all BGCs in library)
print("\n=== Baseline kNN (all types included) ===")
nn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(E_k)
_, nbr = nn.kneighbors(E_k)
baseline = {}
for target in CLASSES:
    mask = (types_k == target)
    if mask.sum() < 5: continue
    hits = []
    for i in np.where(mask)[0]:
        hr = sum(1 for j in nbr[i,1:6] if is_bgc_k[j]) / 5.0
        hits.append(hr)
    baseline[target] = np.mean(hits)
    print(f"  {target}: n={mask.sum()}, hit_rate={baseline[target]:.3f}")

# LOTO: for each type, remove it from library, query its samples
print("\n=== Leave-One-Type-Out kNN ===")
loto_results = {}
for target in CLASSES:
    target_mask = (types_k == target)
    if target_mask.sum() < 5: continue
    # library: all samples EXCEPT target type BGCs (includes bg + other 6 types)
    lib_mask = ~target_mask
    lib_E = E_k[lib_mask]
    lib_is_bgc = is_bgc_k[lib_mask]
    lib_types = types_k[lib_mask]
    # query: target type BGCs
    query_E = E_k[target_mask]
    nn2 = NearestNeighbors(n_neighbors=5, metric='cosine').fit(lib_E)
    _, q_nbr = nn2.kneighbors(query_E)
    # For each query: how many of top-5 are BGCs (any of remaining 6 types)?
    bgc_hits = []      # fraction that are BGC
    other_type_hits = [] # which remaining types are hit
    for q in range(len(query_E)):
        neighbors = q_nbr[q]
        n_bgc = sum(1 for j in neighbors if lib_is_bgc[j])
        bgc_hits.append(n_bgc / 5.0)
    mean_hit = np.mean(bgc_hits)
    n_query = target_mask.sum()
    # prior in library
    lib_prior = lib_is_bgc.mean()
    loto_results[target] = {
        'n': int(n_query),
        'hit_rate_after_loto': float(mean_hit),
        'hit_rate_baseline': baseline.get(target, 0),
        'lib_prior': float(lib_prior),
        'enrichment_loto': float(mean_hit / lib_prior) if lib_prior>0 else 0,
        'retention_ratio': float(mean_hit / baseline.get(target, mean_hit)) if baseline.get(target, 0) > 0 else 1.0,
    }
    print(f"  {target}: n={n_query}, baseline={baseline.get(target,0):.3f}, LOTO hit={mean_hit:.3f}, lib_prior={lib_prior:.3f}, retention={loto_results[target]['retention_ratio']:.2f}")

# Save
with open('/data/muskliu/np_master/data/diagnostic_day1/loto_results.json','w') as f:
    json.dump({'baseline':baseline,'loto':loto_results}, f, indent=2)

print("\n=== Summary ===")
print(f"{'Type':<12} {'N':<5} {'Baseline':<10} {'LOTO':<10} {'Retention':<10}")
for t in CLASSES:
    if t not in loto_results: continue
    r = loto_results[t]
    print(f"{t:<12} {r['n']:<5} {r['hit_rate_baseline']:<10.3f} {r['hit_rate_after_loto']:<10.3f} {r['retention_ratio']:<10.3f}")
