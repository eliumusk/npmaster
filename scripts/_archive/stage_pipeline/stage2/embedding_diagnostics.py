#!/usr/bin/env python3
"""
Stage 2.4: Embedding quality diagnostics.
- UMAP visualization (by type, phylum, length)
- kNN enrichment (global + phylum-controlled)
- Distance distributions
- Source bias check (CRITICAL)
"""
import csv
import json
import os
import numpy as np
from collections import defaultdict

def load_metadata(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def cosine_dist_batch(a, b):
    """Pairwise cosine distance between rows of a and b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return 1 - (a_norm * b_norm).sum(axis=1)

def main():
    emb_path = "data/embeddings/embeddings.npy"
    meta_path = "data/windows/windows_metadata.csv"
    out_dir = "data/embeddings/diagnostics"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading embeddings...", flush=True)
    emb = np.load(emb_path)  # (948362, 4096)
    meta = load_metadata(meta_path)
    assert len(meta) == emb.shape[0]

    labels = np.array([r["label"] for r in meta])
    genomes = np.array([r["genome_id"] for r in meta])
    phyla = np.array([r["phylum"] for r in meta])
    splits = np.array([r["split"] for r in meta])

    # get primary BGC type for positive windows
    classes = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]
    def get_type(row):
        for c in classes:
            if row.get(c) == "1":
                return c
        return "negative"
    types = np.array([get_type(r) for r in meta])

    pos_mask = labels == "positive"
    neg_mask = labels == "negative"
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]
    print(f"Positive: {pos_mask.sum()}, Negative: {neg_mask.sum()}", flush=True)

    rng = np.random.default_rng(42)

    # ====== 1. UMAP ======
    print("Computing UMAP...", flush=True)
    try:
        import umap
        n_sample = 5000
        sample_pos = rng.choice(pos_idx, min(n_sample, len(pos_idx)), replace=False)
        sample_neg = rng.choice(neg_idx, min(n_sample, len(neg_idx)), replace=False)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        rng.shuffle(sample_idx)

        emb_sample = emb[sample_idx]
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
        coords = reducer.fit_transform(emb_sample)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample_types = types[sample_idx]
        sample_phyla = phyla[sample_idx]
        sample_labels = labels[sample_idx]

        # by type
        fig, ax = plt.subplots(figsize=(10, 8))
        for t in classes + ["negative"]:
            mask = sample_types == t
            if mask.sum() == 0: continue
            ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.5, label=f"{t} ({mask.sum()})")
        ax.legend(markerscale=5, fontsize=8)
        ax.set_title("UMAP by BGC type")
        fig.savefig(f"{out_dir}/umap_by_type.png", dpi=150, bbox_inches="tight")
        plt.close()

        # by phylum (top 5)
        fig, ax = plt.subplots(figsize=(10, 8))
        from collections import Counter
        phy_counts = Counter(sample_phyla)
        top_phy = [p for p, _ in phy_counts.most_common(6)]
        for p in top_phy:
            mask = sample_phyla == p
            ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.5, label=f"{p} ({mask.sum()})")
        other_mask = ~np.isin(sample_phyla, top_phy)
        if other_mask.sum():
            ax.scatter(coords[other_mask, 0], coords[other_mask, 1], s=2, alpha=0.2, c="gray", label="other")
        ax.legend(markerscale=5, fontsize=8)
        ax.set_title("UMAP by phylum")
        fig.savefig(f"{out_dir}/umap_by_phylum.png", dpi=150, bbox_inches="tight")
        plt.close()

        # by label (BGC vs non-BGC)
        fig, ax = plt.subplots(figsize=(10, 8))
        for lab, color in [("negative", "gray"), ("positive", "red")]:
            mask = sample_labels == lab
            ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.4, c=color, label=f"{lab} ({mask.sum()})")
        ax.legend(markerscale=5)
        ax.set_title("UMAP: BGC vs non-BGC")
        fig.savefig(f"{out_dir}/umap_by_label.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("UMAP done.", flush=True)
    except Exception as e:
        print(f"UMAP failed: {e}", flush=True)

    # ====== 2. kNN enrichment ======
    print("Computing kNN enrichment...", flush=True)
    from sklearn.neighbors import NearestNeighbors
    k = 5

    # global
    n_knn = 6000
    knn_pos = rng.choice(pos_idx, min(n_knn, len(pos_idx)), replace=False)
    knn_neg = rng.choice(neg_idx, min(n_knn, len(neg_idx)), replace=False)
    knn_idx = np.concatenate([knn_pos, knn_neg])
    knn_emb = emb[knn_idx]
    knn_labels = labels[knn_idx]

    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(knn_emb)
    dists, indices = nn.kneighbors(knn_emb)
    # exclude self (first neighbor)
    neighbor_idx = indices[:, 1:k+1]
    neighbor_labels = knn_labels[neighbor_idx]
    query_labels = knn_labels[:, None]
    hits = (neighbor_labels == query_labels).sum()
    total = len(knn_idx) * k
    hit_rate = hits / total
    prior = knn_labels[knn_labels == "positive"].shape[0] / len(knn_labels)
    # enrichment vs random for positive class
    pos_queries = knn_labels == "positive"
    pos_hits = (neighbor_labels[pos_queries] == "positive").sum()
    pos_total = pos_queries.sum() * k
    pos_hit_rate = pos_hits / pos_total
    enrichment_global = pos_hit_rate / prior if prior > 0 else 0

    # phylum-controlled (Actinomycetota only)
    actino_mask = np.array([r["phylum"] == "p__Actinomycetota" for r in meta])
    actino_pos = np.where(actino_mask & pos_mask)[0]
    actino_neg = np.where(actino_mask & neg_mask)[0]
    n_ap = min(3000, len(actino_pos))
    n_an = min(3000, len(actino_neg))
    ap = rng.choice(actino_pos, n_ap, replace=False)
    an = rng.choice(actino_neg, n_an, replace=False)
    actino_idx = np.concatenate([ap, an])
    actino_emb = emb[actino_idx]
    actino_labels = labels[actino_idx]

    nn2 = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(actino_emb)
    _, indices2 = nn2.kneighbors(actino_emb)
    neighbor_labels2 = actino_labels[indices2[:, 1:k+1]]
    actino_pos_q = actino_labels == "positive"
    actino_pos_hits = (neighbor_labels2[actino_pos_q] == "positive").sum()
    actino_prior = actino_pos_q.sum() / len(actino_labels)
    actino_pos_hit_rate = actino_pos_hits / (actino_pos_q.sum() * k)
    enrichment_actino = actino_pos_hit_rate / actino_prior if actino_prior > 0 else 0

    knn_result = {
        "global_enrichment": round(float(enrichment_global), 3),
        "global_pos_hit_rate": round(float(pos_hit_rate), 4),
        "global_prior": round(float(prior), 4),
        "actino_enrichment": round(float(enrichment_actino), 3),
        "actino_pos_hit_rate": round(float(actino_pos_hit_rate), 4),
        "actino_prior": round(float(actino_prior), 4),
    }
    print(f"kNN: {json.dumps(knn_result, indent=2)}", flush=True)

    # ====== 3. Distance distributions ======
    print("Computing distance distributions...", flush=True)
    n_pairs = 10000

    def sample_pairs(idx_a, idx_b, n):
        a = rng.choice(idx_a, n, replace=True)
        b = rng.choice(idx_b, n, replace=True)
        return cosine_dist_batch(emb[a], emb[b])

    # same-type BGC pairs
    pks_idx = np.where((types == "Polyketide") & pos_mask)[0]
    nrp_idx = np.where((types == "NRP") & pos_mask)[0]
    same_type = sample_pairs(pks_idx, pks_idx, n_pairs) if len(pks_idx) > 100 else np.array([])
    diff_type = sample_pairs(pks_idx, nrp_idx, n_pairs) if len(pks_idx) > 100 and len(nrp_idx) > 100 else np.array([])
    bgc_bg = sample_pairs(pos_idx, neg_idx, n_pairs)
    bg_bg = sample_pairs(neg_idx, neg_idx, n_pairs)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for arr, label in [(same_type, "PKS-PKS"), (diff_type, "PKS-NRP"),
                           (bgc_bg, "BGC-background"), (bg_bg, "bg-bg")]:
            if len(arr) > 0:
                ax.hist(arr, bins=80, alpha=0.5, density=True, label=f"{label} (med={np.median(arr):.3f})")
        ax.legend()
        ax.set_xlabel("Cosine distance")
        ax.set_title("Distance distributions")
        fig.savefig(f"{out_dir}/distance_distributions.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Distance plot failed: {e}")

    dist_result = {
        "same_type_median": float(np.median(same_type)) if len(same_type) else None,
        "diff_type_median": float(np.median(diff_type)) if len(diff_type) else None,
        "bgc_bg_median": float(np.median(bgc_bg)),
        "bg_bg_median": float(np.median(bg_bg)),
    }
    print(f"Distances: {json.dumps(dist_result, indent=2)}", flush=True)

    # ====== 4. SOURCE BIAS CHECK (CRITICAL) ======
    print("Computing source bias check...", flush=True)
    # For each genome with both pos and neg windows:
    #   intra_dist = mean cosine distance between pos and neg windows within genome
    # Then:
    #   inter_dist = mean cosine distance between pos windows of genome A and neg windows of genome B
    # If intra >= inter, source bias is eliminated.

    genome_pos = defaultdict(list)
    genome_neg = defaultdict(list)
    for i, r in enumerate(meta):
        if r["label"] == "positive":
            genome_pos[r["genome_id"]].append(i)
        elif r["label"] == "negative":
            genome_neg[r["genome_id"]].append(i)

    # genomes with both pos and neg
    both_genomes = [g for g in genome_pos if g in genome_neg
                    and len(genome_pos[g]) >= 3 and len(genome_neg[g]) >= 3]
    print(f"Genomes with both pos+neg (≥3 each): {len(both_genomes)}", flush=True)

    # sample genomes
    sample_g = rng.choice(both_genomes, min(200, len(both_genomes)), replace=False)

    intra_dists = []
    for g in sample_g:
        pi = rng.choice(genome_pos[g], min(10, len(genome_pos[g])), replace=False)
        ni = rng.choice(genome_neg[g], min(10, len(genome_neg[g])), replace=False)
        # pairwise cosine dist between pos and neg within genome
        for p in pi:
            d = cosine_dist_batch(emb[p:p+1].repeat(len(ni), axis=0), emb[ni])
            intra_dists.extend(d.tolist())

    # inter-genome: pos from genome A, neg from genome B
    inter_dists = []
    g_pairs = rng.choice(len(sample_g), size=(2000, 2), replace=True)
    for a, b in g_pairs:
        ga, gb = sample_g[a], sample_g[b]
        if ga == gb: continue
        pi = rng.choice(genome_pos[ga], 1)[0]
        ni = rng.choice(genome_neg[gb], 1)[0]
        d = cosine_dist_batch(emb[pi:pi+1], emb[ni:ni+1])
        inter_dists.append(float(d[0]))

    intra_median = float(np.median(intra_dists))
    inter_median = float(np.median(inter_dists))

    source_bias = {
        "intra_genome_bgc_vs_bg_median": round(intra_median, 4),
        "inter_genome_bgc_vs_bg_median": round(inter_median, 4),
        "intra_n_pairs": len(intra_dists),
        "inter_n_pairs": len(inter_dists),
        "bias_eliminated": intra_median >= inter_median,
        "interpretation": (
            "PASS: intra >= inter, source bias eliminated"
            if intra_median >= inter_median
            else "FAIL: intra < inter, source bias still present"
        ),
    }
    print(f"Source bias: {json.dumps(source_bias, indent=2)}", flush=True)

    # ====== Save summary ======
    summary = {
        "knn_enrichment": knn_result,
        "distance_distributions": dist_result,
        "source_bias_check": source_bias,
    }
    with open(f"{out_dir}/diagnostics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}/diagnostics_summary.json")
    print("\n=== STAGE 2.4 COMPLETE ===")

if __name__ == "__main__":
    main()
