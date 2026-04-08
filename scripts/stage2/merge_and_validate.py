#!/usr/bin/env python3
"""
Stage 2.3: Merge per-genome embeddings into single matrices + validate.
"""
import csv
import json
import os
import sys
import numpy as np
from collections import defaultdict

def main():
    emb_dir = "data/embeddings/per_genome"
    meta_csv = "data/windows/windows_metadata.csv"
    out_emb = "data/embeddings/embeddings.npy"
    out_nll = "data/embeddings/nll_features.npy"
    out_summary = "data/embeddings/extraction_summary.json"

    # read metadata to get genome order and window counts
    genome_order = []
    genome_wcount = defaultdict(int)
    total_windows = 0
    with open(meta_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            g = row["genome_id"]
            if not genome_order or genome_order[-1] != g:
                genome_order.append(g)
            genome_wcount[g] += 1
            total_windows += 1

    # deduplicate genome_order (preserve first-seen order)
    seen = set()
    unique_order = []
    for g in genome_order:
        if g not in seen:
            unique_order.append(g)
            seen.add(g)
    genome_order = unique_order

    print(f"Total windows in metadata: {total_windows}")
    print(f"Total genomes: {len(genome_order)}")

    # merge
    all_emb = []
    all_nll = []
    failed = []
    success = 0
    has_nll = True

    for gi, g in enumerate(genome_order):
        emb_path = os.path.join(emb_dir, f"emb_{g}.npy")
        nll_path = os.path.join(emb_dir, f"nll_{g}.npy")
        expected = genome_wcount[g]

        if not os.path.exists(emb_path):
            failed.append({"genome": g, "reason": "emb_file_missing", "n_windows": expected})
            # fill with zeros
            all_emb.append(np.zeros((expected, 4096), dtype=np.float32))
            if has_nll:
                all_nll.append(np.zeros((expected, 8), dtype=np.float32))
            continue

        emb = np.load(emb_path)
        if emb.shape[0] != expected:
            failed.append({"genome": g, "reason": f"shape_mismatch_{emb.shape[0]}_vs_{expected}",
                          "n_windows": expected})
            # pad or truncate
            if emb.shape[0] < expected:
                pad = np.zeros((expected - emb.shape[0], 4096), dtype=np.float32)
                emb = np.vstack([emb, pad])
            else:
                emb = emb[:expected]

        all_emb.append(emb)
        success += 1

        if has_nll and os.path.exists(nll_path):
            nll = np.load(nll_path)
            if nll.shape[0] != expected:
                nll_pad = np.zeros((expected, 8), dtype=np.float32)
                nll_pad[:min(nll.shape[0], expected)] = nll[:expected]
                nll = nll_pad
            all_nll.append(nll)
        elif has_nll:
            all_nll.append(np.zeros((expected, 8), dtype=np.float32))

        if (gi + 1) % 200 == 0:
            print(f"  merged {gi+1}/{len(genome_order)} genomes...", flush=True)

    print(f"Merging {len(all_emb)} genome blocks...")
    embeddings = np.vstack(all_emb)
    print(f"Embeddings shape: {embeddings.shape}")
    np.save(out_emb, embeddings)
    print(f"Saved {out_emb} ({embeddings.nbytes/1e9:.1f} GB)")

    if all_nll:
        nll_features = np.vstack(all_nll)
        print(f"NLL features shape: {nll_features.shape}")
        np.save(out_nll, nll_features)
        print(f"Saved {out_nll} ({nll_features.nbytes/1e6:.0f} MB)")

    # validation
    checks = {}
    checks["total_rows"] = int(embeddings.shape[0])
    checks["expected_rows"] = total_windows
    checks["rows_match"] = embeddings.shape[0] == total_windows
    checks["embedding_dim"] = int(embeddings.shape[1])
    checks["has_nan"] = bool(np.isnan(embeddings).any())
    checks["has_inf"] = bool(np.isinf(embeddings).any())
    checks["n_zero_rows"] = int((embeddings == 0).all(axis=1).sum())

    # per-dim stats
    means = embeddings.mean(axis=0)
    stds = embeddings.std(axis=0)
    checks["dim_mean_range"] = [float(means.min()), float(means.max())]
    checks["dim_std_range"] = [float(stds.min()), float(stds.max())]
    checks["mean_in_range"] = bool(abs(means).max() < 10)
    checks["std_in_range"] = bool(stds.min() > 0.01 and stds.max() < 10)

    # spot check: random 10 rows
    rng = np.random.default_rng(42)
    idx = rng.choice(embeddings.shape[0], min(10, embeddings.shape[0]), replace=False)
    checks["spot_check_nonzero"] = bool(all(embeddings[i].any() for i in idx))

    # NLL diff_var check (col 4) if available
    if all_nll:
        nll_f = nll_features
        # load labels
        labels = []
        with open(meta_csv) as f:
            for row in csv.DictReader(f):
                labels.append(row["label"])
        labels = np.array(labels)
        pos_mask = labels == "positive"
        neg_mask = labels == "negative"
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            diff_var_pos = nll_f[pos_mask, 4].mean()
            diff_var_neg = nll_f[neg_mask, 4].mean()
            checks["nll_diff_var_positive_mean"] = float(diff_var_pos)
            checks["nll_diff_var_negative_mean"] = float(diff_var_neg)
            checks["nll_has_nan"] = bool(np.isnan(nll_f).any())
            checks["nll_has_inf"] = bool(np.isinf(nll_f).any())

    summary = {
        "total_genomes_processed": success,
        "total_genomes_failed": len(failed),
        "failed_genomes": failed[:20],
        "total_windows_extracted": int(embeddings.shape[0]),
        "embedding_dim": 4096,
        "nll_dim": 8,
        "layer_name": "blocks.20.mlp.l3",
        "validation": checks,
    }

    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # collect GPU logs
    gpu_hours = 0
    for fn in sorted(os.listdir(emb_dir)):
        if fn.endswith("_log.csv"):
            with open(os.path.join(emb_dir, fn)) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        try:
                            gpu_hours += float(parts[3]) / 3600
                        except:
                            pass
    print(f"\nTotal GPU-hours: {gpu_hours:.1f}")

if __name__ == "__main__":
    main()
