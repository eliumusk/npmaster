#!/usr/bin/env python3
"""
P1-T2b merge: Reassemble GPU shards into ordered embeddings and merge with existing.

Usage:
  python p1_t2_merge_shards.py \
    --shard-dir /data/muskliu/np_master/data/embeddings/edge_padding \
    --n-gpus 7 \
    --n-total 123953 \
    --existing-emb /data/muskliu/np_master/data/embeddings/embeddings.npy \
    --existing-nll /data/muskliu/np_master/data/embeddings/nll_features.npy
"""
import argparse
import json
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--n-gpus", type=int, required=True)
    ap.add_argument("--n-total", type=int, required=True)
    ap.add_argument("--existing-emb", required=True)
    ap.add_argument("--existing-nll", default=None)
    args = ap.parse_args()

    print(f"Merging {args.n_gpus} shards, {args.n_total} total sequences")

    # Preallocate
    emb_dim = None
    all_emb = [None] * args.n_total
    all_nll = [None] * args.n_total
    all_ids = [None] * args.n_total
    n_loaded = 0

    for g in range(args.n_gpus):
        tag = f"gpu{g}"
        emb_path = os.path.join(args.shard_dir, f"shard_{tag}_emb.npy")
        nll_path = os.path.join(args.shard_dir, f"shard_{tag}_nll.npy")
        meta_path = os.path.join(args.shard_dir, f"shard_{tag}_meta.json")

        if not os.path.exists(emb_path):
            print(f"  WARNING: Missing shard {tag}")
            continue

        emb = np.load(emb_path)
        nll = np.load(nll_path)
        with open(meta_path) as f:
            meta = json.load(f)

        if emb_dim is None:
            emb_dim = emb.shape[1]

        indices = meta["global_indices"]
        ids = meta["window_ids"]

        for j, idx in enumerate(indices):
            all_emb[idx] = emb[j]
            all_nll[idx] = nll[j]
            all_ids[idx] = ids[j]
            n_loaded += 1

        print(f"  Shard {tag}: {len(indices)} sequences loaded")

    missing = sum(1 for e in all_emb if e is None)
    print(f"\nLoaded: {n_loaded}, Missing: {missing}")

    # Filter valid
    valid_mask = [e is not None for e in all_emb]
    new_emb = np.stack([all_emb[i] for i in range(args.n_total) if valid_mask[i]]).astype(np.float32)
    new_nll = np.stack([all_nll[i] for i in range(args.n_total) if valid_mask[i]]).astype(np.float32)
    new_ids = [all_ids[i] for i in range(args.n_total) if valid_mask[i]]

    print(f"New embeddings: {new_emb.shape}")
    np.save(os.path.join(args.shard_dir, "padded_embeddings.npy"), new_emb)
    np.save(os.path.join(args.shard_dir, "padded_nll_features.npy"), new_nll)
    with open(os.path.join(args.shard_dir, "padded_window_ids.json"), "w") as f:
        json.dump(new_ids, f)

    # Merge with existing
    print("Merging with existing embeddings...")
    existing_emb = np.load(args.existing_emb)
    merged = np.vstack([existing_emb, new_emb])
    merged_path = os.path.join(args.shard_dir, "embeddings_merged.npy")
    np.save(merged_path, merged)
    print(f"  Merged embeddings: {merged.shape} -> {merged_path}")

    if args.existing_nll and os.path.exists(args.existing_nll):
        existing_nll = np.load(args.existing_nll)
        merged_nll = np.vstack([existing_nll, new_nll])
        np.save(os.path.join(args.shard_dir, "nll_features_merged.npy"), merged_nll)
        print(f"  Merged NLL: {merged_nll.shape}")

    summary = {
        "n_total": args.n_total,
        "n_loaded": n_loaded,
        "n_missing": missing,
        "emb_dim": int(emb_dim) if emb_dim else 0,
        "merged_shape": list(merged.shape),
    }
    with open(os.path.join(args.shard_dir, "merge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. {merged_path}")


if __name__ == "__main__":
    main()
