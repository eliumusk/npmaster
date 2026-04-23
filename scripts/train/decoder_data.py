"""Build per-contig decoder training data.

Steps:
 1. Apply LR_A1 (W, b) to all 1.064M training embeddings (memory-mapped, chunked)
 2. Write per-window score column aligned with windows_metadata_edgepad_filtered.csv
 3. Group by (genome_id, contig_id), filter contigs with ≥10 windows, sort by window_start
 4. Emit decoder_data.npz with per-contig arrays (+ split assignment)

The resulting file contains a list of examples; each example is a contig-level
sequence of (score, binary label). Decoder training consumes these.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path

OUT_DIR = Path("/data/muskliu/npmaster/experiments/evo2_first_principles/decoder_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = "/data/muskliu/npmaster/data/embeddings/edge_padding/embeddings_merged.npy"
MD_PATH = "/data/muskliu/npmaster/data/windows/edge_padding/windows_metadata_edgepad_filtered.csv"
LR_PATH = "/data/muskliu/npmaster/experiments/evo2_first_principles/baseline_a/lr_A1.npz"
MIN_WINDOWS_PER_CONTIG = 10


def main():
    t0 = time.time()
    print("=== Loading LR_A1 ===")
    lr = np.load(LR_PATH)
    W = lr["W"].astype(np.float32)  # [4096]
    b = float(lr["b"])
    print(f"  W shape={W.shape}, b={b:.4f}")

    print("\n=== Loading metadata ===")
    md = pd.read_csv(MD_PATH)
    print(f"  {len(md)} rows")

    print("\n=== Applying LR to embeddings (memory-mapped) ===")
    emb = np.load(EMB_PATH, mmap_mode="r")
    assert emb.shape[0] == len(md), f"row mismatch: {emb.shape[0]} vs {len(md)}"

    N = emb.shape[0]
    chunk = 50_000
    scores = np.empty(N, dtype=np.float32)
    for i in range(0, N, chunk):
        e = emb[i:i+chunk].astype(np.float32)  # copy to RAM
        logits = e @ W + b
        scores[i:i+chunk] = 1.0 / (1.0 + np.exp(-logits))
        if (i // chunk) % 4 == 0:
            print(f"  row {i:>8} / {N}  pos-rate={(scores[:i+chunk] > 0.5).mean():.3f}")
    print(f"  done in {time.time()-t0:.1f}s")

    print("\n=== Joining scores + labels ===")
    md["lr_score"] = scores
    # Label: 1 if positive, 0 if negative, -1 if ambiguous (masked in loss)
    md["y"] = -1
    md.loc[md.label == "positive", "y"] = 1
    md.loc[md.label == "negative", "y"] = 0
    print(f"  label dist: pos={sum(md.y==1)}, neg={sum(md.y==0)}, ambig={sum(md.y==-1)}")

    # Calibration sanity: mean score on pos vs neg
    pos_mean = md.loc[md.y == 1, "lr_score"].mean()
    neg_mean = md.loc[md.y == 0, "lr_score"].mean()
    print(f"  mean LR score: pos={pos_mean:.3f}, neg={neg_mean:.3f}")

    print("\n=== Grouping by contig, filtering ≥ 10 windows ===")
    # Sort for grouping + per-contig order
    md_sorted = md.sort_values(["genome_id", "contig_id", "window_start"]).reset_index(drop=True)
    gc = md_sorted.groupby(["genome_id", "contig_id"])

    examples = []
    skipped = 0
    for (gid, cid), sub in gc:
        if len(sub) < MIN_WINDOWS_PER_CONTIG:
            skipped += 1
            continue
        # All windows in this contig share same split
        split = sub["split"].iloc[0]
        examples.append({
            "genome_id": gid,
            "contig_id": cid,
            "split": split,
            "starts": sub["window_start"].to_numpy(dtype=np.int32),
            "ends": sub["window_end"].to_numpy(dtype=np.int32),
            "scores": sub["lr_score"].to_numpy(dtype=np.float32),
            "labels": sub["y"].to_numpy(dtype=np.int8),
            "overlap": sub["max_bgc_overlap_ratio"].to_numpy(dtype=np.float32),
        })
    print(f"  kept {len(examples)} contigs, skipped {skipped} (<{MIN_WINDOWS_PER_CONTIG} windows)")

    # Split summary
    from collections import Counter
    sc = Counter(e["split"] for e in examples)
    print(f"  split counts: {dict(sc)}")

    out = OUT_DIR / "decoder_data.pkl"
    with open(out, "wb") as fh:
        pickle.dump(examples, fh, protocol=4)
    print(f"\n[write] {out} ({out.stat().st_size/1e6:.1f} MB)")

    # Also write a small summary
    summary = {
        "n_contigs": len(examples),
        "total_windows": sum(len(e["scores"]) for e in examples),
        "split_counts": {k: int(v) for k, v in sc.items()},
        "min_windows_per_contig": MIN_WINDOWS_PER_CONTIG,
        "pos_rate_all_windows": float(sum((e["labels"] == 1).sum() for e in examples) /
                                      sum(len(e["labels"]) for e in examples)),
        "mean_lr_score_pos": float(pos_mean),
        "mean_lr_score_neg": float(neg_mean),
    }
    import json
    with open(OUT_DIR / "data_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[write] {OUT_DIR/'data_summary.json'}")


if __name__ == "__main__":
    main()
