#!/usr/bin/env python3
"""
P3-T3: Build overlapping chunk index for Route B training.

Each contig is divided into overlapping chunks:
  - chunk length: L_chunk (default 32 windows)
  - stride: L_chunk // 2 (50% overlap)
  - loss only computed on center L_chunk // 2 windows (avoid boundary artifacts)
  - first/last chunks extend loss region to contig edges

Output: chunk_index.pkl with list of chunk descriptors.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def build_chunks(contig_index, meta_arrays, L_chunk=32, split="train"):
    """Build overlapping chunk descriptors for a given split.

    Returns list of dicts:
      {contig_key, global_indices, valid_mask, loss_mask, n_positive}
    """
    stride = L_chunk // 2  # 50% overlap
    margin = L_chunk // 4  # each side margin (context-only, no loss)

    chunks = []
    splits = meta_arrays["split"]
    labels = meta_arrays["labels"]
    label_strs = meta_arrays["label"]

    for contig_key, indices in contig_index.items():
        # Check if contig has any windows in this split
        split_mask = np.array([splits[i] == split for i in indices])
        if not split_mask.any():
            continue

        L = len(indices)

        if L <= L_chunk:
            # Single chunk, pad right
            global_indices = list(indices) + [-1] * (L_chunk - L)
            valid_mask = [True] * L + [False] * (L_chunk - L)

            # Loss on all valid non-ambiguous windows
            loss_mask = []
            for i, g_idx in enumerate(global_indices):
                if g_idx == -1:
                    loss_mask.append(False)
                elif splits[g_idx] != split:
                    loss_mask.append(False)
                elif label_strs[g_idx] == "ambiguous":
                    loss_mask.append(False)
                else:
                    loss_mask.append(True)

            n_pos = sum(1 for i, g_idx in enumerate(global_indices)
                       if g_idx != -1 and loss_mask[i] and labels[g_idx, 7] == 0)

            chunks.append({
                "contig_key": contig_key,
                "global_indices": global_indices,
                "valid_mask": valid_mask,
                "loss_mask": loss_mask,
                "n_positive": n_pos,
            })
        else:
            # Multiple overlapping chunks
            starts = list(range(0, L - L_chunk + 1, stride))
            # Ensure last chunk covers contig end
            if starts[-1] + L_chunk < L:
                starts.append(L - L_chunk)

            for ci, start in enumerate(starts):
                end = start + L_chunk
                global_indices = list(indices[start:end])
                valid_mask = [True] * L_chunk

                # Loss region: center portion only
                # First chunk: loss from 0 to stride (extend to left edge)
                # Last chunk: loss from margin to L_chunk (extend to right edge)
                # Middle chunks: loss from margin to margin+stride
                is_first = (ci == 0)
                is_last = (ci == len(starts) - 1)

                loss_start = 0 if is_first else margin
                loss_end = L_chunk if is_last else margin + stride

                loss_mask = []
                for pos in range(L_chunk):
                    g_idx = global_indices[pos]
                    in_loss_region = (loss_start <= pos < loss_end)
                    is_valid_split = (splits[g_idx] == split)
                    is_not_ambiguous = (label_strs[g_idx] != "ambiguous")

                    loss_mask.append(in_loss_region and is_valid_split and is_not_ambiguous)

                n_pos = sum(1 for pos in range(L_chunk)
                           if loss_mask[pos] and labels[global_indices[pos], 7] == 0)

                chunks.append({
                    "contig_key": contig_key,
                    "global_indices": global_indices,
                    "valid_mask": valid_mask,
                    "loss_mask": loss_mask,
                    "n_positive": n_pos,
                })

    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contig-index", required=True)
    p.add_argument("--meta-arrays", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--L-chunk", type=int, default=32)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.contig_index, "rb") as f:
        contig_index = pickle.load(f)
    with open(args.meta_arrays, "rb") as f:
        meta_arrays = pickle.load(f)

    all_chunks = {}
    for split in ["train", "val", "test_id", "test_ood"]:
        chunks = build_chunks(contig_index, meta_arrays, L_chunk=args.L_chunk, split=split)

        n_loss_windows = sum(sum(c["loss_mask"]) for c in chunks)
        n_pos = sum(c["n_positive"] for c in chunks)
        print(f"{split}: {len(chunks)} chunks, {n_loss_windows} loss windows, {n_pos} positive")
        all_chunks[split] = chunks

    out_path = out_dir / "chunk_index.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"chunks": all_chunks, "L_chunk": args.L_chunk}, f)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
