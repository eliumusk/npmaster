#!/usr/bin/env python3
"""
P1-T2b: Extract Evo2 embeddings for padded windows only
========================================================
Reads the padded_sequences.fasta from P1-T2, extracts Evo2 embeddings,
then merges with existing embeddings.npy.

Usage (on GPU node):
  python p1_t2_extract_padded_embeddings.py \
    --fasta /data/muskliu/np_master/data/windows/edge_padding/padded_sequences.fasta \
    --existing-emb /data/muskliu/np_master/data/embeddings/embeddings.npy \
    --existing-nll /data/muskliu/np_master/data/embeddings/nll_features.npy \
    --out /data/muskliu/np_master/data/embeddings/edge_padding \
    --layer blocks.20.mlp.l3
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch


def load_fasta(path):
    """Load sequences from FASTA file. Returns list of (id, sequence)."""
    seqs = []
    cur_id = None
    cur_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_id is not None:
                    seqs.append((cur_id, "".join(cur_seq)))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            seqs.append((cur_id, "".join(cur_seq)))
    return seqs


def clean_sequence(seq):
    """Replace non-ACGT characters with random bases."""
    import random
    bases = "ACGT"
    return "".join(c if c in bases else random.choice(bases) for c in seq.upper())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="padded_sequences.fasta from P1-T2")
    ap.add_argument("--existing-emb", required=True, help="existing embeddings.npy")
    ap.add_argument("--existing-nll", default=None, help="existing nll_features.npy (optional)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--layer", default="blocks.20.mlp.l3", help="Evo2 layer to extract from")
    ap.add_argument("--batch-size", type=int, default=1, help="batch size (1 for safety)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load sequences
    print("Loading padded sequences...")
    seqs = load_fasta(args.fasta)
    print(f"  {len(seqs)} sequences to extract")

    if len(seqs) == 0:
        print("No sequences to extract. Exiting.")
        return

    # Load Evo2
    print("Loading Evo2 model...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")
    model.model.eval()

    # Register hook for hidden states
    target_layer = args.layer
    hidden_states = {}

    def hook_fn(module, input, output):
        hidden_states["out"] = output.detach()

    # Navigate to the target layer
    parts = target_layer.split(".")
    layer = model.model
    for p in parts:
        if p.isdigit():
            layer = layer[int(p)]
        else:
            layer = getattr(layer, p)
    hook = layer.register_forward_hook(hook_fn)

    # Extract
    print(f"Extracting embeddings from layer {target_layer}...")
    embeddings = []
    nll_features = []

    for i, (wid, seq) in enumerate(seqs):
        seq_clean = clean_sequence(seq)

        t0 = time.time()
        # Tokenize
        input_ids = model.tokenizer(seq_clean, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(args.device)

        with torch.no_grad():
            logits = model.model(input_ids).logits

            # Embedding: mean pool over sequence dimension
            h = hidden_states["out"]
            if h.dim() == 3:
                emb = h[0].mean(dim=0).cpu().numpy()  # (hidden_dim,)
            else:
                emb = h.mean(dim=0).cpu().numpy()

            embeddings.append(emb)

            # NLL features (same 8 features as Stage 2)
            log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
            target_ids = input_ids[0, 1:]
            nll = -log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            nll_np = nll.cpu().numpy()

            nll_mean = float(np.mean(nll_np))
            nll_std = float(np.std(nll_np))
            nll_median = float(np.median(nll_np))
            nll_p75 = float(np.percentile(nll_np, 75))

            diffs = np.diff(nll_np)
            diff_var = float(np.var(diffs)) if len(diffs) > 0 else 0.0

            # Autocorrelation at lag 100 and 1000
            def autocorr(x, lag):
                if len(x) <= lag:
                    return 0.0
                return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

            ac100 = autocorr(nll_np, 100)
            ac1000 = autocorr(nll_np, 1000)

            # Entropy
            probs = torch.softmax(logits[0, :-1], dim=-1)
            entropy = float((-probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().cpu())

            nll_features.append([nll_mean, nll_std, nll_median, nll_p75,
                                diff_var, ac100, ac1000, entropy])

        dt = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(seqs)}] {wid} {dt:.2f}s emb_shape={emb.shape}")

    hook.remove()

    # Stack
    new_emb = np.stack(embeddings, axis=0).astype(np.float32)
    new_nll = np.array(nll_features, dtype=np.float32)
    print(f"\nNew embeddings shape: {new_emb.shape}")
    print(f"New NLL features shape: {new_nll.shape}")

    # Save new-only arrays
    np.save(os.path.join(args.out, "padded_embeddings.npy"), new_emb)
    np.save(os.path.join(args.out, "padded_nll_features.npy"), new_nll)

    # Merge with existing
    print("\nMerging with existing embeddings...")
    existing_emb = np.load(args.existing_emb)
    merged_emb = np.vstack([existing_emb, new_emb])
    merged_emb_path = os.path.join(args.out, "embeddings_merged.npy")
    np.save(merged_emb_path, merged_emb)
    print(f"  Merged embeddings: {merged_emb.shape} -> {merged_emb_path}")

    if args.existing_nll and os.path.exists(args.existing_nll):
        existing_nll = np.load(args.existing_nll)
        merged_nll = np.vstack([existing_nll, new_nll])
        merged_nll_path = os.path.join(args.out, "nll_features_merged.npy")
        np.save(merged_nll_path, merged_nll)
        print(f"  Merged NLL: {merged_nll.shape} -> {merged_nll_path}")

    # Save extraction summary
    window_ids = [wid for wid, _ in seqs]
    summary = {
        "task_id": "P1-T2b",
        "n_extracted": len(seqs),
        "emb_dim": int(new_emb.shape[1]),
        "layer": target_layer,
        "window_ids": window_ids,
        "merged_shape": list(merged_emb.shape),
        "outputs": {
            "padded_embeddings": os.path.join(args.out, "padded_embeddings.npy"),
            "padded_nll": os.path.join(args.out, "padded_nll_features.npy"),
            "merged_embeddings": merged_emb_path,
        },
    }
    with open(os.path.join(args.out, "extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Summary: {args.out}/extraction_summary.json")


if __name__ == "__main__":
    main()
