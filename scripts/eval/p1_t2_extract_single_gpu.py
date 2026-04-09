#!/usr/bin/env python3
"""
P1-T2b: Single-GPU Evo2 extraction for padded windows
======================================================
Processes a shard of padded_sequences.fasta (interleaved by gpu_id).
Launch one process per GPU, same pattern as Stage 2.

Usage:
  CUDA_VISIBLE_DEVICES=0 python p1_t2_extract_single_gpu.py \
    --fasta padded_sequences.fasta \
    --gpu-id 0 --n-gpus 7 \
    --out /data/muskliu/np_master/data/embeddings/edge_padding
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F


def load_fasta(path):
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
    bases = "ACGT"
    return "".join(c if c in bases else random.choice(bases) for c in seq.upper())


def compute_nll_features(logits, input_ids):
    log_probs = F.log_softmax(logits[:-1].float(), dim=-1)
    targets = input_ids[1:]
    per_pos_nll = -log_probs[torch.arange(len(targets)), targets]
    nll = per_pos_nll.cpu().numpy()

    diff_var = np.diff(nll).var() if len(nll) > 1 else 0.0
    ac100 = np.corrcoef(nll[:-100], nll[100:])[0, 1] if len(nll) > 100 else 0.0
    ac1000 = np.corrcoef(nll[:-1000], nll[1000:])[0, 1] if len(nll) > 1000 else 0.0
    hist, _ = np.histogram(nll, bins=50, density=True)
    hist = hist + 1e-10
    nll_entropy = float(-np.sum(hist * np.log(hist)) * (nll.max() - nll.min()) / 50)

    return np.array([
        nll.mean(), nll.std(), np.median(nll), np.percentile(nll, 75),
        diff_var, ac100, ac1000, nll_entropy,
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--n-gpus", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", default="blocks.20.mlp.l3")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = f"gpu{args.gpu_id}"

    # Check done flag
    done_flag = os.path.join(args.out, f"shard_{tag}_done.flag")
    if os.path.exists(done_flag):
        print(f"[{tag}] Already done. Skipping.")
        return

    # Load sequences and take this GPU's shard
    print(f"[{tag}] Loading sequences...")
    all_seqs = load_fasta(args.fasta)
    my_seqs = [(i, all_seqs[i]) for i in range(args.gpu_id, len(all_seqs), args.n_gpus)]
    print(f"[{tag}] Assigned {len(my_seqs)}/{len(all_seqs)} sequences")

    if not my_seqs:
        print(f"[{tag}] No sequences. Done.")
        return

    # Load model
    print(f"[{tag}] Loading Evo2 model...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")
    model.model.eval()

    # Register hook (same as Stage 2: use named_modules search)
    hook_output = {}
    def hook_fn(module, input, output):
        hook_output["value"] = output

    target_module = None
    for name, module in model.model.named_modules():
        if name == args.layer:
            target_module = module
            break
    if target_module is None:
        available = [n for n, _ in model.model.named_modules() if "mlp" in n and "l3" in n]
        print(f"[{tag}] ERROR: layer '{args.layer}' not found. Available: {available[:10]}")
        return
    hook = target_module.register_forward_hook(hook_fn)
    print(f"[{tag}] Hook registered on {args.layer}", flush=True)

    device = f"cuda:0"  # CUDA_VISIBLE_DEVICES maps this

    # Extract
    embeddings = []
    nll_features = []
    global_indices = []
    window_ids = []

    t_start = time.time()
    for j, (global_idx, (wid, seq)) in enumerate(my_seqs):
        seq_clean = clean_sequence(seq)
        tokens = model.tokenizer.tokenize(seq_clean)
        input_ids = torch.tensor(tokens, dtype=torch.int).unsqueeze(0).to(device)

        with torch.no_grad():
            result = model(input_ids)
            # Handle Evo2 return: result is tuple, result[0] may also be tuple
            logits = result[0] if isinstance(result, tuple) else result
            if isinstance(logits, tuple):
                logits = logits[0]

            # Extract hidden state from hook (same as Stage 2)
            hidden = hook_output["value"]
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            emb = hidden.float().mean(dim=1).cpu().numpy().squeeze()
            embeddings.append(emb)

            logits_2d = logits.squeeze(0)
            nll_feat = compute_nll_features(logits_2d, input_ids[0])
            nll_features.append(nll_feat)

        global_indices.append(global_idx)
        window_ids.append(wid)

        if (j + 1) % 100 == 0 or j == 0:
            elapsed = time.time() - t_start
            rate = (j + 1) / elapsed
            eta = (len(my_seqs) - j - 1) / rate / 3600
            print(f"[{tag}] {j+1}/{len(my_seqs)} ({rate:.2f}/s, ETA {eta:.1f}h)")

    hook.remove()

    # Save
    emb_arr = np.stack(embeddings).astype(np.float32)
    nll_arr = np.stack(nll_features).astype(np.float32)

    np.save(os.path.join(args.out, f"shard_{tag}_emb.npy"), emb_arr)
    np.save(os.path.join(args.out, f"shard_{tag}_nll.npy"), nll_arr)
    with open(os.path.join(args.out, f"shard_{tag}_meta.json"), "w") as f:
        json.dump({"global_indices": global_indices, "window_ids": window_ids}, f)

    with open(done_flag, "w") as f:
        elapsed = time.time() - t_start
        f.write(f"done {len(my_seqs)} seqs in {elapsed:.0f}s\n")

    print(f"[{tag}] Done: {len(my_seqs)} seqs, {emb_arr.shape}, {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
