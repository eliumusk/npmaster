"""Audit Route B evaluation: compare full-contig vs chunked inference, BN stats."""
import pickle, sys
import numpy as np
import torch

sys.path.insert(0, "/data/muskliu/np_master/code/np-master/bgc-gcd")
from model.context_full_model import ContextFullModel

device = "cuda"
ckpt = torch.load("/data/muskliu/np_master/data/phase3/route_b/best_route_b_s0.pt",
                  map_location=device, weights_only=False)
model = ContextFullModel(input_dim=4096, proj_hidden=2048, proj_out=256, proj_dropout=0.3)
model.load_state_dict(ckpt["model"])
model.to(device).eval()
print(f"alpha={model.gate.alpha.item():.4f}")

# Check BN running stats
bn = model.projector.bn1
print(f"\nBN running_mean: abs mean={bn.running_mean.abs().mean():.4f} max={bn.running_mean.abs().max():.4f}")
print(f"BN running_var:  mean={bn.running_var.mean():.4f} min={bn.running_var.min():.6f}")
print(f"BN num_batches_tracked: {bn.num_batches_tracked.item()}")

# Load data
emb = np.load("/data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy", mmap_mode="r")
with open("/data/muskliu/np_master/data/phase3/route_a/contig_index.pkl", "rb") as f:
    contig_index = pickle.load(f)
with open("/data/muskliu/np_master/data/phase3/route_a/meta_arrays.pkl", "rb") as f:
    meta = pickle.load(f)

# Pick 5 val contigs of different sizes
val_contigs = []
for key, indices in contig_index.items():
    if any(meta["split"][i] == "val" for i in indices):
        val_contigs.append((key, indices))

# Sample: 1 small, 1 medium, 1 large
sizes = {"small": None, "medium": None, "large": None}
for key, indices in val_contigs:
    L = len(indices)
    if L <= 5 and sizes["small"] is None:
        sizes["small"] = (key, indices)
    elif 30 <= L <= 50 and sizes["medium"] is None:
        sizes["medium"] = (key, indices)
    elif L >= 200 and sizes["large"] is None:
        sizes["large"] = (key, indices)

# Test 1: Determinism — same input gives same output
print("\n=== Test 1: Determinism ===")
for size_name, (key, indices) in sizes.items():
    if indices is None: continue
    e = np.stack([emb[i] for i in indices])
    e_t = torch.from_numpy(e).unsqueeze(0).to(device)
    v = torch.ones(1, len(indices), dtype=torch.bool, device=device)
    with torch.no_grad():
        _, l1 = model(e_t, v)
        _, l2 = model(e_t, v)
    diff = (l1 - l2).abs().max().item()
    print(f"  {size_name} (L={len(indices)}): max diff = {diff:.2e}")

# Test 2: Batch independence — single contig vs batched with another
print("\n=== Test 2: Batch Independence ===")
for size_name, (key, indices) in sizes.items():
    if indices is None: continue
    e1 = np.stack([emb[i] for i in indices])
    # Pad to same length as medium for batching
    L = len(indices)
    e1_t = torch.from_numpy(e1).unsqueeze(0).to(device)
    v1 = torch.ones(1, L, dtype=torch.bool, device=device)
    with torch.no_grad():
        _, solo = model(e1_t, v1)
    # Now batch with a padded version of itself
    e_batch = torch.cat([e1_t, e1_t], dim=0)  # (2, L, 4096)
    v_batch = torch.cat([v1, v1], dim=0)
    with torch.no_grad():
        _, batched = model(e_batch, v_batch)
    diff = (solo[0] - batched[0]).abs().max().item()
    print(f"  {size_name} (L={L}): solo vs batched max diff = {diff:.2e}")

# Test 3: Length effect — predict same center window in chunks of different sizes
print("\n=== Test 3: Sequence Length Sensitivity ===")
if sizes["large"] is not None:
    key, indices = sizes["large"]
    center = len(indices) // 2
    print(f"  Center window at pos {center} of contig (L={len(indices)})")
    
    for L_test in [16, 32, 64, 128, 256]:
        half = L_test // 2
        start = max(0, center - half)
        end = min(len(indices), start + L_test)
        start = max(0, end - L_test)
        actual_L = end - start
        if actual_L < L_test:
            continue
        sub_indices = indices[start:end]
        e = np.stack([emb[i] for i in sub_indices])
        e_t = torch.from_numpy(e).unsqueeze(0).to(device)
        v = torch.ones(1, actual_L, dtype=torch.bool, device=device)
        with torch.no_grad():
            _, logits = model(e_t, v)
        center_in_sub = center - start
        probs = torch.sigmoid(logits[0, center_in_sub, :7]).cpu().numpy()
        score = probs.max()
        print(f"    L={L_test:4d}: center score = {score:.4f}, max type = {probs.argmax()}")

# Test 4: Full-contig vs chunk with training-matched L=32
print("\n=== Test 4: Full-contig vs L=32 chunked prediction ===")
if sizes["large"] is not None:
    key, indices = sizes["large"]
    L_full = len(indices)
    
    # Full contig prediction
    e = np.stack([emb[i] for i in indices])
    e_t = torch.from_numpy(e).unsqueeze(0).to(device)
    v = torch.ones(1, L_full, dtype=torch.bool, device=device)
    with torch.no_grad():
        _, logits_full = model(e_t, v)
    scores_full = torch.sigmoid(logits_full[0, :, :7]).max(dim=-1).values.cpu().numpy()
    
    # Chunked prediction (non-overlapping L=32, center region only)
    scores_chunk = np.full(L_full, np.nan)
    L_chunk = 32
    for start in range(0, L_full - L_chunk + 1, L_chunk):
        sub = indices[start:start+L_chunk]
        e_sub = np.stack([emb[i] for i in sub])
        e_t = torch.from_numpy(e_sub).unsqueeze(0).to(device)
        v = torch.ones(1, L_chunk, dtype=torch.bool, device=device)
        with torch.no_grad():
            _, logits = model(e_t, v)
        probs = torch.sigmoid(logits[0, :, :7]).max(dim=-1).values.cpu().numpy()
        scores_chunk[start:start+L_chunk] = probs
    
    # Compare on middle portion where both are defined
    valid = ~np.isnan(scores_chunk)
    diffs = np.abs(scores_full[valid] - scores_chunk[valid])
    print(f"  Full vs chunked: mean diff={diffs.mean():.4f}, max diff={diffs.max():.4f}")
    print(f"    Full mean score: {scores_full[valid].mean():.4f}")
    print(f"    Chunked mean score: {scores_chunk[valid].mean():.4f}")

# Test 5: Check if any predictions are sensitive to padding (simulate chunk padding)
print("\n=== Test 5: Padding Sensitivity ===")
if sizes["medium"] is not None:
    key, indices = sizes["medium"]
    L_real = len(indices)
    
    # Real contig
    e = np.stack([emb[i] for i in indices])
    e_t = torch.from_numpy(e).unsqueeze(0).to(device)
    v = torch.ones(1, L_real, dtype=torch.bool, device=device)
    with torch.no_grad():
        _, logits_real = model(e_t, v)
    scores_real = torch.sigmoid(logits_real[0, :, :7]).max(dim=-1).values.cpu().numpy()
    
    # Padded to 64 with zeros + valid_mask=False
    pad_len = 64
    e_pad = np.zeros((pad_len, 4096), dtype=np.float32)
    e_pad[:L_real] = e
    e_t = torch.from_numpy(e_pad).unsqueeze(0).to(device)
    v = torch.zeros(1, pad_len, dtype=torch.bool, device=device)
    v[0, :L_real] = True
    with torch.no_grad():
        _, logits_pad = model(e_t, v)
    scores_pad = torch.sigmoid(logits_pad[0, :L_real, :7]).max(dim=-1).values.cpu().numpy()
    
    diffs = np.abs(scores_real - scores_pad)
    print(f"  Real vs padded (same contig): mean diff={diffs.mean():.4f}, max diff={diffs.max():.4f}")
