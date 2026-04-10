"""Sanity check: verify positive windows get high scores."""
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

emb = np.load("/data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy", mmap_mode="r")
with open("/data/muskliu/np_master/data/phase3/route_a/contig_index.pkl", "rb") as f:
    contig_index = pickle.load(f)
with open("/data/muskliu/np_master/data/phase3/route_a/meta_arrays.pkl", "rb") as f:
    meta = pickle.load(f)

# Find a positive val contig with multiple BGC windows
for key, indices in contig_index.items():
    pos_count = sum(1 for i in indices if meta["split"][i] == "val" and meta["label"][i] == "positive")
    if pos_count >= 3 and len(indices) >= 10 and len(indices) <= 100:
        test_key = key
        test_indices = indices
        break

print(f"Contig: {test_key}, L={len(test_indices)}, pos count={pos_count}")

# Predict full contig
e = np.stack([emb[i] for i in test_indices])
e_t = torch.from_numpy(e).unsqueeze(0).to(device)
v = torch.ones(1, len(test_indices), dtype=torch.bool, device=device)
with torch.no_grad():
    _, logits = model(e_t, v)
probs = torch.sigmoid(logits[0]).cpu().numpy()  # (L, 8)
scores = probs[:, :7].max(axis=1)

# Compare scores for positive vs negative windows
pos_scores = []
neg_scores = []
amb_scores = []
for pos, g_idx in enumerate(test_indices):
    label = meta["label"][g_idx]
    if label == "positive":
        pos_scores.append(scores[pos])
    elif label == "negative":
        neg_scores.append(scores[pos])
    else:
        amb_scores.append(scores[pos])

print(f"  Positive windows: n={len(pos_scores)} scores={[f'{s:.3f}' for s in pos_scores]}")
print(f"  Negative windows: n={len(neg_scores)} mean={np.mean(neg_scores):.3f} max={np.max(neg_scores):.3f}")
print(f"  Ambiguous:        n={len(amb_scores)} scores={[f'{s:.3f}' for s in amb_scores]}")

# Summary: does the model separate positive from negative?
if pos_scores and neg_scores:
    print(f"\n  Positive mean score: {np.mean(pos_scores):.4f}")
    print(f"  Negative mean score: {np.mean(neg_scores):.4f}")
    print(f"  Separation: {np.mean(pos_scores) - np.mean(neg_scores):.4f}")

# Also verify by scanning 20 random positive windows in val
print("\n=== Random positive windows in val ===")
val_pos_idx = [i for i, s in enumerate(meta["split"]) 
               if s == "val" and meta["label"][i] == "positive"]
sample = np.random.default_rng(42).choice(val_pos_idx, size=20, replace=False)
scores_pos = []
for g_idx in sample:
    # Find contig and get full prediction
    for key, indices in contig_index.items():
        if g_idx in indices:
            e = np.stack([emb[i] for i in indices])
            e_t = torch.from_numpy(e).unsqueeze(0).to(device)
            v = torch.ones(1, len(indices), dtype=torch.bool, device=device)
            with torch.no_grad():
                _, logits = model(e_t, v)
            probs = torch.sigmoid(logits[0]).cpu().numpy()
            pos_in_contig = indices.index(g_idx)
            scores_pos.append(probs[pos_in_contig, :7].max())
            break
scores_pos = np.array(scores_pos)
print(f"20 random val positive window scores:")
print(f"  mean={scores_pos.mean():.4f} median={np.median(scores_pos):.4f}")
print(f"  >0.5: {(scores_pos > 0.5).sum()}/20")
print(f"  >0.86: {(scores_pos > 0.86).sum()}/20")
