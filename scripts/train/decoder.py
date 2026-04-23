"""Train 1D CNN decoder over per-window LR scores (src/decoding/learned.py model).

16 configs = arch(2) × hidden(2) × label(2) × loss(2) grid.
Each config takes ~1-2 min on a single A800. Parallelize across 16 GPUs
(node7 + node8) with a shell launch loop — see `docs/learned-decoder.md`.

Input: decoder_data.pkl from scripts/train/decoder_data.py (per-contig scores + labels).
Output: <out-root>/<tag>/{decoder.pt, config.json, history.json, results.json}

Usage:
  python scripts/train/decoder.py --config-id K --gpu G --out-root <dir> --data <pkl>
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoding.learned import Decoder1DCNN  # noqa: E402


# ================ 16-config grid ================
def make_grid():
    configs = []
    for arch in ["cnn3", "cnn5"]:
        for hidden in [32, 64]:
            for label_src in ["antismash", "tight"]:  # "tight": overlap>=0.7
                for loss_type in ["bce", "focal"]:
                    configs.append({
                        "arch": arch, "hidden": hidden,
                        "label_src": label_src, "loss_type": loss_type,
                    })
    return configs


# ================ Dataset ================
class ContigDataset(Dataset):
    def __init__(self, examples, label_src="antismash", max_len=512):
        self.examples = examples
        self.label_src = label_src
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        scores = e["scores"]
        if self.label_src == "antismash":
            labels = e["labels"]  # {-1, 0, 1}
        else:
            ov = e["overlap"]
            labels = np.where(ov >= 0.7, 1, np.where(ov <= 0.05, 0, -1)).astype(np.int8)
        # random crop for training
        T = len(scores)
        if T > self.max_len:
            s = np.random.randint(0, T - self.max_len)
            scores = scores[s:s+self.max_len]
            labels = labels[s:s+self.max_len]
        return torch.from_numpy(scores.astype(np.float32)), torch.from_numpy(labels.astype(np.int64))


def collate_fn(batch):
    lens = [len(s) for (s, _) in batch]
    L = max(lens)
    B = len(batch)
    scores = torch.zeros(B, L)
    labels = torch.full((B, L), -1, dtype=torch.int64)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, (s, y) in enumerate(batch):
        t = len(s)
        scores[i, :t] = s
        labels[i, :t] = y
        mask[i, :t] = True
    return scores, labels, mask


# ================ Losses ================
def bce_loss(logits, labels, mask, pos_weight=3.0):
    # labels: {-1, 0, 1}. Skip -1.
    valid = mask & (labels >= 0)
    y = labels.float().clamp(min=0)
    loss = F.binary_cross_entropy_with_logits(
        logits, y, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device))
    return loss[valid].mean()


def focal_loss(logits, labels, mask, gamma=2.0, pos_weight=3.0):
    valid = mask & (labels >= 0)
    y = labels.float().clamp(min=0)
    p = torch.sigmoid(logits)
    p_t = p * y + (1 - p) * (1 - y)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    weight = pos_weight * y + (1 - y)
    loss = weight * (1 - p_t).pow(gamma) * ce
    return loss[valid].mean()


def pick_loss(loss_type):
    return bce_loss if loss_type == "bce" else focal_loss


# ================ Eval ================
def metrics_from_flat(p, y):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ap = average_precision_score(y, p) if len(set(y)) > 1 else float("nan")
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    yhat = (p >= 0.5).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"ap": float(ap), "auc": float(auc), "f1_at_0.5": float(f1),
            "precision_at_0.5": float(prec), "recall_at_0.5": float(rec)}


def compute_metrics(logits, labels, mask):
    # Kept for API compat, single-batch usage
    valid = mask & (labels >= 0)
    p = torch.sigmoid(logits)[valid].cpu().numpy()
    y = labels[valid].cpu().numpy()
    return metrics_from_flat(p, y)


# ================ Train loop ================
def train_one(config, out_dir, data_path, device, max_epochs=50, patience=5):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    print(f"[{config}] loading data")
    examples = pickle.load(open(data_path, "rb"))
    train = [e for e in examples if e["split"] == "train"]
    val = [e for e in examples if e["split"] == "val"]
    test_id = [e for e in examples if e["split"] == "test_id"]
    test_ood = [e for e in examples if e["split"] == "test_ood"]
    print(f"  train={len(train)} val={len(val)} test_id={len(test_id)} test_ood={len(test_ood)}")

    train_ds = ContigDataset(train, config["label_src"])
    val_ds = ContigDataset(val, config["label_src"], max_len=10_000)  # no crop
    test_id_ds = ContigDataset(test_id, config["label_src"], max_len=10_000)
    test_ood_ds = ContigDataset(test_ood, config["label_src"], max_len=10_000)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                               collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn, num_workers=1)

    depth = 3 if config["arch"] == "cnn3" else 5
    model = Decoder1DCNN(depth=depth, hidden=config["hidden"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: depth={depth} hidden={config['hidden']} params={n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = pick_loss(config["loss_type"])

    best_val_ap = -1.0
    best_state = None
    no_improve = 0
    history = []

    for ep in range(max_epochs):
        t0 = time.time()
        model.train()
        losses = []
        for scores, labels, mask in train_loader:
            scores = scores.to(device); labels = labels.to(device); mask = mask.to(device)
            logits = model(scores)
            loss = loss_fn(logits, labels, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        model.eval()
        probs_flat, y_flat = [], []
        with torch.no_grad():
            for scores, labels, mask in val_loader:
                scores = scores.to(device); labels = labels.to(device); mask = mask.to(device)
                logits = model(scores)
                valid = mask & (labels >= 0)
                probs_flat.append(torch.sigmoid(logits)[valid].cpu().numpy())
                y_flat.append(labels[valid].cpu().numpy())
        p = np.concatenate(probs_flat); y = np.concatenate(y_flat)
        m = metrics_from_flat(p, y)
        print(f"  ep{ep:>2} train_loss={np.mean(losses):.4f}  val_AP={m['ap']:.4f} AUC={m['auc']:.4f} F1@0.5={m['f1_at_0.5']:.4f}  {time.time()-t0:.1f}s")
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), **m})
        if m["ap"] > best_val_ap:
            best_val_ap = m["ap"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop at ep{ep}")
                break

    # Save best
    torch.save(best_state, out_dir / "decoder.pt")
    with open(out_dir / "history.json", "w") as fh:
        json.dump({"history": history, "best_val_ap": best_val_ap}, fh, indent=2)

    # Test eval
    model.load_state_dict(best_state)
    model.eval()
    results = {"val": metrics_from_flat(p, y)}
    for name, ds in [("test_id", test_id_ds), ("test_ood", test_ood_ds)]:
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, num_workers=1)
        probs_flat, y_flat = [], []
        with torch.no_grad():
            for scores, labels, mask in loader:
                scores = scores.to(device); labels = labels.to(device); mask = mask.to(device)
                logits = model(scores)
                valid = mask & (labels >= 0)
                probs_flat.append(torch.sigmoid(logits)[valid].cpu().numpy())
                y_flat.append(labels[valid].cpu().numpy())
        pp = np.concatenate(probs_flat); yy = np.concatenate(y_flat)
        results[name] = metrics_from_flat(pp, yy)

    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  [{config}] DONE. val_AP={results['val']['ap']:.4f} test_ood_AP={results['test_ood']['ap']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-id", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--out-root", default="/data/muskliu/npmaster/experiments/evo2_first_principles/decoder_v1/runs")
    p.add_argument("--data", default="/data/muskliu/npmaster/experiments/evo2_first_principles/decoder_v1/decoder_data.pkl")
    p.add_argument("--max-epochs", type=int, default=40)
    args = p.parse_args()

    grid = make_grid()
    config = grid[args.config_id]
    tag = f"c{args.config_id:02d}_{config['arch']}_h{config['hidden']}_{config['label_src']}_{config['loss_type']}"
    out_dir = Path(args.out_root) / tag
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    # seed
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"[config_id={args.config_id} gpu={args.gpu} device={device}] tag={tag}")
    train_one(config, out_dir, args.data, device, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
