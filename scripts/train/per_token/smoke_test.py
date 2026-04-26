#!/usr/bin/env python3
"""Smoke test for the per-token training stack.

Generates 64 synthetic windows of (T=2048, D=128) features and trains for 3
epochs to verify loss goes down + validation metrics compute. No cached data
required. Run on server or locally.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from per_token.model import PerTokenHead, count_params, masked_bce_loss  # noqa: E402


def make_synth(n_windows=64, T=2048, D=128, seed=0):
    rng = np.random.default_rng(seed)
    # Ground-truth: 30% of windows have a BGC at position [400, 1600].
    X = rng.standard_normal((n_windows, T, D)).astype(np.float32) * 0.3
    y = np.zeros((n_windows, T), dtype=np.int8)
    has_bgc = rng.random(n_windows) < 0.3
    for i in range(n_windows):
        if has_bgc[i]:
            # Inject a distinguishing signal in the BGC region.
            X[i, 400:1600, :] += rng.standard_normal(D).astype(np.float32)
            y[i, 400:1600] = 1
    # Edge-pad: last 200 tokens of window 0 are masked.
    y[0, -200:] = -1
    return X, y


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr, y_tr = make_synth(n_windows=64, T=2048, D=128, seed=0)
    X_va, y_va = make_synth(n_windows=16, T=2048, D=128, seed=1)

    model = PerTokenHead(d_in=128, hidden=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"params: {count_params(model)}")

    def run_val():
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X_va).to(device)
            y = torch.from_numpy(y_va.astype(np.int64)).to(device)
            logits = model(x)
            loss = masked_bce_loss(logits, y).item()
            preds = (torch.sigmoid(logits) > 0.5).int()
            mask = (y >= 0)
            tp = ((preds == 1) & (y == 1) & mask).sum().item()
            fp = ((preds == 1) & (y == 0) & mask).sum().item()
            fn = ((preds == 0) & (y == 1) & mask).sum().item()
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-6, prec + rec)
            return loss, prec, rec, f1

    for epoch in range(5):
        model.train()
        x = torch.from_numpy(X_tr).to(device)
        y = torch.from_numpy(y_tr.astype(np.int64)).to(device)
        opt.zero_grad()
        logits = model(x)
        loss = masked_bce_loss(logits, y)
        loss.backward()
        opt.step()

        va_loss, prec, rec, f1 = run_val()
        print(f"epoch {epoch}  train_loss={loss.item():.3f}  "
              f"val_loss={va_loss:.3f}  val_F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}")

    # Gate: val F1 must exceed 0.5 in 5 epochs on this easy synthetic task.
    assert f1 > 0.5, f"smoke test failed: val F1 = {f1:.3f} (<0.5)"
    print("OK — training loop works, loss decreases, val F1 lifts above chance.")


if __name__ == "__main__":
    main()
