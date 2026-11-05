#!/usr/bin/env python3
"""Train the multi-label product-type head on a MIBiG domains bundle.

Pipeline:
  1. Load the domains bundle (per-BGC domain hits + multi-hot labels).
  2. Family-disjoint split (mmseqs cluster TSV if given, else id-identity with a
     leakage warning).
  3. Build the domain vocabulary on the TRAIN split only (min document freq).
  4. Vectorize, fit RandomForest TypeHead, evaluate on val (+ test).
  5. Save model (.joblib, bundles classes+vocab), metrics.json, split.json.

Server invocation:
  python scripts/train/type_head/train.py \
      --bundle experiments/evo2_mibig_type/mibig_domains.json \
      --mmseqs-tsv experiments/evo2_mibig_type/families_cluster.tsv \
      --out-dir experiments/evo2_mibig_type/run01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classification.domains import build_vocab, stack_vectors  # noqa: E402
from classification.metrics import (  # noqa: E402
    format_confusion,
    multilabel_report,
    single_label_confusion,
)
from classification.splits import (  # noqa: E402
    family_from_mmseqs,
    group_split,
    identity_families,
)
from classification.type_head import TypeHead  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--mmseqs-tsv", type=Path, default=None,
                    help="mmseqs easy-cluster TSV for family-disjoint split")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-domain-count", type=int, default=3)
    ap.add_argument("--binary-features", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bundle = json.loads(args.bundle.read_text())
    classes = bundle["classes"]
    entries = bundle["entries"]
    ids = [e["id"] for e in entries]
    by_id = {e["id"]: e for e in entries}
    print(f"[load] {len(entries)} entries, {len(classes)} classes", flush=True)

    # 2. family-disjoint split
    if args.mmseqs_tsv:
        family_of = family_from_mmseqs(args.mmseqs_tsv)
        print(f"[split] family map from {args.mmseqs_tsv}", flush=True)
    else:
        family_of = identity_families(ids)
        print("[split] WARNING: no mmseqs TSV → id-identity families; "
              "metrics will be leakage-optimistic", flush=True)
    split = group_split(ids, family_of, fractions=tuple(args.fractions), seed=args.seed)
    print(f"[split] train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])}", flush=True)

    # 3. vocab on TRAIN only
    train_hits = [by_id[i]["hits"] for i in split["train"]]
    vocab = build_vocab(train_hits, min_count=args.min_domain_count)
    print(f"[vocab] {len(vocab)} domains (min_count={args.min_domain_count})", flush=True)

    def xy(split_ids):
        hits = [by_id[i]["hits"] for i in split_ids]
        X = stack_vectors(hits, vocab, binary=args.binary_features)
        Y = np.array([by_id[i]["label"] for i in split_ids], dtype=int)
        return X, Y

    X_tr, Y_tr = xy(split["train"])
    X_va, Y_va = xy(split["val"])
    X_te, Y_te = xy(split["test"])

    # 4. fit
    head = TypeHead(
        classes=classes, vocab=vocab, binary_features=args.binary_features,
        n_estimators=args.n_estimators, max_depth=args.max_depth, seed=args.seed,
    ).fit(X_tr, Y_tr)
    print(f"[fit] RandomForest on {X_tr.shape[0]}×{X_tr.shape[1]}", flush=True)

    metrics = {}
    for name, (X, Y) in {"val": (X_va, Y_va), "test": (X_te, Y_te)}.items():
        if X.shape[0] == 0:
            continue
        P = head.predict_proba(X)
        rep = multilabel_report(Y, P, classes)
        conf = single_label_confusion(Y, P, classes)
        metrics[name] = {"report": rep, "confusion": conf}
        print(f"\n==== {name} ====")
        print(f"macro-F1={rep['macro_f1']}  micro-F1={rep['micro_f1']}  "
              f"macro-AUROC={rep['macro_auroc']}")
        print(format_confusion(conf))

    # 5. persist
    args.out_dir.mkdir(parents=True, exist_ok=True)
    head.save(args.out_dir / "type_head.joblib")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "split.json").write_text(json.dumps(split, indent=2))
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"\n[done] model + metrics → {args.out_dir}")


if __name__ == "__main__":
    main()
