#!/usr/bin/env python3
"""Evaluate a trained product-type head on a domains bundle.

Reusable for: the held-out MIBiG test split, a 9-genome report-only bundle, or
the T1PKS regression smoke test. Loads the model (which carries its own vocab),
vectorizes the bundle, and prints macro-F1 + per-class P/R + the single-label
confusion matrix.

  python scripts/evaluate/type_head_mibig.py \
      --model experiments/evo2_mibig_type/run01/type_head.joblib \
      --bundle experiments/evo2_mibig_type/mibig_domains.json \
      --ids experiments/evo2_mibig_type/run01/split.json:test \
      --out-json experiments/evo2_mibig_type/run01/eval_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classification.metrics import (  # noqa: E402
    format_confusion,
    multilabel_report,
    single_label_confusion,
)
from classification.type_head import TypeHead  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--ids", default=None,
                    help="optional 'split.json:test' to restrict to a split's ids")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    head = TypeHead.load(args.model)
    bundle = json.loads(args.bundle.read_text())
    by_id = {e["id"]: e for e in bundle["entries"]}

    ids = list(by_id)
    if args.ids:
        sp_path, _, key = args.ids.partition(":")
        ids = [i for i in json.loads(Path(sp_path).read_text())[key] if i in by_id]

    hits = [by_id[i]["hits"] for i in ids]
    Y = np.array([by_id[i]["label"] for i in ids], dtype=int)
    P = head.predict_from_hits(hits)

    rep = multilabel_report(Y, P, head.classes)
    conf = single_label_confusion(Y, P, head.classes)
    print(f"[eval] n={len(ids)}")
    print(f"macro-F1={rep['macro_f1']}  micro-F1={rep['micro_f1']}  "
          f"macro-AUROC={rep['macro_auroc']}")
    for c, m in rep["per_class"].items():
        print(f"  {c:<12} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} n={m['support']}")
    print(format_confusion(conf))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(
            {"n": len(ids), "report": rep, "confusion": conf}, indent=2))
        print(f"[done] → {args.out_json}")


if __name__ == "__main__":
    main()
