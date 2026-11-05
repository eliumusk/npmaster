#!/usr/bin/env python3
"""Augment a MIBiG-4.0 domains bundle with Alkaloid labels from a legacy release.

MIBiG 4.0 dropped the structure-based `Alkaloid` class, so the canonical 7-class
head has zero alkaloid training data. Older MIBiG (2.0/3.1) kept Alkaloid; this
script seeds it the same way DeepBGC/SanntiS did.

For each legacy entry whose biosynthetic class includes Alkaloid:
  - if the accession is already in the bundle (kept in 4.0 under a different
    class), ADD 'Alkaloid' to its multi-label (e.g. NRP+Alkaloid) — no rescan;
  - if the accession is gone from 4.0, extract its legacy proteins, HMM-scan
    them, and add a fresh Alkaloid entry.

  python scripts/prepare/add_alkaloid_legacy.py \
      --bundle experiments/evo2_mibig_type/mibig_domains.json \
      --legacy-dir data/mibig_3.1 --hmm-db data/pfam/Pfam-A.hmm \
      --out experiments/evo2_mibig_type/mibig_domains_with_alkaloid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classification.domains import proteins_from_genbank, scan_proteins_batch  # noqa: E402
from classification.mibig import multihot, parse_mibig_metadata  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--legacy-dir", type=Path, required=True)
    ap.add_argument("--hmm-db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bit-cutoffs", default="gathering")
    ap.add_argument("--e-value", type=float, default=1e-5)
    args = ap.parse_args()

    bundle = json.loads(args.bundle.read_text())
    classes = bundle["classes"]
    by_id = {e["id"]: e for e in bundle["entries"]}
    bit_cutoffs = None if args.bit_cutoffs == "none" else args.bit_cutoffs

    # find legacy alkaloid entries
    gbks = {p.stem: p for p in args.legacy_dir.rglob("*.gbk")}
    jsons = {p.stem: p for p in args.legacy_dir.rglob("*.json")}
    alk_in_bundle = 0
    new_pairs = []  # (accession, gbk)
    for stem in sorted(gbks.keys() & jsons.keys()):
        meta = parse_mibig_metadata(jsons[stem])
        if meta is None or "Alkaloid" not in meta.classes:
            continue
        if meta.accession in by_id:
            e = by_id[meta.accession]
            if "Alkaloid" not in e["classes"]:
                e["classes"].append("Alkaloid")
                e["label"] = multihot(e["classes"])
                alk_in_bundle += 1
        else:
            new_pairs.append((meta.accession, gbks[stem]))

    print(f"[legacy] {alk_in_bundle} existing entries gained Alkaloid label; "
          f"{len(new_pairs)} alkaloid-only entries to add", flush=True)

    # scan proteins for the brand-new alkaloid entries (one batched pass)
    named = []
    new_entries = []
    for ai, (acc, gbk) in enumerate(new_pairs):
        try:
            prots = proteins_from_genbank(gbk)
        except Exception:
            prots = []
        if not prots:
            continue
        names = []
        for k, p in enumerate(prots):
            nm = f"alk{ai}_{k}"
            named.append((nm, p))
            names.append(nm)
        new_entries.append({"id": acc, "organism": "", "classes": ["Alkaloid"],
                            "label": multihot(["Alkaloid"]), "n_proteins": len(prots),
                            "_names": names, "hits": []})
    if named:
        hits_by_name = scan_proteins_batch(named, args.hmm_db,
                                           bit_cutoffs=bit_cutoffs, e_value=args.e_value)
        for e in new_entries:
            hits = []
            for nm in e.pop("_names"):
                hits.extend(hits_by_name.get(nm, ()))
            e["hits"] = hits
        bundle["entries"].extend(new_entries)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2))
    dist = {c: 0 for c in classes}
    for e in bundle["entries"]:
        for c in e["classes"]:
            dist[c] = dist.get(c, 0) + 1
    print(f"[done] {len(bundle['entries'])} entries → {args.out}")
    print("[class distribution]")
    for c in classes:
        print(f"  {c:<12} {dist.get(c,0)}")


if __name__ == "__main__":
    main()
