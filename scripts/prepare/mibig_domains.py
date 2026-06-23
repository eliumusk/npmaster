#!/usr/bin/env python3
"""Extract per-BGC Pfam-domain hits + multi-label classes from a MIBiG release.

Output is a JSON "domains bundle" — the expensive HMM-scan result — that the
train script consumes. Vocabulary is deliberately NOT built here: it must be
fit on the train split only (in train.py) to avoid leakage.

Expects a MIBiG directory containing, per BGC, a `.gbk` (sequence/CDS) and a
`.json` (metadata with biosynthetic class). Pass `--hmm-db Pfam-A.hmm`.

Server invocation (node7, `conda activate bgc`):
  python scripts/prepare/mibig_domains.py \
      --mibig-dir /data/muskliu/npmaster/data/mibig_4.0 \
      --hmm-db    /data/muskliu/npmaster/data/pfam/Pfam-A.hmm \
      --out       experiments/evo2_mibig_type/mibig_domains.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classification.domains import proteins_from_genbank, scan_proteins_batch  # noqa: E402
from classification.mibig import (  # noqa: E402
    CANONICAL_CLASSES,
    multihot,
    parse_mibig_metadata,
)


def find_pairs(mibig_dir: Path) -> list[tuple[str, Path, Path]]:
    """Match {accession}.gbk with {accession}.json by stem."""
    gbks = {p.stem: p for p in mibig_dir.rglob("*.gbk")}
    jsons = {p.stem: p for p in mibig_dir.rglob("*.json")}
    pairs = []
    for stem in sorted(gbks.keys() & jsons.keys()):
        pairs.append((stem, gbks[stem], jsons[stem]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mibig-dir", type=Path, nargs="+", required=True,
                    help="one or more MIBiG dirs (e.g. 4.0 + a legacy dir for alkaloid)")
    ap.add_argument("--hmm-db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bit-cutoffs", default="gathering",
                    help="'gathering' (Pfam GA, default) or 'none' to use --e-value")
    ap.add_argument("--e-value", type=float, default=1e-5)
    args = ap.parse_args()

    pairs = []
    seen_stems: set[str] = set()
    for d in args.mibig_dir:
        for stem, gbk, js in find_pairs(d):
            if stem in seen_stems:   # earlier dirs win (4.0 before legacy)
                continue
            seen_stems.add(stem)
            pairs.append((stem, gbk, js))
    print(f"[scan] {len(pairs)} MIBiG entries with gbk+json", flush=True)
    bit_cutoffs = None if args.bit_cutoffs == "none" else args.bit_cutoffs

    # Pass 1: parse metadata + extract proteins; collect ALL proteins for one
    # batched HMM scan (re-reading full Pfam-A per BGC would be catastrophic).
    entries = []
    skipped = 0
    failed = 0
    named_proteins: list[tuple[str, str]] = []   # (uniq_name, seq)
    for i, (stem, gbk, js) in enumerate(pairs):
        meta = parse_mibig_metadata(js)
        if meta is None or not meta.classes:
            skipped += 1
            continue
        try:
            proteins = proteins_from_genbank(gbk)
        except Exception as exc:
            failed += 1
            print(f"[warn] {stem}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not proteins:
            failed += 1
            continue
        ei = len(entries)
        names = []
        for k, p in enumerate(proteins):
            nm = f"e{ei}_{k}"
            named_proteins.append((nm, p))
            names.append(nm)
        entries.append({
            "id": meta.accession,
            "organism": meta.organism,
            "classes": meta.classes,
            "label": multihot(meta.classes),
            "n_proteins": len(proteins),
            "_names": names,
            "hits": [],
        })
        if (i + 1) % 200 == 0:
            print(f"  parsed {i + 1}/{len(pairs)}", flush=True)

    # Pass 2: one batched scan over the whole protein set
    print(f"[hmm] scanning {len(named_proteins)} proteins from {len(entries)} BGCs "
          f"against {args.hmm_db.name} (one pass)...", flush=True)
    hits_by_name = scan_proteins_batch(
        named_proteins, args.hmm_db, bit_cutoffs=bit_cutoffs, e_value=args.e_value)

    for e in entries:
        hits = []
        for nm in e.pop("_names"):
            hits.extend(hits_by_name.get(nm, ()))
        e["hits"] = hits

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"classes": CANONICAL_CLASSES, "entries": entries}, indent=2))
    print(f"[done] {len(entries)} entries ({skipped} no-class, {failed} parse-failed) "
          f"→ {args.out}")

    # class distribution sanity print
    dist = {c: 0 for c in CANONICAL_CLASSES}
    for e in entries:
        for c in e["classes"]:
            dist[c] += 1
    print("[class distribution]")
    for c, n in dist.items():
        print(f"  {c:<12} {n}")


if __name__ == "__main__":
    main()
