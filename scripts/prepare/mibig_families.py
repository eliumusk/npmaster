#!/usr/bin/env python3
"""Build a family-disjoint split key for a domains bundle via mmseqs2.

Writes one representative protein per BGC (its longest CDS) to a FASTA, runs
`mmseqs easy-cluster`, and emits the member→representative TSV that train.py
consumes (`--mmseqs-tsv`) to keep homologous clusters out of separate splits.

Needs `mmseqs` on PATH and the MIBiG GenBanks (4.0 + any legacy dir used for
alkaloid). Proteins are looked up by `{accession}.gbk` across the given dirs.

  python scripts/prepare/mibig_families.py \
      --bundle experiments/evo2_mibig_type/bundle.json \
      --mibig-dir data/mibig_4.0 data/mibig_3.1 \
      --out-tsv experiments/evo2_mibig_type/families_cluster.tsv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classification.domains import proteins_from_genbank  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--mibig-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument("--min-seq-id", type=float, default=0.5)
    ap.add_argument("--coverage", type=float, default=0.7)
    args = ap.parse_args()

    gbks: dict[str, Path] = {}
    for d in args.mibig_dir:
        for p in d.rglob("*.gbk"):
            gbks.setdefault(p.stem, p)

    ids = [e["id"] for e in json.loads(args.bundle.read_text())["entries"]]
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fasta = args.out_tsv.with_suffix(".rep.fasta")
    n = 0
    with open(fasta, "w") as f:
        for acc in ids:
            g = gbks.get(acc)
            if not g:
                continue
            try:
                prots = proteins_from_genbank(g)
            except Exception:
                prots = []
            if not prots:
                continue
            f.write(f">{acc}\n{max(prots, key=len)}\n")
            n += 1
    print(f"[fasta] {n} representative proteins → {fasta}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(args.out_tsv.with_suffix(""))
        cmd = ["mmseqs", "easy-cluster", str(fasta), prefix, tmp,
               "--min-seq-id", str(args.min_seq_id), "-c", str(args.coverage),
               "--cov-mode", "0", "-v", "1"]
        print(f"[mmseqs] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    # mmseqs writes {prefix}_cluster.tsv
    produced = Path(f"{prefix}_cluster.tsv")
    if produced != args.out_tsv:
        produced.replace(args.out_tsv)
    n_fam = len({line.split("\t")[0] for line in args.out_tsv.read_text().splitlines() if line})
    print(f"[done] {n_fam} families → {args.out_tsv}")


if __name__ == "__main__":
    main()
