#!/usr/bin/env python3
"""Classify antiSMASH region GenBanks with the domain-based type head, vs antiSMASH.

For a directory of antiSMASH results (one sub-dir per genome, each containing
`{contig}.region{NNN}.gbk` files), this:
  1. reads each region's CDS proteins,
  2. HMM-scans them against Pfam-A (batched),
  3. predicts the BGC type with the trained TypeHead,
  4. parses antiSMASH's own product call from the region GenBank,
and writes a per-region comparison CSV + prints an agreement summary that
highlights the PKS-vs-NRP cases.

  python scripts/classify/antismash_regions_type.py \
      --antismash-dir /data/syh/bench_34/antismash_out \
      --model experiments/evo2_mibig_type/run_full/type_head.joblib \
      --hmm-db /data/muskliu/npmaster/data/pfam/Pfam-A.hmm \
      --out-csv experiments/evo2_mibig_type/bench_34_typed.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classification.domains import proteins_from_genbank, scan_proteins_batch  # noqa: E402
from classification.type_head import TypeHead  # noqa: E402


def antismash_product(gbk_path: Path) -> str:
    """Pull antiSMASH's product call(s) from a region GenBank."""
    from Bio import SeqIO

    prods: list[str] = []
    try:
        for rec in SeqIO.parse(str(gbk_path), "genbank"):
            for feat in rec.features:
                if feat.type in ("region", "cand_cluster", "protocluster"):
                    for p in feat.qualifiers.get("product", []):
                        if p not in prods:
                            prods.append(p)
            if prods:
                break
    except Exception:
        return ""
    return "|".join(prods)


def as_canon(product: str) -> str:
    """Collapse an antiSMASH product string to a coarse class for comparison."""
    p = (product or "").lower()
    has_pks = "pks" in p
    has_nrp = "nrps" in p or "nrp" in p
    if has_pks and has_nrp:
        return "PKS+NRP"
    if has_pks:
        return "PKS"
    if has_nrp:
        return "NRP"
    if "terpene" in p:
        return "Terpene"
    if any(s in p for s in ("ripp", "lanthi", "lasso", "thiopeptide", "bacteriocin",
                            "lantipeptide", "sactipeptide", "lap", "ranthipeptide")):
        return "RiPP"
    if "saccharide" in p or "glycoc" in p:
        return "Saccharide"
    if not p:
        return ""
    return "Other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--antismash-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--hmm-db", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--bit-cutoffs", default="gathering")
    ap.add_argument("--e-value", type=float, default=1e-5)
    args = ap.parse_args()

    head = TypeHead.load(args.model)
    bit_cutoffs = None if args.bit_cutoffs == "none" else args.bit_cutoffs

    # collect all region gbks
    region_gbks = sorted(args.antismash_dir.rglob("*.region*.gbk"))
    print(f"[scan] {len(region_gbks)} antiSMASH region GenBanks", flush=True)

    # pass 1: proteins + antiSMASH product per region
    regions = []          # dicts
    named = []            # (name, seq)
    for ri, gbk in enumerate(region_gbks):
        genome = gbk.parent.name
        prods = antismash_product(gbk)
        try:
            prots = proteins_from_genbank(gbk)
        except Exception:
            prots = []
        names = []
        for k, p in enumerate(prots):
            nm = f"r{ri}_{k}"
            named.append((nm, p))
            names.append(nm)
        regions.append({"genome": genome, "region": gbk.name,
                        "as_product": prods, "as_canon": as_canon(prods),
                        "_names": names})
        if (ri + 1) % 200 == 0:
            print(f"  read {ri + 1}/{len(region_gbks)}", flush=True)

    # pass 2: one batched HMM scan
    print(f"[hmm] scanning {len(named)} proteins against {args.hmm_db.name}...", flush=True)
    hits_by_name = scan_proteins_batch(named, args.hmm_db,
                                       bit_cutoffs=bit_cutoffs, e_value=args.e_value)

    # pass 3: predict per region
    pks_i, nrp_i = head.classes.index("Polyketide"), head.classes.index("NRP")
    for r in regions:
        hits = []
        for nm in r.pop("_names"):
            hits.extend(hits_by_name.get(nm, ()))
        P = head.predict_from_hits([hits])[0]
        r["our_type"] = head.classes[int(np.argmax(P))]
        r["our_multi"] = "+".join(head.classes[j] for j in range(len(P)) if P[j] >= 0.5) or "-"
        r["pks_p"] = round(float(P[pks_i]), 3)
        r["nrp_p"] = round(float(P[nrp_i]), 3)
        r["n_hits"] = len(hits)

    # write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["genome", "region", "as_product", "as_canon", "our_type", "our_multi",
            "pks_p", "nrp_p", "n_hits"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(regions)
    print(f"[done] {len(regions)} regions → {args.out_csv}")

    # ---- comparison summary ----
    print("\n=== antiSMASH coarse-class vs our type (agreement on coarse classes) ===")
    AS = ["PKS", "NRP", "PKS+NRP", "Terpene", "RiPP", "Saccharide", "Other"]
    def our_coarse(t):
        return "PKS" if t == "Polyketide" else "NRP" if t == "NRP" else t
    agree = tot = 0
    for asc in AS:
        sub = [r for r in regions if r["as_canon"] == asc]
        if not sub:
            continue
        cnt = Counter(our_coarse(r["our_type"]) for r in sub)
        # 'agreement' lenient for hybrids
        ok = sum(n for k, n in cnt.items()
                 if k == asc or (asc == "PKS+NRP" and k in ("PKS", "NRP")))
        agree += ok; tot += len(sub)
        print(f"  antiSMASH={asc:<9} n={len(sub):<4} our: {dict(cnt)}")
    print(f"\ncoarse agreement: {agree}/{tot} ({agree/max(1,tot)*100:.0f}%)")

    # PKS focus: of antiSMASH-PKS regions, how many do we call NRP (the old bug)?
    pks_regions = [r for r in regions if r["as_canon"] == "PKS"]
    bad = [r for r in pks_regions if r["our_type"] == "NRP"]
    print(f"\n=== PKS focus ===")
    print(f"antiSMASH 纯 PKS 区域: {len(pks_regions)}; 我们判成 NRP 的: {len(bad)} "
          f"({len(bad)/max(1,len(pks_regions))*100:.0f}%)  [旧分类器这里是 18%]")


if __name__ == "__main__":
    main()
