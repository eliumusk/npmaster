#!/usr/bin/env python3
"""Intersect v4.1 OER regions with antiSMASH regions -> 'antiSMASH-missed' novel list.

Inputs:
  --regions-csv   v4.1 regions CSV with columns (genome, contig, start, end, max_score, ...);
                  the ``contig`` field may be prefixed like ``OER00425668|k141_xxx`` —
                  script strips the prefix for matching antiSMASH contig ids.
  --antismash-dir per-sample antiSMASH output dirs with ``<sample>.json``
                  (default layout: OER00425668.ge10000/OER00425668.ge10000.json)
  --out-dir       output dir; writes candidates_novel.csv and antismash_regions_all.tsv

A v4.1 region is "novel" (antiSMASH-missed) if no antiSMASH region on the same contig
has any position overlap (even 1 bp).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_antismash_regions(json_path):
    """Yield (contig, start_0based, end_0based, product)."""
    with open(json_path) as fh:
        data = json.load(fh)
    for rec in data.get("records", []):
        contig = rec.get("id") or rec.get("name")
        if not contig:
            continue
        for feat in rec.get("features", []):
            if feat.get("type") != "region":
                continue
            loc = feat.get("location") or ""
            m = re.search(r"\[(\d+):[><]?(\d+)\]", loc)
            if not m:
                continue
            product = feat.get("qualifiers", {}).get("product", ["?"])[0]
            yield contig, int(m.group(1)), int(m.group(2)), product


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-csv", type=Path, required=True)
    p.add_argument("--antismash-dir", type=Path, required=True,
                   help="dir of per-sample antiSMASH outputs (one subdir per sample)")
    p.add_argument("--sample-dir-glob", default="*.ge10000",
                   help="glob matching antiSMASH sample subdirs")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] parse antiSMASH regions")
    contig_to_regions: dict[str, list[tuple]] = defaultdict(list)
    all_rows: list[tuple] = []
    for sample_dir in sorted(args.antismash_dir.glob(args.sample_dir_glob)):
        sample_id = sample_dir.name.split(".", 1)[0]
        json_files = list(sample_dir.glob("*.json"))
        if not json_files:
            continue
        n = 0
        for contig, s, e, product in parse_antismash_regions(json_files[0]):
            contig_to_regions[contig].append((s, e, product, sample_id))
            all_rows.append((sample_id, contig, s, e, product))
            n += 1
        print(f"  {sample_id}: {n} regions")
    print(f"  total antiSMASH regions: {len(all_rows)}")

    with open(args.out_dir / "antismash_regions_all.tsv", "w") as fh:
        fh.write("sample\tcontig\tstart\tend\tproduct\n")
        for row in all_rows:
            fh.write("\t".join(str(x) for x in row) + "\n")

    print("\n[2] load v4.1 regions")
    v41_rows = list(csv.DictReader(open(args.regions_csv)))
    print(f"  {len(v41_rows)} v4.1 regions")

    print("\n[3] intersect")
    novel = []
    overlap_n = 0
    for row in v41_rows:
        full_contig = row["contig"]
        if "|" in full_contig:
            sample_prefix, contig_short = full_contig.split("|", 1)
        else:
            sample_prefix, contig_short = "", full_contig

        s_v, e_v = int(row["start"]), int(row["end"])
        cands = contig_to_regions.get(contig_short, [])
        overlap_any = False
        min_dist = None; nearest = None
        for (s_a, e_a, product, sample_id) in cands:
            ov = min(e_v, e_a) - max(s_v, s_a)
            if ov > 0:
                overlap_any = True; break
            d = (s_a - e_v) if e_v <= s_a else (s_v - e_a) if e_a <= s_v else 0
            if min_dist is None or d < min_dist:
                min_dist = d; nearest = (s_a, e_a, product, sample_id)

        if overlap_any:
            overlap_n += 1; continue

        entry = dict(row)
        entry["_contig_short"] = contig_short
        entry["_sample_prefix"] = sample_prefix
        entry["_length"] = e_v - s_v
        entry["_n_antismash_on_contig"] = len(cands)
        if nearest:
            sa, ea, prod, samp = nearest
            entry["_nearest_antismash_start"] = sa
            entry["_nearest_antismash_end"] = ea
            entry["_nearest_antismash_product"] = prod
            entry["_nearest_antismash_dist"] = min_dist
        else:
            entry["_nearest_antismash_start"] = ""
            entry["_nearest_antismash_end"] = ""
            entry["_nearest_antismash_product"] = ""
            entry["_nearest_antismash_dist"] = ""
        novel.append(entry)

    print(f"  overlapping: {overlap_n}")
    print(f"  novel (antiSMASH-missed): {len(novel)}")

    out_csv = args.out_dir / "candidates_novel.csv"
    if novel:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(novel[0].keys()))
            w.writeheader()
            w.writerows(novel)
        print(f"\n[write] {out_csv}")


if __name__ == "__main__":
    main()
