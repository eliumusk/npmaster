#!/usr/bin/env python3
"""
Stage 1.2 (post-antiSMASH): parse all antiSMASH JSONs → bgc_annotations.csv

Schema:
  bgc_id, genome_id, contig_id, start_bp, end_bp, bgc_length,
  bgc_types (semicolon-joined 7-class labels),
  bgc_product (semicolon-joined raw antiSMASH product types),
  is_truncated (bool, from contig_edge),
  phylum

Type mapping (antiSMASH 8 → 7-class):
  PKS:     T1PKS,T2PKS,T3PKS,transAT-PKS,hglE-KS,HR-T2PKS,arylpolyene,PKS-like
  NRP:     NRPS,NRPS-like,NAPAA,CDPS,NRP-metallophore
  Terpene: terpene,terpene-precursor
  RiPP:    lanthipeptide-class-*,RiPP-like,lassopeptide,sactipeptide,bottromycin,cyanobactin,
           LAP,microcin,proteusin,ranthipeptide,head_to_tail,lipolanthine,linaridin,glycocin,
           azole-containing-RiPP,epipeptide,triceptide,thioamitides,atropopeptide,darobactin,
           guanidinotides,spliceotide,crocagin,RRE-containing,RaS-RiPP,ribosomal,lanthidin,thiopeptide
  Saccharide: saccharide,oligosaccharide,amglyccycl
  Alkaloid:   indole,alkaloid
  Other:      everything else (siderophore,ectoine,phenazine,melanin,betalactone, ...)
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PKS = {"T1PKS","T2PKS","T3PKS","transAT-PKS","hglE-KS","HR-T2PKS","arylpolyene","PKS-like",
       "hr_t2pks","ladderane","resorcinol","PpyS-KS","PUFA","transAT-PKS-like"}
NRP = {"NRPS","NRPS-like","NAPAA","CDPS","NRP-metallophore","thioamide-NRP"}
TERPENE = {"terpene","terpene-precursor"}
RIPP_PREFIXES = ("lanthipeptide-class",)
RIPP = {"RiPP-like","lassopeptide","sactipeptide","bottromycin","cyanobactin","LAP","microcin",
        "proteusin","ranthipeptide","head_to_tail","lipolanthine","linaridin","glycocin",
        "azole-containing-RiPP","epipeptide","triceptide","thioamitides","atropopeptide",
        "darobactin","guanidinotides","spliceotide","crocagin","RRE-containing","RaS-RiPP",
        "ribosomal","lanthidin","thiopeptide","mycosporine-like","spliceotide",
        "lanthipeptide","lanthipeptide-class-i","lanthipeptide-class-ii",
        "lanthipeptide-class-iii","lanthipeptide-class-iv","lanthipeptide-class-v"}
SACCHARIDE = {"saccharide","oligosaccharide","amglyccycl"}
ALKALOID = {"indole","alkaloid"}
# all else -> Other

def map7(product):
    if product in PKS: return "Polyketide"
    if product in NRP: return "NRP"
    if product in TERPENE: return "Terpene"
    if product in RIPP or product.startswith(RIPP_PREFIXES): return "RiPP"
    if product in SACCHARIDE: return "Saccharide"
    if product in ALKALOID: return "Alkaloid"
    return "Other"

def parse_location(loc):
    # "[101866:143012](+)" or "[<0:5000](+)" or compound
    nums = re.findall(r"(\d+)", loc)
    if len(nums) < 2: return None, None
    return int(nums[0]), int(nums[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--antismash-dir", required=True)
    ap.add_argument("--selected", required=True, help="selected_genomes.tsv (for phylum)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    # load phylum lookup
    phy_of = {}
    with open(args.selected) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            phy_of[row["ncbi_accession"]] = row["phylum"]

    rows = []
    per_genome = defaultdict(int)
    products_raw = Counter()
    class7_count = Counter()
    multi = 0
    failed = []
    all_genomes = sorted(os.listdir(args.antismash_dir))
    all_genomes = [g for g in all_genomes if os.path.isdir(os.path.join(args.antismash_dir, g))]
    for g in all_genomes:
        jp = os.path.join(args.antismash_dir, g, f"{g}.json")
        if not os.path.exists(jp):
            failed.append(g)
            continue
        try:
            d = json.load(open(jp))
        except Exception as e:
            failed.append(g)
            continue
        phylum = phy_of.get(g, "")
        reg_idx = 0
        for rec in d.get("records", []):
            contig_id = rec["id"]
            for feat in rec.get("features", []):
                if feat.get("type") != "region": continue
                q = feat.get("qualifiers", {})
                products = q.get("product", [])
                if not products: continue
                s, e = parse_location(feat.get("location",""))
                if s is None: continue
                edge = q.get("contig_edge", ["False"])[0] == "True"
                classes = sorted(set(map7(p) for p in products))
                reg_idx += 1
                bgc_id = f"{g}_BGC_{reg_idx:03d}"
                rows.append({
                    "bgc_id": bgc_id, "genome_id": g, "contig_id": contig_id,
                    "start_bp": s, "end_bp": e, "bgc_length": e-s,
                    "bgc_types": ";".join(classes),
                    "bgc_product": ";".join(products),
                    "is_truncated": "1" if edge else "0",
                    "phylum": phylum,
                })
                per_genome[g] += 1
                for p in products: products_raw[p] += 1
                for c in classes: class7_count[c] += 1
                if len(classes) > 1: multi += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bgc_id","genome_id","contig_id","start_bp","end_bp",
            "bgc_length","bgc_types","bgc_product","is_truncated","phylum"])
        w.writeheader()
        for r in rows: w.writerow(r)

    # summary
    from statistics import median
    lengths = [r["bgc_length"] for r in rows]
    lengths_by_class = defaultdict(list)
    for r in rows:
        for c in r["bgc_types"].split(";"):
            lengths_by_class[c].append(r["bgc_length"])
    truncated_n = sum(1 for r in rows if r["is_truncated"]=="1")

    # per-phylum density
    genomes_per_phylum = Counter()
    bgcs_per_phylum = Counter()
    for g, phy in phy_of.items():
        if os.path.isdir(os.path.join(args.antismash_dir, g)):
            genomes_per_phylum[phy] += 1
    for r in rows:
        bgcs_per_phylum[r["phylum"]] += 1

    combo_counter = Counter()
    for r in rows:
        if ";" in r["bgc_types"]:
            combo_counter[r["bgc_types"]] += 1

    summary = {
        "n_genomes_processed": len(all_genomes),
        "n_genomes_failed": len(failed),
        "failed_genomes": failed[:50],
        "n_bgcs_total": len(rows),
        "n_bgcs_truncated": truncated_n,
        "pct_truncated": round(100*truncated_n/max(1,len(rows)), 2),
        "n_bgcs_multilabel": multi,
        "pct_multilabel": round(100*multi/max(1,len(rows)), 2),
        "class7_distribution": dict(class7_count),
        "top_multilabel_combos": dict(combo_counter.most_common(10)),
        "raw_product_distribution": dict(products_raw.most_common()),
        "bgc_length_overall": {
            "min": min(lengths) if lengths else 0,
            "median": int(median(lengths)) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": int(sum(lengths)/len(lengths)) if lengths else 0,
        },
        "bgc_length_by_class": {
            c: {"n":len(L), "min":min(L), "median":int(median(L)), "max":max(L)}
            for c,L in lengths_by_class.items()
        },
        "bgcs_per_genome": {
            "min": min(per_genome.values()) if per_genome else 0,
            "median": int(median(per_genome.values())) if per_genome else 0,
            "max": max(per_genome.values()) if per_genome else 0,
            "mean": round(sum(per_genome.values())/max(1,len(per_genome)), 2),
        },
        "bgc_density_by_phylum": {
            phy: round(bgcs_per_phylum[phy]/genomes_per_phylum[phy], 2)
            for phy in genomes_per_phylum if genomes_per_phylum[phy]>0
        },
        "genomes_per_phylum": dict(genomes_per_phylum),
        "bgcs_per_phylum": dict(bgcs_per_phylum),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # print key parts
    print(json.dumps({k:v for k,v in summary.items() if k not in ("raw_product_distribution","bgc_length_by_class","failed_genomes")}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
