#!/usr/bin/env python3
"""
Stage 1.1: Filter GTDB bac120 metadata and stratified sample ~2000 genomes.

Filters:
  - checkm_completeness >= 95
  - checkm_contamination <= 5
  - assembly_level in {Complete Genome, Scaffold}  (exclude Contig)
  - one representative per species (highest completeness - contamination)

Phylum strata (targets):
  Actinomycetota   ~600
  Pseudomonadota   ~500
  Bacillota        ~500
  Bacteroidota     ~200   (test_ood)
  Cyanobacteriota  ~100   (test_ood)
  other phyla      ~100
"""
import argparse
import gzip
import csv
import sys
import random
from collections import defaultdict, Counter
from pathlib import Path

PHYLUM_TARGETS = {
    "p__Actinomycetota":   600,
    "p__Pseudomonadota":   500,
    "p__Bacillota":        500,
    "p__Bacteroidota":     200,
    "p__Cyanobacteriota":  100,
}
OTHER_TARGET = 100

def parse_taxonomy(gtdb_tax):
    # d__Bacteria;p__Actinomycetota;c__Actinomycetia;o__...;f__...;g__...;s__...
    parts = gtdb_tax.split(";")
    out = {"domain":"","phylum":"","class":"","order":"","family":"","genus":"","species":""}
    keys = ["domain","phylum","class","order","family","genus","species"]
    for p, k in zip(parts, keys):
        out[k] = p.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="bac120_metadata.tsv(.gz)")
    ap.add_argument("--out",  required=True, help="output selected genomes TSV")
    ap.add_argument("--report", required=True, help="output filter report JSON")
    ap.add_argument("--seed", type=int, default=20260405)
    args = ap.parse_args()

    random.seed(args.seed)

    opener = gzip.open if args.meta.endswith(".gz") else open
    mode = "rt"

    # discover columns
    with opener(args.meta, mode) as f:
        header = f.readline().rstrip("\n").split("\t")
    col = {name:i for i,name in enumerate(header)}
    need = ["accession","gtdb_taxonomy","gtdb_representative",
            "checkm_completeness","checkm_contamination",
            "ncbi_assembly_level","genome_size",
            "contig_count","ncbi_species_taxid",
            "gtdb_genome_representative"]
    missing = [n for n in need if n not in col]
    if missing:
        print(f"MISSING COLUMNS: {missing}", file=sys.stderr)
        print(f"Available columns sample: {header[:40]}", file=sys.stderr)
        sys.exit(1)

    stats = Counter()
    # species -> best record (dict)
    per_species_best = {}

    with opener(args.meta, mode) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            stats["total"] += 1
            try:
                comp = float(parts[col["checkm_completeness"]])
                cont = float(parts[col["checkm_contamination"]])
            except Exception:
                stats["bad_qc_numeric"] += 1
                continue
            if comp < 95.0:
                stats["fail_completeness"] += 1; continue
            if cont > 5.0:
                stats["fail_contamination"] += 1; continue
            lvl = parts[col["ncbi_assembly_level"]].strip()
            # levels: "Complete Genome","Chromosome","Scaffold","Contig"
            if lvl not in ("Complete Genome","Chromosome","Scaffold"):
                stats["fail_assembly_level"] += 1; continue

            tax = parse_taxonomy(parts[col["gtdb_taxonomy"]])
            species = tax["species"]
            if not species or species == "s__":
                stats["no_species"] += 1; continue

            score = comp - 5.0*cont
            rec = {
                "accession": parts[col["accession"]],
                "completeness": comp,
                "contamination": cont,
                "assembly_level": lvl,
                "genome_size": parts[col["genome_size"]],
                "n_contigs": parts[col["contig_count"]],
                "score": score,
                **tax,
            }
            cur = per_species_best.get(species)
            if cur is None or rec["score"] > cur["score"]:
                per_species_best[species] = rec

    stats["unique_species_passed"] = len(per_species_best)

    # bucket by phylum
    by_phy = defaultdict(list)
    for rec in per_species_best.values():
        by_phy[rec["phylum"]].append(rec)

    # stratified sampling
    selected = []
    phylum_counts_pre = {p:len(v) for p,v in by_phy.items()}
    for phy, tgt in PHYLUM_TARGETS.items():
        pool = by_phy.get(phy, [])
        random.shuffle(pool)
        take = pool[:tgt]
        selected.extend(take)
        stats[f"sampled_{phy}"] = len(take)

    # "other" bucket: everything not in PHYLUM_TARGETS
    others = []
    for phy, recs in by_phy.items():
        if phy in PHYLUM_TARGETS: continue
        others.extend(recs)
    random.shuffle(others)
    selected.extend(others[:OTHER_TARGET])
    stats["sampled_other"] = min(OTHER_TARGET, len(others))

    # write selected genomes
    out_cols = ["accession","species","phylum","class","order","family","genus",
                "genome_size_bp","n_contigs","checkm_completeness","checkm_contamination",
                "assembly_level","ncbi_accession"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(out_cols)
        for rec in selected:
            ncbi_acc = rec["accession"]
            if ncbi_acc.startswith(("RS_","GB_")):
                ncbi_acc = ncbi_acc[3:]
            w.writerow([
                rec["accession"], rec["species"], rec["phylum"], rec["class"],
                rec["order"], rec["family"], rec["genus"],
                rec["genome_size"], rec["n_contigs"],
                f'{rec["completeness"]:.2f}', f'{rec["contamination"]:.2f}',
                rec["assembly_level"], ncbi_acc,
            ])

    # write report
    import json
    report = {
        "version": "GTDB v226",
        "total_rows": stats["total"],
        "filtered_out": {
            "bad_qc_numeric": stats["bad_qc_numeric"],
            "fail_completeness(<95)": stats["fail_completeness"],
            "fail_contamination(>5)": stats["fail_contamination"],
            "fail_assembly_level(Contig)": stats["fail_assembly_level"],
            "no_species": stats["no_species"],
        },
        "unique_species_passed_qc": stats["unique_species_passed"],
        "phylum_pool_sizes_after_qc": phylum_counts_pre,
        "sampled": {k.replace("sampled_",""): v for k,v in stats.items() if k.startswith("sampled_")},
        "total_selected": len(selected),
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # console summary
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
