#!/usr/bin/env python3
"""Build genome_metadata.csv per stage1_detailed.md schema."""
import argparse, csv, os, json
from statistics import median
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--genomes-dir", required=True)
    ap.add_argument("--withdrawn", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    withdrawn = set()
    if args.withdrawn and os.path.exists(args.withdrawn):
        withdrawn = set(open(args.withdrawn).read().split())

    rows = []
    sizes_by_phy = defaultdict(list)
    phy_count = defaultdict(int)
    with open(args.selected) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            acc = row["ncbi_accession"]
            fpath = os.path.join(args.genomes_dir, f"{acc}.fna.gz")
            if acc in withdrawn or not os.path.exists(fpath):
                continue
            # local size
            size = int(row["genome_size_bp"])
            sizes_by_phy[row["phylum"]].append(size)
            phy_count[row["phylum"]] += 1
            rows.append({
                "genome_id": acc,
                "species": row["species"],
                "phylum": row["phylum"],
                "class": row["class"],
                "order": row["order"],
                "family": row["family"],
                "genus": row["genus"],
                "genome_size_bp": row["genome_size_bp"],
                "n_contigs": row["n_contigs"],
                "checkm_completeness": row["checkm_completeness"],
                "checkm_contamination": row["checkm_contamination"],
                "assembly_level": row["assembly_level"],
                "ncbi_accession": acc,
                "fasta_path": fpath,
            })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)

    report = {
        "n_genomes_final": len(rows),
        "n_withdrawn": len(withdrawn),
        "phylum_counts": dict(phy_count),
        "genome_size_by_phylum": {
            phy: {
                "n": len(L),
                "min": min(L), "median": int(median(L)), "max": max(L),
                "mean": int(sum(L)/len(L)),
            } for phy, L in sizes_by_phy.items()
        },
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
