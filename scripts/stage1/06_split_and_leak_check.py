#!/usr/bin/env python3
"""
Stage 1.3c: data split + GCF leak check.

Split:
  - test_ood: all genomes in Bacteroidota + Cyanobacteriota  (phylum-level hold-out)
  - train/val/test_id: in remaining phyla
      20% genomes → test_id
      from rest, 12.5% → val (= 10% of total)
      the rest → train

GCF proxy: cluster BGCs by (primary_class, genus). This is the stage1_detailed.md
fallback when BiG-SCAPE is unavailable. We treat each (primary_class, genus) as one GCF.
A GCF is "leaked" if it contains BGCs from both train and test/val genomes.

Leak fix: move the test/val BGCs of leaked GCFs to train (move their windows too).
We move GENOMES, not just BGCs, to keep split by-genome clean. For each leaked
GCF, pick the test/val genomes with that GCF and move them to train.

Outputs:
  - windows_metadata.csv updated with `split` column
  - split_summary.json
  - train_windows.txt / val_windows.txt / test_id_windows.txt / test_ood_windows.txt
"""
import argparse
import csv
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

def primary_class(types_str):
    # priority order for multi-label BGCs
    priority = ["Polyketide","NRP","Terpene","RiPP","Saccharide","Alkaloid","Other"]
    classes = set(types_str.split(";")) if types_str else set()
    for c in priority:
        if c in classes: return c
    return "Unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="windows_metadata.csv")
    ap.add_argument("--bgc-csv", required=True, help="bgc_annotations.csv")
    ap.add_argument("--selected", required=True, help="selected_genomes.tsv")
    ap.add_argument("--out-windows", required=True, help="output windows_metadata.csv w/ split")
    ap.add_argument("--out-summary", required=True, help="split_summary.json")
    ap.add_argument("--out-lists-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260405)
    args = ap.parse_args()

    random.seed(args.seed)

    # load genomes + phylum + genus
    genome_info = {}
    with open(args.selected) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            genome_info[row["ncbi_accession"]] = {
                "phylum": row["phylum"], "genus": row["genus"], "species": row["species"],
            }

    # load BGCs and compute GCF per BGC, GCF→genomes map
    bgc_gcf = {}
    gcf_genomes = defaultdict(set)
    with open(args.bgc_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            g = row["genome_id"]
            genus = genome_info.get(g,{}).get("genus","?")
            species = genome_info.get(g,{}).get("species","?")
            gcf = f'{primary_class(row["bgc_types"])}|{genus}|{species}'
            bgc_gcf[row["bgc_id"]] = gcf
            gcf_genomes[gcf].add(g)

    # phylum-level OOD
    OOD_PHYLA = {"p__Bacteroidota","p__Cyanobacteriota"}

    # list all genomes (those in selected + those in bgc_csv)
    all_genomes = sorted(genome_info.keys())
    genome_split = {}
    remaining = []
    for g in all_genomes:
        if genome_info[g]["phylum"] in OOD_PHYLA:
            genome_split[g] = "test_ood"
        else:
            remaining.append(g)

    random.shuffle(remaining)
    n = len(remaining)
    n_test = int(round(0.20*n))
    n_val = int(round(0.125*(n-n_test)))
    test_id_set = set(remaining[:n_test])
    val_set = set(remaining[n_test:n_test+n_val])
    train_set = set(remaining[n_test+n_val:])
    for g in test_id_set: genome_split[g] = "test_id"
    for g in val_set: genome_split[g] = "val"
    for g in train_set: genome_split[g] = "train"

    split_init = Counter(genome_split.values())

    # GCF leak check: GCF with genomes in train AND (val OR test_id OR test_ood)
    # For each leaked GCF, MOVE its non-train genomes into train (conservative)
    leaked_gcfs = []
    moved_genomes = []
    for gcf, gs in gcf_genomes.items():
        spl = set(genome_split.get(g,"?") for g in gs)
        if "train" in spl and (spl & {"val","test_id","test_ood"}):
            leaked_gcfs.append(gcf)
            for g in gs:
                if genome_split.get(g) in ("val","test_id","test_ood"):
                    genome_split[g] = "train"
                    moved_genomes.append((g, gcf))

    # verify no leak remaining
    leaked_after = 0
    for gcf, gs in gcf_genomes.items():
        spl = set(genome_split.get(g,"?") for g in gs)
        if "train" in spl and (spl & {"val","test_id","test_ood"}):
            leaked_after += 1

    # write windows with split column (read, add, write)
    header_added = False
    split_counts = {s: {"windows":0, "positive":0, "negative":0, "ambiguous":0} for s in ["train","val","test_id","test_ood"]}
    class_dist = {s: Counter() for s in ["train","val","test_id","test_ood"]}
    combo_dist = {s: Counter() for s in ["train","val","test_id","test_ood"]}
    phylum_pos = defaultdict(lambda: defaultdict(lambda: {"pos":0,"neg":0}))
    win_by_split = defaultdict(list)
    n_ambig_total = 0
    n_total = 0

    Path(args.out_lists_dir).mkdir(parents=True, exist_ok=True)
    with open(args.windows) as fin, open(args.out_windows, "w", newline="") as fout:
        r = csv.DictReader(fin)
        cols = r.fieldnames + ["split"]
        w = csv.DictWriter(fout, fieldnames=cols)
        w.writeheader()
        classes7 = ["Polyketide","NRP","Terpene","RiPP","Saccharide","Alkaloid","Other"]
        for row in r:
            g = row["genome_id"]
            spl = genome_split.get(g, "")
            row["split"] = spl
            w.writerow(row)
            n_total += 1
            if spl not in split_counts: continue
            split_counts[spl]["windows"] += 1
            if row["label"] == "positive":
                split_counts[spl]["positive"] += 1
                active = tuple(sorted(c for c in classes7 if row[c]=="1"))
                for c in active: class_dist[spl][c] += 1
                if len(active) > 1: combo_dist[spl][";".join(active)] += 1
                phylum_pos[row["phylum"]][spl]["pos"] += 1
            elif row["label"] == "negative":
                split_counts[spl]["negative"] += 1
                phylum_pos[row["phylum"]][spl]["neg"] += 1
            else:
                split_counts[spl]["ambiguous"] += 1
                n_ambig_total += 1
            win_by_split[spl].append(row["window_id"])

    for spl, ids in win_by_split.items():
        with open(Path(args.out_lists_dir)/f"{spl}_windows.txt", "w") as fo:
            for wid in ids: fo.write(wid+"\n")

    summary = {
        "split_genomes_initial": dict(split_init),
        "gcf_leakage_check": {
            "n_leaked_gcfs_before_fix": len(leaked_gcfs),
            "n_leaked_gcfs_after_fix": leaked_after,
            "n_moved_genomes": len(moved_genomes),
            "example_leaked_gcfs": leaked_gcfs[:20],
        },
        "split_genomes_final": dict(Counter(genome_split.values())),
        "split_counts": split_counts,
        "class_distribution_by_split": {s:dict(d) for s,d in class_dist.items()},
        "top_multilabel_combos_by_split": {s: dict(d.most_common(10)) for s,d in combo_dist.items()},
        "ambiguous_pct": round(100*n_ambig_total/max(1,n_total), 2),
        "n_windows_total": n_total,
        "pos_neg_ratio_by_phylum_split": {
            phy: {spl: v for spl,v in splits.items()} for phy,splits in phylum_pos.items()
        },
    }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k:v for k,v in summary.items() if k!="pos_neg_ratio_by_phylum_split"}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
