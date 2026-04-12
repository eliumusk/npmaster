#!/usr/bin/env python3
"""
P2-T1: OOD False-Positive Decomposition
========================================
Analyze where test_ood false positives cluster:
  1. By phylum (Bacteroidota vs Cyanobacteriota)
  2. By predicted type
  3. By distance to nearest antiSMASH BGC
  4. By BGC signature domain content (Pfam validation on top FPs)
  5. By predicted region size

Also compare OOD FP patterns with ID FP patterns to identify OOD-specific failure modes.

Usage:
  python p2_t1_ood_fp_decomposition.py \
    --eval-json /data/muskliu/np_master/data/stage4_results/ws3_cap08_eval/score_swap_results.json \
    --bgc-csv /data/muskliu/np_master/data/bgc_annotations.csv \
    --meta /data/muskliu/np_master/data/windows/windows_metadata_cap08.csv \
    --out /data/muskliu/np_master/data/stage4_results/p2_t1_ood_decomposition
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np


CLASSES = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]


def load_eval_results(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True,
                    help="score_swap_results.json from ws3_cap08_eval")
    ap.add_argument("--bgc-csv", required=True, help="bgc_annotations.csv")
    ap.add_argument("--meta", required=True, help="windows_metadata_cap08.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ─── 1. Load eval results ───
    print("Loading evaluation results...")
    eval_data = load_eval_results(args.eval_json)

    # Use score_maxtype (canonical score)
    score_name = "score_maxtype"

    # ─── 2. Extract unmatched predictions for test_ood and test_id ───
    ood_result = eval_data["results"]["test_ood"][score_name]
    id_result = eval_data["results"]["test_id"][score_name]

    ood_unmatched = ood_result["unmatched_details"]
    id_unmatched = id_result["unmatched_details"]

    print(f"\ntest_ood: {ood_result['n_predicted_regions']} regions, "
          f"{ood_result['n_matched']} matched, {len(ood_unmatched)} unmatched")
    print(f"test_id:  {id_result['n_predicted_regions']} regions, "
          f"{id_result['n_matched']} matched, {len(id_unmatched)} unmatched")

    # ─── 3. Load genome metadata for phylum lookup ───
    print("Loading metadata for phylum lookup...")
    genome_phylum = {}
    with open(args.meta) as f:
        for row in csv.DictReader(f):
            if row["genome_id"] not in genome_phylum and row.get("phylum"):
                genome_phylum[row["genome_id"]] = row["phylum"]

    # ─── 4. Decompose OOD FPs ───
    print("\n" + "=" * 80)
    print("OOD FALSE-POSITIVE DECOMPOSITION")
    print("=" * 80)

    # 4.1 By phylum
    print("\n--- By Phylum ---")
    ood_by_phylum = defaultdict(list)
    for u in ood_unmatched:
        phylum = genome_phylum.get(u["genome_id"], "unknown")
        ood_by_phylum[phylum].append(u)

    phylum_summary = {}
    for phylum in sorted(ood_by_phylum.keys()):
        fps = ood_by_phylum[phylum]
        cats = Counter(u["category"] for u in fps)
        types = Counter(u["primary_type"] for u in fps)
        dists = [u["dist_to_nearest_bgc"] for u in fps if u["dist_to_nearest_bgc"] >= 0]
        lengths = [u["length"] for u in fps]

        phylum_summary[phylum] = {
            "n_fps": len(fps),
            "categories": dict(cats),
            "types": dict(types),
            "median_dist_to_bgc": float(np.median(dists)) if dists else -1,
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
        }
        print(f"  {phylum}: {len(fps)} FPs")
        print(f"    Categories: {dict(cats)}")
        print(f"    Types: {dict(types)}")
        if dists:
            print(f"    Median dist to BGC: {np.median(dists):.0f} bp")

    # 4.2 By predicted type
    print("\n--- By Predicted Type ---")
    ood_by_type = defaultdict(list)
    for u in ood_unmatched:
        ood_by_type[u["primary_type"]].append(u)

    type_summary = {}
    for t in CLASSES:
        fps = ood_by_type.get(t, [])
        if not fps:
            type_summary[t] = {"n_fps": 0}
            continue
        cats = Counter(u["category"] for u in fps)
        dists = [u["dist_to_nearest_bgc"] for u in fps if u["dist_to_nearest_bgc"] >= 0]
        scores = [u["max_score"] for u in fps]
        lengths = [u["length"] for u in fps]

        type_summary[t] = {
            "n_fps": len(fps),
            "fraction": round(len(fps) / max(1, len(ood_unmatched)), 4),
            "categories": dict(cats),
            "median_dist": float(np.median(dists)) if dists else -1,
            "mean_score": float(np.mean(scores)),
            "mean_length": float(np.mean(lengths)),
        }
        print(f"  {t}: {len(fps)} ({type_summary[t]['fraction']:.1%})")
        print(f"    Categories: {dict(cats)}")
        if dists:
            print(f"    Median dist: {np.median(dists):.0f} bp")
        print(f"    Mean score: {np.mean(scores):.3f}, Mean length: {np.mean(lengths):.0f} bp")

    # 4.3 By distance regime
    print("\n--- By Distance to Nearest antiSMASH BGC ---")
    dist_buckets = [
        ("0-5kb (boundary)", 0, 5000),
        ("5-20kb (ambiguous)", 5000, 20000),
        ("20-50kb (near)", 20000, 50000),
        ("50-100kb", 50000, 100000),
        (">100kb (far)", 100000, float("inf")),
        ("no BGC on contig", -1, 0),
    ]

    dist_summary = {}
    for bname, blo, bhi in dist_buckets:
        if bname == "no BGC on contig":
            in_bucket = [u for u in ood_unmatched if u["dist_to_nearest_bgc"] < 0]
        else:
            in_bucket = [u for u in ood_unmatched if blo <= u["dist_to_nearest_bgc"] < bhi]
        if in_bucket:
            types = Counter(u["primary_type"] for u in in_bucket)
            dist_summary[bname] = {
                "n_fps": len(in_bucket),
                "fraction": round(len(in_bucket) / max(1, len(ood_unmatched)), 4),
                "types": dict(types),
            }
            print(f"  {bname}: {len(in_bucket)} ({dist_summary[bname]['fraction']:.1%})")
            print(f"    Types: {dict(types)}")

    # 4.4 By region size
    print("\n--- By Region Size ---")
    size_buckets = [
        ("3-10kb", 3000, 10000),
        ("10-20kb", 10000, 20000),
        ("20-50kb", 20000, 50000),
        ("50-100kb", 50000, 100000),
        (">100kb", 100000, float("inf")),
    ]

    size_summary = {}
    for bname, blo, bhi in size_buckets:
        in_bucket = [u for u in ood_unmatched if blo <= u["length"] < bhi]
        if in_bucket:
            size_summary[bname] = {
                "n_fps": len(in_bucket),
                "fraction": round(len(in_bucket) / max(1, len(ood_unmatched)), 4),
            }
            print(f"  {bname}: {len(in_bucket)} ({size_summary[bname]['fraction']:.1%})")

    # 4.5 Score distribution of OOD FPs
    print("\n--- Score Distribution ---")
    ood_scores = [u["max_score"] for u in ood_unmatched]
    if ood_scores:
        score_percentiles = {
            "min": float(np.min(ood_scores)),
            "p25": float(np.percentile(ood_scores, 25)),
            "median": float(np.median(ood_scores)),
            "p75": float(np.percentile(ood_scores, 75)),
            "max": float(np.max(ood_scores)),
            "mean": float(np.mean(ood_scores)),
        }
        print(f"  OOD FP scores: min={score_percentiles['min']:.3f}, "
              f"median={score_percentiles['median']:.3f}, "
              f"max={score_percentiles['max']:.3f}")

    # ─── 5. Compare OOD vs ID FP patterns ───
    print("\n" + "=" * 80)
    print("OOD vs ID FALSE-POSITIVE COMPARISON")
    print("=" * 80)

    id_type_counts = Counter(u["primary_type"] for u in id_unmatched)
    ood_type_counts = Counter(u["primary_type"] for u in ood_unmatched)

    comparison = {}
    print(f"\n  {'Type':>12s} | {'ID FPs':>8s} {'(%)':>6s} | {'OOD FPs':>8s} {'(%)':>6s} | {'Shift':>6s}")
    print("  " + "-" * 65)
    for t in CLASSES:
        id_n = id_type_counts.get(t, 0)
        ood_n = ood_type_counts.get(t, 0)
        id_pct = id_n / max(1, len(id_unmatched))
        ood_pct = ood_n / max(1, len(ood_unmatched))
        shift = ood_pct - id_pct
        comparison[t] = {
            "id_count": id_n, "id_fraction": round(id_pct, 4),
            "ood_count": ood_n, "ood_fraction": round(ood_pct, 4),
            "shift": round(shift, 4),
        }
        print(f"  {t:>12s} | {id_n:>8d} {id_pct:>5.1%} | {ood_n:>8d} {ood_pct:>5.1%} | {shift:>+5.1%}")

    # Category comparison
    id_cats = Counter(u["category"] for u in id_unmatched)
    ood_cats = Counter(u["category"] for u in ood_unmatched)
    print(f"\n  {'Category':>25s} | {'ID':>6s} | {'OOD':>6s}")
    print("  " + "-" * 45)
    for cat in ["boundary_expansion", "ambiguous_unmatched", "clean_missed"]:
        print(f"  {cat:>25s} | {id_cats.get(cat,0):>6d} | {ood_cats.get(cat,0):>6d}")

    # Per-genome FP concentration
    print("\n--- Per-Genome FP Concentration (OOD) ---")
    genome_fp_counts = Counter(u["genome_id"] for u in ood_unmatched)
    fp_per_genome = sorted(genome_fp_counts.values(), reverse=True)
    if fp_per_genome:
        print(f"  Genomes with FPs: {len(genome_fp_counts)}")
        print(f"  FPs per genome: min={min(fp_per_genome)}, median={np.median(fp_per_genome):.0f}, "
              f"max={max(fp_per_genome)}, mean={np.mean(fp_per_genome):.1f}")
        # Top 5 genomes
        top5 = genome_fp_counts.most_common(5)
        print("  Top 5 genomes:")
        for gid, count in top5:
            phylum = genome_phylum.get(gid, "?")
            types_in_genome = Counter(u["primary_type"] for u in ood_unmatched if u["genome_id"] == gid)
            print(f"    {gid} ({phylum}): {count} FPs — {dict(types_in_genome)}")

    # ─── 6. Identify dominant OOD failure mode ───
    print("\n" + "=" * 80)
    print("DOMINANT FAILURE MODE ANALYSIS")
    print("=" * 80)

    # Which type is most over-represented in OOD FPs vs ID?
    max_shift_type = max(CLASSES, key=lambda t: comparison[t]["shift"])
    print(f"\n  Most OOD-shifted type: {max_shift_type} "
          f"(shift={comparison[max_shift_type]['shift']:+.1%})")

    # Which phylum contributes most OOD FPs?
    max_phylum = max(phylum_summary.keys(), key=lambda p: phylum_summary[p]["n_fps"])
    print(f"  Dominant OOD phylum: {max_phylum} ({phylum_summary[max_phylum]['n_fps']} FPs)")

    # What fraction of OOD FPs are clean-missed (far from known BGCs)?
    n_clean = sum(1 for u in ood_unmatched if u["category"] == "clean_missed")
    print(f"  Clean-missed (>20kb from BGC): {n_clean}/{len(ood_unmatched)} "
          f"({n_clean/max(1,len(ood_unmatched)):.1%})")

    # High-score FPs (potentially real BGCs)
    high_score_fps = [u for u in ood_unmatched if u["max_score"] >= 0.8]
    print(f"  High-score FPs (score>=0.8): {len(high_score_fps)}")
    if high_score_fps:
        hs_types = Counter(u["primary_type"] for u in high_score_fps)
        print(f"    Types: {dict(hs_types)}")

    # ─── 7. Recommendations ───
    print("\n" + "=" * 80)
    print("PRELIMINARY RECOMMENDATIONS")
    print("=" * 80)

    recommendations = []

    # Check if NRP is dominant
    nrp_ood_frac = comparison.get("NRP", {}).get("ood_fraction", 0)
    if nrp_ood_frac > 0.4:
        rec = ("NRP-specific hard negatives: NRP accounts for "
               f"{nrp_ood_frac:.0%} of OOD FPs. Consider adding OOD-phylum "
               "NRP-like negative examples to training.")
        recommendations.append(rec)
        print(f"\n  1. {rec}")

    # Check phylum imbalance
    if phylum_summary:
        phylum_fracs = {p: s["n_fps"] / max(1, len(ood_unmatched))
                       for p, s in phylum_summary.items()}
        if max(phylum_fracs.values()) > 0.7:
            dominant = max(phylum_fracs, key=phylum_fracs.get)
            rec = (f"Phylum-concentrated FPs: {dominant} contributes "
                   f"{phylum_fracs[dominant]:.0%} of OOD FPs. "
                   "May benefit from phylum-specific calibration or "
                   "harder negative mining from this phylum.")
            recommendations.append(rec)
            print(f"\n  2. {rec}")

    # Check clean-missed fraction
    clean_frac = n_clean / max(1, len(ood_unmatched))
    if clean_frac > 0.5:
        rec = (f"Most OOD FPs ({clean_frac:.0%}) are far from known BGCs. "
               "These might be real cryptic BGCs or systematic overcalling. "
               "Pfam validation on top OOD FPs would discriminate.")
        recommendations.append(rec)
        print(f"\n  3. {rec}")

    # ─── 8. Save ───
    output = {
        "task_id": "P2-T1",
        "description": "OOD false-positive decomposition",
        "score_name": score_name,
        "evaluation_type": eval_data.get("evaluation_type", "unknown"),
        "ood_summary": {
            "n_predicted": ood_result["n_predicted_regions"],
            "n_matched": ood_result["n_matched"],
            "n_unmatched": len(ood_unmatched),
            "recall": ood_result["recall"],
            "precision_pred_level": ood_result["precision_pred_level"],
        },
        "id_summary": {
            "n_predicted": id_result["n_predicted_regions"],
            "n_matched": id_result["n_matched"],
            "n_unmatched": len(id_unmatched),
            "recall": id_result["recall"],
            "precision_pred_level": id_result["precision_pred_level"],
        },
        "ood_by_phylum": phylum_summary,
        "ood_by_type": type_summary,
        "ood_by_distance": dist_summary,
        "ood_by_size": size_summary,
        "ood_score_distribution": score_percentiles if ood_scores else {},
        "ood_vs_id_type_comparison": comparison,
        "ood_category_comparison": {
            "id": dict(id_cats),
            "ood": dict(ood_cats),
        },
        "ood_per_genome": {
            "n_genomes_with_fps": len(genome_fp_counts),
            "median_fps_per_genome": float(np.median(fp_per_genome)) if fp_per_genome else 0,
            "max_fps_per_genome": max(fp_per_genome) if fp_per_genome else 0,
            "top5": [{"genome_id": gid, "n_fps": count,
                      "phylum": genome_phylum.get(gid, "")}
                     for gid, count in genome_fp_counts.most_common(5)],
        },
        "dominant_failure_mode": {
            "most_shifted_type": max_shift_type,
            "dominant_phylum": max_phylum,
            "clean_missed_fraction": round(clean_frac, 4),
            "n_high_score_fps": len(high_score_fps),
        },
        "recommendations": recommendations,
    }

    out_path = os.path.join(args.out, "ood_fp_decomposition.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
