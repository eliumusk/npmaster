"""Score-stratified 'apparent novel' analysis on 9-genome.

For each v4.1 multi-scale region on 9-genome:
  - antiSMASH covered?  (overlap any antiSMASH region on same contig)
  - GT covered?         (overlap any 9-genome human-curated GT region)
  - score bin:          [0.95, 0.98), [0.98, 0.99), [0.99, 0.995), [0.995, 1.0]

Report rates so we can calibrate what "v4.1 says yes, antiSMASH says no, score>=0.98"
means on a benchmark where we *do* have GT.

Inputs (all server-side):
  /data/muskliu/npmaster/experiments/evo2_first_principles/9g_preds_lrA1_w8192/
  /data/muskliu/npmaster/experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096/
  /data/syh/benchmark/nine_genome/results/antismash/<gid>/<gid>.json
  /data/syh/benchmark/nine_genome/ground_truth.json

Output:
  /data/muskliu/oer_bgc_scan/novel_triage/prior_fdr_9genome.json
  /data/muskliu/oer_bgc_scan/novel_triage/prior_fdr_9genome_table.md
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/data/muskliu/npmaster")
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoding.v4_1 import PRIMARY_OP, SHORT_OP, SHORT_LENGTH_CAP, IOU_DEDUP, decode_dir, union_dedup  # noqa: E402
from baselines.parse_antismash import parse_antismash_dir  # noqa: E402

PRIMARY_DIR = REPO_ROOT / "experiments/evo2_first_principles/9g_preds_lrA1_w8192"
SHORT_DIR = REPO_ROOT / "experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096"
ANTISMASH_DIR = Path("/data/syh/benchmark/nine_genome/results/antismash")
GT_JSON = Path("/data/syh/benchmark/nine_genome/ground_truth.json")
OUT_DIR = Path("/data/muskliu/oer_bgc_scan/novel_triage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_BINS = [
    ("0.95-0.98", 0.95, 0.98),
    ("0.98-0.99", 0.98, 0.99),
    ("0.99-0.995", 0.99, 0.995),
    ("0.995-1.0", 0.995, 1.0 + 1e-9),
]


def interval_overlap(a_s, a_e, b_s, b_e):
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def main():
    print("=== Decode v4.1 regions on 9-genome ===")
    print(f"PRIMARY_OP={PRIMARY_OP}")
    print(f"SHORT_OP  ={SHORT_OP}  (length cap <{SHORT_LENGTH_CAP})")

    primary = decode_dir(PRIMARY_DIR, PRIMARY_OP)
    short = decode_dir(SHORT_DIR, SHORT_OP)
    short_filt = {g: [r for r in regs if (r[2] - r[1]) < SHORT_LENGTH_CAP]
                  for g, regs in short.items()}
    genomes = sorted(set(primary) | set(short_filt))

    # decode_dir returns [(contig, start, end, score), ...]; union_dedup expects two lists.
    # Track source via which list the final region came from.
    v41_regions = {}
    for g in genomes:
        prim = list(primary.get(g, []))
        sh = list(short_filt.get(g, []))
        prim_set = {(c, s, e, round(sc, 6)) for (c, s, e, sc) in prim}
        kept = union_dedup(prim, sh, IOU_DEDUP)
        # Tag source
        tagged = []
        for (c, s, e, sc) in kept:
            src = "primary" if (c, s, e, round(sc, 6)) in prim_set else "short_rescue"
            tagged.append((c, s, e, sc, src))
        v41_regions[g] = tagged

    total_v41 = sum(len(v) for v in v41_regions.values())
    print(f"v4.1 regions total (9 genomes): {total_v41}")

    # antiSMASH regions
    print("\n=== Parse antiSMASH 9g regions ===")
    as_regions = parse_antismash_dir(ANTISMASH_DIR, genome_ids=genomes)
    total_as = sum(len(v) for v in as_regions.values())
    print(f"antiSMASH regions total: {total_as}")

    # GT
    print("\n=== Load 9-genome ground truth ===")
    gt = json.load(open(GT_JSON))["ground_truth"]
    total_gt = sum(len(v) for v in gt.values())
    print(f"GT regions total: {total_gt}")

    # ==== Score-stratified analysis ====
    print("\n=== Score-stratified analysis ===\n")
    rows = []
    all_apparent_novel = []  # details of candidate-like calls

    # Aggregate counters: per score bin
    bin_stats = {b: {"total": 0, "as_cov": 0, "as_missed": 0,
                      "as_missed_and_gt": 0, "as_missed_no_gt": 0,
                      "gt_cov": 0}
                 for b, _lo, _hi in SCORE_BINS}

    for g in genomes:
        v41 = v41_regions[g]
        as_list = [(r["contig_id"], r["start_bp"], r["end_bp"]) for r in as_regions.get(g, [])]
        gt_list = [(c, s, e) for (c, s, e) in gt.get(g, [])]
        # Group as/gt by contig for faster lookup
        as_by_contig = defaultdict(list)
        for c, s, e in as_list: as_by_contig[c].append((s, e))
        gt_by_contig = defaultdict(list)
        for c, s, e in gt_list: gt_by_contig[c].append((s, e))

        for (c, s, e, sc, src) in v41:
            # antiSMASH overlap?
            as_cov = any(interval_overlap(s, e, a_s, a_e) > 0 for (a_s, a_e) in as_by_contig.get(c, []))
            gt_cov = any(interval_overlap(s, e, a_s, a_e) > 0 for (a_s, a_e) in gt_by_contig.get(c, []))

            for b, lo, hi in SCORE_BINS:
                if lo <= sc < hi:
                    st = bin_stats[b]
                    st["total"] += 1
                    if as_cov: st["as_cov"] += 1
                    else:
                        st["as_missed"] += 1
                        if gt_cov:
                            st["as_missed_and_gt"] += 1
                        else:
                            st["as_missed_no_gt"] += 1
                    if gt_cov: st["gt_cov"] += 1
                    break

            if (not as_cov) and sc >= 0.95:
                all_apparent_novel.append({
                    "genome": g, "contig": c, "start": s, "end": e,
                    "length": e - s, "score": sc, "source": src,
                    "gt_covered": gt_cov,
                })

    # Print table
    print(f"{'bin':12s} {'total':>6} {'AS_cov':>8} {'AS_miss':>8} {'AS_miss&GT':>12} "
          f"{'AS_miss_noGT':>13} {'P(GT|AS_miss)':>14}")
    md_lines = ["# 9-genome score-stratified prior FDR\n",
                f"- v4.1 total regions: **{total_v41}**  (decoded from existing preds, frozen OP)",
                f"- antiSMASH regions:  **{total_as}**",
                f"- 9-genome GT:        **{total_gt}**  (human-curated tight-core)",
                "",
                "## Calls broken down by max_score bin",
                "",
                "| score bin | total | AS covered | AS missed | AS miss & GT-hit | AS miss & no GT | **P(GT-hit \\| AS-missed)** |",
                "|---|---:|---:|---:|---:|---:|---:|"]
    for b, _lo, _hi in SCORE_BINS:
        st = bin_stats[b]
        p = st["as_missed_and_gt"] / max(1, st["as_missed"])
        print(f"{b:12s} {st['total']:>6} {st['as_cov']:>8} {st['as_missed']:>8} "
              f"{st['as_missed_and_gt']:>12} {st['as_missed_no_gt']:>13} {p:>14.3f}")
        md_lines.append(f"| {b} | {st['total']} | {st['as_cov']} | {st['as_missed']} | "
                        f"{st['as_missed_and_gt']} | {st['as_missed_no_gt']} | **{p:.3f}** |")

    # Also a pure-novel aggregate: at score>=0.98 (our OER cohort's threshold)
    total_hi = sum(bin_stats[b]["total"] for b, lo, hi in SCORE_BINS if lo >= 0.98)
    miss_hi = sum(bin_stats[b]["as_missed"] for b, lo, hi in SCORE_BINS if lo >= 0.98)
    gt_hi = sum(bin_stats[b]["as_missed_and_gt"] for b, lo, hi in SCORE_BINS if lo >= 0.98)
    p_hi = gt_hi / max(1, miss_hi)
    print(f"\nAggregated score >= 0.98:")
    print(f"  total calls: {total_hi}")
    print(f"  AS-missed:  {miss_hi}")
    print(f"  AS-missed & GT-hit: {gt_hi}")
    print(f"  P(GT-hit | AS-missed, score>=0.98) = {p_hi:.3f}")
    print(f"  (OER-novel analogue rate: expected ~{p_hi*15:.1f} of 15 OER candidates are BGC-like in GT-space)")

    md_lines += [
        "",
        f"## Aggregated for score >= 0.98 (OER-candidate threshold)",
        f"- Total v4.1 calls: **{total_hi}**",
        f"- AS-missed: **{miss_hi}**",
        f"- AS-missed & GT-hit: **{gt_hi}**",
        f"- **P(GT-hit | AS-missed, score>=0.98) = {p_hi:.3f}**",
        "",
        f"## Interpretation for OER 15 candidates",
        f"Assuming 9-genome calibration transfers: of 15 OER candidates at score>=0.98 AS-missed,",
        f"expected GT-hit-rate-analogue = {p_hi:.1%} → ~{p_hi*15:.1f} are true novel-style BGCs,",
        f"~{(1-p_hi)*15:.1f} may be model calls outside curated BGC space.",
        f"**Caveats:** 9-genome GT itself may miss novel BGCs (lower-bound), and OER is metagenomic",
        f"(different composition). Use as *directional* prior, not exact FDR.",
        "",
    ]

    # Write outputs
    out_json = OUT_DIR / "prior_fdr_9genome.json"
    with open(out_json, "w") as fh:
        json.dump({
            "v41_total": total_v41,
            "antismash_total": total_as,
            "gt_total": total_gt,
            "score_bins": {b: bin_stats[b] for b, _lo, _hi in SCORE_BINS},
            "score_ge_0_98": {"total": total_hi, "as_missed": miss_hi,
                              "as_missed_and_gt": gt_hi, "p_gt_given_as_missed": p_hi},
            "apparent_novel_calls": all_apparent_novel,
        }, fh, indent=2)
    print(f"\nwrote {out_json}")

    out_md = OUT_DIR / "prior_fdr_9genome_table.md"
    out_md.write_text("\n".join(md_lines))
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
