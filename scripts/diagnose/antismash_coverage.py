"""Comprehensive coverage diagnostic: antiSMASH vs v4.1 on OER004256 vs 9-genome.

For each dataset (OER, 9g):
  1. Parse antiSMASH regions: contig, start, end, product, n_CDS (from JSON features)
  2. Load v4.1 regions (OER: CSV; 9g: decode from preds)
  3. Load v4.1 per-window scores (primary + short where available)
  4. Categorize each antiSMASH region as:
       - BOTH: overlap with any v4.1 region
       - AS_ONLY: no v4.1 region overlap
     And each v4.1 region as:
       - BOTH or V41_ONLY
  5. For AS_ONLY regions, query v4.1 per-window max score in that coordinate range
     -> "model saw but scored low" vs "model never covered"
  6. Stratify by length buckets; compare OER vs 9g ratios.

Outputs:
  /data/muskliu/oer_bgc_scan/novel_triage/coverage_diag/
      oer_report.json
      9g_report.json
      comparison.md
      as_only_samples.tsv  (top 20 AS_ONLY by product × length for qualitative review)
"""
from __future__ import annotations
import csv
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

OUT_DIR = Path("/data/muskliu/oer_bgc_scan/novel_triage/coverage_diag")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- OER paths ----------------
OER_AS_ROOT = Path("/data/muskliu/oer_bgc_scan/results/antismash")
OER_FILTERED = Path("/data/muskliu/oer_bgc_scan/filtered")
OER_V41_CSV = Path("/data/muskliu/npmaster/experiments/OER004256/v41_regions_fully_typed.csv")
OER_V41_PRIMARY_PREDS = Path("/data/muskliu/npmaster/experiments/OER004256/v41_primary_preds")
OER_V41_SHORT_PREDS = Path("/data/muskliu/npmaster/experiments/OER004256/v41_short_preds")

# ---------------- 9g paths ----------------
NG_AS_ROOT = Path("/data/syh/benchmark/nine_genome/results/antismash")
NG_FASTA_DIR = Path("/data/syh/benchmark/nine_genome/fna")
NG_V41_PRIMARY_PREDS = Path("/data/muskliu/npmaster/experiments/evo2_first_principles/9g_preds_lrA1_w8192")
NG_V41_SHORT_PREDS = Path("/data/muskliu/npmaster/experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096")

# v4.1 OP
sys.path.insert(0, "/data/muskliu/npmaster/src")
from decoding.v4_1 import (PRIMARY_OP, SHORT_OP, SHORT_LENGTH_CAP, IOU_DEDUP,
                           decode_dir, union_dedup)  # noqa: E402


# ------------------- helpers -------------------
def interval_overlap(a_s, a_e, b_s, b_e):
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def length_bucket(L):
    if L < 5_000: return "<5kb"
    if L < 10_000: return "5-10kb"
    if L < 20_000: return "10-20kb"
    if L < 50_000: return "20-50kb"
    return ">=50kb"

BUCKETS = ["<5kb", "5-10kb", "10-20kb", "20-50kb", ">=50kb"]


def contig_len_bucket(L):
    if L < 15_000: return "<15kb"
    if L < 30_000: return "15-30kb"
    if L < 100_000: return "30-100kb"
    if L < 500_000: return "100-500kb"
    return ">=500kb"

CONTIG_BUCKETS = ["<15kb", "15-30kb", "30-100kb", "100-500kb", ">=500kb"]


def load_fasta_lens(fa_path):
    """Return dict contig_id -> length (fast, no seq keep)."""
    out = {}
    cur = None; L = 0
    with open(fa_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur: out[cur] = L
                cur = line[1:].split()[0]
                L = 0
            else:
                L += len(line.rstrip())
        if cur: out[cur] = L
    return out


def parse_antismash_json(json_path):
    """Yield (contig, r_start, r_end, product, n_cds) per region from antiSMASH JSON.
    Also returns dict contig_id -> record_len (nt)."""
    with open(json_path) as fh:
        data = json.load(fh)
    regions = []
    contig_lens = {}
    for rec in data.get("records", []):
        contig = rec.get("id") or rec.get("name")
        if not contig:
            continue
        # Record nucleotide length
        rec_len = rec.get("length") or 0
        if not rec_len:
            # Some schemas store seq; avoid loading it. Try annotations.
            seq = rec.get("seq") or ""
            rec_len = len(seq) if seq else 0
        contig_lens[contig] = rec_len

        # Collect CDS ranges and region features in one pass
        cds_ranges = []
        region_feats = []
        for feat in rec.get("features", []):
            ft = feat.get("type")
            loc = feat.get("location", "")
            m = re.search(r"\[(\d+):[><]?(\d+)\]", loc)
            if not m:
                continue
            fs, fe = int(m.group(1)), int(m.group(2))
            if ft == "CDS":
                cds_ranges.append((fs, fe))
            elif ft == "region":
                product = feat.get("qualifiers", {}).get("product", ["?"])[0]
                region_feats.append((fs, fe, product))

        # Count CDS per region
        for (rs, re_, product) in region_feats:
            n_cds = sum(1 for (cs, ce) in cds_ranges if cs < re_ and ce > rs)
            regions.append((contig, rs, re_, product, n_cds))
    return regions, contig_lens


# ------------------- core dataset analysis -------------------
def analyze_dataset(
    name,
    antismash_regions,              # list of (contig, s, e, product, n_cds)
    contig_lens,                    # dict contig -> length
    v41_regions,                    # list of (contig, s, e, score[, source])
    v41_primary_windows,            # dict contig -> list of (start, end, score)  -- w=8192
    v41_short_windows,              # dict contig -> list of (start, end, score)  -- w=4096
):
    report = {"name": name}

    total_as = len(antismash_regions)
    total_v41 = len(v41_regions)
    as_contigs = set(r[0] for r in antismash_regions)
    v41_contigs = set(r[0] for r in v41_regions)

    # All contigs covered by either tool:
    all_called_contigs = as_contigs | v41_contigs
    report["n_antismash_regions"] = total_as
    report["n_v41_regions"] = total_v41
    report["as_region_density_per_Gb"] = total_as * 1e9 / max(1, sum(contig_lens.values()))
    report["v41_region_density_per_Gb"] = total_v41 * 1e9 / max(1, sum(contig_lens.values()))

    # Length distributions
    as_lens = Counter(length_bucket(e - s) for (c, s, e, p, n) in antismash_regions)
    v41_lens = Counter(length_bucket(e - s) for (c, s, e, *_) in v41_regions)
    report["antismash_length_dist"] = {b: as_lens.get(b, 0) for b in BUCKETS}
    report["v41_length_dist"] = {b: v41_lens.get(b, 0) for b in BUCKETS}

    # Median antiSMASH region length
    as_lens_raw = sorted(e - s for (c, s, e, p, n) in antismash_regions)
    if as_lens_raw:
        report["antismash_median_length"] = as_lens_raw[len(as_lens_raw) // 2]
        report["antismash_mean_length"] = sum(as_lens_raw) / len(as_lens_raw)

    # Contig length bucket distribution (for AS regions)
    cl_dist_as = Counter()
    for (c, s, e, p, n) in antismash_regions:
        L = contig_lens.get(c, 0)
        cl_dist_as[contig_len_bucket(L)] += 1
    report["as_region_contig_length_dist"] = {b: cl_dist_as.get(b, 0) for b in CONTIG_BUCKETS}

    # Regions per contig (clustering)
    as_per_contig = Counter(r[0] for r in antismash_regions)
    v41_per_contig = Counter(r[0] for r in v41_regions)
    report["as_regions_per_contig_hist"] = {
        "1": sum(1 for c, n in as_per_contig.items() if n == 1),
        "2-3": sum(1 for c, n in as_per_contig.items() if 2 <= n <= 3),
        "4-10": sum(1 for c, n in as_per_contig.items() if 4 <= n <= 10),
        ">10": sum(1 for c, n in as_per_contig.items() if n > 10),
    }
    report["as_contigs_total"] = len(as_per_contig)
    report["as_max_regions_on_single_contig"] = max(as_per_contig.values()) if as_per_contig else 0
    report["v41_max_regions_on_single_contig"] = max(v41_per_contig.values()) if v41_per_contig else 0

    # Overlap classification
    as_by_contig = defaultdict(list)
    for i, (c, s, e, p, n) in enumerate(antismash_regions):
        as_by_contig[c].append((s, e, i))
    v41_by_contig = defaultdict(list)
    for j, (c, s, e, *_rest) in enumerate(v41_regions):
        v41_by_contig[c].append((s, e, j))

    as_category = ["AS_ONLY"] * total_as
    v41_category = ["V41_ONLY"] * total_v41
    # For each AS region, mark if any v4.1 region overlaps
    for c, items in as_by_contig.items():
        v = v41_by_contig.get(c, [])
        for (s_a, e_a, ia) in items:
            for (s_v, e_v, iv) in v:
                if interval_overlap(s_a, e_a, s_v, e_v) > 0:
                    as_category[ia] = "BOTH"
                    v41_category[iv] = "BOTH"
                    break

    as_only_idx = [i for i, cat in enumerate(as_category) if cat == "AS_ONLY"]
    v41_only_idx = [j for j, cat in enumerate(v41_category) if cat == "V41_ONLY"]
    both_as_idx = [i for i, cat in enumerate(as_category) if cat == "BOTH"]

    report["categories"] = {
        "BOTH_from_antismash_side": len(both_as_idx),
        "AS_ONLY": len(as_only_idx),
        "V41_ONLY": len(v41_only_idx),
    }

    # AS_ONLY stratification
    as_only_len_dist = Counter()
    as_only_contig_len_dist = Counter()
    as_only_ncds_dist = Counter()
    as_only_product_dist = Counter()
    for i in as_only_idx:
        c, s, e, p, n = antismash_regions[i]
        as_only_len_dist[length_bucket(e - s)] += 1
        as_only_contig_len_dist[contig_len_bucket(contig_lens.get(c, 0))] += 1
        if n <= 3: as_only_ncds_dist["1-3_CDS"] += 1
        elif n <= 7: as_only_ncds_dist["4-7_CDS"] += 1
        elif n <= 15: as_only_ncds_dist["8-15_CDS"] += 1
        else: as_only_ncds_dist[">15_CDS"] += 1
        as_only_product_dist[p] += 1

    report["as_only_length_dist"] = {b: as_only_len_dist.get(b, 0) for b in BUCKETS}
    report["as_only_contig_length_dist"] = {b: as_only_contig_len_dist.get(b, 0) for b in CONTIG_BUCKETS}
    report["as_only_ncds_dist"] = dict(as_only_ncds_dist)
    report["as_only_product_top10"] = dict(as_only_product_dist.most_common(10))

    # Same for BOTH for comparison
    both_len_dist = Counter()
    both_contig_len_dist = Counter()
    both_ncds_dist = Counter()
    both_product_dist = Counter()
    for i in both_as_idx:
        c, s, e, p, n = antismash_regions[i]
        both_len_dist[length_bucket(e - s)] += 1
        both_contig_len_dist[contig_len_bucket(contig_lens.get(c, 0))] += 1
        if n <= 3: both_ncds_dist["1-3_CDS"] += 1
        elif n <= 7: both_ncds_dist["4-7_CDS"] += 1
        elif n <= 15: both_ncds_dist["8-15_CDS"] += 1
        else: both_ncds_dist[">15_CDS"] += 1
        both_product_dist[p] += 1

    report["both_length_dist"] = {b: both_len_dist.get(b, 0) for b in BUCKETS}
    report["both_contig_length_dist"] = {b: both_contig_len_dist.get(b, 0) for b in CONTIG_BUCKETS}
    report["both_ncds_dist"] = dict(both_ncds_dist)
    report["both_product_top10"] = dict(both_product_dist.most_common(10))

    # "Did v4.1 see the AS_ONLY regions?" -> look at per-window max score
    # v41_primary_windows and v41_short_windows are dicts contig -> sorted list of (s, e, score)
    v41_covered = 0
    v41_saw_lowscore = 0  # saw windows with score >= 0.5 but decode didn't produce region
    v41_saw_highscore = 0  # saw windows with score >= 0.95 but region failed to meet support_windows/extend criteria
    v41_never_covered = 0
    per_region_max_scores = []
    for i in as_only_idx:
        c, s, e, p, n = antismash_regions[i]
        # Primary windows in the region
        p_wins = v41_primary_windows.get(c, [])
        s_wins = v41_short_windows.get(c, [])
        max_sc = 0.0
        cov_count = 0
        for (ws, we, sc) in p_wins:
            if we > s and ws < e:
                cov_count += 1
                if sc > max_sc: max_sc = sc
        for (ws, we, sc) in s_wins:
            if we > s and ws < e:
                cov_count += 1
                if sc > max_sc: max_sc = sc
        per_region_max_scores.append(max_sc)
        if cov_count == 0:
            v41_never_covered += 1
        elif max_sc >= 0.95:
            v41_saw_highscore += 1
        elif max_sc >= 0.5:
            v41_saw_lowscore += 1
        else:
            v41_covered += 1  # covered but low

    report["as_only_v41_signal"] = {
        "never_covered_no_windows": v41_never_covered,
        "covered_low_score(<0.5)": v41_covered,
        "covered_mid_score(0.5-0.95)": v41_saw_lowscore,
        "covered_high_score(>=0.95)_but_decoder_rejected": v41_saw_highscore,
    }
    # Max-score percentiles
    if per_region_max_scores:
        ps = sorted(per_region_max_scores)
        report["as_only_v41_max_score_percentiles"] = {
            "p10": round(ps[int(len(ps) * 0.10)], 3),
            "p50": round(ps[int(len(ps) * 0.50)], 3),
            "p90": round(ps[int(len(ps) * 0.90)], 3),
            "p99": round(ps[int(len(ps) * 0.99)], 3),
        }

    report["as_only_max_score_v41"] = per_region_max_scores  # for later plot if needed

    return report, as_only_idx, v41_only_idx, both_as_idx, as_category


# ------------------- Per-window loader -------------------
def load_preds_dir(preds_dir, flavor):
    """Return dict contig -> list of (start, end, bgc_score)."""
    windows = defaultdict(list)
    n_csv = 0
    n_rows = 0
    for csv_path in sorted(Path(preds_dir).glob("*_predictions.csv")):
        n_csv += 1
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                c = row["contig"]
                s = int(row["start"]); e = int(row["end"])
                sc = float(row["bgc_score"])
                windows[c].append((s, e, sc))
                n_rows += 1
    print(f"  [{flavor}] {n_csv} CSVs, {n_rows} windows, {len(windows)} contigs")
    return windows


# ------------------- OER loader -------------------
def load_oer():
    print("=== OER: parsing antiSMASH regions ===")
    regs = []
    contig_lens_all = {}
    for sample_dir in sorted(OER_AS_ROOT.glob("OER*.ge10000")):
        sample_id = sample_dir.name.replace(".ge10000", "")
        json_path = sample_dir / f"{sample_dir.name}.json"
        if not json_path.exists():
            continue
        sample_regs, sample_contig_lens = parse_antismash_json(json_path)
        regs.extend(sample_regs)
        contig_lens_all.update(sample_contig_lens)
        print(f"  {sample_id}: {len(sample_regs)} regions")

    # If antiSMASH JSON didn't give us contig lengths, fall back to filtered FASTAs
    if sum(contig_lens_all.values()) == 0:
        print("  (backfilling contig lengths from filtered FASTAs)")
        for fa in OER_FILTERED.glob("*.fa"):
            contig_lens_all.update(load_fasta_lens(fa))

    print(f"  OER total: {len(regs)} antiSMASH regions, {len(contig_lens_all)} contigs "
          f"(total {sum(contig_lens_all.values())/1e9:.2f} Gb)")

    # v4.1 regions from CSV: contig like "OER00425668|k141_xxx", antismash keys are bare "k141_xxx"
    print("=== OER: loading v4.1 regions ===")
    v41 = []
    for row in csv.DictReader(open(OER_V41_CSV)):
        full_c = row["contig"]
        contig = full_c.split("|")[-1]
        v41.append((contig, int(row["start"]), int(row["end"]), float(row["max_score"])))
    print(f"  OER v4.1: {len(v41)} regions")

    # Per-window preds — prefix strip for matching
    print("=== OER: loading v4.1 per-window preds ===")
    p = load_preds_dir(OER_V41_PRIMARY_PREDS, "primary")
    s = load_preds_dir(OER_V41_SHORT_PREDS, "short")
    # Strip sample prefix
    def strip_prefix(d):
        out = defaultdict(list)
        for k, v in d.items():
            short_k = k.split("|")[-1]
            out[short_k].extend(v)
        return out
    p = strip_prefix(p)
    s = strip_prefix(s)
    return regs, contig_lens_all, v41, p, s


# ------------------- 9g loader -------------------
def load_9g():
    print("\n=== 9g: parsing antiSMASH regions ===")
    regs = []
    contig_lens_all = {}
    for gdir in sorted(NG_AS_ROOT.iterdir()):
        if not gdir.is_dir():
            continue
        json_path = gdir / f"{gdir.name}.json"
        if not json_path.exists():
            continue
        sample_regs, sample_contig_lens = parse_antismash_json(json_path)
        regs.extend(sample_regs)
        contig_lens_all.update(sample_contig_lens)
        print(f"  {gdir.name}: {len(sample_regs)} regions")

    if sum(contig_lens_all.values()) == 0:
        for fa in NG_FASTA_DIR.glob("*.fna"):
            contig_lens_all.update(load_fasta_lens(fa))
        for fa in NG_FASTA_DIR.glob("*.fa"):
            contig_lens_all.update(load_fasta_lens(fa))
    print(f"  9g total: {len(regs)} antiSMASH regions, {len(contig_lens_all)} contigs "
          f"(total {sum(contig_lens_all.values())/1e9:.2f} Gb)")

    # v4.1 regions from decode
    print("=== 9g: decoding v4.1 regions ===")
    prim_map = decode_dir(NG_V41_PRIMARY_PREDS, PRIMARY_OP)
    short_map = decode_dir(NG_V41_SHORT_PREDS, SHORT_OP)
    short_filt = {g: [r for r in rs if (r[2] - r[1]) < SHORT_LENGTH_CAP]
                  for g, rs in short_map.items()}
    v41 = []
    genomes = sorted(set(prim_map) | set(short_filt))
    for g in genomes:
        kept = union_dedup(prim_map.get(g, []), short_filt.get(g, []), IOU_DEDUP)
        for (c, s, e, sc) in kept:
            v41.append((c, s, e, sc))
    print(f"  9g v4.1: {len(v41)} regions")

    print("=== 9g: loading per-window preds ===")
    p = load_preds_dir(NG_V41_PRIMARY_PREDS, "primary")
    s = load_preds_dir(NG_V41_SHORT_PREDS, "short")
    return regs, contig_lens_all, v41, p, s


# ------------------- Main -------------------
def main():
    oer_regs, oer_cl, oer_v41, oer_p, oer_s = load_oer()
    ng_regs, ng_cl, ng_v41, ng_p, ng_s = load_9g()

    print("\n=== Analyzing OER ===")
    oer_rep, oer_as_only_idx, oer_v41_only_idx, oer_both_idx, oer_cats = analyze_dataset(
        "OER", oer_regs, oer_cl, oer_v41, oer_p, oer_s)
    print("\n=== Analyzing 9g ===")
    ng_rep, ng_as_only_idx, ng_v41_only_idx, ng_both_idx, ng_cats = analyze_dataset(
        "9g", ng_regs, ng_cl, ng_v41, ng_p, ng_s)

    # Save JSON reports
    def clean_for_json(d):
        # remove any large arrays
        out = {}
        for k, v in d.items():
            if k == "as_only_max_score_v41":
                out[k + "_summary"] = {"n": len(v), "mean": sum(v)/max(1,len(v)),
                                       "max": max(v) if v else 0}
            else:
                out[k] = v
        return out

    with open(OUT_DIR / "oer_report.json", "w") as fh:
        json.dump(clean_for_json(oer_rep), fh, indent=2)
    with open(OUT_DIR / "9g_report.json", "w") as fh:
        json.dump(clean_for_json(ng_rep), fh, indent=2)

    # ---------- Emit samples of AS_ONLY for qualitative review ----------
    # Top 25 AS_ONLY on OER, stratified across length + product
    samples = []
    # Sort by (product, length descending) so we get diverse coverage
    scored = [(oer_regs[i], i) for i in oer_as_only_idx]
    # Group by length bucket, pick up to 5 per bucket
    by_bucket = defaultdict(list)
    for (r, i) in scored:
        by_bucket[length_bucket(r[2] - r[1])].append((r, i))
    for b in BUCKETS:
        for (r, i) in by_bucket[b][:5]:
            c, s, e, p, n = r
            max_sc = oer_rep["as_only_max_score_v41"][oer_as_only_idx.index(i)] if oer_as_only_idx else 0
            samples.append({"bucket": b, "contig": c, "start": s, "end": e,
                            "length": e - s, "product": p, "n_cds": n,
                            "contig_len": oer_cl.get(c, 0),
                            "v41_max_score_in_region": round(max_sc, 3)})
    with open(OUT_DIR / "as_only_samples.tsv", "w") as fh:
        cols = ["bucket", "contig", "start", "end", "length", "product", "n_cds",
                "contig_len", "v41_max_score_in_region"]
        fh.write("\t".join(cols) + "\n")
        for s in samples:
            fh.write("\t".join(str(s[k]) for k in cols) + "\n")

    # ---------- Comparison markdown ----------
    lines = []
    lines.append("# Coverage diagnostic — OER vs 9-genome\n")
    lines.append("## Top-line ratio")
    lines.append("")
    lines.append("| dataset | antiSMASH regions | v4.1 regions | AS/v4.1 | DNA (Gb) | AS density/Gb | v4.1 density/Gb |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in [oer_rep, ng_rep]:
        total_bp = sum(oer_cl.values()) if r["name"] == "OER" else sum(ng_cl.values())
        lines.append(f"| {r['name']} | {r['n_antismash_regions']} | {r['n_v41_regions']} | "
                     f"{r['n_antismash_regions']/max(1,r['n_v41_regions']):.2f}× | "
                     f"{total_bp/1e9:.2f} | "
                     f"{r['as_region_density_per_Gb']:.1f} | "
                     f"{r['v41_region_density_per_Gb']:.1f} |")

    lines.append("\n## antiSMASH region length distribution")
    lines.append("")
    lines.append("| bucket | OER | 9g | OER % | 9g % |")
    lines.append("|---|---:|---:|---:|---:|")
    total_oer = sum(oer_rep["antismash_length_dist"].values())
    total_ng = sum(ng_rep["antismash_length_dist"].values())
    for b in BUCKETS:
        o = oer_rep["antismash_length_dist"][b]
        n = ng_rep["antismash_length_dist"][b]
        lines.append(f"| {b} | {o} | {n} | {100*o/max(1,total_oer):.1f}% | {100*n/max(1,total_ng):.1f}% |")
    lines.append(f"| **median length** | **{oer_rep.get('antismash_median_length', 0)}** | "
                 f"**{ng_rep.get('antismash_median_length', 0)}** | | |")

    lines.append("\n## Contig length where antiSMASH regions live")
    lines.append("")
    lines.append("| contig bucket | OER | 9g | OER % | 9g % |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in CONTIG_BUCKETS:
        o = oer_rep["as_region_contig_length_dist"][b]
        n = ng_rep["as_region_contig_length_dist"][b]
        lines.append(f"| {b} | {o} | {n} | {100*o/max(1,total_oer):.1f}% | {100*n/max(1,total_ng):.1f}% |")

    lines.append("\n## Category split")
    lines.append("")
    lines.append("| category | OER | 9g |")
    lines.append("|---|---:|---:|")
    for k in ["BOTH_from_antismash_side", "AS_ONLY", "V41_ONLY"]:
        lines.append(f"| {k} | {oer_rep['categories'][k]} | {ng_rep['categories'][k]} |")

    lines.append("\n## AS_ONLY length distribution (v4.1 missed these)")
    lines.append("")
    lines.append("| bucket | OER AS_ONLY | 9g AS_ONLY | OER % | 9g % |")
    lines.append("|---|---:|---:|---:|---:|")
    oas = sum(oer_rep["as_only_length_dist"].values())
    nas = sum(ng_rep["as_only_length_dist"].values())
    for b in BUCKETS:
        o = oer_rep["as_only_length_dist"][b]
        n = ng_rep["as_only_length_dist"][b]
        lines.append(f"| {b} | {o} | {n} | {100*o/max(1,oas):.1f}% | {100*n/max(1,nas):.1f}% |")

    lines.append("\n## AS_ONLY — ORF count distribution")
    lines.append("")
    lines.append("| #CDS bucket | OER AS_ONLY | 9g AS_ONLY |")
    lines.append("|---|---:|---:|")
    for k in ["1-3_CDS", "4-7_CDS", "8-15_CDS", ">15_CDS"]:
        lines.append(f"| {k} | {oer_rep['as_only_ncds_dist'].get(k, 0)} | "
                     f"{ng_rep['as_only_ncds_dist'].get(k, 0)} |")

    lines.append("\n## AS_ONLY — did v4.1 see them?")
    lines.append("")
    lines.append("| case | OER | 9g |")
    lines.append("|---|---:|---:|")
    for k in ["never_covered_no_windows", "covered_low_score(<0.5)",
              "covered_mid_score(0.5-0.95)",
              "covered_high_score(>=0.95)_but_decoder_rejected"]:
        lines.append(f"| {k} | {oer_rep['as_only_v41_signal'].get(k, 0)} | "
                     f"{ng_rep['as_only_v41_signal'].get(k, 0)} |")

    lines.append("\n## AS_ONLY — top 10 antiSMASH product types")
    lines.append("")
    lines.append("**OER:**")
    for p, n in sorted(oer_rep["as_only_product_top10"].items(), key=lambda x: -x[1]):
        lines.append(f"- {p}: {n}")
    lines.append("\n**9g:**")
    for p, n in sorted(ng_rep["as_only_product_top10"].items(), key=lambda x: -x[1]):
        lines.append(f"- {p}: {n}")

    lines.append("\n## antiSMASH regions per contig (clustering on same contig)")
    lines.append("")
    lines.append("| regions/contig | OER | 9g |")
    lines.append("|---|---:|---:|")
    for k in ["1", "2-3", "4-10", ">10"]:
        lines.append(f"| {k} | {oer_rep['as_regions_per_contig_hist'].get(k, 0)} | "
                     f"{ng_rep['as_regions_per_contig_hist'].get(k, 0)} |")
    lines.append(f"| total contigs with AS calls | {oer_rep['as_contigs_total']} | "
                 f"{ng_rep['as_contigs_total']} |")

    lines.append("\n## v4.1 max-score percentiles on AS_ONLY regions")
    lines.append("")
    lines.append("| percentile | OER | 9g |")
    lines.append("|---|---:|---:|")
    for k in ["p10", "p50", "p90", "p99"]:
        lines.append(f"| {k} | {oer_rep['as_only_v41_max_score_percentiles'].get(k, 0)} | "
                     f"{ng_rep['as_only_v41_max_score_percentiles'].get(k, 0)} |")

    with open(OUT_DIR / "comparison.md", "w") as fh:
        fh.write("\n".join(lines))

    # Also a concise summary to stdout
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"OER AS regions: {oer_rep['n_antismash_regions']}, v4.1: {oer_rep['n_v41_regions']} "
          f"({oer_rep['n_antismash_regions']/max(1,oer_rep['n_v41_regions']):.1f}x)")
    print(f" 9g AS regions: {ng_rep['n_antismash_regions']}, v4.1: {ng_rep['n_v41_regions']} "
          f"({ng_rep['n_antismash_regions']/max(1,ng_rep['n_v41_regions']):.1f}x)")
    print(f"\nAS median length: OER={oer_rep.get('antismash_median_length', 0)} | "
          f"9g={ng_rep.get('antismash_median_length', 0)}")
    print(f"\nAS_ONLY count: OER={oer_rep['categories']['AS_ONLY']}, 9g={ng_rep['categories']['AS_ONLY']}")
    print(f"AS_ONLY v4.1 never covered (structural miss): OER={oer_rep['as_only_v41_signal'].get('never_covered_no_windows', 0)}, 9g={ng_rep['as_only_v41_signal'].get('never_covered_no_windows', 0)}")
    print(f"\nOER AS_ONLY ORF distribution:")
    for k in ["1-3_CDS", "4-7_CDS", "8-15_CDS", ">15_CDS"]:
        print(f"  {k}: {oer_rep['as_only_ncds_dist'].get(k, 0)} ({100*oer_rep['as_only_ncds_dist'].get(k, 0)/max(1,oer_rep['categories']['AS_ONLY']):.1f}%)")
    print(f"\nWrote:")
    for f in OUT_DIR.iterdir():
        print(f"  {f}")


if __name__ == "__main__":
    main()
