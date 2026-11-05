#!/usr/bin/env python3
"""Compare antiSMASH and evidence-aware NP-Master annotations.

The comparison reports two complementary views:
  * antiSMASH recall: antiSMASH regions covered by an NP-Master region by at
    least a fraction of the antiSMASH interval.
  * NP-Master overlap/novelty: NP-Master regions with any antiSMASH overlap.

It also stratifies raw NP-Master calls vs conservative `safe_pass` calls from
scripts/annotate/safe_bgc_annotation.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def as_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def overlap(a_s: int, a_e: int, b_s: int, b_e: int) -> int:
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def parse_location(loc: str) -> tuple[int, int] | None:
    nums = [(int(a), int(b)) for a, b in re.findall(r"\[(\d+):(\d+)\]", loc or "")]
    if not nums:
        return None
    return min(a for a, _ in nums), max(b for _, b in nums)


def as_product_to_coarse(products: str) -> str:
    s = products.lower()
    if any(x in s for x in ["nrps", "nrp", "nonribosomal"]): return "NRP"
    if any(x in s for x in ["t1pks", "t2pks", "t3pks", "pks", "polyketide"]): return "Polyketide"
    if "terpene" in s: return "Terpene"
    if any(x in s for x in ["ripp", "lanth", "lasso", "thiopeptide", "microcin"]): return "RiPP"
    if any(x in s for x in ["saccharide", "glycosyl", "butyrolactone"]): return "Saccharide"
    if "alkaloid" in s: return "Alkaloid"
    return "Other"


def load_antismash_root(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        json_paths = sorted(sample_dir.glob("*.json"))
        if not json_paths:
            continue
        # Prefer the JSON with the sample directory name.
        preferred = sample_dir / f"{sample_dir.name}.json"
        jp = preferred if preferred.exists() else json_paths[0]
        data = json.load(open(jp))
        for rec in data.get("records", []):
            contig = rec.get("id") or rec.get("name") or ""
            for feat in rec.get("features", []):
                if feat.get("type") != "region":
                    continue
                loc = parse_location(feat.get("location") or "")
                if loc is None:
                    continue
                quals = feat.get("qualifiers", {}) or {}
                product = quals.get("product") or quals.get("products") or [""]
                if isinstance(product, list):
                    product_s = ",".join(str(x) for x in product)
                else:
                    product_s = str(product)
                rows.append({
                    "sample": sample_dir.name,
                    "contig": contig,
                    "start": loc[0],
                    "end": loc[1],
                    "length": loc[1] - loc[0],
                    "products": product_s,
                    "coarse_type": as_product_to_coarse(product_s),
                })
    return rows


def load_npmaster(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = root / "safe_regions.csv"
    if aggregate.exists():
        with open(aggregate) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
        return rows
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        p = sample_dir / "regions.safe.csv"
        if not p.exists():
            p = sample_dir / "regions.csv"
        if not p.exists():
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):
                r.setdefault("sample", sample_dir.name)
                rows.append(r)
    return rows


def best_overlap(query: dict[str, Any], refs_by_key: dict[tuple[str, str], list[dict[str, Any]]]) -> tuple[int, dict[str, Any] | None]:
    best = 0; best_r = None
    key = (str(query.get("sample", "")), str(query.get("contig", "")))
    qs, qe = as_int(query.get("start")), as_int(query.get("end"))
    for r in refs_by_key.get(key, []):
        ov = overlap(qs, qe, as_int(r.get("start")), as_int(r.get("end")))
        if ov > best:
            best = ov; best_r = r
    return best, best_r


def summarize(as_rows: list[dict[str, Any]], np_rows: list[dict[str, Any]], safe_only: bool) -> dict[str, Any]:
    preds = [r for r in np_rows if (str(r.get("safe_pass")) == "True" or not safe_only)]
    as_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pred_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in as_rows:
        as_by_key[(str(r["sample"]), str(r["contig"]))].append(r)
    for r in preds:
        pred_by_key[(str(r.get("sample", "")), str(r.get("contig", "")))].append(r)

    as_cov_any = as_cov_10 = as_cov_50 = 0
    as_type_counter = Counter()
    for a in as_rows:
        ov, p = best_overlap(a, pred_by_key)
        frac = ov / max(1, as_int(a.get("length")))
        if ov > 0: as_cov_any += 1
        if frac >= 0.10: as_cov_10 += 1
        if frac >= 0.50: as_cov_50 += 1
        if p: as_type_counter[a.get("coarse_type", "Other")] += 1

    pred_overlap_any = 0
    type_agree = type_compared = 0
    novel_by_tier = Counter()
    overlap_by_tier = Counter()
    for p in preds:
        ov, a = best_overlap(p, as_by_key)
        tier = str(p.get("evidence_tier") or "unstratified")
        if ov > 0:
            pred_overlap_any += 1
            overlap_by_tier[tier] += 1
            model_type = str(p.get("model_top_type") or p.get("v4_1_type") or "")
            if a and model_type:
                type_compared += 1
                if model_type == a.get("coarse_type"):
                    type_agree += 1
        else:
            novel_by_tier[tier] += 1

    n_as = len(as_rows); n_pred = len(preds)
    return {
        "mode": "safe_pass_only" if safe_only else "all_npmaster_regions",
        "n_antismash": n_as,
        "n_npmaster": n_pred,
        "antismash_covered_any": as_cov_any,
        "antismash_covered_10pct": as_cov_10,
        "antismash_covered_50pct": as_cov_50,
        "antismash_recall_any": as_cov_any / n_as if n_as else 0,
        "antismash_recall_10pct": as_cov_10 / n_as if n_as else 0,
        "antismash_recall_50pct": as_cov_50 / n_as if n_as else 0,
        "npmaster_overlap_any": pred_overlap_any,
        "npmaster_novel_any": n_pred - pred_overlap_any,
        "npmaster_overlap_rate": pred_overlap_any / n_pred if n_pred else 0,
        "type_agreement_on_overlap": type_agree / type_compared if type_compared else 0,
        "type_compared": type_compared,
        "novel_by_tier": dict(novel_by_tier),
        "overlap_by_tier": dict(overlap_by_tier),
        "antismash_overlap_type_counts": dict(as_type_counter),
    }


def per_sample_table(as_rows: list[dict[str, Any]], np_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = sorted({r["sample"] for r in as_rows} | {r.get("sample", "") for r in np_rows})
    out = []
    for sample in samples:
        a = [r for r in as_rows if r["sample"] == sample]
        p = [r for r in np_rows if r.get("sample") == sample]
        psafe = [r for r in p if str(r.get("safe_pass")) == "True"]
        raw = summarize(a, p, safe_only=False)
        safe = summarize(a, p, safe_only=True)
        out.append({
            "sample": sample,
            "antismash_regions": len(a),
            "npmaster_regions": len(p),
            "npmaster_safe_pass": len(psafe),
            "raw_as_recall_50pct": "{:.4f}".format(raw["antismash_recall_50pct"]),
            "safe_as_recall_50pct": "{:.4f}".format(safe["antismash_recall_50pct"]),
            "raw_overlap_rate": "{:.4f}".format(raw["npmaster_overlap_rate"]),
            "safe_overlap_rate": "{:.4f}".format(safe["npmaster_overlap_rate"]),
            "raw_novel": raw["npmaster_novel_any"],
            "safe_novel": safe["npmaster_novel_any"],
        })
    return out


def write_markdown(path: Path, raw: dict[str, Any], safe: dict[str, Any], safe_summary: dict[str, Any] | None) -> None:
    lines = []
    lines.append("# Sequenced Genomes: antiSMASH vs NP-Master Safe Annotation")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Mode | antiSMASH regions | NP-Master regions | AS recall >=50% | NP overlap rate | NP novel | Type agreement on overlaps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in [raw, safe]:
        lines.append(
            "| {mode} | {n_as} | {n_np} | {as_cov}/{n_as} ({as_rec:.3f}) | "
            "{np_ov}/{n_np} ({np_rate:.3f}) | {novel} | {type_agree:.3f} |".format(
                mode=s["mode"],
                n_as=s["n_antismash"],
                n_np=s["n_npmaster"],
                as_cov=s["antismash_covered_50pct"],
                as_rec=s["antismash_recall_50pct"],
                np_ov=s["npmaster_overlap_any"],
                np_rate=s["npmaster_overlap_rate"],
                novel=s["npmaster_novel_any"],
                type_agree=s["type_agreement_on_overlap"],
            )
        )
    lines.append("")
    lines.append("Interpretation: raw NP-Master is the recall-first DNA-model callset; safe_pass is the conservative evidence-aware subset that excludes fragmentary, primary-metabolism-risk, and embedding-only calls.")
    if safe_summary:
        lines.append("")
        lines.append("## NP-Master Evidence Tiers")
        lines.append("")
        for k, v in sorted((safe_summary.get("tiers") or {}).items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Safe Type Labels")
        lines.append("")
        for k, v in sorted((safe_summary.get("safe_type_labels") or {}).items(), key=lambda kv: (-kv[1], kv[0]))[:40]:
            lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--antismash-root", type=Path, required=True)
    ap.add_argument("--npmaster-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    as_rows = load_antismash_root(args.antismash_root)
    np_rows = load_npmaster(args.npmaster_root)
    raw = summarize(as_rows, np_rows, safe_only=False)
    safe = summarize(as_rows, np_rows, safe_only=True)
    safe_summary = None
    if (args.npmaster_root / "safe_summary.json").exists():
        safe_summary = json.load(open(args.npmaster_root / "safe_summary.json"))
    result = {"raw": raw, "safe_pass": safe}
    with open(args.out_dir / "comparison_summary.json", "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    table = per_sample_table(as_rows, np_rows)
    with open(args.out_dir / "comparison_by_sample.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0].keys()) if table else [])
        if table:
            wr.writeheader(); wr.writerows(table)
    write_markdown(args.out_dir / "comparison_report.md", raw, safe, safe_summary)
    print(f"[write] {args.out_dir / 'comparison_summary.json'}")
    print(f"[write] {args.out_dir / 'comparison_by_sample.csv'}")
    print(f"[write] {args.out_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
