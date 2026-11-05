#!/usr/bin/env python3
"""Evidence-aware safety layer for NP-Master BGC annotations.

This script does *not* replace the DNA-model detector. It takes NP-Master
region calls and adds conservative biological interpretation so broad model
labels (for example "Polyketide") are not over-read as complete mechanism-level
BGC annotations.

Inputs are the NP-Master web/batch output layout:
  <npmaster-root>/<genome>/regions.csv
  <npmaster-root>/<genome>/cds_features.json
  <npmaster-root>/<genome>/mibig_hits.json        (optional)
  <npmaster-root>/<genome>/input.fasta.fai        (optional)

Outputs:
  <genome>/regions.safe.csv
  <out-root>/safe_regions.csv
  <out-root>/safe_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BGC_TYPES = ["Alkaloid", "Terpene", "NRP", "Polyketide", "RiPP", "Saccharide", "Other"]

# Domain/product term sets are intentionally conservative and mechanism-aware.
PKS_KS_TERMS = ["ketoacyl-synt", "ketosynthase", "pks_ks", "ks domain", " beta-ketoacyl", "fabf", "type iii pks", "chal_sti_synt"]
PKS_AT_TERMS = ["acyl_transf_1", "acyltransferase", "pks_at", "malonyl-coa:acp transacylase"]
PKS_ACP_TERMS = ["pp-binding", "acyl carrier", " acp", "acp_", "phosphopantetheine"]
PKS_REDUCING_TERMS = ["pks_kr", " kr", "ketoreductase", "dehydratase", "pks_dh", "enoyl", "pks_er", "thioesterase"]
TYPEII_PKS_TERMS = ["t2pks", "t2ks", "chain length factor", "clf", "act_ks", "type ii"]
FAS_RISK_TERMS = ["fatty acid", "fabf", "fabh", "fabd", "fabg", "fabl", "t2fas", "beta-ketoacyl-acp"]

NRPS_A_TERMS = ["amp-binding", "adenylation", "a_domain", "a domain", "acyl-coa synthetase"]
NRPS_C_TERMS = ["condensation", "condens", "c_domain", "c domain"]
NRPS_PCP_TERMS = ["pp-binding", "pcp", "peptidyl carrier", "thiolation"]
NRPS_TE_TERMS = ["thioesterase", " te domain"]

RIPP_TERMS = ["ycaO".lower(), "lanc_like", "lanth", "tomm", "microcin", "thiostrepton", "ripp", "radical sam", "lap_dehydr"]
TERPENE_TERMS = ["terpene", "prenyltransferase", "squalene", "phytoene", "lycopene", "isoprenoid", "hopene", "cyclase"]
SACCHARIDE_TERMS = ["glycosyl", "glyco_transf", "sugar", "polysacc", "dtdp", "mgt", "sacch"]
TAILORING_TERMS = ["p450", "methyltransferase", "oxidoreductase", "monooxygenase", "dioxygenase", "dehydrogenase", "aminotransferase", "halogenase", "glycosyltransferase", "radical sam", "reductase", "hydroxylase", "cyclase"]
TRANSPORT_TERMS = ["transporter", "abc", "efflux", "mate", "permease", "export"]
REGULATORY_TERMS = ["regulator", "transcription", "response regulator", "luxr", "tetr", "lysr", "hth"]
PRIMARY_RISK_TERMS = ["ribosomal protein", "dna polymerase", "rna polymerase", "trna", "rrna", "housekeeping", "central metabolism"]


def parse_type_scores(raw: str | None) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not raw:
        return scores
    for part in str(raw).split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            scores[k.strip()] = float(v)
        except ValueError:
            continue
    return scores


def type_stats(row: dict[str, str]) -> dict[str, Any]:
    scores = parse_type_scores(row.get("v4_1_type_scores") or row.get("evo2_lr_type_scores"))
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_type, top_score = ranked[0]
        top2_type, top2_score = ranked[1] if len(ranked) > 1 else ("", math.nan)
        margin = top_score - top2_score if len(ranked) > 1 else math.nan
        n_high = sum(v >= 0.5 for v in scores.values())
    else:
        top_type = row.get("v4_1_type") or row.get("evo2_lr_type") or row.get("type") or ""
        try:
            top_score = float(row.get("v4_1_type_score") or row.get("evo2_lr_type_score") or "nan")
        except ValueError:
            top_score = math.nan
        top2_type = row.get("v4_1_type_top2") or row.get("evo2_lr_type_top2") or ""
        top2_score = math.nan
        margin = math.nan
        n_high = 0
    if math.isnan(margin):
        conf = "unknown"
    elif margin < 0.10:
        conf = "low"
    elif margin < 0.20:
        conf = "medium"
    else:
        conf = "high"
    return {
        "model_top_type": top_type,
        "model_top_score": top_score,
        "model_top2_type": top2_type,
        "model_top2_score": top2_score,
        "type_margin": margin,
        "type_confidence": conf,
        "n_type_scores_ge_0_5": n_high,
    }


def read_fai(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                try:
                    out[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    return out


def text_for_cds(cds: dict[str, Any]) -> str:
    bits: list[str] = []
    for key in ("product", "function_class", "locus_tag"):
        if cds.get(key):
            bits.append(str(cds[key]))
    for dom in cds.get("pfam_domains") or []:
        for key in ("name", "accession", "description"):
            if dom.get(key):
                bits.append(str(dom[key]))
    return " ".join(bits).lower()


def has_any(text: str, terms: list[str]) -> bool:
    return any(t.lower() in text for t in terms)


def collect_evidence(cds_list: list[dict[str, Any]], mibig_hits: list[dict[str, Any]]) -> dict[str, Any]:
    ev: dict[str, Any] = Counter()
    cds_texts = [text_for_cds(c) for c in cds_list]
    ev["n_cds"] = len(cds_list)
    ev["max_cds_aa"] = max([int(c.get("length_aa") or 0) for c in cds_list] or [0])
    ev["n_long_cds_ge_1000aa"] = sum(1 for c in cds_list if int(c.get("length_aa") or 0) >= 1000)
    ev["n_small_cds_le_120aa"] = sum(1 for c in cds_list if 0 < int(c.get("length_aa") or 0) <= 120)
    ev["has_pks_ks"] = any(has_any(t, PKS_KS_TERMS) for t in cds_texts)
    ev["has_pks_at"] = any(has_any(t, PKS_AT_TERMS) for t in cds_texts)
    ev["has_pks_acp"] = any(has_any(t, PKS_ACP_TERMS) for t in cds_texts)
    ev["has_pks_reducing"] = any(has_any(t, PKS_REDUCING_TERMS) for t in cds_texts)
    ev["has_typeii_pks_signal"] = any(has_any(t, TYPEII_PKS_TERMS) for t in cds_texts)
    ev["has_fas_risk"] = any(has_any(t, FAS_RISK_TERMS) for t in cds_texts)
    ev["has_nrps_a"] = any(has_any(t, NRPS_A_TERMS) for t in cds_texts)
    ev["has_nrps_c"] = any(has_any(t, NRPS_C_TERMS) for t in cds_texts)
    ev["has_nrps_pcp"] = any(has_any(t, NRPS_PCP_TERMS) for t in cds_texts)
    ev["has_nrps_te"] = any(has_any(t, NRPS_TE_TERMS) for t in cds_texts)
    ev["has_ripp_mod"] = any(has_any(t, RIPP_TERMS) for t in cds_texts)
    ev["has_terpene_core"] = any(has_any(t, TERPENE_TERMS) for t in cds_texts)
    ev["has_saccharide_core"] = any(has_any(t, SACCHARIDE_TERMS) for t in cds_texts)
    ev["n_tailoring_cds"] = sum(1 for t in cds_texts if has_any(t, TAILORING_TERMS))
    ev["n_transport_cds"] = sum(1 for t in cds_texts if has_any(t, TRANSPORT_TERMS))
    ev["n_regulatory_cds"] = sum(1 for t in cds_texts if has_any(t, REGULATORY_TERMS))
    ev["n_primary_risk_cds"] = sum(1 for t in cds_texts if has_any(t, PRIMARY_RISK_TERMS))
    ev["n_core_signals"] = sum(bool(ev[k]) for k in [
        "has_pks_ks", "has_pks_at", "has_pks_acp", "has_nrps_a", "has_nrps_c",
        "has_nrps_pcp", "has_ripp_mod", "has_terpene_core", "has_saccharide_core",
    ])
    ev["n_mibig_hits"] = len(mibig_hits)
    max_ident = 0.0
    top_product = ""
    for hit in mibig_hits:
        try:
            ident = float(hit.get("identity") or 0.0)
        except (TypeError, ValueError):
            ident = 0.0
        if ident > max_ident:
            max_ident = ident
            top_product = str(hit.get("product") or "")
    ev["max_mibig_identity"] = max_ident
    ev["top_mibig_product"] = top_product
    return dict(ev)


def mechanism_label(model_type: str, ev: dict[str, Any], type_confidence: str) -> str:
    low = type_confidence in {"low", "unknown"}
    if model_type == "Polyketide":
        if ev["has_pks_ks"] and ev["has_pks_at"] and ev["has_pks_acp"] and ev["max_cds_aa"] >= 1000:
            return "typeI_PKS_supported"
        if ev["has_typeii_pks_signal"] or (ev["has_pks_ks"] and not ev["has_pks_at"]):
            return "typeII_or_FAS_like_risk" if ev["has_fas_risk"] else "typeII_PKS_like"
        if ev["has_pks_ks"] or ev["has_pks_reducing"] or ev["has_pks_acp"]:
            return "partial_PKS_like"
        return "polyketide_embedding_only_ambiguous" if low else "polyketide_like_no_domain_support"
    if model_type == "NRP":
        if ev["has_nrps_a"] and ev["has_nrps_pcp"] and (ev["has_nrps_c"] or ev["has_nrps_te"]):
            return "NRPS_supported"
        if ev["has_nrps_a"] or ev["has_nrps_pcp"] or ev["has_nrps_c"]:
            return "partial_NRPS_like"
        return "NRP_embedding_only_ambiguous" if low else "NRP_like_no_domain_support"
    if model_type == "RiPP":
        if ev["has_ripp_mod"] and ev["n_small_cds_le_120aa"] >= 1:
            return "RiPP_supported"
        if ev["has_ripp_mod"]:
            return "RiPP_like_no_clear_precursor"
        return "RiPP_embedding_only_ambiguous" if low else "RiPP_like_no_domain_support"
    if model_type == "Terpene":
        if ev["has_terpene_core"]:
            return "terpene_supported"
        return "terpene_embedding_only_ambiguous" if low else "terpene_like_no_domain_support"
    if model_type == "Saccharide":
        if ev["has_saccharide_core"] and ev["n_tailoring_cds"] >= 1:
            return "saccharide_supported"
        return "saccharide_embedding_only_ambiguous" if low else "saccharide_like_no_domain_support"
    if model_type == "Alkaloid":
        return "alkaloid_like_tentative" if not low else "alkaloid_embedding_only_ambiguous"
    if model_type == "Other":
        if ev["n_core_signals"] or ev["n_tailoring_cds"] >= 2:
            return "other_biosynthetic_supported"
        return "other_embedding_only_ambiguous"
    return "untyped_bgc_like"


def evidence_tier(length_bp: int, at_edge: bool, ev: dict[str, Any], label: str, type_confidence: str) -> str:
    has_bio = ev["n_core_signals"] >= 2 or ev["n_tailoring_cds"] >= 2 or ev["max_mibig_identity"] >= 0.35
    has_strong = ev["n_core_signals"] >= 3 or label.endswith("_supported") or ev["max_mibig_identity"] >= 0.70
    if length_bp < 3000 or ev["n_cds"] <= 2:
        return "Tier4_fragmentary_or_single_gene"
    if at_edge or length_bp < 5000:
        return "Tier4_fragmentary_or_contig_edge"
    if not has_bio:
        return "Tier5_embedding_only_or_likely_FP"
    if "FAS_like_risk" in label or ev["n_primary_risk_cds"] >= max(2, ev["n_cds"] // 2):
        return "Tier5_primary_metabolism_risk"
    if has_strong and type_confidence == "high":
        return "Tier1_known_like_high_confidence"
    if has_strong:
        return "Tier2_biosynthetic_supported"
    return "Tier3_novel_architecture_or_low_confidence"


def safe_pass(tier: str, label: str) -> bool:
    if tier.startswith("Tier5"):
        return False
    if tier.startswith("Tier4"):
        return False
    if "embedding_only" in label or "no_domain_support" in label:
        return False
    return True


def annotate_genome(genome_dir: Path, out_dir: Path | None = None) -> list[dict[str, Any]]:
    regions_csv = genome_dir / "regions.csv"
    cds_json = genome_dir / "cds_features.json"
    if not regions_csv.exists() or not cds_json.exists():
        return []
    out_dir = out_dir or genome_dir
    with open(regions_csv) as fh:
        rows = list(csv.DictReader(fh))
    cds_by_region = json.load(open(cds_json))
    mibig_by_region = {}
    if (genome_dir / "mibig_hits.json").exists():
        mibig_by_region = json.load(open(genome_dir / "mibig_hits.json"))
    contig_lens = read_fai(genome_dir / "input.fasta.fai")
    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        bgc_id = f"BGC_{i:04d}"
        start = int(float(row.get("start") or 0))
        end = int(float(row.get("end") or 0))
        length_bp = max(0, end - start)
        contig = row.get("contig", "")
        contig_len = contig_lens.get(contig, 0)
        at_start = start <= 500
        at_end = bool(contig_len and contig_len - end <= 500)
        tstats = type_stats(row)
        cds_list = cds_by_region.get(bgc_id, [])
        ev = collect_evidence(cds_list, mibig_by_region.get(bgc_id, []))
        label = mechanism_label(str(tstats["model_top_type"]), ev, str(tstats["type_confidence"]))
        tier = evidence_tier(length_bp, at_start or at_end, ev, label, str(tstats["type_confidence"]))
        out = dict(row)
        out.update({
            "bgc_id": bgc_id,
            "region_length_bp": length_bp,
            "contig_length_bp": contig_len,
            "at_contig_start": at_start,
            "at_contig_end": at_end,
            **{k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in tstats.items()},
            **ev,
            "safe_type_label": label,
            "evidence_tier": tier,
            "safe_pass": safe_pass(tier, label),
        })
        out_rows.append(out)
    if out_rows:
        out_path = out_dir / "regions.safe.csv"
        fieldnames: list[str] = []
        for r in out_rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(out_path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fieldnames)
            wr.writeheader(); wr.writerows(out_rows)
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npmaster-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args()
    out_root = args.out_root or args.npmaster_root
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for genome_dir in sorted(p for p in args.npmaster_root.iterdir() if p.is_dir()):
        rows = annotate_genome(genome_dir)
        for r in rows:
            r["sample"] = genome_dir.name
        all_rows.extend(rows)
        if rows:
            print(f"[safe] {genome_dir.name}: {len(rows)} regions")
    if not all_rows:
        raise SystemExit("no NP-Master regions found")
    fieldnames: list[str] = []
    for r in all_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_root / "safe_regions.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader(); wr.writerows(all_rows)
    summary = {
        "n_samples": len({r["sample"] for r in all_rows}),
        "n_regions": len(all_rows),
        "safe_pass": sum(str(r.get("safe_pass")) == "True" for r in all_rows),
        "tiers": Counter(str(r.get("evidence_tier")) for r in all_rows),
        "safe_type_labels": Counter(str(r.get("safe_type_label")) for r in all_rows),
        "model_top_types": Counter(str(r.get("model_top_type")) for r in all_rows),
    }
    # Convert Counters for JSON.
    summary = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in summary.items()}
    with open(out_root / "safe_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[write] {out_root / 'safe_regions.csv'}")
    print(f"[write] {out_root / 'safe_summary.json'}")


if __name__ == "__main__":
    main()
