"""MIBiG 4.0 novelty check for the 15 v4.1 novel candidates.

DIAMOND blastp candidate_orfs.faa vs MIBiG clustercompare proteins.dmnd,
then aggregate per candidate to judge novelty:
  - novel   : best-hit pident < 30% (no ORF has pident >= 30%)
  - variant : 30% <= best-hit pident < 70%
  - known   : best-hit pident >= 70%  (may be a fragment of an already-characterized BGC)

We also report:
  - best_mibig_bgc_id, best_compound, best_biosyn_class
  - number of ORFs with pident >= 30% to the best BGC (syntenic evidence)
  - top 3 MIBiG BGCs by total bitscore summed over candidate's ORFs

Inputs:
  /data/muskliu/oer_bgc_scan/novel_triage/candidate_orfs.faa                       (125 ORFs)
  /data/conda/miniconda/envs/antismash-env/.../clustercompare/mibig/4.0/proteins.dmnd
  /data/syh/benchmark/mibig/mibig_json_4.0/{BGCxxxxxxx}.json                        (compound metadata)

Output:
  /data/muskliu/oer_bgc_scan/novel_triage/mibig_novelty.csv                        (per candidate)
  /data/muskliu/oer_bgc_scan/novel_triage/mibig_blastp_raw.tsv                     (raw diamond hits)
"""
from __future__ import annotations
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CANDIDATES_CSV = Path("/data/muskliu/oer_bgc_scan/novel_triage/candidates_novel.csv")
ORFS_FAA = Path("/data/muskliu/oer_bgc_scan/novel_triage/candidate_orfs.faa")
TRIAGE_CSV = Path("/data/muskliu/oer_bgc_scan/novel_triage/novel_candidates_triage.csv")

DIAMOND = "/data/conda/miniconda/envs/antismash-env/bin/diamond"
MIBIG_DB = Path("/data/conda/miniconda/envs/antismash-env/lib/python3.12/site-packages/antismash/databases/clustercompare/mibig/4.0/proteins.dmnd")
MIBIG_JSON_DIR = Path("/data/syh/benchmark/mibig/mibig_json_4.0")

OUT_DIR = Path("/data/muskliu/oer_bgc_scan/novel_triage")
RAW_TSV = OUT_DIR / "mibig_blastp_raw.tsv"
OUT_CSV = OUT_DIR / "mibig_novelty.csv"

NOVEL_THRESH = 30.0
KNOWN_THRESH = 70.0


def load_mibig_metadata():
    """Return {BGC_id_bare: {compounds: [...], classes: [...], completeness: str, taxonomy: str}}."""
    print(f"[meta] loading MIBiG JSONs from {MIBIG_JSON_DIR}")
    meta = {}
    for jf in sorted(MIBIG_JSON_DIR.glob("BGC*.json")):
        try:
            j = json.load(open(jf))
        except Exception:
            continue
        bgc_id = j.get("accession") or jf.stem
        compounds = j.get("compounds") or []
        names = [c.get("name", "") for c in compounds if c.get("name")]
        classes = []
        bio = j.get("biosynthesis") or {}
        if isinstance(bio, dict):
            cls = bio.get("classes") or []
            for c in cls:
                if isinstance(c, dict):
                    classes.append(c.get("class", ""))
                elif isinstance(c, str):
                    classes.append(c)
        tax = ""
        t = j.get("taxonomy") or {}
        if isinstance(t, dict):
            tax = t.get("name") or ""
        meta[bgc_id] = {
            "compounds": names,
            "classes": classes,
            "completeness": j.get("completeness", ""),
            "taxonomy": tax,
        }
    print(f"  loaded metadata for {len(meta)} MIBiG BGCs")
    return meta


def run_diamond(orfs_faa, db, out_tsv):
    print(f"[diamond] blastp {orfs_faa.name} vs MIBiG 4.0 ({db.stat().st_size/1e6:.0f} MB)")
    # Output: qseqid sseqid pident length qcovhsp scovhsp evalue bitscore
    cmd = [
        DIAMOND, "blastp",
        "--query", str(orfs_faa),
        "--db", str(db),
        "--out", str(out_tsv),
        "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "qcovhsp", "scovhsp", "evalue", "bitscore",
        "--max-target-seqs", "25",     # top 25 MIBiG proteins per query ORF
        "--evalue", "1e-5",
        "--threads", "8",
        "--quiet",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("diamond stderr:", r.stderr)
        raise RuntimeError("diamond blastp failed")
    n = sum(1 for _ in open(out_tsv))
    print(f"  wrote {n} hits to {out_tsv}")


def parse_bgc_id(sseqid):
    """Extract BGC accession from MIBiG header. Format: BGC0000001.5|1 -> BGC0000001"""
    head = sseqid.split("|")[0]
    return head.split(".")[0]


def main():
    if not ORFS_FAA.exists():
        print(f"[!] {ORFS_FAA} missing"); sys.exit(1)

    meta = load_mibig_metadata()

    # Load candidate_orf -> cand_idx map (derived from FAA headers)
    print(f"[orfs] loading ORF -> cand map from {ORFS_FAA}")
    orf_to_cand = {}
    with open(ORFS_FAA) as fh:
        for line in fh:
            if line.startswith(">"):
                header = line[1:].strip()
                orf_id = header.split()[0]   # cand{N}_orf{M}
                m = re.match(r"cand(\d+)_orf\d+", orf_id)
                if m:
                    orf_to_cand[orf_id] = int(m.group(1))
    print(f"  {len(orf_to_cand)} ORFs mapped")

    if not RAW_TSV.exists() or RAW_TSV.stat().st_size == 0:
        run_diamond(ORFS_FAA, MIBIG_DB, RAW_TSV)
    else:
        print(f"[cache] reusing {RAW_TSV}")

    # Aggregate
    # Per ORF: best hit by bitscore
    orf_best = {}
    # Per candidate: (ORF -> best BGC pident) + (BGC -> total bitscore across this cand's ORFs)
    cand_best_per_orf = defaultdict(dict)   # cand -> {orf: (bgc_id, pident, cov_q, cov_s, bitscore)}
    cand_bgc_score = defaultdict(lambda: defaultdict(float))  # cand -> {bgc: total_bitscore}

    with open(RAW_TSV) as fh:
        for line in fh:
            parts = line.rstrip().split("\t")
            if len(parts) < 8: continue
            q, s, pident, length, qcov, scov, ev, bit = parts[:8]
            pident = float(pident); qcov = float(qcov); scov = float(scov); bit = float(bit)
            bgc_id = parse_bgc_id(s)
            if q not in orf_to_cand: continue
            ci = orf_to_cand[q]
            # Best hit per (orf, bgc) collapsing multi-HSP
            key = (q, bgc_id)
            if (q not in orf_best) or bit > orf_best[q][-1]:
                orf_best[q] = (bgc_id, pident, qcov, scov, bit)
            cand_bgc_score[ci][bgc_id] += bit
            prev = cand_best_per_orf[ci].get(q)
            if (prev is None) or bit > prev[-1]:
                cand_best_per_orf[ci][q] = (bgc_id, pident, qcov, scov, bit)

    # Candidate-level summary
    cands = list(csv.DictReader(open(CANDIDATES_CSV)))
    triage = {int(r["cand_idx"]): r for r in csv.DictReader(open(TRIAGE_CSV))}

    rows = []
    for idx, cand in enumerate(cands):
        per_orf = cand_best_per_orf.get(idx, {})
        bgc_scores = cand_bgc_score.get(idx, {})
        # Top-3 BGCs by aggregate bitscore
        top3 = sorted(bgc_scores.items(), key=lambda kv: -kv[1])[:3]
        top_bgc = top3[0][0] if top3 else ""
        top_bgc_names = []
        for bgc_id, bit in top3:
            m = meta.get(bgc_id, {})
            comp = ",".join(m.get("compounds", [])[:2]) or "(no compound)"
            cls = ",".join(m.get("classes", [])) or "?"
            top_bgc_names.append(f"{bgc_id}:{comp} [{cls}] agg_bit={bit:.0f}")

        # Best ORF-level pident to the top BGC
        hits_to_top_bgc = [v for v in per_orf.values() if v[0] == top_bgc]
        best_pident_top = max((v[1] for v in hits_to_top_bgc), default=0.0)
        n_orfs_hit_top_ge30 = sum(1 for v in hits_to_top_bgc if v[1] >= 30.0)
        n_orfs_hit_top_ge70 = sum(1 for v in hits_to_top_bgc if v[1] >= 70.0)

        # Best ORF-level pident across ANY MIBiG BGC
        best_any = max((v[1] for v in per_orf.values()), default=0.0)
        # Number of ORFs with ANY hit >= 30 / >= 70
        orf_ids = set()
        orf_ge30 = set(); orf_ge70 = set()
        for v in per_orf.values():
            orf_ids  # not used
            if v[1] >= 30: orf_ge30.add(v)
            if v[1] >= 70: orf_ge70.add(v)

        total_orfs = int(triage.get(idx, {}).get("n_orfs", 0) or 0)

        # Verdict — use best-per-ORF hit distribution
        #   known   : ≥2 ORFs with pident ≥ 70%
        #   variant : ≥2 ORFs with pident ≥ 30% (but not known-level)
        #   novel   : fewer than 2 ORFs with pident ≥ 30%
        n_ge70 = sum(1 for v in per_orf.values() if v[1] >= 70)
        n_ge30 = sum(1 for v in per_orf.values() if v[1] >= 30)
        if n_ge70 >= 2:
            verdict = f"KNOWN (≥{n_ge70} ORFs >70% to MIBiG)"
        elif n_ge30 >= 2:
            verdict = f"VARIANT ({n_ge30} ORFs >30%, max={best_any:.0f}%)"
        elif n_ge30 == 1:
            verdict = f"LIKELY_NOVEL (only 1 ORF >30%, max={best_any:.0f}%)"
        else:
            verdict = f"NOVEL (max pident to MIBiG = {best_any:.0f}%)"

        row = {
            "cand_idx": idx,
            "v41_type": cand["v41_type"],
            "defrag_v2_type": cand.get("defrag_v2_type", ""),
            "contig": cand["_contig_short"],
            "start": cand["start"], "end": cand["end"],
            "length": cand["_length"],
            "max_score": cand["max_score"],
            "n_orfs": total_orfs,
            "n_orfs_hit_any_mibig": len(per_orf),
            "n_orfs_pident_ge30": n_ge30,
            "n_orfs_pident_ge70": n_ge70,
            "best_pident_any": f"{best_any:.1f}",
            "top_mibig_bgc": top_bgc,
            "top_mibig_compound": ",".join(meta.get(top_bgc, {}).get("compounds", [])[:2]),
            "top_mibig_class": ",".join(meta.get(top_bgc, {}).get("classes", [])),
            "top_mibig_best_pident": f"{best_pident_top:.1f}",
            "top_mibig_n_orfs_ge30": n_orfs_hit_top_ge30,
            "top_mibig_n_orfs_ge70": n_orfs_hit_top_ge70,
            "top3_mibig": " || ".join(top_bgc_names),
            "verdict": verdict,
        }
        rows.append(row)

    # Sort by novelty (NOVEL > VARIANT > KNOWN)
    rank = {"NOVEL": 0, "LIKELY_NOVEL": 1, "VARIANT": 2, "KNOWN": 3}
    def _rk(v):
        for k, n in rank.items():
            if v.startswith(k): return n
        return 9
    rows.sort(key=lambda r: (_rk(r["verdict"]), -float(r["best_pident_any"])))

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[write] {OUT_CSV}")

    # Console summary
    print("\n=== Per-candidate novelty verdict ===\n")
    for r in rows:
        print(f"cand{r['cand_idx']:>2} {r['v41_type']:>10s} | {r['contig']}:{r['start']}-{r['end']} "
              f"({r['length']}bp) score={r['max_score']}")
        print(f"   {r['n_orfs_hit_any_mibig']}/{r['n_orfs']} ORFs hit MIBiG. "
              f"{r['n_orfs_pident_ge30']} ≥30% id, {r['n_orfs_pident_ge70']} ≥70% id. "
              f"max pident = {r['best_pident_any']}%")
        print(f"   top MIBiG: {r['top3_mibig'][:250]}")
        print(f"   VERDICT: {r['verdict']}")
        print()

    # Global summary
    from collections import Counter
    vc = Counter(r["verdict"].split(" ")[0] for r in rows)
    print("\n=== Global summary ===")
    for v, n in vc.most_common():
        print(f"  {v}: {n}")


if __name__ == "__main__":
    main()
