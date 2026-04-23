"""HMM + MIBiG novelty on the 429 v4.1-v2 novel candidates.

Pipeline:
  1. Read v41_v2_novel_candidates.csv (429 regions with contig="OER00425668|k141_xxx")
  2. For each: slice DNA from filtered FASTA, pyrodigal -> ORFs (translated)
  3. pyhmmer vs antiSMASH biosynth HMMs (in-memory load, skip disk concat)
  4. pyhmmer vs Pfam-A 35.0
  5. DIAMOND blastp ORFs vs MIBiG 4.0 proteins
  6. Aggregate per candidate -> triage table with verdict + novelty

Output:
  /data/muskliu/npmaster/experiments/OER004256/v41_v2/novel_triage/
     candidate_orfs.faa
     hmm_hits.tsv
     mibig_blastp.tsv
     novel_triage_full.csv
"""
from __future__ import annotations
import csv
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyhmmer
import pyrodigal

# Inputs
CAND_CSV = Path("/data/muskliu/npmaster/experiments/OER004256/v41_v2/v41_v2_novel_candidates.csv")
FILTERED_DIR = Path("/data/muskliu/oer_bgc_scan/filtered")

AS_ROOT = Path("/data/conda/miniconda/envs/antismash-env/lib/python3.12/site-packages/antismash")
BIOSYNTH_SRC_DIRS = [
    AS_ROOT / "detection" / "hmm_detection" / "data",
    AS_ROOT / "detection" / "nrps_pks_domains" / "data",
]
PFAM_DB = AS_ROOT / "databases" / "pfam" / "35.0" / "Pfam-A.hmm"

DIAMOND = "/data/conda/miniconda/envs/antismash-env/bin/diamond"
MIBIG_DB = AS_ROOT / "databases" / "clustercompare" / "mibig" / "4.0" / "proteins.dmnd"
MIBIG_JSON_DIR = Path("/data/syh/benchmark/mibig/mibig_json_4.0")

OUT_DIR = Path("/data/muskliu/npmaster/experiments/OER004256/v41_v2/novel_triage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIOSYNTH_EVAL = 1e-5
PFAM_EVAL = 1e-5

# Reused classification markers
PKS_MARKERS = {"PKS_KS", "KS", "ksdomain", "PKS_AT", "AT", "PKS_TE", "Thioesterase", "PKS_DH",
               "PKS_ER", "PKS_KR", "ACP_PKS", "PKS_PP", "CondensK", "condensing"}
NRPS_MARKERS = {"Condensation", "AMP-binding", "A_domain", "PCP", "PP-binding", "Thioesterase",
                "TD", "Epimerase", "TIGR01733"}
TERPENE_MARKERS = {"T1TS", "T2TS", "Terpene_synth_C", "Terpene_synth", "Terpene_syn",
                   "Lycopene_cycl", "squalene", "TRI5", "SQHop_cyclase_N", "SQHop_cyclase_C"}
RIPP_MARKERS = {"LANC_like", "Lant_dehydr_C", "Lant_dehydr_N", "TOMM", "YcaO", "LAP_dehydr",
                "thiostrepton", "PatD", "PatG", "Flavoprotein", "micJ", "RRE"}
SAC_MARKERS = {"Glyco_transf", "Glycos_transf_1", "DXS", "MGT", "Sugar_tr", "Polysacc_synt",
               "DOIS"}


def classify_domain(hmm_name):
    lname = hmm_name.lower()
    hits = []
    for tag, markers in [("PKS", PKS_MARKERS), ("NRPS", NRPS_MARKERS),
                         ("Terpene", TERPENE_MARKERS), ("RiPP", RIPP_MARKERS),
                         ("Saccharide", SAC_MARKERS)]:
        for m in markers:
            if m.lower() in lname:
                hits.append(tag); break
    return hits


def read_fasta_contigs(fa_path):
    out = {}
    cur_id = None; cur_chunks = []
    with open(fa_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id: out[cur_id] = "".join(cur_chunks).upper()
                cur_id = line[1:].split()[0]; cur_chunks = []
            else:
                cur_chunks.append(line)
        if cur_id: out[cur_id] = "".join(cur_chunks).upper()
    return out


def load_biosynth_hmms():
    hmms = []
    for d in BIOSYNTH_SRC_DIRS:
        for f in sorted(d.glob("*.hmm")):
            with pyhmmer.plan7.HMMFile(str(f)) as hf:
                for hmm in hf: hmms.append(hmm)
    return hmms


def scan_hmms(proteins_path, source, evalue, label):
    alphabet = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.easel.SequenceFile(str(proteins_path), digital=True, alphabet=alphabet) as sf:
        seqs = list(sf)
    if not seqs: return []
    print(f"[{label}] {len(seqs)} proteins (E<{evalue})")
    hits = []
    def consume(profiles):
        for top in pyhmmer.hmmer.hmmscan(queries=seqs, profiles=profiles, E=evalue, cpus=16):
            q_name = top.query.name.decode() if isinstance(top.query.name, bytes) else top.query.name
            for hit in top:
                if hit.evalue > evalue: continue
                hn = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
                hits.append((q_name, hn, hit.evalue, hit.score))
    if isinstance(source, list):
        consume(source)
    else:
        with pyhmmer.plan7.HMMFile(str(source)) as hf:
            consume(hf)
    return hits


def load_mibig_meta():
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
            for c in bio.get("classes") or []:
                if isinstance(c, dict):
                    classes.append(c.get("class", ""))
                elif isinstance(c, str):
                    classes.append(c)
        meta[bgc_id] = {"compounds": names, "classes": classes}
    return meta


def parse_bgc_id(sseqid):
    head = sseqid.split("|")[0]
    return head.split(".")[0]


def main():
    t0 = time.time()
    print(f"[load] candidates from {CAND_CSV}")
    cands = list(csv.DictReader(open(CAND_CSV)))
    print(f"  {len(cands)} candidates")

    # Normalize: add _contig_short, _sample_prefix, _length
    for c in cands:
        full = c["contig"]
        if "|" in full:
            sp, cs = full.split("|", 1)
        else:
            sp, cs = "", full
        c["_sample_prefix"] = sp
        c["_contig_short"] = cs
        c["_length"] = int(c["end"]) - int(c["start"])

    # Step 1: extract ORFs per candidate via pyrodigal
    print("\n[1] pyrodigal ORF extraction")
    fasta_cache = {}
    orf_meta = {}
    pyrodigal_finder = pyrodigal.GeneFinder(meta=True)
    orfs_faa = OUT_DIR / "candidate_orfs.faa"
    per_cand_stats = []

    with open(orfs_faa, "w") as pf:
        for idx, cand in enumerate(cands):
            sp = cand["_sample_prefix"]; cs = cand["_contig_short"]
            s = int(cand["start"]); e = int(cand["end"])
            if sp not in fasta_cache:
                fa = FILTERED_DIR / f"{sp}.ge10000.fa"
                if fa.exists():
                    print(f"  [load] {fa.name}")
                    fasta_cache[sp] = read_fasta_contigs(fa)
                else:
                    fasta_cache[sp] = {}
            seq = fasta_cache[sp].get(cs)
            if seq is None:
                per_cand_stats.append({"cand_idx": idx, "ok": False})
                continue
            contig_len = len(seq)
            region_seq = seq[s:e]
            gc = 100.0 * (region_seq.count("G") + region_seq.count("C")) / max(1, len(region_seq))
            genes = pyrodigal_finder.find_genes(bytes(region_seq, "ascii"))
            per_cand_stats.append({
                "cand_idx": idx, "ok": True,
                "sample": sp, "contig": cs,
                "contig_length": contig_len,
                "region_start": s, "region_end": e, "region_length": e - s,
                "max_score": float(cand["max_score"]),
                "source_scale": cand.get("source_scale", ""),
                "v41_type": cand.get("v41_type", ""),
                "defrag_v2_type": cand.get("defrag_v2_type", ""),
                "v41_type_score": cand.get("v41_type_score", ""),
                "gc_pct": gc, "n_orfs": len(genes),
            })
            for oi, g in enumerate(genes):
                orf_id = f"cand{idx}_orf{oi}"
                pf.write(f">{orf_id}\n{g.translate()}\n")
                orf_meta[orf_id] = {"cand_idx": idx}

    n_orfs = len(orf_meta)
    print(f"  {n_orfs} ORFs from {sum(1 for s in per_cand_stats if s.get('ok'))} candidates")

    # Step 2: HMM scans (biosynth + Pfam)
    print("\n[2] biosynth HMM scan")
    t1 = time.time()
    biosynth_hmms = load_biosynth_hmms()
    print(f"  loaded {len(biosynth_hmms)} biosynth HMMs in {time.time()-t1:.1f}s")
    t1 = time.time()
    biosynth_hits = scan_hmms(orfs_faa, biosynth_hmms, BIOSYNTH_EVAL, "biosynth")
    print(f"  {len(biosynth_hits)} hits in {time.time()-t1:.1f}s")

    print("\n[3] Pfam-A 35.0 scan")
    t1 = time.time()
    pfam_hits = scan_hmms(orfs_faa, PFAM_DB, PFAM_EVAL, "pfam")
    print(f"  {len(pfam_hits)} hits in {time.time()-t1:.1f}s")

    # Step 3: DIAMOND vs MIBiG
    print("\n[4] DIAMOND blastp vs MIBiG 4.0")
    mibig_tsv = OUT_DIR / "mibig_blastp.tsv"
    t1 = time.time()
    cmd = [
        DIAMOND, "blastp", "--query", str(orfs_faa),
        "--db", str(MIBIG_DB), "--out", str(mibig_tsv),
        "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "qcovhsp", "scovhsp", "evalue", "bitscore",
        "--max-target-seqs", "25", "--evalue", "1e-5",
        "--threads", "16", "--quiet",
    ]
    subprocess.run(cmd, check=True)
    print(f"  done in {time.time()-t1:.1f}s")

    # Step 4: Aggregate per candidate
    print("\n[5] Aggregate")
    meta = load_mibig_meta()
    print(f"  MIBiG metadata: {len(meta)} BGCs")

    # Per-candidate biosynth + pfam
    cand_biosynth = defaultdict(list)
    cand_pfam = defaultdict(list)
    for (orf, hmm, ev, sc) in biosynth_hits:
        if orf in orf_meta:
            cand_biosynth[orf_meta[orf]["cand_idx"]].append((hmm, ev, sc))
    for (orf, hmm, ev, sc) in pfam_hits:
        if orf in orf_meta:
            cand_pfam[orf_meta[orf]["cand_idx"]].append((hmm, ev, sc))

    # Per-candidate MIBiG
    cand_orf_mibig_best = defaultdict(dict)   # cand_idx -> {orf: (bgc, pident, bit)}
    cand_bgc_agg_bit = defaultdict(lambda: defaultdict(float))
    with open(mibig_tsv) as fh:
        for line in fh:
            parts = line.rstrip().split("\t")
            if len(parts) < 8: continue
            q, s, pident, length, qcov, scov, ev, bit = parts[:8]
            pident = float(pident); bit = float(bit)
            bgc = parse_bgc_id(s)
            if q not in orf_meta: continue
            ci = orf_meta[q]["cand_idx"]
            cand_bgc_agg_bit[ci][bgc] += bit
            prev = cand_orf_mibig_best[ci].get(q)
            if (prev is None) or bit > prev[2]:
                cand_orf_mibig_best[ci][q] = (bgc, pident, bit)

    # Build output rows
    out_rows = []
    for stats in per_cand_stats:
        if not stats.get("ok"):
            continue
        ci = stats["cand_idx"]
        bh = cand_biosynth.get(ci, [])
        ph = cand_pfam.get(ci, [])
        # classes
        tag_counts = defaultdict(int)
        for (h, _e, _s) in bh:
            for t in classify_domain(h):
                tag_counts[t] += 1

        # Top hits
        bh_top = sorted(bh, key=lambda x: x[1])[:5]
        biosynth_str = "; ".join(f"{h}({e:.1e})" for (h, e, _s) in bh_top)

        # Verdict 1: biosynth strength
        if len(bh) >= 3:
            bio_verdict = "STRONG"
        elif len(bh) >= 1:
            bio_verdict = "MODERATE"
        elif len(ph) >= 5:
            bio_verdict = "WEAK_pfam_only"
        else:
            bio_verdict = "LIKELY_FP"

        # Novelty: MIBiG
        orf_best = cand_orf_mibig_best.get(ci, {})
        per_orf_pident = [v[1] for v in orf_best.values()]
        n_ge30 = sum(1 for p in per_orf_pident if p >= 30)
        n_ge70 = sum(1 for p in per_orf_pident if p >= 70)
        max_pident = max(per_orf_pident) if per_orf_pident else 0.0
        if n_ge70 >= 2:
            novelty = "KNOWN"
        elif n_ge30 >= 2:
            novelty = "VARIANT"
        elif n_ge30 == 1:
            novelty = "LIKELY_NOVEL"
        else:
            novelty = "NOVEL"
        # Top MIBiG hit
        bgc_scores = cand_bgc_agg_bit.get(ci, {})
        top3 = sorted(bgc_scores.items(), key=lambda kv: -kv[1])[:3]
        top_mibig = ""
        if top3:
            bgc0, agg0 = top3[0]
            m = meta.get(bgc0, {})
            comp = ",".join(m.get("compounds", [])[:2]) or "(no compound)"
            cls = ",".join(m.get("classes", [])) or "?"
            top_mibig = f"{bgc0}:{comp}[{cls}]agg_bit={agg0:.0f}"

        row = dict(stats)
        row["n_biosynth_hits"] = len(bh)
        row["n_biosynth_orfs"] = len(set(o for (o, _, _, _) in [(orf, hmm, e, s) for (orf, hmm, e, s) in biosynth_hits if orf in orf_meta and orf_meta[orf]["cand_idx"] == ci]))
        row["biosynth_classes"] = ",".join(f"{k}:{v}" for k, v in sorted(tag_counts.items()))
        row["top_biosynth_hits"] = biosynth_str
        row["n_pfam_hits"] = len(ph)
        row["bio_verdict"] = bio_verdict
        row["max_mibig_pident"] = round(max_pident, 1)
        row["n_orfs_pident_ge30"] = n_ge30
        row["n_orfs_pident_ge70"] = n_ge70
        row["top_mibig_hit"] = top_mibig
        row["novelty_verdict"] = novelty
        # Combined flag
        row["PASS_for_wetlab"] = bio_verdict in ("STRONG", "MODERATE") and novelty in ("NOVEL", "LIKELY_NOVEL", "VARIANT")
        out_rows.append(row)

    # Sort: PASS + STRONG + NOVEL first, then by biosynth count
    out_rows.sort(key=lambda r: (not r["PASS_for_wetlab"],
                                 0 if r["bio_verdict"] == "STRONG" else (1 if r["bio_verdict"] == "MODERATE" else 2),
                                 -r["n_biosynth_hits"]))

    triage_csv = OUT_DIR / "novel_triage_full.csv"
    if out_rows:
        cols = list(out_rows[0].keys())
        with open(triage_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in out_rows: w.writerow(r)
    print(f"\n[write] {triage_csv}  ({len(out_rows)} candidates)")

    # Summary
    from collections import Counter
    bv = Counter(r["bio_verdict"] for r in out_rows)
    nv = Counter(r["novelty_verdict"] for r in out_rows)
    passes = sum(1 for r in out_rows if r["PASS_for_wetlab"])
    print(f"\n[summary] {len(out_rows)} total candidates")
    print(f"  biosynth verdict: {dict(bv)}")
    print(f"  novelty verdict:  {dict(nv)}")
    print(f"  PASS_for_wetlab:  {passes} ({passes/max(1,len(out_rows)):.1%})")
    # Cross-tab
    ct = defaultdict(int)
    for r in out_rows:
        ct[(r["bio_verdict"], r["novelty_verdict"])] += 1
    print(f"\n  bio × novelty cross-tab:")
    for (bv_, nv_), n in sorted(ct.items(), key=lambda x: -x[1]):
        print(f"    {bv_:>15s} × {nv_:>15s}: {n}")

    print(f"\nTOTAL {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
