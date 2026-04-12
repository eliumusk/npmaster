#!/usr/bin/env python3
"""
Validate top N novel candidates via Pfam domain search.
1. Extract region sequences from genomes
2. prodigal → protein predictions
3. hmmsearch vs Pfam
4. Cross-reference with antiSMASH's biosynthetic Pfam list
"""
import argparse
import csv
import gzip
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

# Key BGC signature domains (Pfam accessions)
BGC_SIGNATURE_DOMAINS = {
    # PKS
    "PF00109": "KS (ketosynthase)",
    "PF02801": "KS_C-terminal",
    "PF00698": "AT (acyltransferase)",
    "PF00975": "TE (thioesterase)",
    "PF14765": "KR (ketoreductase)",
    "PF08659": "KR_C",
    "PF00106": "SDR (short-chain dehydrogenase)",
    # NRP
    "PF00668": "C (condensation)",
    "PF00501": "AMP-binding (adenylation)",
    "PF00550": "PCP/T (peptidyl carrier)",
    "PF13193": "AMP-binding_C",
    # Shared PKS/NRP
    "PF00667": "FAD_binding (enoylreductase)",
    # Terpene
    "PF01397": "Terpene_synth",
    "PF03936": "Terpene_synth_C",
    "PF00494": "SQS_PSY (squalene/phytoene synthase)",
    # RiPP
    "PF13575": "DUF4135 (lanthipeptide)",
    "PF04055": "Radical_SAM",
    "PF00881": "Nitroreductase",
    # Saccharide
    "PF00535": "Glycos_transf_2",
    "PF00534": "Glycos_transf_1",
    # General biosynthetic
    "PF00067": "p450",
    "PF00891": "Methyltransf_2",
    "PF08241": "Methyltransf_11",
    "PF00155": "Aminotran_1_2",
    "PF13561": "Enoyl_CoA_hyd",
    "PF00441": "Acyl_CoA_dh_N",
    "PF02771": "Acyl_CoA_dh_M",
}


def load_fasta(path):
    op = gzip.open if path.endswith(".gz") else open
    contigs = {}
    cur_id = None
    chunks = []
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur_id:
                    contigs[cur_id] = "".join(chunks)
                cur_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if cur_id:
            contigs[cur_id] = "".join(chunks)
    return contigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="candidates_ranked.csv")
    ap.add_argument("--genomes-dir", required=True)
    ap.add_argument("--pfam-db", required=True, help="Pfam-A.hmm path")
    ap.add_argument("--biosynthetic-list", required=True, help="pfam_biosynthetic.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--cpus", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # load biosynthetic pfam list
    biosyn_pfams = set()
    with open(args.biosynthetic_list) as f:
        for line in f:
            line = line.strip()
            if line:
                biosyn_pfams.add(line)
    print(f"Biosynthetic Pfam domains: {len(biosyn_pfams)}")

    # load candidates
    candidates = []
    with open(args.candidates) as f:
        for row in csv.DictReader(f):
            candidates.append(row)
    candidates = candidates[:args.top_n]
    print(f"Analyzing top {len(candidates)} candidates")

    # process each candidate
    results = []
    genome_cache = {}

    for ci, cand in enumerate(candidates):
        gid = cand["genome_id"]
        contig = cand["contig"]
        start = int(cand["start"])
        end = int(cand["end"])
        region_id = f"cand_{ci:03d}_{gid}_{start}_{end}"

        # load genome
        if gid not in genome_cache:
            fasta_path = os.path.join(args.genomes_dir, f"{gid}.fna.gz")
            if not os.path.exists(fasta_path):
                print(f"  SKIP {gid}: fasta not found")
                continue
            genome_cache[gid] = load_fasta(fasta_path)

        contigs = genome_cache[gid]
        seq = contigs.get(contig, "")
        region_seq = seq[start:end]
        if len(region_seq) < 1000:
            print(f"  SKIP {region_id}: too short ({len(region_seq)})")
            continue

        # write region fasta
        region_fasta = out / f"{region_id}.fna"
        with open(region_fasta, "w") as f:
            f.write(f">{region_id}\n{region_seq}\n")

        # prodigal
        protein_fasta = out / f"{region_id}.faa"
        subprocess.run(
            ["prodigal", "-i", str(region_fasta), "-a", str(protein_fasta),
             "-p", "meta", "-q"],
            capture_output=True
        )

        if not protein_fasta.exists() or protein_fasta.stat().st_size == 0:
            results.append({
                "candidate_idx": ci, **cand,
                "n_genes": 0, "n_pfam_hits": 0,
                "n_biosynthetic_hits": 0, "n_bgc_signature_hits": 0,
                "biosynthetic_domains": [], "bgc_signature_domains": [],
            })
            continue

        # count genes
        n_genes = sum(1 for line in open(protein_fasta) if line.startswith(">"))

        # hmmsearch
        domtbl = out / f"{region_id}.domtbl"
        subprocess.run(
            ["hmmsearch", "--domtblout", str(domtbl), "--noali",
             "-E", "1e-5", "--cpu", str(args.cpus),
             args.pfam_db, str(protein_fasta)],
            capture_output=True
        )

        # parse domtbl
        pfam_hits = set()
        all_hits = []
        if domtbl.exists():
            with open(domtbl) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 22:
                        continue
                    target = parts[0]  # protein
                    query_acc = parts[4].split(".")[0]  # PFxxxxx
                    query_name = parts[3]
                    evalue = float(parts[12])
                    pfam_hits.add(query_acc)
                    all_hits.append({
                        "pfam": query_acc, "name": query_name,
                        "evalue": evalue, "protein": target,
                    })

        biosyn_hits = pfam_hits & biosyn_pfams
        signature_hits = {d for d in pfam_hits if d in BGC_SIGNATURE_DOMAINS}

        results.append({
            "candidate_idx": ci,
            "genome_id": gid, "contig": contig,
            "start": start, "end": end, "length": end - start,
            "pred_type": cand["primary_type"],
            "max_prob": cand["max_prob"],
            "confidence": cand["confidence"],
            "dist_to_nearest_bgc": cand["dist_to_nearest_bgc"],
            "n_genes": n_genes,
            "n_pfam_hits": len(pfam_hits),
            "n_biosynthetic_hits": len(biosyn_hits),
            "n_bgc_signature_hits": len(signature_hits),
            "biosynthetic_domains": sorted(biosyn_hits),
            "bgc_signature_domains": {d: BGC_SIGNATURE_DOMAINS[d] for d in sorted(signature_hits)},
            "all_pfam_hits": sorted(pfam_hits),
        })

        sig_str = ", ".join(f"{d}({BGC_SIGNATURE_DOMAINS[d]})" for d in sorted(signature_hits))
        print(f"  [{ci+1:2d}/{len(candidates)}] {region_id}: {n_genes} genes, "
              f"{len(pfam_hits)} pfam, {len(biosyn_hits)} biosyn, "
              f"{len(signature_hits)} BGC-sig [{sig_str}]")

        # cleanup
        region_fasta.unlink(missing_ok=True)
        protein_fasta.unlink(missing_ok=True)
        domtbl.unlink(missing_ok=True)

    # summary
    n_with_biosyn = sum(1 for r in results if r["n_biosynthetic_hits"] > 0)
    n_with_signature = sum(1 for r in results if r["n_bgc_signature_hits"] > 0)
    all_sig_domains = Counter()
    for r in results:
        for d in r.get("bgc_signature_domains", {}):
            all_sig_domains[d] += 1

    summary = {
        "n_candidates_analyzed": len(results),
        "n_with_any_biosynthetic_pfam": n_with_biosyn,
        "pct_with_biosynthetic": round(100 * n_with_biosyn / max(1, len(results)), 1),
        "n_with_bgc_signature_domain": n_with_signature,
        "pct_with_bgc_signature": round(100 * n_with_signature / max(1, len(results)), 1),
        "bgc_signature_domain_frequency": {
            f"{d} ({BGC_SIGNATURE_DOMAINS.get(d, '')})": n
            for d, n in all_sig_domains.most_common()
        },
        "per_candidate": results,
    }

    with open(out / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== SUMMARY ===")
    print(f"Candidates analyzed: {len(results)}")
    print(f"With biosynthetic Pfam:    {n_with_biosyn}/{len(results)} ({100*n_with_biosyn/max(1,len(results)):.0f}%)")
    print(f"With BGC signature domain: {n_with_signature}/{len(results)} ({100*n_with_signature/max(1,len(results)):.0f}%)")
    print(f"Top signature domains:")
    for d, n in all_sig_domains.most_common(10):
        print(f"  {d} ({BGC_SIGNATURE_DOMAINS.get(d,'')}): {n}")


if __name__ == "__main__":
    main()
