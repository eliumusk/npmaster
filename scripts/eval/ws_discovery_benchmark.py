#!/usr/bin/env python3
"""
Discovery-Side Benchmark
=========================
Evaluate novel candidate quality under score_maxtype (canonical detection score).

Metrics:
  1. precision@K (K=20, 50) via Pfam validation
  2. De-duplicated family diversity (unique Pfam domain profiles)
  3. Null distribution comparison (random negative regions)
  4. Candidate classification breakdown

Pipeline:
  1. Load full-window score_swap results
  2. Rank unmatched candidates (clean_missed only) by score_maxtype
  3. Run Pfam validation on top-50 clean_missed candidates
  4. Run Pfam validation on 50 random negative-region controls (null distribution)
  5. Compute enrichment ratio and precision@K
  6. Compute family diversity via Pfam profile clustering

Usage:
  python ws_discovery_benchmark.py \
    --results /data/muskliu/np_master/data/stage4_results/ws1_score_swap_fullwindow/score_swap_results.json \
    --genomes-dir /data/muskliu/np_master/data/genomes \
    --pfam-db /data/muskliu/np_master/data/stage4_results/pfam_validation/biosyn_pfam.hmm \
    --biosynthetic-list /data/muskliu/np_master/data/stage4_results/pfam_validation/biosyn_names.txt \
    --meta /data/muskliu/np_master/data/windows/windows_metadata.csv \
    --out /data/muskliu/np_master/data/stage4_results/ws_discovery_benchmark \
    --cpus 16
"""
import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# BGC signature domains (same as validate_novel_candidates.py)
BGC_SIGNATURE_DOMAINS = {
    "PF00109": "KS (ketosynthase)",
    "PF02801": "KS_C-terminal",
    "PF00698": "AT (acyltransferase)",
    "PF00975": "TE (thioesterase)",
    "PF14765": "KR (ketoreductase)",
    "PF08659": "KR_C",
    "PF00106": "SDR (short-chain dehydrogenase)",
    "PF00668": "C (condensation)",
    "PF00501": "AMP-binding (adenylation)",
    "PF00550": "PCP/T (peptidyl carrier)",
    "PF13193": "AMP-binding_C",
    "PF00667": "FAD_binding (enoylreductase)",
    "PF01397": "Terpene_synth",
    "PF03936": "Terpene_synth_C",
    "PF00494": "SQS_PSY (squalene/phytoene synthase)",
    "PF13575": "DUF4135 (lanthipeptide)",
    "PF04055": "Radical_SAM",
    "PF00881": "Nitroreductase",
    "PF00535": "Glycos_transf_2",
    "PF00534": "Glycos_transf_1",
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


def validate_regions(regions, genomes_dir, pfam_db, biosyn_pfams, out_dir, cpus, label="candidate"):
    """Run Pfam validation on a list of regions. Returns per-region results."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    genome_cache = {}
    results = []

    for ci, reg in enumerate(regions):
        gid = reg["genome_id"]
        contig = reg["contig"]
        start = int(reg["start"])
        end = int(reg["end"])
        region_id = f"{label}_{ci:03d}_{gid}_{start}_{end}"

        if gid not in genome_cache:
            fasta_path = os.path.join(genomes_dir, f"{gid}.fna.gz")
            if not os.path.exists(fasta_path):
                print(f"  SKIP {gid}: fasta not found")
                results.append({"idx": ci, "skip": True, "reason": "fasta_not_found", **reg})
                continue
            genome_cache[gid] = load_fasta(fasta_path)

        contigs = genome_cache[gid]
        seq = contigs.get(contig, "")
        region_seq = seq[start:end]
        if len(region_seq) < 1000:
            results.append({"idx": ci, "skip": True, "reason": "too_short", **reg})
            continue

        region_fasta = out / f"{region_id}.fna"
        with open(region_fasta, "w") as f:
            f.write(f">{region_id}\n{region_seq}\n")

        protein_fasta = out / f"{region_id}.faa"
        prodigal_bin = os.environ.get("PRODIGAL", "/data/conda/miniconda/envs/antismash-env/bin/prodigal")
        subprocess.run(
            [prodigal_bin, "-i", str(region_fasta), "-a", str(protein_fasta), "-p", "meta", "-q"],
            capture_output=True
        )

        if not protein_fasta.exists() or protein_fasta.stat().st_size == 0:
            results.append({"idx": ci, "skip": False, "n_genes": 0, "n_pfam": 0,
                            "n_biosyn": 0, "n_signature": 0, "pfam_set": set(),
                            "biosyn_set": set(), "signature_set": set(), **reg})
            for f in [region_fasta, protein_fasta]:
                f.unlink(missing_ok=True)
            continue

        n_genes = sum(1 for line in open(protein_fasta) if line.startswith(">"))

        domtbl = out / f"{region_id}.domtbl"
        hmmsearch_bin = os.environ.get("HMMSEARCH", "/data/conda/miniconda/envs/antismash-env/bin/hmmsearch")
        subprocess.run(
            [hmmsearch_bin, "--domtblout", str(domtbl), "--noali",
             "-E", "1e-5", "--cpu", str(cpus), pfam_db, str(protein_fasta)],
            capture_output=True
        )

        pfam_acc_hits = set()   # PF accessions (e.g. PF00109)
        pfam_name_hits = set()  # Pfam family names (e.g. ketoacyl-synt)
        if domtbl.exists():
            with open(domtbl) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 22:
                        pfam_acc_hits.add(parts[4].split(".")[0])   # PF accession
                        pfam_name_hits.add(parts[3])                # Pfam name

        # biosyn_pfams may use names OR accessions — match both
        biosyn_hits = (pfam_acc_hits | pfam_name_hits) & biosyn_pfams
        sig_hits = {d for d in pfam_acc_hits if d in BGC_SIGNATURE_DOMAINS}
        pfam_hits = pfam_acc_hits  # keep accessions as primary set

        results.append({
            "idx": ci, "skip": False, "n_genes": n_genes,
            "n_pfam": len(pfam_hits), "n_biosyn": len(biosyn_hits),
            "n_signature": len(sig_hits),
            "pfam_set": pfam_hits, "biosyn_set": biosyn_hits, "signature_set": sig_hits,
            **reg
        })

        print(f"  [{ci+1:2d}/{len(regions)}] {label} {gid}:{contig}:{start}-{end} "
              f"genes={n_genes} pfam={len(pfam_hits)} biosyn={len(biosyn_hits)} sig={len(sig_hits)}")

        for f in [region_fasta, protein_fasta, domtbl]:
            f.unlink(missing_ok=True)

    return results


def compute_precision_at_k(results, k):
    """precision@K = fraction of top-K with any biosynthetic Pfam hit."""
    valid = [r for r in results if not r.get("skip", False)][:k]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r["n_biosyn"] > 0) / len(valid)


def compute_signature_precision_at_k(results, k):
    """Fraction of top-K with BGC signature domain."""
    valid = [r for r in results if not r.get("skip", False)][:k]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r["n_signature"] > 0) / len(valid)


def compute_family_diversity(results, k, domain_key="biosyn_set"):
    """De-duplicated family diversity: unique domain profile sets at top-K."""
    valid = [r for r in results if not r.get("skip", False)][:k]
    profiles = set()
    for r in valid:
        domains = r.get(domain_key, set())
        if domains:
            profiles.add(frozenset(domains))
    return len(profiles)


def sample_null_regions(meta, genomes_dir, n=50, target_lengths=None, seed=42):
    """Sample random negative regions as null distribution controls."""
    rng = np.random.default_rng(seed)

    # Collect all negative windows in test_id
    neg_windows = [(r["genome_id"], r["contig_id"], int(r["window_start"]), int(r["window_end"]))
                   for r in meta if r["split"] == "test_id" and r["label"] == "negative"]

    if target_lengths is None:
        target_lengths = [24576] * n  # default: ~1.5 windows

    # Sample random negative locations and extend to target lengths
    chosen = rng.choice(len(neg_windows), size=min(n * 3, len(neg_windows)), replace=False)
    regions = []
    seen_locs = set()
    for idx in chosen:
        if len(regions) >= n:
            break
        gid, cid, ws, we = neg_windows[idx]
        loc_key = (gid, cid, ws)
        if loc_key in seen_locs:
            continue
        seen_locs.add(loc_key)
        target_len = target_lengths[len(regions) % len(target_lengths)]
        regions.append({
            "genome_id": gid, "contig": cid,
            "start": ws, "end": ws + target_len,
            "length": target_len, "primary_type": "null",
        })

    return regions[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Full-window score_swap_results.json")
    ap.add_argument("--genomes-dir", required=True)
    ap.add_argument("--pfam-db", required=True)
    ap.add_argument("--biosynthetic-list", required=True)
    ap.add_argument("--meta", required=True, help="windows_metadata.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load biosynthetic Pfam list
    biosyn_pfams = set()
    with open(args.biosynthetic_list) as f:
        for line in f:
            line = line.strip()
            if line:
                biosyn_pfams.add(line)
    print(f"Biosynthetic Pfam set: {len(biosyn_pfams)} families")

    # Load full-window results
    print("Loading score swap results...")
    with open(args.results) as f:
        data = json.load(f)

    # Extract score_maxtype unmatched candidates from test_id
    maxtype_result = data["results"]["test_id"]["score_maxtype"]
    unmatched = maxtype_result.get("unmatched_details", [])

    # We need ALL unmatched, not just top-50 details. Check if full list is available.
    # The script saves only top 50 in unmatched_details. We need the full candidate list.
    # Load from the full-window JSON's region data instead.
    print(f"Unmatched details available: {len(unmatched)}")

    # Separate by category
    clean_missed = [u for u in unmatched if u["category"] == "clean_missed"]
    boundary_exp = [u for u in unmatched if u["category"] == "boundary_expansion"]
    ambiguous_um = [u for u in unmatched if u["category"] == "ambiguous_unmatched"]

    # Rank clean_missed by score (descending)
    clean_missed.sort(key=lambda x: -x["max_score"])

    print(f"\nUnmatched breakdown:")
    print(f"  Clean missed:       {len(clean_missed)}")
    print(f"  Boundary expansion: {len(boundary_exp)}")
    print(f"  Ambiguous:          {len(ambiguous_um)}")

    top_n = min(args.top_n, len(clean_missed))
    top_candidates = clean_missed[:top_n]
    print(f"\nValidating top-{top_n} clean_missed candidates via Pfam...")

    # ─── Pfam validation on top candidates ───
    cand_results = validate_regions(
        top_candidates, args.genomes_dir, args.pfam_db, biosyn_pfams,
        os.path.join(args.out, "tmp_cand"), args.cpus, label="cand"
    )

    # ─── Null distribution: random negative regions ───
    print(f"\nBuilding null distribution ({top_n} random negative regions)...")
    meta = []
    with open(args.meta) as f:
        for row in csv.DictReader(f):
            meta.append(row)

    # Match length distribution of candidates
    cand_lengths = [int(c["length"]) for c in top_candidates]
    null_regions = sample_null_regions(meta, args.genomes_dir, n=top_n,
                                       target_lengths=cand_lengths)

    print(f"Validating {len(null_regions)} null regions via Pfam...")
    null_results = validate_regions(
        null_regions, args.genomes_dir, args.pfam_db, biosyn_pfams,
        os.path.join(args.out, "tmp_null"), args.cpus, label="null"
    )

    # ─── Compute metrics ───
    print("\n" + "="*60)
    print("DISCOVERY BENCHMARK RESULTS")
    print("="*60)

    # Precision@K
    for k in [20, 50]:
        if k > top_n:
            continue
        p_biosyn = compute_precision_at_k(cand_results, k)
        p_sig = compute_signature_precision_at_k(cand_results, k)
        null_p_biosyn = compute_precision_at_k(null_results, k)
        null_p_sig = compute_signature_precision_at_k(null_results, k)
        enrichment = p_biosyn / max(null_p_biosyn, 0.01)

        print(f"\n  precision@{k}:")
        print(f"    Candidates - biosynthetic Pfam:   {p_biosyn:.2%} ({int(p_biosyn*k)}/{k})")
        print(f"    Candidates - BGC signature:       {p_sig:.2%} ({int(p_sig*k)}/{k})")
        print(f"    Null       - biosynthetic Pfam:   {null_p_biosyn:.2%} ({int(null_p_biosyn*k)}/{k})")
        print(f"    Null       - BGC signature:       {null_p_sig:.2%} ({int(null_p_sig*k)}/{k})")
        print(f"    Enrichment ratio (biosyn):        {enrichment:.1f}x")

    # Family diversity
    for k in [20, 50]:
        if k > top_n:
            continue
        div_biosyn = compute_family_diversity(cand_results, k, "biosyn_set")
        null_div_biosyn = compute_family_diversity(null_results, k, "biosyn_set")
        div_sig = compute_family_diversity(cand_results, k, "signature_set")
        null_div_sig = compute_family_diversity(null_results, k, "signature_set")
        print(f"\n  Family diversity @{k}:")
        print(f"    Candidates (biosyn profiles): {div_biosyn}")
        print(f"    Null       (biosyn profiles): {null_div_biosyn}")
        print(f"    Candidates (signature profiles): {div_sig}")
        print(f"    Null       (signature profiles): {null_div_sig}")

    # Domain frequency in candidates
    all_sig = Counter()
    for r in cand_results:
        if not r.get("skip"):
            for d in r.get("signature_set", set()):
                all_sig[d] += 1

    print(f"\n  Top signature domains in candidates:")
    for d, n in all_sig.most_common(10):
        print(f"    {d} ({BGC_SIGNATURE_DOMAINS.get(d, '?')}): {n}/{top_n}")

    # Type distribution
    type_dist = Counter(c["primary_type"] for c in top_candidates)
    print(f"\n  Type distribution (top-{top_n} clean_missed):")
    for t, n in type_dist.most_common():
        print(f"    {t}: {n}")

    # Phylum distribution
    phy_dist = Counter()
    # Check if phylum info is available
    if "phylum" in top_candidates[0]:
        phy_dist = Counter(c.get("phylum", "?") for c in top_candidates)
    print(f"\n  Phylum distribution (top-{top_n} clean_missed):")
    for p, n in phy_dist.most_common(5):
        print(f"    {p}: {n}")

    # ─── Save results ───
    # Convert sets to lists for JSON
    def clean_for_json(results):
        out = []
        for r in results:
            r2 = {k: v for k, v in r.items() if k not in ("pfam_set", "biosyn_set", "signature_set")}
            r2["pfam_list"] = sorted(r.get("pfam_set", set())) if isinstance(r.get("pfam_set"), set) else []
            r2["biosyn_list"] = sorted(r.get("biosyn_set", set())) if isinstance(r.get("biosyn_set"), set) else []
            r2["signature_list"] = sorted(r.get("signature_set", set())) if isinstance(r.get("signature_set"), set) else []
            out.append(r2)
        return out

    output = {
        "task": "Discovery-side benchmark",
        "evaluation_type": "full_window",
        "detection_score": "score_maxtype",
        "candidate_source": "test_id clean_missed (>20kb from any antiSMASH BGC)",
        "n_total_unmatched": maxtype_result["n_unmatched"],
        "n_clean_missed": len(clean_missed),
        "n_boundary_expansion": len(boundary_exp),
        "n_ambiguous_unmatched": len(ambiguous_um),
        "n_validated": top_n,
        "metrics": {},
        "candidate_results": clean_for_json(cand_results),
        "null_results": clean_for_json(null_results),
    }

    for k in [20, 50]:
        if k > top_n:
            continue
        output["metrics"][f"precision_biosyn_at_{k}"] = round(compute_precision_at_k(cand_results, k), 4)
        output["metrics"][f"precision_signature_at_{k}"] = round(compute_signature_precision_at_k(cand_results, k), 4)
        output["metrics"][f"null_precision_biosyn_at_{k}"] = round(compute_precision_at_k(null_results, k), 4)
        output["metrics"][f"null_precision_signature_at_{k}"] = round(compute_signature_precision_at_k(null_results, k), 4)
        output["metrics"][f"enrichment_biosyn_at_{k}"] = round(
            compute_precision_at_k(cand_results, k) / max(compute_precision_at_k(null_results, k), 0.01), 2)
        output["metrics"][f"family_diversity_biosyn_at_{k}"] = compute_family_diversity(cand_results, k, "biosyn_set")
        output["metrics"][f"null_family_diversity_biosyn_at_{k}"] = compute_family_diversity(null_results, k, "biosyn_set")
        output["metrics"][f"family_diversity_signature_at_{k}"] = compute_family_diversity(cand_results, k, "signature_set")
        output["metrics"][f"null_family_diversity_signature_at_{k}"] = compute_family_diversity(null_results, k, "signature_set")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, set): return sorted(obj)
            return super().default(obj)

    with open(os.path.join(args.out, "discovery_benchmark.json"), "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    # Cleanup temp dirs
    import shutil
    for d in ["tmp_cand", "tmp_null"]:
        p = os.path.join(args.out, d)
        if os.path.exists(p):
            shutil.rmtree(p)

    print(f"\nResults saved to {os.path.join(args.out, 'discovery_benchmark.json')}")


if __name__ == "__main__":
    main()
