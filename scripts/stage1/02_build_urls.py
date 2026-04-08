#!/usr/bin/env python3
"""
Stage 1.2: Resolve NCBI FTP URLs for selected accessions.

For each accession like GCF_003428645.1, NCBI stores at:
  https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/003/428/645/GCF_003428645.1_<name>/

We list that parent dir, find the matching subdir, and build the fna.gz URL.
Writes an aria2 input file (one url + out= line per genome).
"""
import argparse
import csv
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all"

def parent_url(acc):
    # acc e.g. "GCF_003428645.1"
    prefix, rest = acc.split("_", 1)
    num = rest.split(".")[0]
    num = num.zfill(9)
    return f"{BASE}/{prefix}/{num[0:3]}/{num[3:6]}/{num[6:9]}/"

def resolve(acc, retries=3, timeout=30):
    url = parent_url(acc)
    last_err = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"curl/7.88"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                html = r.read().decode("utf-8","replace")
            m = re.search(rf'href="({re.escape(acc)}_[^/"]+)/"', html)
            if m:
                d = m.group(1)
                fna = f"{url}{d}/{d}_genomic.fna.gz"
                return (acc, fna, None)
            return (acc, None, "no_matching_subdir")
        except Exception as e:
            last_err = str(e)
            time.sleep(2)
    return (acc, None, last_err)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--out-urls", required=True, help="aria2 input file")
    ap.add_argument("--out-manifest", required=True, help="tsv: accession\\tfna_url\\tlocal_fname\\terror")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    accs = []
    with open(args.selected) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            accs.append(row["ncbi_accession"])
    if args.limit:
        accs = accs[:args.limit]

    results = {}
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(resolve, a): a for a in accs}
        for f in as_completed(futs):
            acc, fna, err = f.result()
            results[acc] = (fna, err)
            done += 1
            if done % 50 == 0 or done == len(accs):
                dt = time.time() - t0
                print(f"[{done}/{len(accs)}] elapsed {dt:.0f}s", file=sys.stderr)

    Path(args.out_urls).parent.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with open(args.out_urls, "w") as fu, open(args.out_manifest, "w") as fm:
        fm.write("accession\tfna_url\tlocal_fname\terror\n")
        for acc in accs:
            fna, err = results[acc]
            if fna:
                local = f"{acc}.fna.gz"
                fu.write(f"{fna}\n  out={local}\n")
                fm.write(f"{acc}\t{fna}\t{local}\t\n")
                ok += 1
            else:
                fm.write(f"{acc}\t\t\t{err or 'unknown'}\n")
                fail += 1
    print(f"resolved ok={ok} fail={fail} of {len(accs)}", file=sys.stderr)

if __name__ == "__main__":
    main()
