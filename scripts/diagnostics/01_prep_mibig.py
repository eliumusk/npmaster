"""
Day-1 Diagnostic: prepare MIBiG variable-length DNA samples.
- Parse 2502 MIBiG GBK files
- Extract full DNA sequence
- If len > 16384: take center 16384
- If len <= 16384: keep full sequence (no padding)
- Extract 7-class BGC type from 'biosyn_class' qualifier / MIBiG JSON-like comment
Output: mibig_variable.pkl with list of dicts
"""
import os, glob, pickle, re
from collections import Counter

MIBIG_DIR = "/Users/musk/Documents/code/np-master/data/mibig_gbk_3.1"
OUT_PKL = "/Users/musk/Documents/code/np-master/data/diagnostic_day1/mibig_variable.pkl"
MAX_LEN = 16384

# MIBiG 7 classes
CLASSES = ["Alkaloid", "NRP", "Polyketide", "RiPP", "Saccharide", "Terpene", "Other"]

def parse_gbk(path):
    """Return (bgc_id, dna_seq, categories_list, organism)."""
    bgc_id = os.path.basename(path).replace('.gbk','')
    with open(path) as f:
        content = f.read()
    # extract DNA between ORIGIN and //
    m = re.search(r'ORIGIN\s*\n(.*?)(?:^//|\Z)', content, re.DOTALL | re.MULTILINE)
    if not m:
        return None
    dna = re.sub(r'[^atcgATCG]', '', m.group(1)).upper()
    if len(dna) < 100:
        return None
    # extract biosyn_class / categories
    cats = []
    # Method 1: MIBiG stores biosynth class in /label="Polyketide, Alkaloid" style fields
    # Search in feature qualifiers near the top of the record
    label_matches = re.findall(r'/label="([^"]+)"', content)
    for label in label_matches:
        # split by comma
        parts = [p.strip() for p in label.split(',')]
        hit = False
        for p in parts:
            for canonical in CLASSES:
                if p.lower() == canonical.lower():
                    if canonical not in cats:
                        cats.append(canonical)
                    hit = True
        # Only trust the label if it contained at least one canonical class
        if hit:
            break  # first valid label is enough (usually on cluster feature)
    # Method 1b: biosyn_class qualifier (MIBiG JSON-extracted format)
    if not cats:
        class_match = re.findall(r'biosyn_class[\"\s:=]*[\[\(\"]*([^\]\)\"\n,]+)', content, re.IGNORECASE)
        for c in class_match:
            c = c.strip().strip('"').strip("'").strip()
            c_norm = c.replace('nonribosomal peptide','NRP').replace('Nonribosomal peptide','NRP')
            for canonical in CLASSES:
                if canonical.lower() in c_norm.lower() or c_norm.lower() in canonical.lower():
                    if canonical not in cats:
                        cats.append(canonical)
                    break
    # Method 2: fallback - look for /subclass= / common_name and guess type
    if not cats:
        # check for PKS/NRPS signatures in gene_functions
        if re.search(r'polyketide synthase|\bPKS\b|\bT1PKS\b|\bT2PKS\b', content, re.IGNORECASE):
            cats.append('Polyketide')
        if re.search(r'nonribosomal peptide|\bNRPS\b', content, re.IGNORECASE):
            if 'NRP' not in cats: cats.append('NRP')
        if re.search(r'terpene synthase|terpene cyclase', content, re.IGNORECASE):
            cats.append('Terpene')
        if re.search(r'lantipeptide|lasso peptide|\bRiPP\b|thiopeptide', content, re.IGNORECASE):
            cats.append('RiPP')
        if re.search(r'saccharide|glycosyltransferase', content, re.IGNORECASE) and 'Polyketide' not in cats:
            cats.append('Saccharide')
    if not cats:
        cats = ['Other']
    # Organism
    org_m = re.search(r'^\s*ORGANISM\s+(.+)$', content, re.MULTILINE)
    organism = org_m.group(1).strip() if org_m else "unknown"
    # Truncate if needed
    full_len = len(dna)
    if full_len > MAX_LEN:
        start = (full_len - MAX_LEN) // 2
        dna = dna[start:start + MAX_LEN]
    return {
        'id': bgc_id,
        'dna': dna,
        'full_length': full_len,
        'used_length': len(dna),
        'categories': cats,
        'organism': organism,
        'is_bgc': 1,
        'source_tag': 'MIBiG',
    }

def main():
    files = sorted(glob.glob(f"{MIBIG_DIR}/*.gbk"))
    print(f"Found {len(files)} GBK files")
    samples = []
    cls_counter = Counter()
    skipped = 0
    for i, f in enumerate(files):
        s = parse_gbk(f)
        if s is None:
            skipped += 1
            continue
        samples.append(s)
        for c in s['categories']:
            cls_counter[c] += 1
        if (i+1) % 500 == 0:
            print(f"  parsed {i+1}/{len(files)}")
    print(f"\nTotal samples: {len(samples)}, skipped: {skipped}")
    print(f"Class distribution (multi-label):")
    for c in CLASSES:
        print(f"  {c}: {cls_counter[c]}")
    # Length distribution
    lens = [s['full_length'] for s in samples]
    used = [s['used_length'] for s in samples]
    import numpy as np
    print(f"\nFull BGC length: min={min(lens)}, max={max(lens)}, median={int(np.median(lens))}, mean={int(np.mean(lens))}")
    print(f"Used length (after truncation): min={min(used)}, max={max(used)}, median={int(np.median(used))}")
    print(f"  BGCs shorter than 16384: {sum(1 for l in lens if l < 16384)}")
    print(f"  BGCs longer than 16384 (truncated): {sum(1 for l in lens if l > 16384)}")
    print(f"  BGCs shorter than 5000: {sum(1 for l in lens if l < 5000)}")
    # Save
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(samples, f)
    size_mb = os.path.getsize(OUT_PKL) / 1024 / 1024
    print(f"\nSaved to {OUT_PKL} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
