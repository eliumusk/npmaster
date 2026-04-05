"""
Day-1 Diagnostic: sample background windows from diverse genomes.
- Pseudomonas (Proteobacteria phylum)
- Nostoc (Cyanobacteria phylum)
- Bacillus (Firmicutes phylum)
Each genome: sample random 16384bp windows, up to N windows.
(Strep background is reused from server's existing expanded_strep pkl)
"""
import os, glob, pickle, random
random.seed(42)

GENOME_ROOT = "/Users/musk/Documents/code/np-master/data/ref_genomes_batch"
OUT_PKL = "/Users/musk/Documents/code/np-master/data/diagnostic_day1/bg_multigenus.pkl"
WIN = 16384
# Genera to include (non-Strep)
GENERA = {
    'pseudomonas':  {'phylum': 'Proteobacteria', 'n_per_genome': 50, 'max_genomes': 20},
    'nostoc':       {'phylum': 'Cyanobacteria', 'n_per_genome': 50, 'max_genomes': 10},
    'bacillus':     {'phylum': 'Firmicutes', 'n_per_genome': 50, 'max_genomes': 20},
}

def read_fna(path):
    """Read FASTA, concat contigs with 'N' separator, uppercase."""
    seqs = []
    cur = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if cur:
                    seqs.append(''.join(cur))
                cur = []
            else:
                cur.append(line.strip())
        if cur:
            seqs.append(''.join(cur))
    # Keep only main chromosome(s), skip tiny plasmids
    seqs = [s.upper() for s in seqs if len(s) > 50000]
    return seqs

def sample_windows_from_genome(fna_path, n_windows, win=16384):
    seqs = read_fna(fna_path)
    if not seqs:
        return []
    # weight by length
    total_len = sum(len(s) for s in seqs)
    out = []
    for _ in range(n_windows):
        # pick contig weighted by length
        r = random.uniform(0, total_len)
        cum = 0
        for s in seqs:
            cum += len(s)
            if r <= cum:
                seq = s
                break
        if len(seq) < win:
            continue
        # random window position
        start = random.randint(0, len(seq) - win)
        w = seq[start:start+win]
        # filter out high-N content (bad assembly)
        n_frac = (w.count('N') + w.count('n')) / len(w)
        if n_frac > 0.05:
            continue
        # clean: only ATCG
        w_clean = ''.join(c for c in w if c in 'ATCG')
        if len(w_clean) < win * 0.95:
            continue
        out.append(w_clean)
    return out

def main():
    all_samples = []
    for genus, cfg in GENERA.items():
        data_dir = f"{GENOME_ROOT}/{genus}/ncbi_dataset/data"
        if not os.path.isdir(data_dir):
            print(f"SKIP {genus}: {data_dir} missing")
            continue
        genome_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith("GCF_") or d.startswith("GCA_")])
        genome_dirs = genome_dirs[:cfg['max_genomes']]
        print(f"\n=== {genus} ({cfg['phylum']}): {len(genome_dirs)} genomes ===")
        per_genus_count = 0
        for gdir in genome_dirs:
            fnas = glob.glob(f"{data_dir}/{gdir}/*.fna")
            if not fnas:
                continue
            fna = fnas[0]
            windows = sample_windows_from_genome(fna, cfg['n_per_genome'], WIN)
            for i, w in enumerate(windows):
                all_samples.append({
                    'id': f'BG_{genus}_{gdir}_{i}',
                    'dna': w,
                    'full_length': WIN,
                    'used_length': WIN,
                    'categories': [],
                    'organism': f'{genus}_{gdir}',
                    'phylum': cfg['phylum'],
                    'is_bgc': 0,
                    'source_tag': f'BG_{genus}',
                })
            per_genus_count += len(windows)
            print(f"  {gdir}: {len(windows)} windows")
        print(f"  Total {genus}: {per_genus_count}")
    print(f"\nTotal background samples: {len(all_samples)}")
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(all_samples, f)
    size_mb = os.path.getsize(OUT_PKL) / 1024 / 1024
    print(f"Saved to {OUT_PKL} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
