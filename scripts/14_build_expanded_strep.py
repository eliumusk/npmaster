"""
扩充版 Streptomyces 数据集构建
- 使用所有可用的 antiSMASH 标注基因组（~105个）做负样本
- 支持多窗口大小（用于消融实验）
- 输出: data/expanded_strep/w{size}/train.pkl, val.pkl, test.pkl

用法:
  python3 14_build_expanded_strep.py                    # 默认8192
  python3 14_build_expanded_strep.py --window 4096      # 4k窗口
  python3 14_build_expanded_strep.py --window 16384     # 16k窗口
  python3 14_build_expanded_strep.py --all-windows      # 跑所有窗口大小
"""

import zipfile, os, pickle, random, json, re, argparse
import numpy as np
from pathlib import Path
from collections import Counter
from Bio import SeqIO

MIBIG_DIR = Path("data/mibig_gbk_3.1")
ANTISMASH_DIR = Path("data/antismash_result")
REF_GENOME_DIR = Path("data/ref_genomes_batch/streptomyces/ncbi_dataset/data")
RANDOM_SEED = 42
CATEGORY_MAP = {
    'Polyketide': 0, 'NRP': 1, 'RiPP': 2, 'Terpene': 3,
    'Saccharide': 4, 'Alkaloid': 5, 'Other': 6,
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def parse_antismash_regions_from_dir(result_dir):
    regions = {}
    for rf in sorted(Path(result_dir).glob("*.region*.gbk")):
        match = re.match(r'(.+)\.region(\d+)\.gbk', rf.name)
        if not match:
            continue
        contig_id = match.group(1)
        with open(rf) as f:
            content = f.read()
            loc_match = re.search(r'source\s+(\d+)\.\.(\d+)', content[:3000])
            if loc_match:
                start = int(loc_match.group(1)) - 1
                end = int(loc_match.group(2))
            else:
                for line in content.split('\n')[:30]:
                    m = re.search(r'(\d+)\.\.(\d+)', line)
                    if m:
                        s, e = int(m.group(1)), int(m.group(2))
                        if e - s > 1000:
                            start, end = s - 1, e
                            break
                else:
                    continue
            if contig_id not in regions:
                regions[contig_id] = []
            regions[contig_id].append((start, end))
    return regions


def parse_antismash_regions_from_zip(zip_path):
    regions = {}
    with zipfile.ZipFile(zip_path) as zf:
        region_files = [n for n in zf.namelist() if '.region' in n and n.endswith('.gbk')]
        for rf in region_files:
            basename = os.path.basename(rf)
            match = re.match(r'(.+)\.region(\d+)\.gbk', basename)
            if not match:
                continue
            contig_id = match.group(1)
            with zf.open(rf) as f:
                content = f.read().decode('utf-8', errors='ignore')
                loc_match = re.search(r'source\s+(\d+)\.\.(\d+)', content[:3000])
                if loc_match:
                    start = int(loc_match.group(1)) - 1
                    end = int(loc_match.group(2))
                else:
                    for line in content.split('\n')[:30]:
                        m = re.search(r'(\d+)\.\.(\d+)', line)
                        if m:
                            s, e = int(m.group(1)), int(m.group(2))
                            if e - s > 1000:
                                start, end = s - 1, e
                                break
                    else:
                        continue
                if contig_id not in regions:
                    regions[contig_id] = []
                regions[contig_id].append((start, end))
    return regions


def get_antismash_regions(gcf_id):
    dir_path = ANTISMASH_DIR / gcf_id
    zip_path = ANTISMASH_DIR / f"{gcf_id}.zip"
    if dir_path.is_dir() and list(dir_path.glob("*.region*.gbk")):
        return parse_antismash_regions_from_dir(dir_path)
    elif zip_path.exists():
        return parse_antismash_regions_from_zip(zip_path)
    return None


def get_nonbgc_segments(seq_len, bgc_regions, min_length):
    if not bgc_regions:
        return [(0, seq_len)]
    sorted_regions = sorted(bgc_regions)
    segments = []
    prev_end = 0
    for start, end in sorted_regions:
        if start > prev_end and (start - prev_end) >= min_length:
            segments.append((prev_end, start))
        prev_end = max(prev_end, end)
    if seq_len > prev_end and (seq_len - prev_end) >= min_length:
        segments.append((prev_end, seq_len))
    return segments


def sample_windows(genome_seq, segments, window_size, num_windows):
    windows = []
    weights = [max(0, end - start - window_size + 1) for start, end in segments]
    total_w = sum(weights)
    if total_w <= 0:
        return windows
    for _ in range(num_windows):
        r = random.random() * total_w
        cumsum = 0
        for i, (start, end) in enumerate(segments):
            cumsum += weights[i]
            if cumsum >= r:
                max_start = end - window_size
                win_start = random.randint(start, max_start)
                window = genome_seq[win_start:win_start + window_size]
                if window.count('N') / len(window) < 0.05:
                    windows.append(window)
                break
    return windows


def gc_content(seq):
    clean = ''.join(x for x in seq if x in 'ATCG')
    if not clean:
        return 0.0
    return (clean.count('G') + clean.count('C')) / len(clean)


def build_dataset(window_size):
    output_dir = Path(f"data/expanded_strep/w{window_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"构建扩充版Strep数据集 (window={window_size})")
    print(f"{'='*60}")

    # Step 1: 找所有有antiSMASH结果的Strep基因组
    print("\n[1/6] 扫描antiSMASH结果...")
    strep_gcfs = sorted(REF_GENOME_DIR.glob("GCF_*"))
    genome_bgc_map = {}
    for gcf_path in strep_gcfs:
        gcf_id = gcf_path.name
        regions = get_antismash_regions(gcf_id)
        if regions is not None:
            genome_bgc_map[gcf_id] = regions

    print(f"  找到 {len(genome_bgc_map)} 个有antiSMASH标注的Strep基因组")
    print(f"  (之前只有25个，现在扩充到{len(genome_bgc_map)}个)")

    # Step 2: 提取负样本
    print(f"\n[2/6] 从{len(genome_bgc_map)}个基因组提取负样本...")
    neg_windows = []
    genome_stats = {}
    for gcf_id, bgc_regions_by_contig in genome_bgc_map.items():
        gcf_dir = REF_GENOME_DIR / gcf_id
        fna_files = list(gcf_dir.glob("*.fna"))
        if not fna_files:
            continue

        gcf_windows = []
        for fna in fna_files:
            for rec in SeqIO.parse(fna, 'fasta'):
                genome_seq = str(rec.seq).upper()
                bgc_regions = bgc_regions_by_contig.get(rec.id, [])
                nonbgc_segments = get_nonbgc_segments(len(genome_seq), bgc_regions, window_size)
                nonbgc_len = sum(e - s for s, e in nonbgc_segments)
                n_windows = nonbgc_len // window_size
                windows = sample_windows(genome_seq, nonbgc_segments, window_size, n_windows)
                for w in windows:
                    gcf_windows.append({'dna': w, 'source': gcf_id, 'contig': rec.id})

        neg_windows.extend(gcf_windows)
        genome_stats[gcf_id] = len(gcf_windows)

    print(f"  提取了 {len(neg_windows)} 个负样本窗口 (from {len(genome_stats)} genomes)")
    stats_vals = sorted(genome_stats.values(), reverse=True)
    print(f"  每基因组窗口数: max={stats_vals[0]}, median={np.median(stats_vals):.0f}, min={stats_vals[-1]}")

    # Step 3: 提取正样本 (只取Streptomyces)
    print(f"\n[3/6] 提取Streptomyces正样本...")
    pos_samples = []
    for f in sorted(MIBIG_DIR.glob("BGC*.gbk")):
        try:
            rec = list(SeqIO.parse(f, 'genbank'))[0]
        except:
            continue

        organism = 'unknown'
        for feat in rec.features:
            if feat.type == 'source' and 'organism' in feat.qualifiers:
                organism = feat.qualifiers['organism'][0]
                break

        if 'streptomyces' not in organism.lower():
            continue

        bgc_id = f.stem
        dna = str(rec.seq).upper()

        cats = set()
        for feat in rec.features:
            if feat.type == 'subregion' and 'label' in feat.qualifiers:
                for l in feat.qualifiers['label']:
                    for sl in l.split(','):
                        sl = sl.strip()
                        cats.add(sl if sl in CATEGORY_MAP else 'Other')
        if not cats:
            cats.add('Other')

        if len(dna) <= window_size:
            pad_l = (window_size - len(dna)) // 2
            pad_r = window_size - len(dna) - pad_l
            windows = ['N' * pad_l + dna + 'N' * pad_r]
        else:
            stride = window_size // 2
            windows = []
            for s in range(0, len(dna) - window_size + 1, stride):
                windows.append(dna[s:s + window_size])
            if (len(dna) - window_size) % stride != 0:
                windows.append(dna[-window_size:])

        for i, w in enumerate(windows):
            pos_samples.append({
                'id': f"{bgc_id}_w{i}",
                'dna': w,
                'is_bgc': 1,
                'category_vector': [1 if c in cats else 0 for c in CATEGORY_MAP],
                'categories': list(cats),
                'source': bgc_id,
                'organism': organism,
            })

    n_bgcs = len(set(s['source'] for s in pos_samples))
    print(f"  {n_bgcs} BGCs -> {len(pos_samples)} windows")

    # Step 4: 平衡
    print(f"\n[4/6] 平衡正负样本...")
    num_needed = len(pos_samples)
    if len(neg_windows) > num_needed:
        random.shuffle(neg_windows)
        neg_windows = neg_windows[:num_needed]
    print(f"  正: {len(pos_samples)}, 负: {len(neg_windows)}")

    neg_samples = []
    for i, nw in enumerate(neg_windows):
        neg_samples.append({
            'id': f"NEG_{i}",
            'dna': nw['dna'],
            'is_bgc': 0,
            'category_vector': [0] * len(CATEGORY_MAP),
            'categories': [],
            'source': nw['source'],
            'organism': f"Streptomyces_{nw['source']}",
        })

    # Step 5: GC检查
    pos_gc = [gc_content(s['dna']) for s in random.sample(pos_samples, min(1000, len(pos_samples)))]
    neg_gc = [gc_content(s['dna']) for s in random.sample(neg_samples, min(1000, len(neg_samples)))]
    gc_diff = abs(np.mean(pos_gc) - np.mean(neg_gc))
    print(f"\n[5/6] GC: pos={np.mean(pos_gc):.3f}, neg={np.mean(neg_gc):.3f}, diff={gc_diff:.4f}", end="")
    print(" ✓" if gc_diff < 0.02 else " ⚠️ BIAS!")

    # Step 6: Split (BGC级别分正样本, 基因组级别分负样本)
    print(f"\n[6/6] Split...")
    bgc_ids = list(set(s['source'] for s in pos_samples))
    random.shuffle(bgc_ids)
    n = len(bgc_ids)
    train_ids = set(bgc_ids[:int(n * 0.8)])
    val_ids = set(bgc_ids[int(n * 0.8):int(n * 0.9)])
    test_ids = set(bgc_ids[int(n * 0.9):])

    train_pos = [s for s in pos_samples if s['source'] in train_ids]
    val_pos = [s for s in pos_samples if s['source'] in val_ids]
    test_pos = [s for s in pos_samples if s['source'] in test_ids]

    neg_gcfs = list(set(s['source'] for s in neg_samples))
    random.shuffle(neg_gcfs)
    n_neg = len(neg_gcfs)
    train_gcfs = set(neg_gcfs[:int(n_neg * 0.8)])
    val_gcfs = set(neg_gcfs[int(n_neg * 0.8):int(n_neg * 0.9)])
    test_gcfs = set(neg_gcfs[int(n_neg * 0.9):])

    train_neg = [s for s in neg_samples if s['source'] in train_gcfs]
    val_neg = [s for s in neg_samples if s['source'] in val_gcfs]
    test_neg = [s for s in neg_samples if s['source'] in test_gcfs]

    splits = {
        'train': train_pos + train_neg,
        'val': val_pos + val_neg,
        'test': test_pos + test_neg,
    }
    for name in splits:
        random.shuffle(splits[name])

    for name, ds in splits.items():
        pos_c = sum(1 for s in ds if s['is_bgc'] == 1)
        neg_c = len(ds) - pos_c
        neg_g = len(set(s['source'] for s in ds if s['is_bgc'] == 0))
        pos_b = len(set(s['source'] for s in ds if s['is_bgc'] == 1))
        print(f"  {name}: {len(ds)} windows (pos:{pos_c} from {pos_b} BGC, neg:{neg_c} from {neg_g} genomes)")

    # 保存
    for name, ds in splits.items():
        path = output_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(ds, f)

    metadata = {
        'window_size': window_size,
        'num_bgcs': n_bgcs,
        'num_positive': len(pos_samples),
        'num_negative': len(neg_samples),
        'num_neg_genomes': len(genome_stats),
        'pos_gc_mean': float(np.mean(pos_gc)),
        'neg_gc_mean': float(np.mean(neg_gc)),
        'gc_diff': float(gc_diff),
        'splits': {
            name: {
                'total': len(ds),
                'pos': sum(1 for s in ds if s['is_bgc'] == 1),
                'neg': sum(1 for s in ds if s['is_bgc'] == 0),
                'pos_bgcs': len(set(s['source'] for s in ds if s['is_bgc'] == 1)),
                'neg_genomes': len(set(s['source'] for s in ds if s['is_bgc'] == 0)),
            } for name, ds in splits.items()
        },
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n  保存到 {output_dir}/")
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=8192)
    parser.add_argument('--all-windows', action='store_true', help='Build 4096, 8192, 16384')
    args = parser.parse_args()

    if args.all_windows:
        for ws in [4096, 8192, 16384]:
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            build_dataset(ws)
    else:
        build_dataset(args.window)
