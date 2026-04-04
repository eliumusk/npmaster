"""
多属数据集构建
- 5个属: Streptomyces, Micromonospora, Amycolatopsis, Pseudomonas, Bacillus
- 每个属独立构建: 正样本(MIBiG该属BGC) + 负样本(该属基因组非BGC区域)
- 输出: 每属一个数据集 + 跨属评估用的混合数据集
- GC偏差检查: 每个属单独报告
"""

import zipfile, os, glob, pickle, random, json, re, sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from Bio import SeqIO

# ============ 配置 ============
BASE = Path("data")
MIBIG_DIR = BASE / "mibig_gbk_3.1"
ANTISMASH_DIR = BASE / "antismash_result"
REF_BASE = BASE / "ref_genomes_batch"
OUTPUT_DIR = BASE / "multigenus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 8192
RANDOM_SEED = 42
CATEGORY_MAP = {
    'Polyketide': 0, 'NRP': 1, 'RiPP': 2, 'Terpene': 3,
    'Saccharide': 4, 'Alkaloid': 5, 'Other': 6,
}

# 属配置: 属名 → ref_genomes_batch下的目录名
GENERA = {
    'Streptomyces':    'streptomyces',
    'Micromonospora':  'micromonospora',
    'Amycolatopsis':   'amycolatopsis',
    'Pseudomonas':     'pseudomonas',
    'Bacillus':        'bacillus',
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============ 工具函数 ============

def gc_content(seq):
    """计算GC含量"""
    clean = ''.join(x for x in seq if x in 'ATCG')
    if not clean:
        return 0.0
    return (clean.count('G') + clean.count('C')) / len(clean)


def parse_antismash_regions_from_dir(result_dir):
    """从antiSMASH输出目录中提取BGC区域坐标"""
    regions = {}  # contig_id → [(start, end), ...]
    region_gbks = sorted(Path(result_dir).glob("*.region*.gbk"))

    for rf in region_gbks:
        match = re.match(r'(.+)\.region(\d+)\.gbk', rf.name)
        if not match:
            continue
        contig_id = match.group(1)

        with open(rf) as f:
            content = f.read()
            # 从LOCUS行和source feature提取坐标
            loc_match = re.search(r'source\s+(\d+)\.\.(\d+)', content[:3000])
            if loc_match:
                start = int(loc_match.group(1)) - 1  # 0-based
                end = int(loc_match.group(2))
            else:
                # fallback: 找第一个大区间
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
    """从antiSMASH zip中提取BGC区域坐标 (兼容旧格式)"""
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
    """自动检测并解析antiSMASH结果 (目录或zip)"""
    dir_path = ANTISMASH_DIR / gcf_id
    zip_path = ANTISMASH_DIR / f"{gcf_id}.zip"

    if dir_path.is_dir():
        return parse_antismash_regions_from_dir(dir_path)
    elif zip_path.exists():
        return parse_antismash_regions_from_zip(zip_path)
    else:
        return None


def get_nonbgc_segments(seq_len, bgc_regions, min_length=8192):
    """从序列中提取非BGC区段"""
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
    """从区段中随机采样窗口"""
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


def extract_mibig_positives(genus_name):
    """从MIBiG提取指定属的BGC正样本"""
    pos_samples = []
    skipped = 0

    for f in sorted(MIBIG_DIR.glob("BGC*.gbk")):
        try:
            rec = list(SeqIO.parse(f, 'genbank'))[0]
        except:
            continue

        # 提取organism
        organism = 'unknown'
        for feat in rec.features:
            if feat.type == 'source' and 'organism' in feat.qualifiers:
                organism = feat.qualifiers['organism'][0]
                break

        # 检查是否属于目标属
        if genus_name.lower() not in organism.lower():
            skipped += 1
            continue

        bgc_id = f.stem
        dna = str(rec.seq).upper()

        # 提取类别
        cats = set()
        for feat in rec.features:
            if feat.type == 'subregion' and 'label' in feat.qualifiers:
                for l in feat.qualifiers['label']:
                    for sl in l.split(','):
                        sl = sl.strip()
                        cats.add(sl if sl in CATEGORY_MAP else 'Other')
        if not cats:
            cats.add('Other')

        # 切窗口
        if len(dna) <= WINDOW_SIZE:
            pad_l = (WINDOW_SIZE - len(dna)) // 2
            pad_r = WINDOW_SIZE - len(dna) - pad_l
            windows = ['N' * pad_l + dna + 'N' * pad_r]
        else:
            stride = WINDOW_SIZE // 2
            windows = []
            for s in range(0, len(dna) - WINDOW_SIZE + 1, stride):
                windows.append(dna[s:s + WINDOW_SIZE])
            if (len(dna) - WINDOW_SIZE) % stride != 0:
                windows.append(dna[-WINDOW_SIZE:])

        for i, w in enumerate(windows):
            pos_samples.append({
                'id': f"{bgc_id}_w{i}",
                'dna': w,
                'is_bgc': 1,
                'category_vector': [1 if c in cats else 0 for c in CATEGORY_MAP],
                'categories': list(cats),
                'source': bgc_id,
                'organism': organism,
                'genus': genus_name,
            })

    return pos_samples


def extract_negatives(genus_name, genus_dir):
    """从指定属的基因组中提取非BGC负样本"""
    ref_dir = REF_BASE / genus_dir / "ncbi_dataset" / "data"
    neg_windows = []

    gcf_dirs = sorted(ref_dir.glob("GCF_*"))
    for gcf_path in gcf_dirs:
        gcf_id = gcf_path.name

        # 获取antiSMASH标注
        bgc_regions_by_contig = get_antismash_regions(gcf_id)
        if bgc_regions_by_contig is None:
            continue

        # 读基因组序列
        fna_files = list(gcf_path.glob("*.fna"))
        if not fna_files:
            continue

        for fna in fna_files:
            for rec in SeqIO.parse(fna, 'fasta'):
                contig_id = rec.id
                genome_seq = str(rec.seq).upper()
                bgc_regions = bgc_regions_by_contig.get(contig_id, [])
                nonbgc_segments = get_nonbgc_segments(len(genome_seq), bgc_regions)
                nonbgc_len = sum(e - s for s, e in nonbgc_segments)
                n_windows = nonbgc_len // WINDOW_SIZE

                windows = sample_windows(genome_seq, nonbgc_segments, WINDOW_SIZE, n_windows)
                for w in windows:
                    neg_windows.append({
                        'dna': w,
                        'source': gcf_id,
                        'contig': contig_id,
                        'genus': genus_name,
                    })

    return neg_windows


def build_genus_dataset(genus_name, genus_dir):
    """构建单个属的数据集"""
    print(f"\n{'='*60}")
    print(f"  {genus_name}")
    print(f"{'='*60}")

    # 正样本
    print(f"  [1] 提取MIBiG正样本...")
    pos_samples = extract_mibig_positives(genus_name)
    n_bgcs = len(set(s['source'] for s in pos_samples))
    print(f"      {n_bgcs} BGCs → {len(pos_samples)} windows")

    if len(pos_samples) == 0:
        print(f"      跳过: 没有{genus_name}的BGC")
        return None

    # 负样本
    print(f"  [2] 提取负样本...")
    neg_windows = extract_negatives(genus_name, genus_dir)
    print(f"      {len(neg_windows)} raw negative windows")

    if len(neg_windows) == 0:
        print(f"      跳过: 没有{genus_name}的负样本")
        return None

    # 平衡
    num_needed = len(pos_samples)
    if len(neg_windows) > num_needed:
        random.shuffle(neg_windows)
        neg_windows = neg_windows[:num_needed]
    elif len(neg_windows) < num_needed:
        print(f"      警告: 负样本不足 ({len(neg_windows)} < {num_needed}), 正样本将下采样")
        random.shuffle(pos_samples)
        pos_samples = pos_samples[:len(neg_windows)]

    neg_samples = []
    for i, nw in enumerate(neg_windows):
        neg_samples.append({
            'id': f"NEG_{genus_name}_{i}",
            'dna': nw['dna'],
            'is_bgc': 0,
            'category_vector': [0] * len(CATEGORY_MAP),
            'categories': [],
            'source': nw['source'],
            'organism': f"{genus_name}_{nw['source']}",
            'genus': genus_name,
        })

    print(f"      平衡后: {len(pos_samples)} pos, {len(neg_samples)} neg")

    # GC检查
    pos_gc = [gc_content(s['dna']) for s in pos_samples]
    neg_gc = [gc_content(s['dna']) for s in neg_samples]
    gc_diff = abs(np.mean(pos_gc) - np.mean(neg_gc))
    print(f"  [3] GC: pos={np.mean(pos_gc):.3f}±{np.std(pos_gc):.3f}, "
          f"neg={np.mean(neg_gc):.3f}±{np.std(neg_gc):.3f}, "
          f"diff={gc_diff:.4f} {'⚠️ BIAS!' if gc_diff > 0.02 else '✓'}")

    # 按BGC ID分组分割 (防泄露)
    bgc_ids = list(set(s['source'] for s in pos_samples))
    random.shuffle(bgc_ids)
    n = len(bgc_ids)
    train_ids = set(bgc_ids[:int(n * 0.8)])
    val_ids = set(bgc_ids[int(n * 0.8):int(n * 0.9)])
    test_ids = set(bgc_ids[int(n * 0.9):])

    train_pos = [s for s in pos_samples if s['source'] in train_ids]
    val_pos = [s for s in pos_samples if s['source'] in val_ids]
    test_pos = [s for s in pos_samples if s['source'] in test_ids]

    # 负样本按基因组分组分割
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

    print(f"  [4] Splits:")
    for name, ds in splits.items():
        pos_c = sum(1 for s in ds if s['is_bgc'] == 1)
        print(f"      {name}: {len(ds)} (pos:{pos_c}, neg:{len(ds)-pos_c})")

    # 保存
    genus_out = OUTPUT_DIR / genus_name.lower()
    genus_out.mkdir(parents=True, exist_ok=True)
    for name, ds in splits.items():
        path = genus_out / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(ds, f)

    metadata = {
        'genus': genus_name,
        'window_size': WINDOW_SIZE,
        'num_bgcs': n_bgcs,
        'num_positive': len(pos_samples),
        'num_negative': len(neg_samples),
        'pos_gc_mean': float(np.mean(pos_gc)),
        'neg_gc_mean': float(np.mean(neg_gc)),
        'gc_diff': float(gc_diff),
        'splits': {
            name: {
                'total': len(ds),
                'pos': sum(1 for s in ds if s['is_bgc'] == 1),
                'neg': sum(1 for s in ds if s['is_bgc'] == 0),
            } for name, ds in splits.items()
        },
    }
    with open(genus_out / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return splits


def build_cross_genus_test(all_genus_data):
    """构建跨属评估数据: 每个属的test集打包"""
    print(f"\n{'='*60}")
    print("  跨属评估数据")
    print(f"{'='*60}")

    cross_dir = OUTPUT_DIR / "cross_genus"
    cross_dir.mkdir(parents=True, exist_ok=True)

    # 保存每个属的test集，用于 train-on-A / test-on-B 实验
    for genus, splits in all_genus_data.items():
        if splits is None:
            continue
        path = cross_dir / f"{genus.lower()}_test.pkl"
        with open(path, 'wb') as f:
            pickle.dump(splits['test'], f)
        n = len(splits['test'])
        pos = sum(1 for s in splits['test'] if s['is_bgc'] == 1)
        print(f"  {genus}: {n} test samples (pos:{pos}, neg:{n-pos})")

    # 保存每个属的train集
    for genus, splits in all_genus_data.items():
        if splits is None:
            continue
        path = cross_dir / f"{genus.lower()}_train.pkl"
        with open(path, 'wb') as f:
            pickle.dump(splits['train'], f)

    # 混合训练集 (所有属的train合并)
    mixed_train = []
    for genus, splits in all_genus_data.items():
        if splits is None:
            continue
        mixed_train.extend(splits['train'])
    random.shuffle(mixed_train)
    with open(cross_dir / "mixed_train.pkl", 'wb') as f:
        pickle.dump(mixed_train, f)
    print(f"  混合训练集: {len(mixed_train)} samples")


def main():
    print("=" * 60)
    print("多属数据集构建")
    print("=" * 60)

    all_genus_data = {}

    for genus_name, genus_dir in GENERA.items():
        all_genus_data[genus_name] = build_genus_dataset(genus_name, genus_dir)

    # 汇总
    print(f"\n{'='*60}")
    print("  汇总")
    print(f"{'='*60}")
    print(f"  {'属':<20} {'BGCs':>6} {'正样本':>8} {'负样本':>8} {'GC diff':>8}")
    print(f"  {'-'*54}")
    for genus_name in GENERA:
        meta_path = OUTPUT_DIR / genus_name.lower() / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                m = json.load(f)
            print(f"  {genus_name:<20} {m['num_bgcs']:>6} {m['num_positive']:>8} "
                  f"{m['num_negative']:>8} {m['gc_diff']:>8.4f}")
        else:
            print(f"  {genus_name:<20} {'SKIPPED':>6}")

    # 构建跨属评估数据
    build_cross_genus_test(all_genus_data)

    print(f"\n完成! 数据在 {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
