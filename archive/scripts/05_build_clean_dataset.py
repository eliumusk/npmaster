"""
最终干净数据集构建
- 正样本: MIBiG 2502个BGC的DNA序列
- 负样本: 25个Streptomyces基因组中，经antiSMASH标注后排除BGC区域的DNA片段
- 防泄露: 按BGC来源分组分割train/val/test
"""

import zipfile, os, glob, pickle, random, json, re
import numpy as np
from pathlib import Path
from collections import Counter
from Bio import SeqIO

MIBIG_DIR = Path("data/mibig_gbk_3.1")
ANTISMASH_DIR = Path("data/antismash_result")
REF_GENOME_DIR = Path("data/ref_genomes_batch/streptomyces/ncbi_dataset/data")
OUTPUT_DIR = Path("data/clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 8192
RANDOM_SEED = 42
CATEGORY_MAP = {
    'Polyketide': 0, 'NRP': 1, 'RiPP': 2, 'Terpene': 3,
    'Saccharide': 4, 'Alkaloid': 5, 'Other': 6,
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def parse_antismash_bgc_regions(zip_path):
    """从antiSMASH zip中提取每个contig的BGC区域坐标"""
    gcf_id = os.path.basename(zip_path).replace('.zip', '')
    regions = {}  # contig_id → [(start, end), ...]

    with zipfile.ZipFile(zip_path) as zf:
        # 找region gbk文件: contig_id.regionXXX.gbk
        region_files = [n for n in zf.namelist() if '.region' in n and n.endswith('.gbk')]

        for rf in region_files:
            # 文件名格式: NC_003155.5.region001.gbk 或 NZ_CP023202.1.region001.gbk
            basename = os.path.basename(rf)
            # 提取contig ID和region长度
            match = re.match(r'(.+)\.region(\d+)\.gbk', basename)
            if not match:
                continue
            contig_id = match.group(1)

            with zf.open(rf) as f:
                first_line = f.readline().decode('utf-8', errors='ignore')
                if first_line.startswith('LOCUS'):
                    parts = first_line.split()
                    region_len = 0
                    for j, p in enumerate(parts):
                        if p == 'bp':
                            region_len = int(parts[j - 1])
                            break

                    # 读取region的实际位置（从FEATURES中的location）
                    content = f.read().decode('utf-8', errors='ignore')
                    # 找 source 的 location
                    loc_match = re.search(r'source\s+(\d+)\.\.(\d+)', first_line + content[:2000])
                    if loc_match:
                        start = int(loc_match.group(1)) - 1  # 转0-based
                        end = int(loc_match.group(2))
                    else:
                        # 从FEATURES第一行找
                        for line in content.split('\n')[:20]:
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

    return gcf_id, regions


def get_nonbgc_segments(genome_seq, bgc_regions, min_length=8192):
    """从基因组序列中提取非BGC区段"""
    seq_len = len(genome_seq)
    if not bgc_regions:
        return [(0, seq_len)]

    # 按起始位置排序
    sorted_regions = sorted(bgc_regions)

    segments = []
    prev_end = 0
    for start, end in sorted_regions:
        if start > prev_end and (start - prev_end) >= min_length:
            segments.append((prev_end, start))
        prev_end = max(prev_end, end)

    # 最后一段
    if seq_len > prev_end and (seq_len - prev_end) >= min_length:
        segments.append((prev_end, seq_len))

    return segments


def sample_windows_from_segments(genome_seq, segments, window_size, num_windows):
    """从非BGC区段中随机采样固定大小窗口"""
    windows = []
    # 计算每个segment可以提供多少窗口
    total_available = sum(max(0, end - start - window_size + 1) for start, end in segments)
    if total_available <= 0:
        return windows

    for _ in range(num_windows):
        # 按segment长度加权随机选择
        weights = [max(0, end - start - window_size + 1) for start, end in segments]
        total_w = sum(weights)
        if total_w <= 0:
            break

        r = random.random() * total_w
        cumsum = 0
        for i, (start, end) in enumerate(segments):
            cumsum += weights[i]
            if cumsum >= r:
                # 在这个segment内随机取窗口
                max_start = end - window_size
                win_start = random.randint(start, max_start)
                window = genome_seq[win_start:win_start + window_size]
                if window.count('N') / len(window) < 0.05:
                    windows.append(window)
                break

    return windows


def main():
    print("=" * 60)
    print("构建干净数据集 (antiSMASH标注排除BGC)")
    print("=" * 60)

    # === Step 1: 解析antiSMASH结果 ===
    print("\n[1/6] 解析antiSMASH结果...")
    antismash_zips = sorted(ANTISMASH_DIR.glob("*.zip"))
    genome_bgc_map = {}  # gcf_id → {contig_id: [(start, end), ...]}

    for zp in antismash_zips:
        gcf_id, regions = parse_antismash_bgc_regions(zp)
        genome_bgc_map[gcf_id] = regions
        total_bgc = sum(len(v) for v in regions.values())
        print(f"  {gcf_id}: {total_bgc}个BGC region")

    print(f"  共{len(genome_bgc_map)}个基因组的BGC标注")

    # === Step 2: 构造负样本 ===
    print(f"\n[2/6] 从参考基因组中提取干净的非BGC区域...")

    neg_windows = []
    for gcf_id, bgc_regions_by_contig in genome_bgc_map.items():
        # 找到对应的FASTA文件
        gcf_dir = REF_GENOME_DIR / gcf_id
        fna_files = list(gcf_dir.glob("*.fna"))
        if not fna_files:
            print(f"  {gcf_id}: 找不到FASTA文件，跳过")
            continue

        for fna in fna_files:
            for rec in SeqIO.parse(fna, 'fasta'):
                contig_id = rec.id
                genome_seq = str(rec.seq).upper()

                # 获取该contig的BGC区域
                bgc_regions = bgc_regions_by_contig.get(contig_id, [])

                # 提取非BGC区段
                nonbgc_segments = get_nonbgc_segments(genome_seq, bgc_regions)

                # 从非BGC区段中采样窗口
                # 每个基因组取的窗口数按非BGC长度成比例
                nonbgc_len = sum(e - s for s, e in nonbgc_segments)
                n_windows = nonbgc_len // WINDOW_SIZE
                windows = sample_windows_from_segments(
                    genome_seq, nonbgc_segments, WINDOW_SIZE, n_windows
                )

                for w in windows:
                    neg_windows.append({
                        'dna': w,
                        'source': gcf_id,
                        'contig': contig_id,
                    })

    print(f"  提取了 {len(neg_windows)} 个干净的负样本窗口")

    # === Step 3: 构造正样本 ===
    print(f"\n[3/6] 构造正样本 (MIBiG BGC)...")
    pos_samples = []
    for f in sorted(MIBIG_DIR.glob("BGC*.gbk")):
        try:
            rec = list(SeqIO.parse(f, 'genbank'))[0]
        except:
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

        organism = 'unknown'
        for feat in rec.features:
            if feat.type == 'source' and 'organism' in feat.qualifiers:
                organism = feat.qualifiers['organism'][0]
                break

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
            })

    print(f"  正样本: {len(pos_samples)}")

    # === Step 4: 平衡正负样本 ===
    print(f"\n[4/6] 平衡正负样本...")
    num_needed = len(pos_samples)
    if len(neg_windows) > num_needed:
        random.shuffle(neg_windows)
        neg_windows = neg_windows[:num_needed]
    print(f"  负样本(截断后): {len(neg_windows)}")

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

    # === Step 5: GC含量检查 ===
    def gc(seq):
        c = ''.join(x for x in seq if x in 'ATCG')
        return (c.count('G') + c.count('C')) / max(len(c), 1)

    pos_gc = [gc(s['dna']) for s in random.sample(pos_samples, min(1000, len(pos_samples)))]
    neg_gc = [gc(s['dna']) for s in random.sample(neg_samples, min(1000, len(neg_samples)))]
    print(f"\n  GC含量: 正={np.mean(pos_gc):.3f}±{np.std(pos_gc):.3f}, 负={np.mean(neg_gc):.3f}±{np.std(neg_gc):.3f}")

    # === Step 6: 分割数据集 ===
    print(f"\n[5/6] 分割数据集...")
    bgc_ids = list(set(s['source'] for s in pos_samples))
    random.shuffle(bgc_ids)
    n = len(bgc_ids)
    train_ids = set(bgc_ids[:int(n * 0.8)])
    val_ids = set(bgc_ids[int(n * 0.8):int(n * 0.9)])
    test_ids = set(bgc_ids[int(n * 0.9):])

    train_pos = [s for s in pos_samples if s['source'] in train_ids]
    val_pos = [s for s in pos_samples if s['source'] in val_ids]
    test_pos = [s for s in pos_samples if s['source'] in test_ids]

    random.shuffle(neg_samples)
    n_neg = len(neg_samples)
    train_neg = neg_samples[:int(n_neg * 0.8)]
    val_neg = neg_samples[int(n_neg * 0.8):int(n_neg * 0.9)]
    test_neg = neg_samples[int(n_neg * 0.9):]

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
        print(f"  {name}: {len(ds)} (正:{pos_c}, 负:{neg_c})")

    # 保存
    print(f"\n[6/6] 保存...")
    for name, ds in splits.items():
        path = OUTPUT_DIR / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(ds, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {path} ({size_mb:.0f}MB)")

    metadata = {
        'window_size': WINDOW_SIZE,
        'num_bgcs': len(bgc_ids),
        'num_positive': len(pos_samples),
        'num_negative': len(neg_samples),
        'num_antismash_genomes': len(genome_bgc_map),
        'total_antismash_bgcs': sum(sum(len(v) for v in regions.values()) for regions in genome_bgc_map.values()),
        'pos_gc_mean': float(np.mean(pos_gc)),
        'neg_gc_mean': float(np.mean(neg_gc)),
        'splits': {name: {'total': len(ds), 'pos': sum(1 for s in ds if s['is_bgc']==1), 'neg': sum(1 for s in ds if s['is_bgc']==0)} for name, ds in splits.items()},
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n完成! 数据在 {OUTPUT_DIR}/")
    return splits


if __name__ == '__main__':
    splits = main()

    # 跑baseline验证
    print(f"\n{'='*60}")
    print("Baseline验证")
    print(f"{'='*60}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    import time

    def kmer_feat(seq, k=4):
        c = ''.join(x for x in seq.upper() if x in 'ATCG')
        if len(c) < k: return np.zeros(260)
        counts = Counter(c[i:i+k] for i in range(len(c)-k+1))
        bases = 'ATCG'
        kmers = []
        def gen(p, r):
            if r==0: kmers.append(p); return
            for b in bases: gen(p+b, r-1)
        gen('', k)
        total = max(sum(counts.values()), 1)
        kf = [counts.get(km,0)/total for km in kmers]
        gc_val = (c.count('G')+c.count('C'))/len(c)
        return np.array(kf + [gc_val, seq.count('N')/len(seq), len(set(c[i:i+4] for i in range(len(c)-3)))/min(256,max(len(c)-3,1)), len(c)/WINDOW_SIZE])

    t0 = time.time()
    X_train = np.array([kmer_feat(s['dna']) for s in splits['train']])
    y_train = np.array([s['is_bgc'] for s in splits['train']])
    X_test = np.array([kmer_feat(s['dna']) for s in splits['test']])
    y_test = np.array([s['is_bgc'] for s in splits['test']])
    print(f"特征提取: {time.time()-t0:.0f}s")

    for name, clf in [
        ('LR', LogisticRegression(max_iter=1000, random_state=42)),
        ('RF', RandomForestClassifier(100, max_depth=20, random_state=42, n_jobs=-1)),
    ]:
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:,1]
        y_pred = clf.predict(X_test)
        auroc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"  {name}: AUROC={auroc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
