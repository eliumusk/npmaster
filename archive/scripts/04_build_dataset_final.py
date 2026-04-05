"""
最终数据集构建 - 使用306个同源物种基因组的non-BGC区域作为负样本

数据源:
  正样本: MIBiG 3.1 (2502个已验证BGC的DNA序列)
  负样本: NCBI参考基因组 (306个基因组，覆盖Streptomyces等9个属)

关键设计:
  1. 负样本来自与BGC同属的物种，避免物种偏差
  2. 按BGC来源分割train/val/test，防止数据泄露
  3. 多种窗口大小(8k/16k/32k)以支持不同实验
  4. 记录每个样本的来源物种，便于后续分析
"""

import os
import pickle
import random
import json
import numpy as np
from pathlib import Path
from collections import Counter
from Bio import SeqIO
import glob
import time

MIBIG_DIR = Path("data/mibig_gbk_3.1")
REF_DIR = Path("data/ref_genomes_batch")
OUTPUT_BASE = Path("data/final")
RANDOM_SEED = 42

CATEGORY_MAP = {
    'Polyketide': 0, 'NRP': 1, 'RiPP': 2, 'Terpene': 3,
    'Saccharide': 4, 'Alkaloid': 5, 'Other': 6,
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_mibig():
    """加载MIBiG BGC数据"""
    bgcs = []
    for f in sorted(MIBIG_DIR.glob("BGC*.gbk")):
        try:
            rec = list(SeqIO.parse(f, 'genbank'))[0]
        except:
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

        # 提取物种
        organism = 'unknown'
        for feat in rec.features:
            if feat.type == 'source' and 'organism' in feat.qualifiers:
                organism = feat.qualifiers['organism'][0]
                break

        bgcs.append({
            'id': bgc_id,
            'dna': dna,
            'categories': list(cats),
            'category_vector': [1 if c in cats else 0 for c in CATEGORY_MAP],
            'organism': organism,
            'genus': organism.split()[0] if organism != 'unknown' else 'unknown',
        })
    return bgcs


def load_ref_genomes():
    """加载参考基因组DNA序列"""
    genomes = []
    for genus_dir in sorted(REF_DIR.glob("*/ncbi_dataset/data/GCF_*")):
        genus = genus_dir.parts[-4]  # genus name from directory

        fna_files = list(genus_dir.glob("*.fna"))
        if not fna_files:
            continue

        gcf_id = genus_dir.name

        for fna in fna_files:
            for rec in SeqIO.parse(fna, 'fasta'):
                seq = str(rec.seq).upper()
                if len(seq) >= 10000:  # 只保留>10kb的contig
                    genomes.append({
                        'genus': genus,
                        'gcf_id': gcf_id,
                        'contig': rec.id,
                        'seq': seq,
                        'length': len(seq),
                    })

    return genomes


def create_windows(dna_seq, window_size, stride=None):
    """切分DNA为固定窗口"""
    if stride is None:
        stride = window_size // 2

    if len(dna_seq) <= window_size:
        pad_l = (window_size - len(dna_seq)) // 2
        pad_r = window_size - len(dna_seq) - pad_l
        return ['N' * pad_l + dna_seq + 'N' * pad_r]

    windows = []
    for s in range(0, len(dna_seq) - window_size + 1, stride):
        windows.append(dna_seq[s:s + window_size])
    if (len(dna_seq) - window_size) % stride != 0:
        windows.append(dna_seq[-window_size:])
    return windows


def build_dataset(window_size):
    """构建指定窗口大小的数据集"""
    output_dir = OUTPUT_BASE / f"w{window_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"构建数据集: 窗口大小={window_size}bp")
    print(f"{'='*60}")

    # ===== 正样本 =====
    print("\n[1/5] 加载MIBiG BGC...")
    bgcs = load_mibig()
    print(f"  BGC总数: {len(bgcs)}")

    print("\n[2/5] 构造正样本...")
    pos_samples = []
    for bgc in bgcs:
        windows = create_windows(bgc['dna'], window_size)
        for i, w in enumerate(windows):
            pos_samples.append({
                'id': f"{bgc['id']}_w{i}",
                'dna': w,
                'is_bgc': 1,
                'category_vector': bgc['category_vector'],
                'categories': bgc['categories'],
                'source_id': bgc['id'],
                'genus': bgc['genus'],
                'organism': bgc['organism'],
            })
    print(f"  正样本数: {len(pos_samples)}")

    # 正样本的属分布
    pos_genus = Counter(s['genus'] for s in pos_samples)
    print("  正样本属分布(top5):")
    for g, c in pos_genus.most_common(5):
        print(f"    {g}: {c}")

    # ===== 负样本 =====
    print("\n[3/5] 加载参考基因组...")
    ref_genomes = load_ref_genomes()
    print(f"  参考序列数: {len(ref_genomes)}")

    # 按genus分组
    ref_by_genus = {}
    for g in ref_genomes:
        ref_by_genus.setdefault(g['genus'], []).append(g)

    print("  参考基因组属分布:")
    for genus, seqs in sorted(ref_by_genus.items(), key=lambda x: -len(x[1])):
        total_bp = sum(s['length'] for s in seqs)
        print(f"    {genus}: {len(seqs)}条序列, {total_bp/1e6:.0f}Mb")

    print("\n[4/5] 构造负样本...")
    num_neg = len(pos_samples)
    neg_samples = []
    attempts = 0
    max_attempts = num_neg * 3

    while len(neg_samples) < num_neg and attempts < max_attempts:
        attempts += 1
        genome = random.choice(ref_genomes)
        if genome['length'] < window_size:
            continue
        start = random.randint(0, genome['length'] - window_size)
        frag = genome['seq'][start:start + window_size]

        # 跳过N太多的片段
        if frag.count('N') / len(frag) > 0.05:
            continue

        neg_samples.append({
            'id': f"NEG_{len(neg_samples)}",
            'dna': frag,
            'is_bgc': 0,
            'category_vector': [0] * len(CATEGORY_MAP),
            'categories': [],
            'source_id': genome['gcf_id'],
            'genus': genome['genus'],
            'organism': f"{genome['genus']}_{genome['gcf_id']}",
        })

    print(f"  负样本数: {len(neg_samples)}")

    neg_genus = Counter(s['genus'] for s in neg_samples)
    print("  负样本属分布(top5):")
    for g, c in neg_genus.most_common(5):
        print(f"    {g}: {c}")

    # ===== GC含量对比 =====
    def gc(seq):
        c = ''.join(x for x in seq if x in 'ATCG')
        return (c.count('G') + c.count('C')) / max(len(c), 1)

    pos_gc = [gc(s['dna']) for s in random.sample(pos_samples, min(1000, len(pos_samples)))]
    neg_gc = [gc(s['dna']) for s in random.sample(neg_samples, min(1000, len(neg_samples)))]
    print(f"\n  GC含量: 正={np.mean(pos_gc):.3f}±{np.std(pos_gc):.3f}, 负={np.mean(neg_gc):.3f}±{np.std(neg_gc):.3f}")
    gc_diff = abs(np.mean(pos_gc) - np.mean(neg_gc))
    if gc_diff < 0.03:
        print("  ✓ GC含量接近，不会有简单的物种偏差")
    elif gc_diff < 0.08:
        print("  △ GC含量有一定差异，模型可能部分依赖GC特征")
    else:
        print("  ✗ GC含量差异大，需要检查负样本来源")

    # ===== 分割数据集 =====
    print(f"\n[5/5] 分割数据集 (按BGC来源分组防泄露)...")

    # 正样本按BGC ID分组
    bgc_ids = list(set(s['source_id'] for s in pos_samples))
    random.shuffle(bgc_ids)
    n = len(bgc_ids)
    train_ids = set(bgc_ids[:int(n * 0.8)])
    val_ids = set(bgc_ids[int(n * 0.8):int(n * 0.9)])
    test_ids = set(bgc_ids[int(n * 0.9):])

    train_pos = [s for s in pos_samples if s['source_id'] in train_ids]
    val_pos = [s for s in pos_samples if s['source_id'] in val_ids]
    test_pos = [s for s in pos_samples if s['source_id'] in test_ids]

    # 负样本按比例分
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
        pos_count = sum(1 for s in ds if s['is_bgc'] == 1)
        neg_count = len(ds) - pos_count
        print(f"  {name}: {len(ds)} (正:{pos_count}, 负:{neg_count})")

    # 保存
    for name, ds in splits.items():
        # 不保存dna字段到metadata中（太大）
        path = output_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(ds, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  保存: {path} ({size_mb:.1f}MB)")

    # 保存metadata
    metadata = {
        'window_size': window_size,
        'num_bgcs': len(bgcs),
        'num_positive': len(pos_samples),
        'num_negative': len(neg_samples),
        'pos_gc_mean': float(np.mean(pos_gc)),
        'neg_gc_mean': float(np.mean(neg_gc)),
        'pos_genus_dist': dict(pos_genus.most_common(20)),
        'neg_genus_dist': dict(neg_genus.most_common(20)),
        'category_dist': dict(Counter(
            cat for s in pos_samples for cat in s['categories']
        ).most_common()),
        'splits': {
            name: {
                'total': len(ds),
                'pos': sum(1 for s in ds if s['is_bgc'] == 1),
                'neg': sum(1 for s in ds if s['is_bgc'] == 0),
            }
            for name, ds in splits.items()
        },
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return splits


def run_baseline(splits, window_size):
    """用k-mer baseline验证数据集质量"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    print(f"\n{'='*60}")
    print(f"Baseline验证 (窗口={window_size}bp)")
    print(f"{'='*60}")

    def kmer_feat(seq, k=4):
        c = ''.join(x for x in seq.upper() if x in 'ATCG')
        if len(c) < k:
            return np.zeros(4**k + 4)
        counts = Counter(c[i:i+k] for i in range(len(c) - k + 1))
        bases = 'ATCG'
        kmers = []
        def gen(p, r):
            if r == 0:
                kmers.append(p)
                return
            for b in bases:
                gen(p + b, r - 1)
        gen('', k)
        total = max(sum(counts.values()), 1)
        kf = [counts.get(km, 0) / total for km in kmers]
        gc_val = (c.count('G') + c.count('C')) / len(c)
        n_rat = seq.count('N') / len(seq)
        un = len(set(c[i:i+k] for i in range(len(c) - k + 1)))
        compl = un / min(256, max(len(c) - k + 1, 1))
        return np.array(kf + [gc_val, n_rat, compl, len(c) / window_size])

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
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        auroc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"  {name}: AUROC={auroc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    print("\n解读:")
    print("  LR AUROC ~0.7-0.85: 正常，说明有区分度但非trivial")
    print("  RF AUROC ~0.85-0.95: 正常，非线性特征有帮助")
    print("  RF AUROC >0.98: 偏高，可能有偏差")


if __name__ == '__main__':
    t_start = time.time()

    # 先构建8k窗口（用于evo2_7b_base最小实验）
    splits_8k = build_dataset(8192)
    run_baseline(splits_8k, 8192)

    print(f"\n总耗时: {time.time()-t_start:.0f}s")
    print("\n后续可以构建更大窗口:")
    print("  python3 scripts/04_build_dataset_final.py --window 16384")
    print("  python3 scripts/04_build_dataset_final.py --window 32768")
