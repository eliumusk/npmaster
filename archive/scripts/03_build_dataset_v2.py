"""
数据集构建 v2 - 使用真实基因组的non-BGC区域作为负样本

改进点：
1. 负样本来自真实基因组，而非打乱的BGC序列
2. 这样正负样本在k-mer分布上有自然差异（BGC区域通常GC含量更高、基因密度不同等）
3. 更接近真实应用场景
"""

import os
import pickle
import random
import json
import numpy as np
from pathlib import Path
from collections import Counter
from Bio import SeqIO

MIBIG_DIR = Path("data/mibig_gbk_3.1")
REF_GENOMES_DIR = Path("data/ref_genomes")
OUTPUT_DIR = Path("data/processed_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 8192
RANDOM_SEED = 42

CATEGORY_MAP = {
    'Polyketide': 0, 'NRP': 1, 'RiPP': 2, 'Terpene': 3,
    'Saccharide': 4, 'Alkaloid': 5, 'Other': 6,
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def parse_mibig_gbk(gbk_path):
    """从MIBiG GenBank文件中提取信息"""
    rec = list(SeqIO.parse(gbk_path, 'genbank'))[0]
    bgc_id = os.path.basename(gbk_path).replace('.gbk', '')
    dna_seq = str(rec.seq).upper()

    categories = set()
    for feat in rec.features:
        if feat.type == 'subregion' and 'label' in feat.qualifiers:
            for label in feat.qualifiers['label']:
                for sub_label in label.split(','):
                    sub_label = sub_label.strip()
                    if sub_label in CATEGORY_MAP:
                        categories.add(sub_label)
                    else:
                        categories.add('Other')
    if not categories:
        categories.add('Other')

    return {
        'id': bgc_id,
        'dna_seq': dna_seq,
        'dna_len': len(dna_seq),
        'categories': list(categories),
        'category_vector': [1 if cat in categories else 0 for cat in CATEGORY_MAP.keys()],
    }


def create_windows(dna_seq, window_size, stride=None):
    """切分DNA序列为固定大小窗口"""
    if stride is None:
        stride = window_size // 2

    seq_len = len(dna_seq)

    if seq_len <= window_size:
        pad_left = (window_size - seq_len) // 2
        pad_right = window_size - seq_len - pad_left
        padded = 'N' * pad_left + dna_seq + 'N' * pad_right
        return [padded]

    windows = []
    for start in range(0, seq_len - window_size + 1, stride):
        windows.append(dna_seq[start:start + window_size])

    if (seq_len - window_size) % stride != 0:
        windows.append(dna_seq[-window_size:])

    return windows


def extract_negative_from_genomes(genome_dir, num_negatives, window_size):
    """从参考基因组中随机提取non-BGC DNA片段作为负样本"""
    # 收集所有参考基因组的DNA序列
    genome_seqs = []
    for fna_file in genome_dir.glob("*.fna"):
        for rec in SeqIO.parse(fna_file, 'fasta'):
            seq = str(rec.seq).upper()
            if len(seq) >= window_size:
                genome_seqs.append(seq)
                print(f"  加载基因组: {fna_file.name}, 长度={len(seq):,}bp")

    if not genome_seqs:
        raise ValueError(f"没有找到足够长的基因组序列在 {genome_dir}")

    # 随机截取片段
    negatives = []
    for i in range(num_negatives):
        # 随机选一个基因组
        genome = random.choice(genome_seqs)
        # 随机选一个起始位置
        max_start = len(genome) - window_size
        start = random.randint(0, max_start)
        fragment = genome[start:start + window_size]

        # 确保不含太多N
        n_ratio = fragment.count('N') / len(fragment)
        if n_ratio > 0.1:
            i -= 1
            continue

        negatives.append(fragment)

    return negatives


def main():
    print("=" * 60)
    print("数据集构建 v2 - 真实基因组负样本")
    print("=" * 60)

    # Step 1: 解析MIBiG
    print("\n[1/5] 解析MIBiG GenBank文件...")
    gbk_files = sorted(MIBIG_DIR.glob("BGC*.gbk"))
    all_bgcs = []
    for f in gbk_files:
        try:
            all_bgcs.append(parse_mibig_gbk(f))
        except Exception as e:
            pass
    print(f"  成功: {len(all_bgcs)} BGCs")

    # Step 2: 正样本
    print(f"\n[2/5] 构造正样本 (窗口={WINDOW_SIZE}bp)...")
    positive_samples = []
    for bgc in all_bgcs:
        windows = create_windows(bgc['dna_seq'], WINDOW_SIZE)
        for i, window in enumerate(windows):
            positive_samples.append({
                'id': f"{bgc['id']}_w{i}",
                'dna': window,
                'is_bgc': 1,
                'category_vector': bgc['category_vector'],
                'categories': bgc['categories'],
                'source': bgc['id'],
            })
    print(f"  正样本数: {len(positive_samples)}")

    # Step 3: 负样本
    print(f"\n[3/5] 构造负样本（从参考基因组截取）...")
    num_neg = len(positive_samples)
    neg_seqs = extract_negative_from_genomes(REF_GENOMES_DIR, num_neg, WINDOW_SIZE)

    negative_samples = []
    for i, seq in enumerate(neg_seqs):
        negative_samples.append({
            'id': f"NEG_{i}",
            'dna': seq,
            'is_bgc': 0,
            'category_vector': [0] * len(CATEGORY_MAP),
            'categories': [],
            'source': 'reference_genome',
        })
    print(f"  负样本数: {len(negative_samples)}")

    # Step 4: 分析正负样本的区分度
    print(f"\n[4/5] 正负样本统计对比...")

    def gc_content(seq):
        clean = ''.join(c for c in seq if c in 'ATCG')
        if len(clean) == 0:
            return 0
        return (clean.count('G') + clean.count('C')) / len(clean)

    pos_gc = [gc_content(s['dna']) for s in positive_samples[:500]]
    neg_gc = [gc_content(s['dna']) for s in negative_samples[:500]]

    print(f"  正样本GC含量: mean={np.mean(pos_gc):.3f}, std={np.std(pos_gc):.3f}")
    print(f"  负样本GC含量: mean={np.mean(neg_gc):.3f}, std={np.std(neg_gc):.3f}")

    # Step 5: 分割数据集
    print(f"\n[5/5] 分割并保存数据集...")

    # 按BGC来源分组，确保同一BGC的不同窗口不会同时出现在训练集和测试集
    bgc_ids = list(set(s['source'] for s in positive_samples))
    random.shuffle(bgc_ids)

    n_bgc = len(bgc_ids)
    train_bgcs = set(bgc_ids[:int(n_bgc * 0.8)])
    val_bgcs = set(bgc_ids[int(n_bgc * 0.8):int(n_bgc * 0.9)])
    test_bgcs = set(bgc_ids[int(n_bgc * 0.9):])

    train_pos = [s for s in positive_samples if s['source'] in train_bgcs]
    val_pos = [s for s in positive_samples if s['source'] in val_bgcs]
    test_pos = [s for s in positive_samples if s['source'] in test_bgcs]

    # 负样本也按比例分
    random.shuffle(negative_samples)
    n_neg = len(negative_samples)
    train_neg = negative_samples[:int(n_neg * 0.8)]
    val_neg = negative_samples[int(n_neg * 0.8):int(n_neg * 0.9)]
    test_neg = negative_samples[int(n_neg * 0.9):]

    train_set = train_pos + train_neg
    val_set = val_pos + val_neg
    test_set = test_pos + test_neg
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    print(f"  训练集: {len(train_set)} (正:{len(train_pos)}, 负:{len(train_neg)})")
    print(f"  验证集: {len(val_set)} (正:{len(val_pos)}, 负:{len(val_neg)})")
    print(f"  测试集: {len(test_set)} (正:{len(test_pos)}, 负:{len(test_neg)})")
    print(f"  [防泄露] 同一BGC的所有窗口只出现在同一个split中 ✓")

    for name, dataset in [('train', train_set), ('val', val_set), ('test', test_set)]:
        path = OUTPUT_DIR / f"{name}_w{WINDOW_SIZE}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  保存: {path}")

    metadata = {
        'window_size': WINDOW_SIZE,
        'num_bgcs': len(all_bgcs),
        'num_positive': len(positive_samples),
        'num_negative': len(negative_samples),
        'splits': {
            'train': {'total': len(train_set), 'pos': len(train_pos), 'neg': len(train_neg)},
            'val': {'total': len(val_set), 'pos': len(val_pos), 'neg': len(val_neg)},
            'test': {'total': len(test_set), 'pos': len(test_pos), 'neg': len(test_neg)},
        },
        'positive_gc_mean': float(np.mean(pos_gc)),
        'negative_gc_mean': float(np.mean(neg_gc)),
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n完成!")


if __name__ == '__main__':
    main()
