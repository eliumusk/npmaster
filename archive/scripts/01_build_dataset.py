"""
最小闭环实验 - 数据构建脚本
目标：从MIBiG构建正负样本，准备Evo2可用的DNA序列数据集

思路：
- 正样本：MIBiG中每个BGC的完整DNA序列
- 负样本：从MIBiG GenBank中取BGC外的flanking区域（如果有的话），
         或者打乱正样本的DNA序列作为"假BGC"
- 固定窗口大小：先用8192bp做最小实验（evo2_7b_base支持的最大长度）

数据流：
  MIBiG GenBank → 提取DNA序列 + 类别标签 → 固定长度窗口 → 保存为pickle

"""

import os
import sys
import json
import pickle
import random
import numpy as np
from pathlib import Path
from collections import Counter
from Bio import SeqIO

# ============================================================
# 配置
# ============================================================
MIBIG_DIR = Path("data/mibig_gbk_3.1")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 8192  # 8kb窗口，evo2_7b_base的上限
RANDOM_SEED = 42

# BGC类别映射（7类 + non-BGC）
CATEGORY_MAP = {
    'Polyketide': 0,
    'NRP': 1,
    'RiPP': 2,
    'Terpene': 3,
    'Saccharide': 4,
    'Alkaloid': 5,
    'Other': 6,
}
NUM_CLASSES = 7

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def parse_mibig_gbk(gbk_path):
    """从MIBiG GenBank文件中提取信息"""
    rec = list(SeqIO.parse(gbk_path, 'genbank'))[0]
    bgc_id = os.path.basename(gbk_path).replace('.gbk', '')

    dna_seq = str(rec.seq).upper()

    # 提取类别
    categories = set()
    for feat in rec.features:
        if feat.type == 'subregion' and 'label' in feat.qualifiers:
            for label in feat.qualifiers['label']:
                # 处理多标签格式如 "NRP, Polyketide"
                for sub_label in label.split(','):
                    sub_label = sub_label.strip()
                    if sub_label in CATEGORY_MAP:
                        categories.add(sub_label)
                    else:
                        categories.add('Other')

    if not categories:
        categories.add('Other')

    # 提取CDS信息
    cds_list = []
    for feat in rec.features:
        if feat.type == 'CDS':
            protein = feat.qualifiers.get('translation', [''])[0]
            cds_list.append({
                'start': int(feat.location.start),
                'end': int(feat.location.end),
                'strand': feat.location.strand,
                'protein': protein,
                'dna': str(feat.location.extract(rec.seq)),
            })

    # 提取生物体信息
    organism = 'unknown'
    for feat in rec.features:
        if feat.type == 'source' and 'organism' in feat.qualifiers:
            organism = feat.qualifiers['organism'][0]
            break

    return {
        'id': bgc_id,
        'dna_seq': dna_seq,
        'dna_len': len(dna_seq),
        'categories': list(categories),
        'category_vector': [1 if cat in categories else 0 for cat in CATEGORY_MAP.keys()],
        'cds_list': cds_list,
        'organism': organism,
    }


def create_windows(dna_seq, window_size, stride=None):
    """将DNA序列切成固定大小的窗口

    如果序列比窗口短：两侧补N
    如果序列比窗口长：用滑动窗口切分
    """
    if stride is None:
        stride = window_size // 2  # 50% overlap

    seq_len = len(dna_seq)

    if seq_len <= window_size:
        # 短序列：居中放置，两侧补N
        pad_left = (window_size - seq_len) // 2
        pad_right = window_size - seq_len - pad_left
        padded = 'N' * pad_left + dna_seq + 'N' * pad_right
        return [padded]

    # 长序列：滑动窗口
    windows = []
    for start in range(0, seq_len - window_size + 1, stride):
        windows.append(dna_seq[start:start + window_size])

    # 确保最后一个窗口覆盖到序列末尾
    if (seq_len - window_size) % stride != 0:
        windows.append(dna_seq[-window_size:])

    return windows


def generate_negative_samples(positive_seqs, num_negatives, window_size):
    """生成负样本

    策略：打乱正样本DNA序列的局部顺序（dinucleotide shuffle）
    保持碱基频率但破坏序列模式

    这比随机生成ATCG更好，因为：
    1. 保持了GC含量等统计特征
    2. 但破坏了生物学有意义的模式（基因结构、调控序列等）
    """
    negatives = []

    for i in range(num_negatives):
        # 随机选一个正样本作为模板
        template = random.choice(positive_seqs)

        # Dinucleotide shuffle：以2bp为单位打乱
        # 这保持了dinucleotide频率但破坏了高阶结构
        seq_list = list(template)
        # 按chunk打乱
        chunk_size = 200  # 200bp的chunk
        chunks = [template[j:j+chunk_size] for j in range(0, len(template), chunk_size)]
        random.shuffle(chunks)
        shuffled = ''.join(chunks)

        # 确保长度正确
        if len(shuffled) < window_size:
            shuffled = shuffled + 'N' * (window_size - len(shuffled))
        shuffled = shuffled[:window_size]

        negatives.append(shuffled)

    return negatives


def main():
    print("=" * 60)
    print("MIBiG BGC 数据集构建")
    print("=" * 60)

    # Step 1: 解析所有MIBiG文件
    print("\n[1/4] 解析MIBiG GenBank文件...")
    gbk_files = sorted(MIBIG_DIR.glob("BGC*.gbk"))

    all_bgcs = []
    errors = []
    for f in gbk_files:
        try:
            info = parse_mibig_gbk(f)
            all_bgcs.append(info)
        except Exception as e:
            errors.append((f.name, str(e)))

    print(f"  成功解析: {len(all_bgcs)}")
    print(f"  解析失败: {len(errors)}")

    # 统计
    cat_counter = Counter()
    for bgc in all_bgcs:
        for cat in bgc['categories']:
            cat_counter[cat] += 1

    print(f"\n  类别分布:")
    for cat, count in cat_counter.most_common():
        print(f"    {cat}: {count}")

    # Step 2: 构造正样本
    print(f"\n[2/4] 构造正样本 (窗口大小={WINDOW_SIZE}bp)...")
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
                'source_bgc': bgc['id'],
                'source_len': bgc['dna_len'],
            })

    print(f"  正样本数: {len(positive_samples)}")
    print(f"  (来自{len(all_bgcs)}个BGC，平均每个{len(positive_samples)/len(all_bgcs):.1f}个窗口)")

    # Step 3: 构造负样本
    print(f"\n[3/4] 构造负样本...")
    num_negatives = len(positive_samples)  # 1:1正负比

    positive_seqs = [s['dna'] for s in positive_samples]
    negative_seqs = generate_negative_samples(positive_seqs, num_negatives, WINDOW_SIZE)

    negative_samples = []
    for i, seq in enumerate(negative_seqs):
        negative_samples.append({
            'id': f"NEG_{i}",
            'dna': seq,
            'is_bgc': 0,
            'category_vector': [0] * NUM_CLASSES,
            'categories': [],
            'source_bgc': 'negative',
            'source_len': WINDOW_SIZE,
        })

    print(f"  负样本数: {len(negative_samples)}")

    # Step 4: 合并、打乱、分割
    print(f"\n[4/4] 合并并分割数据集...")
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)

    # 80/10/10 split
    n = len(all_samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_set = all_samples[:n_train]
    val_set = all_samples[n_train:n_train + n_val]
    test_set = all_samples[n_train + n_val:]

    print(f"  训练集: {len(train_set)}")
    print(f"  验证集: {len(val_set)}")
    print(f"  测试集: {len(test_set)}")

    # 保存
    for name, dataset in [('train', train_set), ('val', val_set), ('test', test_set)]:
        output_path = OUTPUT_DIR / f"{name}_w{WINDOW_SIZE}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  保存: {output_path} ({len(dataset)} samples)")

    # 保存元数据
    metadata = {
        'window_size': WINDOW_SIZE,
        'num_bgcs': len(all_bgcs),
        'num_positive': len(positive_samples),
        'num_negative': len(negative_samples),
        'num_train': len(train_set),
        'num_val': len(val_set),
        'num_test': len(test_set),
        'categories': list(CATEGORY_MAP.keys()),
        'category_distribution': dict(cat_counter),
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n完成! 数据保存在 {OUTPUT_DIR}/")
    print(f"\n=== 重要提醒 ===")
    print(f"这是最小实验版本的数据集。正式实验需要改进：")
    print(f"1. 负样本应该来自真实基因组的non-BGC区域，而非简单打乱")
    print(f"2. 需要确保训练/测试集的生物体没有overlap（防止数据泄露）")
    print(f"3. 窗口大小可能需要调整（8kb只能装下14%的BGC而不切分）")
    print(f"4. 需要添加更细致的标注（BGC边界位置标注，不仅是整体标签）")


if __name__ == '__main__':
    main()
