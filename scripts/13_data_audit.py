"""完整数据审计 - 精确报告每个数字"""

import pickle, numpy as np
from collections import Counter

BASE = "/data/muskliu/np_master/data"
MULTIGENUS = BASE + "/multigenus"

def gc(seq):
    c = "".join(x for x in seq if x in "ATCG")
    return (c.count("G") + c.count("C")) / max(len(c), 1)

# ====== Streptomyces 训练集审计 ======
print("=" * 70)
print("1. Streptomyces 训练集 (用于训练分类头)")
print("=" * 70)

for split in ["train", "val", "test"]:
    with open(f"{BASE}/{split}.pkl", "rb") as f:
        samples = pickle.load(f)
    pos = [s for s in samples if s["is_bgc"] == 1]
    neg = [s for s in samples if s["is_bgc"] == 0]
    pos_bgcs = set(s["source"] for s in pos)
    neg_genomes = set(s["source"] for s in neg)
    pos_per_bgc = Counter(s["source"] for s in pos)
    avg_win = np.mean(list(pos_per_bgc.values())) if pos_per_bgc else 0
    pos_gc_val = np.mean([gc(s["dna"]) for s in pos]) if pos else 0
    neg_gc_val = np.mean([gc(s["dna"]) for s in neg]) if neg else 0

    print(f"\n  {split}:")
    print(f"    窗口: {len(pos)} pos + {len(neg)} neg = {len(samples)}")
    print(f"    独立: {len(pos_bgcs)} BGC, {len(neg_genomes)} genome")
    print(f"    每BGC平均窗口: {avg_win:.1f}")
    print(f"    GC: pos={pos_gc_val:.3f}, neg={neg_gc_val:.3f}, diff={abs(pos_gc_val - neg_gc_val):.4f}")

# BGC泄露
all_bgcs = {}
all_neg_g = {}
for split in ["train", "val", "test"]:
    with open(f"{BASE}/{split}.pkl", "rb") as f:
        samples = pickle.load(f)
    all_bgcs[split] = set(s["source"] for s in samples if s["is_bgc"] == 1)
    all_neg_g[split] = set(s["source"] for s in samples if s["is_bgc"] == 0)

print(f"\n  BGC泄露: train&val={len(all_bgcs['train'] & all_bgcs['val'])}, "
      f"train&test={len(all_bgcs['train'] & all_bgcs['test'])}, "
      f"val&test={len(all_bgcs['val'] & all_bgcs['test'])}")
print(f"  负样本基因组泄露: train&test={len(all_neg_g['train'] & all_neg_g['test'])}")
print(f"  总BGC数: {len(all_bgcs['train'] | all_bgcs['val'] | all_bgcs['test'])}")
print(f"  总负样本基因组: {len(all_neg_g['train'] | all_neg_g['val'] | all_neg_g['test'])}")

# ====== 跨属测试数据审计 ======
print("\n" + "=" * 70)
print("2. 跨属测试数据 (全量pool, 这些数据只用于测试不用于训练)")
print("=" * 70)

for genus in ["micromonospora", "amycolatopsis", "pseudomonas"]:
    all_samples = []
    for split in ["train", "val", "test"]:
        with open(f"{MULTIGENUS}/{genus}/{split}.pkl", "rb") as f:
            all_samples.extend(pickle.load(f))

    pos = [s for s in all_samples if s["is_bgc"] == 1]
    neg = [s for s in all_samples if s["is_bgc"] == 0]
    pos_bgcs = set(s["source"] for s in pos)
    neg_genomes = set(s["source"] for s in neg)
    pos_per_bgc = Counter(s["source"] for s in pos)
    avg_win = np.mean(list(pos_per_bgc.values()))
    pos_gc_val = np.mean([gc(s["dna"]) for s in pos])
    neg_gc_val = np.mean([gc(s["dna"]) for s in neg])

    print(f"\n  === {genus} ===")
    print(f"  窗口: {len(pos)} pos + {len(neg)} neg = {len(all_samples)}")
    print(f"  独立: {len(pos_bgcs)} BGC, {len(neg_genomes)} genomes")
    print(f"  每BGC平均窗口: {avg_win:.1f}")
    print(f"  GC: pos={pos_gc_val:.3f}, neg={neg_gc_val:.3f}, diff={abs(pos_gc_val - neg_gc_val):.4f}")

    # 正样本来源物种
    pos_orgs = Counter(s.get("organism", "unknown") for s in pos)
    print(f"  正样本来源物种 ({len(pos_orgs)}种):")
    for org, cnt in pos_orgs.most_common():
        print(f"    {org}: {cnt} windows")

    # 每个BGC的窗口数分布
    wins_list = sorted(pos_per_bgc.values(), reverse=True)
    print(f"  BGC窗口数分布: max={wins_list[0]}, median={np.median(wins_list):.0f}, min={wins_list[-1]}")

# ====== 核心问题审计 ======
print("\n" + "=" * 70)
print("3. 已知问题和风险")
print("=" * 70)

# 问题1: Strep训练集的负样本基因组太少
print(f"\n  [问题1] Strep训练集只有{len(all_neg_g['train'])}个负样本基因组")
print(f"    风险: 模型可能记住了这20个基因组的特征而非通用的'非BGC'模式")
print(f"    test负样本来自3个不同的基因组, 但如果这3个恰好和train的20个很相似,")
print(f"    测试集AUROC会虚高")

# 问题2: 正负样本来源不对称
print(f"\n  [问题2] 正样本来自MIBiG(实验验证), 负样本来自参考基因组(antiSMASH排除)")
print(f"    MIBiG BGC经过了人工整理, 可能有特定的序列特征(如注释质量更高的物种)")
print(f"    负样本是'antiSMASH没检测到的区域', 但antiSMASH有~30%假阴性率")
print(f"    这意味着负样本中可能混入了真BGC (label noise)")

# 问题3: 跨属实验中正负样本不匹配
print(f"\n  [问题3] 跨属测试的正样本来自MIBiG, 负样本来自我们下载的参考基因组")
print(f"    MIBiG BGC和参考基因组不是来自同一个基因组")
print(f"    如果Evo2能区分'来自MIBiG注释质量的DNA'和'来自NCBI参考基因组的DNA',")
print(f"    这就是artifact而非真正的BGC检测能力")

# 问题4: 多domain训练的可行性
print(f"\n  [问题4] 为什么不在多属上训练?")
for genus in ["micromonospora", "amycolatopsis", "pseudomonas"]:
    all_s = []
    for split in ["train", "val", "test"]:
        with open(f"{MULTIGENUS}/{genus}/{split}.pkl", "rb") as f:
            all_s.extend(pickle.load(f))
    pos = [s for s in all_s if s["is_bgc"] == 1]
    bgcs = set(s["source"] for s in pos)
    print(f"    {genus}: 只有{len(bgcs)}个独立BGC, 训练+测试都不够拆分")
print(f"    Streptomyces有503个BGC, 其他属加起来才83个")
print(f"    多属训练的增益需要: 1)每个属足够的独立样本 2)属间BGC模式有差异")
print(f"    如果Evo2已经跨属泛化(我们的结果显示如此), 加小属数据不会有明显增益")
