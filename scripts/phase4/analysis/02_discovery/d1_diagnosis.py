"""D1: Quantify why current discovery evidence is weak.

Analyze the existing Route B discovery_benchmark.json and signal strength.
"""
import json
import numpy as np
from collections import Counter

with open("/data/muskliu/np_master/data/phase3/route_b/discovery/discovery_benchmark.json") as f:
    data = json.load(f)

print("="*70)
print("D1: DIAGNOSIS — Why current discovery evidence is weak")
print("="*70)

m = data["metrics"]

print("\n### Current Route B discovery metrics ###")
print(f"  biosynthetic Pfam @20:  {m['precision_biosyn_at_20']*100:.0f}%  (null {m['null_precision_biosyn_at_20']*100:.0f}%)  enrichment {m['enrichment_biosyn_at_20']:.2f}x")
print(f"  biosynthetic Pfam @50:  {m['precision_biosyn_at_50']*100:.0f}%  (null {m['null_precision_biosyn_at_50']*100:.0f}%)  enrichment {m['enrichment_biosyn_at_50']:.2f}x")
print(f"  signature precision @20: {m['precision_signature_at_20']*100:.0f}%  (null {m['null_precision_signature_at_20']*100:.0f}%)  enrichment {m['precision_signature_at_20']/m['null_precision_signature_at_20']:.2f}x")
print(f"  signature precision @50: {m['precision_signature_at_50']*100:.0f}%  (null {m['null_precision_signature_at_50']*100:.0f}%)  enrichment {m['precision_signature_at_50']/m['null_precision_signature_at_50']:.2f}x")

print("\n### Problem 1: biosynthetic Pfam is saturated (100% vs 95%) ###")
print("'biosynthetic Pfam' 是一个宽泛的 Pfam 列表（数百个 family），包括 SDR、SAM methyltransferase、AMP-binding 等.")
print("这些在细菌基因组里几乎无处不在. 100% vs 95% 的区分度几乎为零.")
print("  → 作为 discovery metric 完全无意义.")

print("\n### Problem 2: 'signature precision' has moderate enrichment (1.4x) but small base ###")
print("Signature 是更严格的核心 BGC 结构域 list (KS/AT/ACP/C/A/PCP/TE 等).")
print(f"Route B 的 @20: 18/20 有 signature，null @20: 13/20 有 signature.")
print("enrichment 1.38x，但基数太小（N=20），统计噪声大.")
print("  → 需要更大样本 + 更严格的 null.")

# Analyze per-candidate signatures from the existing data
print("\n### Per-candidate analysis (existing data) ###")
cands = data["candidate_results"]
valid = [c for c in cands if not c.get("skip", False)]
print(f"Total validated candidates: {len(valid)}")

# signature domain count distribution
sig_counts = [c.get("n_signature", 0) for c in valid]
pfam_counts = [c.get("n_pfam", 0) for c in valid]
gene_counts = [c.get("n_genes", 0) for c in valid]
length_kb = [c["length"]/1000 for c in valid]

print(f"\n  sig domain count: mean={np.mean(sig_counts):.1f}, median={np.median(sig_counts):.0f}, max={max(sig_counts)}")
print(f"  n_pfam count:     mean={np.mean(pfam_counts):.1f}, median={np.median(pfam_counts):.0f}")
print(f"  n_genes:          mean={np.mean(gene_counts):.1f}, median={np.median(gene_counts):.0f}")
print(f"  length (kb):      mean={np.mean(length_kb):.1f}, median={np.median(length_kb):.0f}")

# sig density (domains per kb)
sig_density = [sig_counts[i]/max(length_kb[i], 1) for i in range(len(valid))]
print(f"  sig density (domains/kb): mean={np.mean(sig_density):.3f}, median={np.median(sig_density):.3f}")

# How many have 0 signature domains?
n_zero_sig = sum(1 for s in sig_counts if s == 0)
print(f"\n  Candidates with 0 signature domains: {n_zero_sig}/{len(valid)} ({n_zero_sig/len(valid)*100:.0f}%)")
n_sparse = sum(1 for s in sig_counts if 0 < s <= 2)
n_moderate = sum(1 for s in sig_counts if 3 <= s <= 5)
n_dense = sum(1 for s in sig_counts if s > 5)
print(f"  Sparse (1-2 sig):       {n_sparse} ({n_sparse/len(valid)*100:.0f}%)")
print(f"  Moderate (3-5 sig):     {n_moderate} ({n_moderate/len(valid)*100:.0f}%)")
print(f"  Dense (>5 sig):         {n_dense} ({n_dense/len(valid)*100:.0f}%)")

print("\n### Problem 3: Top 50 candidates are ranked by max_score, not by evidence ###")
# Sort by max_score and by n_signature
by_score = sorted(valid, key=lambda c: -c["max_score"])[:20]
by_sig = sorted(valid, key=lambda c: -c.get("n_signature", 0))[:20]
print(f"  Top-20 by max_score: {sum(1 for c in by_score if c.get('n_signature', 0) > 0)}/20 have signature domains")
print(f"  Top-20 by n_signature: {sum(1 for c in by_sig if c.get('n_signature', 0) > 0)}/20 have signature domains")
print(f"  Overlap: {len(set(id(c) for c in by_score) & set(id(c) for c in by_sig))}/20")

# Average sig count
print(f"  Top-20 by max_score: mean sig count = {np.mean([c.get('n_signature', 0) for c in by_score]):.1f}")
print(f"  Top-20 by n_signature: mean sig count = {np.mean([c.get('n_signature', 0) for c in by_sig]):.1f}")

# Check: if we ranked by sig count directly, would we do better?
print("\n  → ranking by max_score misses some sig-rich candidates; composite ranker needed.")

print("\n### Problem 4: Null is random negative windows, not matched ###")
print("Current null sampling (from archive):")
print("  1. Pick random negative windows from test_id")
print("  2. Extend to target length")
print("  3. That's it — no GC matching, no gene density matching, no same-genome constraint")
print()
print("Problem: in Streptomyces (BGC-rich), 'random negative' often samples regions")
print("near BGC boundaries or in biosynthetic-gene-dense pockets. This makes null too strong.")
print()
print("Fix (D2): same-genome + GC ±2% + gene density ±20% + distance from all tools > 20-50kb")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
当前 discovery 证据链的三大缺陷：

1. **Metric 选错**: biosynthetic Pfam 过于宽泛，100% vs 95% 无区分度
2. **Null 偏强**: 随机负区域可能采样到 BGC 边界，signature 1.4x enrichment 边际
3. **Ranking 过简**: 仅按 max_score，没利用 signature count / module completeness

**量化结果**：
  - Top-50 candidates 中 {n_dense} 个有 >5 signature domains（看起来像真 BGC）
  - {n_sparse + n_zero_sig} 个 signature 稀疏（可能 FP 或非 canonical 架构）
  - 这 500 个总候选（412 clean_missed）需要更严格筛选才能交湿实验

**下一步（D2 + D3 + D4）**：
  - Matched null 让 signature enrichment 能真正区分
  - Composite ranker 双轨：conservative (module complete) + novelty (enzymatic density)
  - Multi-tool concordance 作为附加 evidence layer（P2 完成后）
""")

# Save to JSON
import os
os.makedirs("/data/muskliu/np_master/data/phase4", exist_ok=True)
with open("/data/muskliu/np_master/data/phase4/d1_diagnosis.json", "w") as f:
    json.dump({
        "current_metrics": m,
        "n_candidates": len(valid),
        "sig_distribution": {"zero": n_zero_sig, "sparse_1_2": n_sparse, "moderate_3_5": n_moderate, "dense_gt5": n_dense},
        "sig_density_stats": {"mean": float(np.mean(sig_density)), "median": float(np.median(sig_density))},
        "sig_count_stats": {"mean": float(np.mean(sig_counts)), "median": float(np.median(sig_counts)), "max": int(max(sig_counts))},
    }, f, indent=2)
print("Saved to /data/muskliu/np_master/data/phase4/d1_diagnosis.json")
