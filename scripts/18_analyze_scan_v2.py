"""
诚实地分析全基因组扫描结果
- BGC级检测率 (不是窗口级)
- 多工具交叉对比 (Evo2+RF vs GECCO vs antiSMASH)
- 找"novel candidates" (三个工具都没检测到 but Evo2说有)
"""

import pickle, json, os, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

SCAN_DIR = Path("data/genome_scan")
GECCO_DIR = SCAN_DIR / "gecco_results"
TARGETS = {
    'S_longhuiensis':  'GCF_020616555.1',
    'S_deccanensis':   'GCF_022385335.1',
    'S_aquilus':        'GCF_003955715.1',
}


def overlap(r1_start, r1_end, r2_start, r2_end):
    """两个区间是否有重叠"""
    return r1_start < r2_end and r2_start < r1_end


def load_gecco_clusters(name):
    """加载GECCO的cluster预测"""
    gecco_dir = GECCO_DIR / name
    tsv_files = list(gecco_dir.glob("*.clusters.tsv"))
    if not tsv_files:
        return []
    df = pd.read_csv(tsv_files[0], sep='\t')
    clusters = []
    for _, row in df.iterrows():
        clusters.append({
            'contig': row['sequence_id'],
            'start': int(row['start']),
            'end': int(row['end']),
            'max_prob': float(row['max_p']),
            'avg_prob': float(row['average_p']),
            'type': row['type'],
        })
    return clusters


def merge_evo2_regions(predictions, prob_threshold=0.5, min_windows=2):
    """合并Evo2的连续高概率窗口为区域"""
    by_contig = defaultdict(list)
    for p in predictions:
        by_contig[p['contig']].append(p)

    regions = []
    for contig_id, preds in by_contig.items():
        preds.sort(key=lambda x: x['start'])

        i = 0
        while i < len(preds):
            if preds[i]['prob'] > prob_threshold:
                start = preds[i]['start']
                probs = [preds[i]['prob']]
                j = i + 1
                while j < len(preds) and preds[j]['prob'] > prob_threshold:
                    probs.append(preds[j]['prob'])
                    j += 1
                end = preds[j-1]['end']

                if len(probs) >= min_windows:
                    regions.append({
                        'contig': contig_id,
                        'start': start,
                        'end': end,
                        'max_prob': max(probs),
                        'mean_prob': float(np.mean(probs)),
                        'n_windows': len(probs),
                    })
                i = j
            else:
                i += 1

    return regions


def bgc_level_detection(predictions, antismash_regions, threshold=0.5):
    """BGC级别的检测率"""
    # 每个BGC: 它的任何一个窗口prob>threshold就算检测到
    detected = 0
    total = 0
    bgc_probs = []

    for contig_id, bgc_list in antismash_regions.items():
        for bgc_start, bgc_end, bgc_type in bgc_list:
            # 找覆盖这个BGC的窗口
            overlapping_probs = []
            for p in predictions:
                if p['contig'] == contig_id:
                    if overlap(p['start'], p['end'], bgc_start, bgc_end):
                        overlapping_probs.append(p['prob'])

            total += 1
            if overlapping_probs:
                max_p = max(overlapping_probs)
                bgc_probs.append((bgc_type, max_p, bgc_end - bgc_start))
                if max_p > threshold:
                    detected += 1

    return detected, total, bgc_probs


def main():
    print("=" * 75)
    print("全基因组扫描结果 - 多工具交叉分析")
    print("=" * 75)

    all_results = {}

    for name, gcf_id in TARGETS.items():
        scan_dir = SCAN_DIR / name

        # 加载数据
        with open(scan_dir / "predictions.pkl", 'rb') as f:
            predictions = pickle.load(f)
        with open(scan_dir / "antismash_regions.json") as f:
            antismash = json.load(f)
        with open(scan_dir / "metadata.json") as f:
            meta = json.load(f)

        gecco_clusters = load_gecco_clusters(name)

        print(f"\n{'='*65}")
        print(f"  {name} ({gcf_id}) - {meta['genome_size']/1e6:.1f} Mb")
        print(f"{'='*65}")

        # BGC级检测率
        print(f"\n  [1] BGC级检测率 (antiSMASH作为ground truth)")
        for thresh in [0.3, 0.5, 0.7]:
            det, tot, bgc_probs = bgc_level_detection(predictions, antismash, threshold=thresh)
            print(f"    threshold={thresh}: {det}/{tot} ({det/tot*100:.0f}%)")

        # 看看哪些BGC类型检测得好
        _, _, bgc_probs = bgc_level_detection(predictions, antismash, threshold=0.5)
        by_type = defaultdict(list)
        for btype, prob, length in bgc_probs:
            by_type[btype].append((prob, length))

        print(f"\n  [2] 按BGC类型分析 (每个BGC的max window prob)")
        print(f"    {'类型':<28} {'N':>3} {'mean_p':>7} {'det@0.5':>7} {'avg_len':>8}")
        for btype, data in sorted(by_type.items(), key=lambda x: -len(x[1])):
            probs = [p for p, _ in data]
            lens = [l for _, l in data]
            det = sum(1 for p in probs if p > 0.5)
            print(f"    {btype:<28} {len(data):>3} {np.mean(probs):>7.3f} "
                  f"{det}/{len(data)}".rjust(48) + f" {np.mean(lens)/1000:>7.1f}kb")

        # 3工具对比
        print(f"\n  [3] 工具对比: antiSMASH vs GECCO vs Evo2+RF")

        # Evo2 regions (merge high-prob windows)
        evo2_regions = merge_evo2_regions(predictions, prob_threshold=0.5, min_windows=2)
        print(f"    antiSMASH: {meta['n_antismash_bgcs']} BGCs")
        print(f"    GECCO:     {len(gecco_clusters)} BGCs")
        print(f"    Evo2+RF:   {len(evo2_regions)} high-prob regions")

        # 三工具的共识 + 独有
        print(f"\n  [4] 工具间一致性分析")

        # antiSMASH regions as reference
        as_regions = []
        for contig_id, bgc_list in antismash.items():
            for s, e, t in bgc_list:
                as_regions.append({'contig': contig_id, 'start': s, 'end': e, 'type': t})

        # 对每个antiSMASH区域, 检查GECCO和Evo2是否也检测到
        all_detected_by_as = 0
        detected_by_gecco = 0
        detected_by_evo2 = 0
        detected_by_both = 0

        for asr in as_regions:
            all_detected_by_as += 1
            g_hit = any(c['contig'] == asr['contig'] and overlap(c['start'], c['end'], asr['start'], asr['end']) for c in gecco_clusters)
            e_hit = any(r['contig'] == asr['contig'] and overlap(r['start'], r['end'], asr['start'], asr['end']) for r in evo2_regions)
            if g_hit:
                detected_by_gecco += 1
            if e_hit:
                detected_by_evo2 += 1
            if g_hit and e_hit:
                detected_by_both += 1

        print(f"    antiSMASH's {all_detected_by_as} BGCs:")
        print(f"      - GECCO also finds: {detected_by_gecco} ({detected_by_gecco/all_detected_by_as*100:.0f}%)")
        print(f"      - Evo2+RF also finds: {detected_by_evo2} ({detected_by_evo2/all_detected_by_as*100:.0f}%)")
        print(f"      - Both GECCO+Evo2 find: {detected_by_both}")

        # Evo2独有的高置信候选 (不被antiSMASH和GECCO检测到)
        print(f"\n  [5] Evo2独有的候选 (可能的novel BGC)")
        novel_candidates = []
        for er in evo2_regions:
            as_hit = any(a['contig'] == er['contig'] and overlap(a['start'], a['end'], er['start'], er['end']) for a in as_regions)
            g_hit = any(g['contig'] == er['contig'] and overlap(g['start'], g['end'], er['start'], er['end']) for g in gecco_clusters)
            if not as_hit and not g_hit:
                novel_candidates.append(er)

        novel_candidates.sort(key=lambda x: -x['max_prob'])
        print(f"    共找到 {len(novel_candidates)} 个Evo2独有的候选区域:")
        for i, nc in enumerate(novel_candidates[:5]):
            print(f"      #{i+1}: {nc['contig']}:{nc['start']:,}-{nc['end']:,} "
                  f"({(nc['end']-nc['start'])/1000:.1f}kb) "
                  f"max_prob={nc['max_prob']:.3f} n_win={nc['n_windows']}")

        # GECCO独有的候选 (Evo2 & antiSMASH没找到)
        gecco_novel = []
        for gc in gecco_clusters:
            as_hit = any(a['contig'] == gc['contig'] and overlap(a['start'], a['end'], gc['start'], gc['end']) for a in as_regions)
            e_hit = any(r['contig'] == gc['contig'] and overlap(r['start'], r['end'], gc['start'], gc['end']) for r in evo2_regions)
            if not as_hit and not e_hit:
                gecco_novel.append(gc)
        print(f"\n  [6] GECCO独有的候选: {len(gecco_novel)} (对照)")

        # Evo2+GECCO一致但antiSMASH没有
        both_novel = []
        for er in evo2_regions:
            as_hit = any(a['contig'] == er['contig'] and overlap(a['start'], a['end'], er['start'], er['end']) for a in as_regions)
            g_hit = any(g['contig'] == er['contig'] and overlap(g['start'], g['end'], er['start'], er['end']) for g in gecco_clusters)
            if not as_hit and g_hit:
                # Find matching GECCO cluster
                matching_gecco = [g for g in gecco_clusters if g['contig'] == er['contig'] and overlap(g['start'], g['end'], er['start'], er['end'])]
                if matching_gecco:
                    er['gecco_prob'] = matching_gecco[0]['max_prob']
                    er['gecco_type'] = matching_gecco[0]['type']
                    both_novel.append(er)

        both_novel.sort(key=lambda x: -x['max_prob'])
        print(f"\n  [7] 高置信novel候选 (Evo2+GECCO都说有, antiSMASH没检测): {len(both_novel)} 个")
        print(f"      这些是最有说服力的候选!")
        for i, nc in enumerate(both_novel[:10]):
            print(f"      #{i+1}: {nc['contig']}:{nc['start']:,}-{nc['end']:,} "
                  f"({(nc['end']-nc['start'])/1000:.1f}kb) "
                  f"Evo2={nc['max_prob']:.3f} GECCO={nc['gecco_prob']:.3f} type={nc['gecco_type']}")

        all_results[name] = {
            'antismash_bgcs': len(as_regions),
            'gecco_clusters': len(gecco_clusters),
            'evo2_regions': len(evo2_regions),
            'evo2_only_novel': len(novel_candidates),
            'gecco_only_novel': len(gecco_novel),
            'evo2_gecco_consensus_novel': len(both_novel),
            'top_consensus_candidates': both_novel[:10],
        }

        # 保存
        with open(scan_dir / "cross_tool_analysis.json", 'w') as f:
            json.dump({
                'evo2_regions': evo2_regions,
                'evo2_only_novel': novel_candidates,
                'evo2_gecco_consensus_novel': both_novel,
            }, f, indent=2)

    # 总结
    print("\n" + "=" * 75)
    print("  汇总")
    print("=" * 75)
    print(f"\n  {'基因组':<20} {'antiSMASH':>10} {'GECCO':>8} {'Evo2':>7} {'Evo2∩GECCO新':>14}")
    print(f"  {'-'*65}")
    for name, r in all_results.items():
        print(f"  {name:<20} {r['antismash_bgcs']:>10} {r['gecco_clusters']:>8} {r['evo2_regions']:>7} {r['evo2_gecco_consensus_novel']:>14}")


if __name__ == '__main__':
    main()
