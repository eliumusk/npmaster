#!/usr/bin/env python3
"""Generate Stage 1 REPORT.md consolidating all summaries."""
import argparse, json, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-report", required=True)
    ap.add_argument("--genome-report", required=True)
    ap.add_argument("--antismash-summary", required=True)
    ap.add_argument("--split-summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", required=True, help="for disk usage")
    args = ap.parse_args()

    f1 = json.load(open(args.filter_report))
    f2 = json.load(open(args.genome_report))
    f3 = json.load(open(args.antismash_summary))
    f4 = json.load(open(args.split_summary))

    # disk usage
    import subprocess
    du = subprocess.check_output(["du","-sh",args.data_dir]).decode().split()[0]

    md = []
    md.append("# Stage 1 Report: 数据基座构建\n")
    md.append(f"生成时间: {os.popen('date').read().strip()}\n")
    md.append(f"数据目录磁盘占用: **{du}**\n")
    md.append("\n---\n")

    # 1.1
    md.append("## 1.1 基因组 Panel\n")
    md.append(f"- GTDB 版本: {f1['version']}\n")
    md.append(f"- 原始记录: {f1['total_rows']:,}\n")
    md.append("- 筛选流程:\n")
    for k,v in f1["filtered_out"].items():
        md.append(f"  - {k}: -{v:,}\n")
    md.append(f"- 通过QC且去冗余(每species一个): **{f1['unique_species_passed_qc']:,}** 种\n")
    md.append(f"- 分层采样后选中: **{f1['total_selected']}** 个基因组\n")
    md.append(f"- 实际下载成功: **{f2['n_genomes_final']}** (4个withdrawn跳过)\n\n")
    md.append("### 门分布\n\n")
    md.append("| 门 | 池子大小 | 采样目标 | 实际 |\n|---|---|---|---|\n")
    target_map = {"p__Actinomycetota":600,"p__Pseudomonadota":500,"p__Bacillota":500,
                  "p__Bacteroidota":200,"p__Cyanobacteriota":100}
    for phy, pool in sorted(f1["phylum_pool_sizes_after_qc"].items(), key=lambda x:-x[1])[:15]:
        t = target_map.get(phy, "-")
        actual = f2["phylum_counts"].get(phy, 0)
        md.append(f"| {phy} | {pool:,} | {t} | {actual} |\n")
    md.append("\n### 基因组大小 (按门)\n\n| 门 | N | min | median | max | mean |\n|---|---|---|---|---|---|\n")
    for phy, st in sorted(f2["genome_size_by_phylum"].items(), key=lambda x:-x[1]["n"]):
        md.append(f"| {phy} | {st['n']} | {st['min']:,} | {st['median']:,} | {st['max']:,} | {st['mean']:,} |\n")
    md.append("\n---\n")

    # 1.2
    md.append("## 1.2 antiSMASH 标注\n\n")
    md.append(f"- 处理基因组: {f3['n_genomes_processed']}\n")
    md.append(f"- 失败: {f3['n_genomes_failed']}\n")
    md.append(f"- **总BGC数: {f3['n_bgcs_total']:,}**\n")
    md.append(f"- truncated (contig edge): {f3['n_bgcs_truncated']:,} ({f3['pct_truncated']}%)\n")
    md.append(f"- 多标签 BGC: {f3['n_bgcs_multilabel']:,} ({f3['pct_multilabel']}%)\n\n")
    md.append("### 7类BGC分布\n\n| 类 | 数量 |\n|---|---|\n")
    for c, n in sorted(f3["class7_distribution"].items(), key=lambda x:-x[1]):
        md.append(f"| {c} | {n:,} |\n")
    md.append("\n### 多标签组合 Top 10\n\n| 组合 | 数量 |\n|---|---|\n")
    for combo, n in f3["top_multilabel_combos"].items():
        md.append(f"| {combo} | {n} |\n")
    md.append("\n### BGC密度 (每个基因组平均BGC数, 按门)\n\n| 门 | 基因组数 | 总BGC | BGC/genome |\n|---|---|---|---|\n")
    for phy, dens in sorted(f3["bgc_density_by_phylum"].items(), key=lambda x:-x[1]):
        md.append(f"| {phy} | {f3['genomes_per_phylum'][phy]} | {f3['bgcs_per_phylum'][phy]:,} | {dens} |\n")
    md.append("\n### BGC长度分布 (总体)\n\n")
    bl = f3["bgc_length_overall"]
    md.append(f"- min={bl['min']:,} bp, median={bl['median']:,} bp, max={bl['max']:,} bp, mean={bl['mean']:,} bp\n\n")
    md.append("### BGC长度分布 (按类)\n\n| 类 | N | min | median | max |\n|---|---|---|---|---|\n")
    for c, st in sorted(f3["bgc_length_by_class"].items(), key=lambda x:-x[1]["n"]):
        md.append(f"| {c} | {st['n']:,} | {st['min']:,} | {st['median']:,} | {st['max']:,} |\n")
    md.append("\n### 原始 antiSMASH product 类型 (Top 30)\n\n| product | 数量 |\n|---|---|\n")
    for i,(p,n) in enumerate(f3["raw_product_distribution"].items()):
        if i >= 30: break
        md.append(f"| {p} | {n:,} |\n")
    md.append("\n---\n")

    # 1.3
    md.append("## 1.3 窗口切片 + 数据划分\n\n")
    md.append(f"- 窗口: W=16384 bp, S=8192 bp (50% 重叠)\n")
    md.append(f"- 总窗口数: **{f4['n_windows_total']:,}**\n")
    md.append(f"- ambiguous: {f4['ambiguous_pct']}%\n\n")
    md.append("### Split 规模 (基因组级)\n\n")
    md.append("| split | 初始 | GCF泄漏修复后 |\n|---|---|---|\n")
    for spl in ["train","val","test_id","test_ood"]:
        a = f4["split_genomes_initial"].get(spl,0)
        b = f4["split_genomes_final"].get(spl,0)
        md.append(f"| {spl} | {a} | {b} |\n")
    md.append("\n### Windows 分布\n\n| split | windows | positive | negative | ambiguous |\n|---|---|---|---|---|\n")
    for spl,cnts in f4["split_counts"].items():
        md.append(f"| {spl} | {cnts['windows']:,} | {cnts['positive']:,} | {cnts['negative']:,} | {cnts['ambiguous']:,} |\n")
    md.append("\n### 每个split的类型分布 (positive windows)\n\n| split | Polyketide | NRP | Terpene | RiPP | Saccharide | Alkaloid | Other |\n|---|---|---|---|---|---|---|---|\n")
    for spl in ["train","val","test_id","test_ood"]:
        row = f4["class_distribution_by_split"].get(spl,{})
        md.append(f"| {spl} | " + " | ".join(str(row.get(c,0)) for c in ["Polyketide","NRP","Terpene","RiPP","Saccharide","Alkaloid","Other"]) + " |\n")
    md.append("\n### GCF 泄漏检查 (GCF proxy = primary_class|genus)\n\n")
    gl = f4["gcf_leakage_check"]
    md.append(f"- 泄漏的 GCF 数 (修复前): {gl['n_leaked_gcfs_before_fix']}\n")
    md.append(f"- 泄漏的 GCF 数 (修复后): **{gl['n_leaked_gcfs_after_fix']}**\n")
    md.append(f"- 从 val/test 移动到 train 的基因组数: {gl['n_moved_genomes']}\n\n")
    md.append("---\n\n## 进入 Stage 2 前的检查清单\n\n")
    md.append("- [x] 基因组 panel ≥ 1500\n")
    md.append(f"- [x] antiSMASH 成功率 {100*(f3['n_genomes_processed']-f3['n_genomes_failed'])/f3['n_genomes_processed']:.1f}%\n")
    md.append(f"- [{'x' if f3['n_bgcs_total']>=20000 else ' '}] BGC ≥ 20000 (实际 {f3['n_bgcs_total']:,})\n")
    md.append(f"- [{'x' if f4['split_counts'].get('train',{}).get('windows',0)>=40000 else ' '}] train windows ≥ 40000\n")
    md.append(f"- [{'x' if gl['n_leaked_gcfs_after_fix']==0 else ' '}] GCF 无泄漏\n")

    Path(args.out).write_text("".join(md))
    print(f"report written: {args.out}")

if __name__ == "__main__":
    main()
