"""Full multi-scale pipeline: learned-decoder primary + learned-decoder short + union-dedup.

Mirrors v4.1 multi-scale but replaces hysteresis with c00 CNN decoder on both arms.
Short arm filtered to length < SHORT_LENGTH_CAP (5-10 kb rescue).
Union-dedup at IOU_DEDUP = 0.3.

Sweeps thresholds for the **primary** arm; short arm uses a moderate threshold (0.8)
since it's a rescue arm (want recall).

Outputs:
  decoder_v1/full_pipeline_report.md
  decoder_v1/full_pipeline.csv
"""
from __future__ import annotations
import csv
import json
import re
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))
from decoder import Decoder1DCNN, make_grid  # noqa
from baselines.parse_antismash import parse_antismash_dir  # noqa
from decoding.v4_1 import (PRIMARY_OP, SHORT_OP, SHORT_LENGTH_CAP, IOU_DEDUP,
                           decode_dir, union_dedup)  # noqa

OUT_DIR = Path("/data/muskliu/npmaster/experiments/evo2_first_principles/decoder_v1")
RUNS_DIR = OUT_DIR / "runs"

NG_PRIMARY = Path("/data/muskliu/npmaster/experiments/evo2_first_principles/9g_preds_lrA1_w8192")
NG_SHORT = Path("/data/muskliu/npmaster/experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096")
NG_AS = Path("/data/syh/benchmark/nine_genome/results/antismash")
NG_GT_JSON = Path("/data/syh/benchmark/nine_genome/ground_truth.json")
OER_PRIMARY = Path("/data/muskliu/npmaster/experiments/OER004256/v41_primary_preds")
OER_SHORT = Path("/data/muskliu/npmaster/experiments/OER004256/v41_short_preds")
OER_AS_ROOT = Path("/data/muskliu/oer_bgc_scan/results/antismash")
OER_NOVEL15_CSV = Path("/data/muskliu/oer_bgc_scan/novel_triage/candidates_novel.csv")

# Chosen based on prior sweep: c00 is Pareto-optimal for novel preservation.
CHOSEN_CONFIG_ID = 0
PRIMARY_THRESHOLDS = [0.9, 0.95]
SHORT_THRESHOLDS = [0.85, 0.9, 0.95, 0.98]  # short arm — need to tame it


def interval_overlap(a_s, a_e, b_s, b_e):
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def iou(a, b):
    # (s, e)
    o = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return o / max(1, max(a[1], b[1]) - min(a[0], b[0]))


def load_preds_fast(preds_dir):
    out = {}
    for csv_path in sorted(Path(preds_dir).glob("*_predictions.csv")):
        key = csv_path.name[:-len("_predictions.csv")]
        df = pd.read_csv(csv_path, usecols=["contig", "start", "end", "bgc_score"])
        df = df.sort_values(["contig", "start"]).reset_index(drop=True)
        by_c = {}
        for c, sub in df.groupby("contig", sort=False):
            by_c[str(c)] = (sub["start"].to_numpy(np.int32),
                            sub["end"].to_numpy(np.int32),
                            sub["bgc_score"].to_numpy(np.float32))
        out[key] = by_c
    return out


def parse_oer_antismash():
    out = defaultdict(list)
    for sample_dir in sorted(OER_AS_ROOT.glob("OER*.ge10000")):
        json_path = sample_dir / f"{sample_dir.name}.json"
        if not json_path.exists(): continue
        data = json.load(open(json_path))
        for rec in data.get("records", []):
            contig = rec.get("id")
            if not contig: continue
            for feat in rec.get("features", []):
                if feat.get("type") != "region": continue
                m = re.search(r"\[(\d+):[><]?(\d+)\]", feat.get("location", ""))
                if not m: continue
                out[contig].append((int(m.group(1)), int(m.group(2))))
    return dict(out)


def decoder_probs(model, windows_by_contig, device):
    out = {}
    model.eval()
    with torch.no_grad():
        for contig, (starts, ends, scores) in windows_by_contig.items():
            x = torch.from_numpy(scores).unsqueeze(0).to(device)
            logits = model(x).squeeze(0).cpu().numpy()
            out[contig] = (starts, ends, scores, 1.0 / (1.0 + np.exp(-logits)))
    return out


def regions_from_probs(probs_by_contig, threshold):
    """Return {contig: [(s, e, max_raw_score)]}"""
    out = defaultdict(list)
    for c, (starts, ends, raw, probs) in probs_by_contig.items():
        mask = probs >= threshold
        if not mask.any(): continue
        diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
        rs = np.where(diff == 1)[0]; re_ = np.where(diff == -1)[0]
        for i, j in zip(rs, re_):
            out[c].append((int(starts[i]), int(ends[j-1]), float(raw[i:j].max())))
    return dict(out)


def merge_multiscale(primary_regions, short_regions, short_length_cap, iou_thresh):
    """Per-genome (primary_by_c, short_by_c) -> union-dedup list of (c, s, e, sc, source)."""
    out = {}
    for key in set(primary_regions) | set(short_regions):
        pc_map = primary_regions.get(key, {})
        sc_map = short_regions.get(key, {})
        merged = []
        for c, lst in pc_map.items():
            for (s, e, sc) in lst:
                merged.append((c, s, e, sc, "primary"))
        for c, lst in sc_map.items():
            for (s, e, sc) in lst:
                if (e - s) < short_length_cap:
                    merged.append((c, s, e, sc, "short_rescue"))
        # Dedup by IoU per contig, keep higher score
        per_contig = defaultdict(list)
        for r in merged:
            per_contig[r[0]].append(r)
        kept = []
        for c, lst in per_contig.items():
            lst_sorted = sorted(lst, key=lambda r: (-r[3], 0 if r[4] == "primary" else 1))
            picked = []
            for r in lst_sorted:
                if any(iou((r[1], r[2]), (k[1], k[2])) >= iou_thresh for k in picked):
                    continue
                picked.append(r)
            kept.extend(picked)
        out[key] = kept
    return out


def dedup_metrics(preds_by_g, gt_by_g):
    tp = 0; total_gt = 0; total_p = 0
    for g, gt_regs in gt_by_g.items():
        total_gt += len(gt_regs)
        p_regs = preds_by_g.get(g, [])
        total_p += len(p_regs)
        used = [False] * len(gt_regs)
        gt_by_c = defaultdict(list)
        for idx, r in enumerate(gt_regs):
            gt_by_c[r[0]].append((idx, r[1], r[2]))
        for r in p_regs:
            pc, ps, pe = r[0], r[1], r[2]
            best_idx = -1; best_ov = 0
            for (gi, gs, ge) in gt_by_c.get(pc, []):
                if used[gi]: continue
                o = interval_overlap(gs, ge, ps, pe)
                if o / max(1, ge - gs) >= 0.5 or o / max(1, pe - ps) >= 0.5:
                    if o > best_ov:
                        best_ov = o; best_idx = gi
            if best_idx >= 0:
                tp += 1; used[best_idx] = True
    prec = tp / max(1, total_p); rec = tp / max(1, total_gt)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"n_gt": total_gt, "n_pred": total_p, "tp": tp,
            "precision": prec, "recall": rec, "f1": f1}


def main():
    device = torch.device("cuda:0")

    print("[load] data")
    t0 = time.time()
    ng_primary = load_preds_fast(NG_PRIMARY)
    ng_short = load_preds_fast(NG_SHORT)
    oer_primary = load_preds_fast(OER_PRIMARY)
    oer_short = load_preds_fast(OER_SHORT)
    gt_all = json.load(open(NG_GT_JSON))["ground_truth"]
    ng_gt = {g: [(c, s, e) for (c, s, e) in gt_all.get(g, [])] for g in ng_primary}
    ng_as_raw = parse_antismash_dir(NG_AS, genome_ids=list(ng_primary))
    ng_as = {g: [(r["contig_id"], r["start_bp"], r["end_bp"]) for r in regs]
             for g, regs in ng_as_raw.items()}
    oer_as = parse_oer_antismash()
    total_oer_as = sum(len(v) for v in oer_as.values())
    novel15 = list(csv.DictReader(open(OER_NOVEL15_CSV)))
    print(f"  done in {time.time()-t0:.1f}s. 9g: {len(ng_primary)} genomes. OER: {len(oer_primary)} shards, AS={total_oer_as}")

    # Baseline v4.1 multi-scale
    print("\n[BASELINE] v4.1 multi-scale hysteresis")
    bl_primary = decode_dir(NG_PRIMARY, PRIMARY_OP)
    bl_short = decode_dir(NG_SHORT, SHORT_OP)
    bl_short_filt = {g: [r for r in rs if (r[2] - r[1]) < SHORT_LENGTH_CAP]
                      for g, rs in bl_short.items()}
    bl_merged = {g: union_dedup(bl_primary.get(g, []), bl_short_filt.get(g, []), IOU_DEDUP)
                 for g in set(bl_primary) | set(bl_short_filt)}
    # To format: [(c, s, e)]
    bl_preds = {g: [(c, s, e) for (c, s, e, _sc) in lst] for g, lst in bl_merged.items()}
    bl_gt = dedup_metrics(bl_preds, ng_gt)
    bl_as = dedup_metrics(bl_preds, ng_as)
    print(f"  9g vs GT: F1={bl_gt['f1']:.4f} P={bl_gt['precision']:.4f} R={bl_gt['recall']:.4f} n_pred={bl_gt['n_pred']}")
    print(f"  9g vs AS: F1={bl_as['f1']:.4f} R={bl_as['recall']:.4f}")

    # OER baseline with v4.1 multi-scale hysteresis
    bl_oer_primary = decode_dir(OER_PRIMARY, PRIMARY_OP)
    bl_oer_short = decode_dir(OER_SHORT, SHORT_OP)
    bl_oer_short_filt = {g: [r for r in rs if (r[2] - r[1]) < SHORT_LENGTH_CAP]
                         for g, rs in bl_oer_short.items()}
    bl_oer_merged = {g: union_dedup(bl_oer_primary.get(g, []), bl_oer_short_filt.get(g, []), IOU_DEDUP)
                     for g in set(bl_oer_primary) | set(bl_oer_short_filt)}
    bl_oer_flat = []
    for shard, lst in bl_oer_merged.items():
        for (c, s, e, sc) in lst:
            bl_oer_flat.append((c.split("|")[-1], s, e, sc))
    bl_oer_count = len(bl_oer_flat)
    # AS-recall
    bl_oer_by_c = defaultdict(list)
    for (c, s, e, sc) in bl_oer_flat:
        bl_oer_by_c[c].append((s, e, sc))
    bl_oer_as_covered = 0
    for c, as_list in oer_as.items():
        ours = bl_oer_by_c.get(c, [])
        for (a_s, a_e) in as_list:
            if any(interval_overlap(a_s, a_e, os, oe) > 0 for (os, oe, _) in ours):
                bl_oer_as_covered += 1
    bl_as_rec = bl_oer_as_covered / max(1, total_oer_as)
    print(f"  OER baseline: {bl_oer_count} regions, AS-recall={bl_as_rec:.4f}")

    # Load chosen decoder
    grid = make_grid()
    config = grid[CHOSEN_CONFIG_ID]
    tag = f"c{CHOSEN_CONFIG_ID:02d}_{config['arch']}_h{config['hidden']}_{config['label_src']}_{config['loss_type']}"
    depth = 3 if config["arch"] == "cnn3" else 5
    model = Decoder1DCNN(depth=depth, hidden=config["hidden"]).to(device)
    model.load_state_dict(torch.load(RUNS_DIR / tag / "decoder.pt", map_location=device))
    print(f"\n[decoder] {tag} (depth={depth}, hidden={config['hidden']})")

    # Compute decoder probs once for primary + short on both 9g and OER
    print("\n[compute probs]")
    t1 = time.time()
    ng_primary_probs = {g: decoder_probs(model, by_c, device) for g, by_c in ng_primary.items()}
    ng_short_probs = {g: decoder_probs(model, by_c, device) for g, by_c in ng_short.items()}
    oer_primary_probs = {s: decoder_probs(model, by_c, device) for s, by_c in oer_primary.items()}
    oer_short_probs = {s: decoder_probs(model, by_c, device) for s, by_c in oer_short.items()}
    print(f"  done in {time.time()-t1:.1f}s")

    # Sweep primary × short threshold grid
    rows = []
    for prim_thr in PRIMARY_THRESHOLDS:
      for short_thr in SHORT_THRESHOLDS:
        print(f"\n[sweep] prim_thr={prim_thr} short_thr={short_thr}")

        # 9g regions at these thresholds
        ng_primary_regs = {g: regions_from_probs(pp, prim_thr) for g, pp in ng_primary_probs.items()}
        ng_short_regs = {g: regions_from_probs(pp, short_thr) for g, pp in ng_short_probs.items()}
        ng_merged = merge_multiscale(ng_primary_regs, ng_short_regs, SHORT_LENGTH_CAP, IOU_DEDUP)
        ng_preds = {g: [(c, s, e, sc) for (c, s, e, sc, _src) in lst] for g, lst in ng_merged.items()}
        ng_gt_m = dedup_metrics(ng_preds, ng_gt)
        ng_as_m = dedup_metrics(ng_preds, ng_as)

        # OER regions
        oer_primary_regs = {s: regions_from_probs(pp, prim_thr) for s, pp in oer_primary_probs.items()}
        oer_short_regs = {s: regions_from_probs(pp, short_thr) for s, pp in oer_short_probs.items()}
        oer_merged = merge_multiscale(oer_primary_regs, oer_short_regs, SHORT_LENGTH_CAP, IOU_DEDUP)
        oer_flat = []
        prim_only = 0; short_only = 0
        for shard, lst in oer_merged.items():
            for (full_c, s, e, sc, src) in lst:
                c = full_c.split("|")[-1]
                oer_flat.append((c, s, e, sc, src))
                if src == "primary": prim_only += 1
                else: short_only += 1

        oer_by_c = defaultdict(list)
        for (c, s, e, sc, src) in oer_flat:
            oer_by_c[c].append((s, e, sc))
        # AS-recall
        as_covered = 0
        for c, as_list in oer_as.items():
            ours = oer_by_c.get(c, [])
            for (a_s, a_e) in as_list:
                if any(interval_overlap(a_s, a_e, os, oe) > 0 for (os, oe, _) in ours):
                    as_covered += 1
        as_recall = as_covered / max(1, total_oer_as)
        # Novel15 kept
        novel_kept = 0
        for row in novel15:
            c = row["_contig_short"]
            s, e = int(row["start"]), int(row["end"])
            if any(interval_overlap(s, e, os, oe) > 0 for (os, oe, _) in oer_by_c.get(c, [])):
                novel_kept += 1
        # v41-only
        v41_only = sum(1 for (c, s, e, sc, _src) in oer_flat
                        if not any(interval_overlap(s, e, a_s, a_e) > 0 for (a_s, a_e) in oer_as.get(c, [])))

        # OER length distribution
        lens = np.array([e - s for (_, s, e, _sc, _src) in oer_flat])
        len_med = int(np.median(lens)) if len(lens) else 0
        len_p10, len_p90 = (int(np.percentile(lens, 10)), int(np.percentile(lens, 90))) if len(lens) else (0, 0)

        r = {
            "prim_thr": prim_thr, "short_thr": short_thr,
            "ng_gt_F1": round(ng_gt_m["f1"], 4),
            "ng_gt_P": round(ng_gt_m["precision"], 4),
            "ng_gt_R": round(ng_gt_m["recall"], 4),
            "ng_gt_npred": ng_gt_m["n_pred"],
            "ng_as_F1": round(ng_as_m["f1"], 4),
            "ng_as_R": round(ng_as_m["recall"], 4),
            "oer_regions": len(oer_flat),
            "oer_primary_count": prim_only,
            "oer_short_count": short_only,
            "oer_as_covered": as_covered,
            "oer_as_recall": round(as_recall, 4),
            "oer_v41_only": v41_only,
            "oer_region_len_med": len_med,
            "oer_region_len_p10p90": f"{len_p10}/{len_p90}",
            "novel15_kept": f"{novel_kept}/15",
        }
        rows.append(r)
        print(f"  9g F1={r['ng_gt_F1']} (P={r['ng_gt_P']} R={r['ng_gt_R']} n={r['ng_gt_npred']})")
        print(f"  OER regions = {r['oer_regions']}  (primary={prim_only} + short={short_only})")
        print(f"  OER AS-recall = {r['oer_as_recall']}  v41-only = {r['oer_v41_only']}")
        print(f"  OER region length median = {len_med}  p10/p90 = {len_p10}/{len_p90}")
        print(f"  novel15 kept = {r['novel15_kept']}")

    # Write
    with open(OUT_DIR / "full_pipeline.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    md = [f"# Full multi-scale pipeline — {tag}\n",
          "## Baseline: v4.1 multi-scale (hysteresis primary + short)",
          f"- 9g vs GT: F1={bl_gt['f1']:.4f} P={bl_gt['precision']:.4f} R={bl_gt['recall']:.4f} n_pred={bl_gt['n_pred']}",
          f"- 9g vs AS: F1={bl_as['f1']:.4f} R={bl_as['recall']:.4f}",
          f"- OER: {bl_oer_count} regions, AS-recall={bl_as_rec:.4f}",
          f"- antiSMASH on OER: {total_oer_as} regions",
          "",
          f"## Learned-decoder multi-scale ({tag})",
          "",
          "| prim | short | 9g F1 | P | R | n_pred | OER regions | primary/short | AS-recall | v41-only | len med | novel15 |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['prim_thr']} | {r['short_thr']} | **{r['ng_gt_F1']}** | {r['ng_gt_P']} | {r['ng_gt_R']} | "
                  f"{r['ng_gt_npred']} | **{r['oer_regions']}** | {r['oer_primary_count']}/{r['oer_short_count']} | "
                  f"**{r['oer_as_recall']}** | {r['oer_v41_only']} | {r['oer_region_len_med']} | {r['novel15_kept']} |")
    (OUT_DIR / "full_pipeline_report.md").write_text("\n".join(md))
    print(f"\n[write] {OUT_DIR / 'full_pipeline.csv'} + full_pipeline_report.md")


if __name__ == "__main__":
    main()
