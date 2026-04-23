"""Threshold sweep for top 4 decoders.

For each (decoder, threshold) pair, report:
  - 9g vs GT: F1 / P / R / n_pred
  - OER: n_regions / AS-recall / novel15 kept

Compares against v4.1 hysteresis baseline.
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
from decoding.v4_1 import PRIMARY_OP, decode_dir  # noqa

# Top configs to sweep — cover best F1 (c0/c3) and best OER recall (c8/c9)
TOP_CONFIG_IDS = [0, 3, 9, 15]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]

RUNS_DIR = Path("/data/muskliu/npmaster/experiments/evo2_first_principles/decoder_v1/runs")
OUT_DIR = Path("/data/muskliu/npmaster/experiments/evo2_first_principles/decoder_v1")
NG_PREDS = Path("/data/muskliu/npmaster/experiments/evo2_first_principles/9g_preds_lrA1_w8192")
NG_AS = Path("/data/syh/benchmark/nine_genome/results/antismash")
NG_GT_JSON = Path("/data/syh/benchmark/nine_genome/ground_truth.json")
OER_PREDS = Path("/data/muskliu/npmaster/experiments/OER004256/v41_primary_preds")
OER_AS_ROOT = Path("/data/muskliu/oer_bgc_scan/results/antismash")
OER_NOVEL15_CSV = Path("/data/muskliu/oer_bgc_scan/novel_triage/candidates_novel.csv")


def interval_overlap(a_s, a_e, b_s, b_e):
    return max(0, min(a_e, b_e) - max(a_s, b_s))


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


def decoder_probs_per_contig(model, windows_by_contig, device):
    """Return {contig: (starts, ends, raw_scores, probs)}."""
    out = {}
    model.eval()
    with torch.no_grad():
        for contig, (starts, ends, scores) in windows_by_contig.items():
            x = torch.from_numpy(scores).unsqueeze(0).to(device)
            logits = model(x).squeeze(0).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            out[contig] = (starts, ends, scores, probs)
    return out


def regions_at_threshold(probs_by_contig, threshold):
    regions = defaultdict(list)
    for contig, (starts, ends, scores, probs) in probs_by_contig.items():
        mask = probs >= threshold
        if not mask.any(): continue
        diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
        rs_arr = np.where(diff == 1)[0]; re_arr = np.where(diff == -1)[0]
        for rs_i, re_i in zip(rs_arr, re_arr):
            regions[contig].append((int(starts[rs_i]), int(ends[re_i - 1]),
                                    float(scores[rs_i:re_i].max())))
    return dict(regions)


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
    ng_preds = load_preds_fast(NG_PREDS)
    gt_all = json.load(open(NG_GT_JSON))["ground_truth"]
    ng_gt = {g: [(c, s, e) for (c, s, e) in gt_all.get(g, [])] for g in ng_preds}
    ng_as_raw = parse_antismash_dir(NG_AS, genome_ids=list(ng_preds))
    ng_as = {g: [(r["contig_id"], r["start_bp"], r["end_bp"]) for r in regs]
             for g, regs in ng_as_raw.items()}
    oer_preds = load_preds_fast(OER_PREDS)
    oer_as = parse_oer_antismash()
    total_oer_as = sum(len(v) for v in oer_as.values())
    novel15 = list(csv.DictReader(open(OER_NOVEL15_CSV)))

    # Baseline
    print("\n[BASELINE]")
    baseline = decode_dir(NG_PREDS, PRIMARY_OP)
    bl_preds = {g: [(c, s, e) for (c, s, e, _sc) in lst] for g, lst in baseline.items()}
    bl_gt = dedup_metrics(bl_preds, ng_gt)
    print(f"  v4.1-hyst 9g vs GT: F1={bl_gt['f1']:.4f}  P={bl_gt['precision']:.4f}  "
          f"R={bl_gt['recall']:.4f}  n_pred={bl_gt['n_pred']}")

    grid = make_grid()
    rows = []
    # Pre-compute decoder probs once per config (threshold-free)
    for cid in TOP_CONFIG_IDS:
        config = grid[cid]
        tag = f"c{cid:02d}_{config['arch']}_h{config['hidden']}_{config['label_src']}_{config['loss_type']}"
        ckpt = RUNS_DIR / tag / "decoder.pt"
        if not ckpt.exists():
            print(f"[{tag}] no ckpt"); continue

        depth = 3 if config["arch"] == "cnn3" else 5
        model = Decoder1DCNN(depth=depth, hidden=config["hidden"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        print(f"\n[{tag}] computing probs...")
        tt = time.time()
        ng_probs = {g: decoder_probs_per_contig(model, by_c, device) for g, by_c in ng_preds.items()}
        oer_probs = {s: decoder_probs_per_contig(model, by_c, device) for s, by_c in oer_preds.items()}
        print(f"  probs done in {time.time()-tt:.1f}s")

        for thr in THRESHOLDS:
            # 9g
            ng_regions = {}
            for g, prob_map in ng_probs.items():
                regs_by_c = regions_at_threshold(prob_map, thr)
                ng_regions[g] = [(c, s, e, sc) for c, lst in regs_by_c.items() for (s, e, sc) in lst]
            ng_vs_gt = dedup_metrics(ng_regions, ng_gt)
            ng_vs_as = dedup_metrics(ng_regions, ng_as)

            # OER
            oer_flat = []
            for shard, prob_map in oer_probs.items():
                regs_by_c = regions_at_threshold(prob_map, thr)
                for full_c, lst in regs_by_c.items():
                    c_short = full_c.split("|")[-1]
                    for (s, e, sc) in lst:
                        oer_flat.append((c_short, s, e, sc))
            oer_by_c = defaultdict(list)
            for (c, s, e, sc) in oer_flat:
                oer_by_c[c].append((s, e, sc))
            as_covered = 0
            for c, as_list in oer_as.items():
                ours = oer_by_c.get(c, [])
                for (a_s, a_e) in as_list:
                    if any(interval_overlap(a_s, a_e, os, oe) > 0 for (os, oe, _) in ours):
                        as_covered += 1
            novel_kept = 0
            for row in novel15:
                c = row["_contig_short"]
                s, e = int(row["start"]), int(row["end"])
                if any(interval_overlap(s, e, os, oe) > 0 for (os, oe, _) in oer_by_c.get(c, [])):
                    novel_kept += 1
            v41_only = sum(1 for (c, s, e, sc) in oer_flat
                            if not any(interval_overlap(s, e, a_s, a_e) > 0 for (a_s, a_e) in oer_as.get(c, [])))

            r = {
                "tag": tag, "thr": thr,
                "ng_gt_F1": round(ng_vs_gt["f1"], 4),
                "ng_gt_R": round(ng_vs_gt["recall"], 4),
                "ng_gt_P": round(ng_vs_gt["precision"], 4),
                "ng_n_pred": ng_vs_gt["n_pred"],
                "ng_as_F1": round(ng_vs_as["f1"], 4),
                "oer_regions": len(oer_flat),
                "oer_as_recall": round(as_covered / max(1, total_oer_as), 4),
                "oer_v41_only": v41_only,
                "novel15": f"{novel_kept}/15",
            }
            rows.append(r)
            print(f"  thr={thr}  F1={r['ng_gt_F1']}  P={r['ng_gt_P']}  R={r['ng_gt_R']}  "
                  f"n_pred={r['ng_n_pred']}  OER={r['oer_regions']}  AS_rec={r['oer_as_recall']}  "
                  f"novel={r['novel15']}")

    # Save CSV + table
    if rows:
        cols = list(rows[0].keys())
        with open(OUT_DIR / "threshold_sweep.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows: w.writerow(r)

        # Markdown table
        md = ["# Decoder threshold sweep — Pareto frontier\n",
              f"- Baseline (hysteresis) 9g vs GT: **F1={bl_gt['f1']:.4f}** P={bl_gt['precision']:.4f} R={bl_gt['recall']:.4f} n=222",
              "- Baseline OER: 254 regions, AS-recall ~12%",
              "",
              "| tag | thr | 9g F1 (vs GT) | P | R | n_pred | OER regs | OER AS-recall | novel15 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        # Sort by tag then thr
        rows_sorted = sorted(rows, key=lambda r: (r["tag"], -r["thr"]))
        for r in rows_sorted:
            flag = " ⭐" if r["ng_gt_F1"] >= bl_gt["f1"] else ""
            md.append(f"| {r['tag']} | {r['thr']} | **{r['ng_gt_F1']}**{flag} | "
                      f"{r['ng_gt_P']} | {r['ng_gt_R']} | {r['ng_n_pred']} | "
                      f"{r['oer_regions']} | {r['oer_as_recall']} | {r['novel15']} |")
        (OUT_DIR / "threshold_sweep.md").write_text("\n".join(md))
        print(f"\n[write] {OUT_DIR / 'threshold_sweep.csv'} + .md")


if __name__ == "__main__":
    main()
