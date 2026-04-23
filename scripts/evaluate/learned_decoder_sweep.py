"""Fast evaluate trained decoders — load data once, apply 16 decoders in sequence.

Load ONCE:
  - 9g per-window preds (9 genomes)
  - 9g GT + 9g antiSMASH regions
  - OER per-window preds (17 shards)
  - OER antiSMASH regions
  - 15 OER novel candidate coords

Per decoder (fast):
  - Forward pass on all contig windows (batched, GPU)
  - Region decoding (vectorized on CPU)
  - Metrics (vectorized)

Target: ~2 min total for 16 decoders.
"""
from __future__ import annotations
import csv
import json
import pickle
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
from decoder import Decoder1DCNN, make_grid  # noqa: E402
from baselines.parse_antismash import parse_antismash_dir  # noqa: E402
from decoding.v4_1 import PRIMARY_OP, decode_dir  # noqa: E402

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
    """Return {file_key: {contig: (starts, ends, scores)}} as numpy arrays."""
    out = {}
    for csv_path in sorted(Path(preds_dir).glob("*_predictions.csv")):
        key = csv_path.name[:-len("_predictions.csv")]
        df = pd.read_csv(csv_path, usecols=["contig", "start", "end", "bgc_score"])
        df = df.sort_values(["contig", "start"]).reset_index(drop=True)
        by_c = {}
        # groupby contig (fast)
        for c, sub in df.groupby("contig", sort=False):
            by_c[str(c)] = (sub["start"].to_numpy(np.int32),
                            sub["end"].to_numpy(np.int32),
                            sub["bgc_score"].to_numpy(np.float32))
        out[key] = by_c
    return out


def parse_oer_antismash():
    """Return {contig_short: [(s, e)]}"""
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


def decoder_forward_all(model, windows_by_contig, device, threshold=0.5):
    """Return {contig: [(s, e, max_raw_score)]}.
    Batched forward across all contigs, sized individually (no padding needed)."""
    regions = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for contig, (starts, ends, scores) in windows_by_contig.items():
            if len(scores) < 1: continue
            x = torch.from_numpy(scores).unsqueeze(0).to(device)  # [1, T]
            logits = model(x).squeeze(0).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            # Decode: consecutive windows with prob >= threshold
            mask = probs >= threshold
            if not mask.any(): continue
            # Find runs
            diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
            run_starts = np.where(diff == 1)[0]
            run_ends = np.where(diff == -1)[0]
            for rs_i, re_i in zip(run_starts, run_ends):
                region_s = int(starts[rs_i])
                region_e = int(ends[re_i - 1])
                max_sc = float(scores[rs_i:re_i].max())
                regions[contig].append((region_s, region_e, max_sc))
    return dict(regions)


def dedup_metrics(preds_by_genome, gt_by_genome):
    """preds: {genome: [(contig, s, e, ...)]}, gt: {genome: [(contig, s, e)]}"""
    tp = 0; total_gt = 0; total_p = 0
    for g_key, gt_regs in gt_by_genome.items():
        total_gt += len(gt_regs)
        p_regs = preds_by_genome.get(g_key, [])
        total_p += len(p_regs)
        gt_used = [False] * len(gt_regs)
        gt_by_c = defaultdict(list)
        for idx, r in enumerate(gt_regs):
            c, s, e = r[0], r[1], r[2]
            gt_by_c[c].append((idx, s, e))
        for r in p_regs:
            pc, ps, pe = r[0], r[1], r[2]
            best_idx = -1; best_ov = 0
            for (gi, gs, ge) in gt_by_c.get(pc, []):
                if gt_used[gi]: continue
                o = interval_overlap(gs, ge, ps, pe)
                if o / max(1, ge - gs) >= 0.5 or o / max(1, pe - ps) >= 0.5:
                    if o > best_ov:
                        best_ov = o; best_idx = gi
            if best_idx >= 0:
                tp += 1; gt_used[best_idx] = True
    prec = tp / max(1, total_p); rec = tp / max(1, total_gt)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"n_gt": total_gt, "n_pred": total_p, "tp": tp,
            "precision": prec, "recall": rec, "f1": f1}


def main():
    device = torch.device("cuda:0")
    t0 = time.time()

    print("[load] 9g per-window preds")
    ng_preds = load_preds_fast(NG_PREDS)  # {genome: {contig: (starts, ends, scores)}}
    print(f"  {len(ng_preds)} genomes, "
          f"{sum(len(c) for g in ng_preds.values() for c in [g]) } contigs")

    print("[load] 9g GT")
    gt_all = json.load(open(NG_GT_JSON))["ground_truth"]
    ng_gt = {g: [(c, s, e) for (c, s, e) in gt_all.get(g, [])] for g in ng_preds}

    print("[load] 9g antiSMASH")
    ng_as_raw = parse_antismash_dir(NG_AS, genome_ids=list(ng_preds))
    ng_as = {g: [(r["contig_id"], r["start_bp"], r["end_bp"]) for r in regs]
             for g, regs in ng_as_raw.items()}

    print("[load] OER per-window preds (17 shards)")
    t1 = time.time()
    oer_preds = load_preds_fast(OER_PREDS)
    print(f"  {len(oer_preds)} shards, {time.time()-t1:.1f}s")

    print("[load] OER antiSMASH regions")
    oer_as = parse_oer_antismash()
    total_oer_as = sum(len(v) for v in oer_as.values())
    print(f"  {total_oer_as} regions on {len(oer_as)} contigs")

    print("[load] 15 novel candidates")
    novel15 = list(csv.DictReader(open(OER_NOVEL15_CSV)))

    # Baseline
    print("\n[BASELINE] v4.1 hysteresis on 9g")
    baseline = decode_dir(NG_PREDS, PRIMARY_OP)
    bl_preds = {g: [(c, s, e) for (c, s, e, _sc) in lst] for g, lst in baseline.items()}
    bl_gt = dedup_metrics(bl_preds, ng_gt)
    bl_as = dedup_metrics(bl_preds, ng_as)
    print(f"  vs GT: F1={bl_gt['f1']:.4f} P={bl_gt['precision']:.4f} R={bl_gt['recall']:.4f} n_pred={bl_gt['n_pred']}")
    print(f"  vs AS: F1={bl_as['f1']:.4f} R={bl_as['recall']:.4f}")

    # OER baseline: existing v4.1 regions (254)
    print(f"  OER baseline regions: 254 (primary arm only), antiSMASH: {total_oer_as}")

    # Loop over 16 decoders
    grid = make_grid()
    rows = []
    for cid, config in enumerate(grid):
        tag = f"c{cid:02d}_{config['arch']}_h{config['hidden']}_{config['label_src']}_{config['loss_type']}"
        run_dir = RUNS_DIR / tag
        ckpt = run_dir / "decoder.pt"
        if not ckpt.exists():
            print(f"[{tag}] SKIP (no ckpt)"); continue

        tt = time.time()
        depth = 3 if config["arch"] == "cnn3" else 5
        model = Decoder1DCNN(depth=depth, hidden=config["hidden"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        # 9g
        ng_regions = {}
        for g, by_c in ng_preds.items():
            regs_by_c = decoder_forward_all(model, by_c, device)
            ng_regions[g] = [(c, s, e, sc) for c, lst in regs_by_c.items() for (s, e, sc) in lst]
        ng_vs_gt = dedup_metrics(ng_regions, ng_gt)
        ng_vs_as = dedup_metrics(ng_regions, ng_as)

        # OER — strip sample prefix for antiSMASH matching
        oer_regions_flat = []  # (contig_short, s, e, score)
        for shard, by_c in oer_preds.items():
            regs_by_c = decoder_forward_all(model, by_c, device)
            for full_c, lst in regs_by_c.items():
                c_short = full_c.split("|")[-1]
                for (s, e, sc) in lst:
                    oer_regions_flat.append((c_short, s, e, sc))

        oer_total = len(oer_regions_flat)
        # Organize by contig
        oer_by_contig = defaultdict(list)
        for (c, s, e, sc) in oer_regions_flat:
            oer_by_contig[c].append((s, e, sc))

        # Count: how many of OER antiSMASH regions are now covered by our decoder?
        as_covered = 0
        for c, as_list in oer_as.items():
            our_list = oer_by_contig.get(c, [])
            for (a_s, a_e) in as_list:
                if any(interval_overlap(a_s, a_e, os, oe) > 0 for (os, oe, _) in our_list):
                    as_covered += 1
        # v41-only on OER
        v41_only = 0
        for (c, s, e, sc) in oer_regions_flat:
            if not any(interval_overlap(s, e, a_s, a_e) > 0 for (a_s, a_e) in oer_as.get(c, [])):
                v41_only += 1

        # Check 15 novel preservation
        novel_kept = 0
        for row in novel15:
            c = row["_contig_short"]
            s, e = int(row["start"]), int(row["end"])
            if any(interval_overlap(s, e, os, oe) > 0 for (os, oe, _) in oer_by_contig.get(c, [])):
                novel_kept += 1

        r = {
            "tag": tag,
            "arch": config["arch"], "hidden": config["hidden"],
            "label_src": config["label_src"], "loss": config["loss_type"],
            "ng_gt_f1": round(ng_vs_gt["f1"], 4),
            "ng_gt_recall": round(ng_vs_gt["recall"], 4),
            "ng_gt_precision": round(ng_vs_gt["precision"], 4),
            "ng_gt_npred": ng_vs_gt["n_pred"],
            "ng_as_f1": round(ng_vs_as["f1"], 4),
            "ng_as_recall": round(ng_vs_as["recall"], 4),
            "oer_regions": oer_total,
            "oer_as_covered": as_covered,
            "oer_as_recall": round(as_covered / max(1, total_oer_as), 4),
            "oer_v41_only": v41_only,
            "novel15_kept": f"{novel_kept}/15",
        }
        rows.append(r)
        print(f"[{tag}] {time.time()-tt:.1f}s  "
              f"ng_gt_F1={r['ng_gt_f1']}(P={r['ng_gt_precision']},R={r['ng_gt_recall']},n={r['ng_gt_npred']})  "
              f"oer_regions={r['oer_regions']}  as_recall={r['oer_as_recall']}  novel={r['novel15_kept']}")

    # Save
    if rows:
        cols = list(rows[0].keys())
        with open(OUT_DIR / "sweep_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows: w.writerow(r)
        rows_sorted = sorted(rows, key=lambda r: -r["ng_gt_f1"])
        md = ["# Learned decoder sweep — OER vs 9g recall recovery\n",
              f"- Baseline v4.1 hysteresis on 9g: F1={bl_gt['f1']:.4f} "
              f"(P={bl_gt['precision']:.4f}, R={bl_gt['recall']:.4f}, n_pred={bl_gt['n_pred']})",
              f"- Baseline v4.1 regions on OER (primary arm): 254  (antiSMASH: {total_oer_as})",
              "",
              "## Ranked by 9g-vs-GT F1",
              "",
              "| tag | ng_gt_F1 | R | P | n_pred | ng_as_F1 | OER regs | OER AS-recall | novel15 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows_sorted:
            md.append(f"| {r['tag']} | **{r['ng_gt_f1']}** | {r['ng_gt_recall']} | "
                      f"{r['ng_gt_precision']} | {r['ng_gt_npred']} | "
                      f"{r['ng_as_f1']} | {r['oer_regions']} | "
                      f"{r['oer_as_recall']} | {r['novel15_kept']} |")
        with open(OUT_DIR / "sweep_summary.md", "w") as fh:
            fh.write("\n".join(md))
        print(f"\n[write] {OUT_DIR/'sweep_summary.csv'} + sweep_summary.md")

    print(f"\nTOTAL {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
