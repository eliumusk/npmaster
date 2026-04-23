"""CLI: python -m evo2_lr <subcommand>.

Subcommands:
  extract   FASTA → Evo2 mean-pool embeddings + windows CSV
  predict   embeddings + windows CSV → per-window LR scores
  decode    per-window scores → BGC regions
  run       end-to-end (FASTA → BGC regions)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evo2_lr",
        description="Linear BGC probe on frozen Evo2 embeddings (v4 frozen).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # extract
    ex = sub.add_parser("extract", help="FASTA → Evo2 mean-pool embeddings")
    ex.add_argument("--fna", type=Path, required=True)
    ex.add_argument("--out-dir", type=Path, required=True)
    ex.add_argument("--window", type=int, default=8192)
    ex.add_argument("--stride", type=int, default=2048)
    ex.add_argument("--layer", default="blocks.20.mlp.l3")
    ex.add_argument("--min-contig-len", type=int, default=0)
    ex.add_argument("--batch-size", type=int, default=1)

    # predict
    pr = sub.add_parser("predict", help="embeddings + windows → per-window LR scores")
    pr.add_argument("--emb-npy", type=Path, required=True)
    pr.add_argument("--windows-csv", type=Path, required=True)
    pr.add_argument("--lr-weights", type=Path, default=None)
    pr.add_argument("--out-csv", type=Path, required=True)

    # decode
    de = sub.add_parser("decode", help="per-window scores → regions")
    de.add_argument("--window-scores", type=Path, required=True)
    de.add_argument("--op", choices=["primary", "alt"], default="alt")
    de.add_argument("--out-csv", type=Path, required=True)

    # run
    rn = sub.add_parser("run", help="FASTA → BGC regions (end-to-end)")
    rn.add_argument("--fna", type=Path, required=True)
    rn.add_argument("--out-dir", type=Path, required=True)
    rn.add_argument("--op", choices=["primary", "alt"], default="alt")
    rn.add_argument("--min-contig-len", type=int, default=0)
    rn.add_argument("--batch-size", type=int, default=1)

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.cmd == "extract":
        from . import extract
        genome = args.fna.stem.replace(".fna", "").replace(".fasta", "")
        emb = args.out_dir / f"{genome}_embeddings.npy"
        win = args.out_dir / f"{genome}_windows.csv"
        mf = extract.extract_embeddings(
            args.fna, emb, win,
            window=args.window, stride=args.stride, layer=args.layer,
            min_contig_len=args.min_contig_len,
            batch_size=args.batch_size,
        )
        print(json.dumps(mf, indent=2))
        return

    if args.cmd == "predict":
        from .predict import load_lr_weights, score_windows_from_embeddings
        from .config import load_frozen_config
        cfg = load_frozen_config()
        lr_path = args.lr_weights or cfg.lr_weights_path
        W, b = load_lr_weights(lr_path)
        emb = np.load(args.emb_npy)
        dfw = pd.read_csv(args.windows_csv)
        probs = score_windows_from_embeddings(emb, W, b)
        out = pd.DataFrame({
            "contig": dfw["contig"], "start": dfw["start"].astype(int),
            "end": dfw["end"].astype(int),
            "bgc_score": probs.astype(np.float32),
            "non_bgc_score": (1 - probs).astype(np.float32),
        })
        out.to_csv(args.out_csv, index=False)
        print(f"wrote {len(out)} rows → {args.out_csv}")
        return

    if args.cmd == "decode":
        from .decode import decode_regions, WindowScore
        from .config import PRIMARY_OP, ALT_OP
        op = PRIMARY_OP if args.op == "primary" else ALT_OP
        dfw = pd.read_csv(args.window_scores)
        wins = [WindowScore(str(c), int(s), int(e), float(bs))
                for c, s, e, bs in zip(
                    dfw["contig"], dfw["start"], dfw["end"], dfw["bgc_score"])]
        regs = decode_regions(
            wins,
            start_threshold=op["start_threshold"],
            extend_threshold=op["extend_threshold"],
            min_support_windows=op["min_support_windows"],
            max_gap=op["max_gap"],
        )
        pd.DataFrame(regs, columns=["contig", "start", "end", "max_score"]).to_csv(
            args.out_csv, index=False)
        print(f"wrote {len(regs)} regions → {args.out_csv}  (OP={args.op})")
        return

    if args.cmd == "run":
        from .pipeline import run_pipeline
        from .config import load_frozen_config, PRIMARY_OP, ALT_OP
        cfg = load_frozen_config()
        op = PRIMARY_OP if args.op == "primary" else ALT_OP
        result = run_pipeline(
            args.fna,
            args.out_dir,
            frozen_cfg=cfg,
            op=op,
            min_contig_len=args.min_contig_len,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
