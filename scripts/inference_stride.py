#!/usr/bin/env python3
"""Configurable 9-genome inference for NP-Master Route B.

This script mirrors the server-side `run_npmaster.py` flow but makes the
window size and stride explicit experiment knobs so we can test whether coarse
tiling is the main source of boundary failures on the 9-genome benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np


DEFAULT_CLASS_NAMES = [
    "Alkaloid",
    "Terpene",
    "NRP",
    "Polyketide",
    "RiPP",
    "Saccharide",
    "Other",
    "non-BGC",
]


def read_fna(fna_path: str | Path) -> list[tuple[str, str]]:
    """Read FASTA and return `(contig_id, sequence)` pairs."""
    contigs = []
    current_id = None
    current_seq: list[str] = []
    with open(fna_path) as f:
        for line in f:
            if line.startswith(">"):
                if current_id is not None:
                    contigs.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip().upper())
    if current_id is not None:
        contigs.append((current_id, "".join(current_seq)))
    return contigs


def make_windows(seq: str, window: int, stride: int) -> list[tuple[int, int, str]]:
    """Slide fixed-size windows across sequence and pad the tail window."""
    windows = []
    for start in range(0, len(seq), stride):
        end = min(start + window, len(seq))
        subseq = seq[start:end]
        if len(subseq) < 100:
            continue
        if len(subseq) < window:
            subseq = subseq + ("N" * (window - len(subseq)))
        windows.append((start, end, subseq))
    return windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fna-dir", default="/data/syh/benchmark/nine_genome/fna")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--src-dir", default=None, help="Path to repo src/ (auto-detected if omitted).")
    parser.add_argument(
        "--genome-ids",
        default=None,
        help="Optional comma-separated subset of genome IDs (without .fna) to process.",
    )
    parser.add_argument("--window", type=int, default=16384)
    parser.add_argument("--stride", type=int, default=16384)
    parser.add_argument("--layer", default="blocks.20.mlp.l3")
    parser.add_argument("--evo-model", default="evo2_7b")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--cache-dir", default="/data/public_models/hub")
    return parser.parse_args()


def extract_embeddings(evo2_model, sequences: list[str], layer_name: str) -> np.ndarray:
    """Extract mean-pooled embeddings for a list of DNA windows."""
    import torch

    embeddings = []
    for seq in sequences:
        clean = "".join(c for c in seq if c in "ATCG")
        if len(clean) < 10:
            embeddings.append(np.zeros(4096, dtype=np.float32))
            continue
        input_ids = torch.tensor(
            evo2_model.tokenizer.tokenize(clean),
            dtype=torch.int,
        ).unsqueeze(0).to("cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, emb_dict = evo2_model(
                input_ids,
                return_embeddings=True,
                layer_names=[layer_name],
            )
        emb = emb_dict[layer_name][0].float().mean(dim=0).cpu().numpy()
        embeddings.append(emb)
    return np.asarray(embeddings, dtype=np.float32)


def resolve_src_dir(cli_value: str | None) -> Path:
    """Resolve the repo `src/` package path for local and piped-stdin execution."""
    if cli_value:
        candidate = Path(cli_value).expanduser().resolve()
        if (candidate / "model" / "context_full_model.py").exists():
            return candidate
        raise FileNotFoundError(f"Invalid --src-dir: {candidate}")

    candidates = []
    try:
        current_file = Path(__file__).resolve()
        candidates.extend(
            [
                current_file.parent.parent / "src",
                current_file.parent.parent.parent / "src",
            ]
        )
    except Exception:
        pass
    cwd = Path.cwd().resolve()
    candidates.extend(
        [
            cwd / "src",
            cwd.parent / "src",
        ]
    )
    for candidate in candidates:
        if (candidate / "model" / "context_full_model.py").exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate src/. Pass --src-dir explicitly. "
        f"Searched: {searched}"
    )


def main() -> None:
    args = parse_args()

    selected_genomes = None
    if args.genome_ids:
        selected_genomes = {part.strip() for part in args.genome_ids.split(",") if part.strip()}

    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Evo2...", flush=True)
    t0 = time.time()
    from evo2 import Evo2

    evo2_model = Evo2(args.evo_model)
    print(f"Evo2 loaded in {time.time() - t0:.1f}s", flush=True)

    print("Loading ContextFullModel...", flush=True)
    import sys

    src_dir = resolve_src_dir(args.src_dir)
    sys.path.insert(0, str(src_dir))
    from model.context_full_model import ContextFullModel

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    context_model = ContextFullModel(
        input_dim=4096,
        proj_hidden=2048,
        proj_out=256,
        proj_dropout=0.0,
        n_labels=8,
        cnn_dropout=0.0,
        init_beta=-4.0,
    )
    context_model.load_state_dict(ckpt["model"])
    context_model.to("cuda").eval()
    print(
        f"ContextFullModel loaded, alpha={context_model.gate.alpha.item():.4f}, "
        f"epoch={ckpt.get('epoch', '?')}",
        flush=True,
    )

    all_results = {}
    for fna_path in sorted(Path(args.fna_dir).glob("*.fna")):
        genome_id = fna_path.stem
        if selected_genomes is not None and genome_id not in selected_genomes:
            continue
        out_csv = Path(args.out_dir) / f"{genome_id}_predictions.csv"
        print(f"\nProcessing {genome_id}...", flush=True)

        genome_predictions = []
        for contig_id, seq in read_fna(fna_path):
            windows = make_windows(seq, window=args.window, stride=args.stride)
            if not windows:
                continue
            print(
                f"  {contig_id}: {len(seq)} bp, {len(windows)} windows "
                f"(window={args.window}, stride={args.stride})",
                flush=True,
            )
            emb_array = extract_embeddings(
                evo2_model,
                [window_seq for _, _, window_seq in windows],
                args.layer,
            )
            emb_tensor = torch.from_numpy(emb_array).unsqueeze(0).to("cuda")
            valid_mask = torch.ones(1, len(windows), dtype=torch.bool, device="cuda")
            with torch.no_grad():
                _, logits = context_model(emb_tensor, valid_mask)
                probs = torch.sigmoid(logits[0]).cpu().numpy()

            for idx, (start, end, _) in enumerate(windows):
                bgc_score = float(probs[idx, :7].max())
                pred_class = int(probs[idx].argmax())
                genome_predictions.append(
                    {
                        "contig": contig_id,
                        "start": start,
                        "end": min(end, len(seq)),
                        "bgc_score": round(bgc_score, 6),
                        "non_bgc_score": round(float(probs[idx, 7]), 6),
                        "pred_class": DEFAULT_CLASS_NAMES[pred_class],
                        "probs": json.dumps([round(float(p), 6) for p in probs[idx].tolist()]),
                    }
                )

        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "contig",
                    "start",
                    "end",
                    "bgc_score",
                    "non_bgc_score",
                    "pred_class",
                    "probs",
                ],
            )
            writer.writeheader()
            writer.writerows(genome_predictions)
        all_results[genome_id] = len(genome_predictions)
        print(f"  saved to {out_csv}", flush=True)

    print(json.dumps({"n_windows_per_genome": all_results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
