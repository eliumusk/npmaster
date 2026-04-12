"""Adapter: diagnostic-day1 embeddings (train/val/test pkl + *_emb.pkl)
-> (embeddings.npy, labels.csv) in the stage3 format.

Diagnostic data source column order (7 types):
  [Polyketide, NRP, RiPP, Terpene, Saccharide, Alkaloid, Other]
Stage-3 target column order (8 cols):
  [Polyketide, NRP, Terpene, RiPP, Saccharide, Alkaloid, Other, is_negative]
We remap RiPP<->Terpene.

Phylum (diagnostic data has none preserved): we synthesize a pseudo-phylum
from the id prefix so DANN has something meaningful to adversarially remove.
  - id starting with "BGC" -> phylum = "MIBiG_src"
  - id starting with "NEG" -> phylum = "Background_src"
This exactly mirrors the known source bias for code-validation purposes.

Split: we honor the existing train/val/test pkl files (no re-splitting).

Usage:
  python -m bgc-gcd.data.adapt_diagnostic \
      --src_dir /data/muskliu/np_master/data \
      --out_dir /data/muskliu/np_master/bgc-gcd/data
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# diagnostic -> stage3 column remap:
# diag index: 0=Polyketide, 1=NRP, 2=RiPP, 3=Terpene, 4=Saccharide,
#             5=Alkaloid, 6=Other
# stage3 order: Polyketide, NRP, Terpene, RiPP, Saccharide, Alkaloid, Other
DIAG_TO_STAGE3 = [0, 1, 3, 2, 4, 5, 6]


def infer_phylum(window_id: str) -> str:
    if window_id.startswith("BGC"):
        return "MIBiG_src"
    if window_id.startswith("NEG"):
        return "Background_src"
    return "Unknown"


def load_split(src_dir: Path, split_pkl: str, emb_pkl: str, split_name: str):
    emb_blob = pickle.load(open(src_dir / emb_pkl, "rb"))
    embeddings = np.asarray(emb_blob["embeddings"], dtype=np.float32)
    cats_diag = np.asarray(emb_blob["categories"], dtype=np.int64)  # (N, 7)
    ids = list(emb_blob["ids"])
    is_bgc = np.asarray(emb_blob["labels"], dtype=np.int64)

    # reorder diagnostic category columns to stage3 order (first 7 cols)
    cats_s3 = cats_diag[:, DIAG_TO_STAGE3]
    is_neg = (1 - is_bgc).reshape(-1, 1)
    labels_8 = np.concatenate([cats_s3, is_neg], axis=1).astype(np.int64)

    phyla = [infer_phylum(x) for x in ids]
    split_arr = [split_name] * len(ids)

    df = pd.DataFrame({
        "window_id": ids,
        "Polyketide": labels_8[:, 0],
        "NRP": labels_8[:, 1],
        "Terpene": labels_8[:, 2],
        "RiPP": labels_8[:, 3],
        "Saccharide": labels_8[:, 4],
        "Alkaloid": labels_8[:, 5],
        "Other": labels_8[:, 6],
        "is_negative": labels_8[:, 7],
        "phylum": phyla,
        "split": split_arr,
    })
    return embeddings, df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    parts = [
        ("train", "train_emb.pkl", "train"),
        ("val", "val_emb.pkl", "val"),
        ("test", "test_emb.pkl", "test_id"),  # reuse diagnostic test as id-test
    ]
    all_emb, all_df = [], []
    for split_name, emb_pkl, out_split in parts:
        emb, df = load_split(src, "", emb_pkl, out_split)
        all_emb.append(emb)
        all_df.append(df)
        print(f"[{split_name}->{out_split}] {len(df)} windows, "
              f"pos={(df['is_negative']==0).sum()}, neg={(df['is_negative']==1).sum()}")

    # (no test_ood in diagnostic data; stage3 still accepts missing splits)
    emb_all = np.concatenate(all_emb, axis=0)
    df_all = pd.concat(all_df, axis=0, ignore_index=True)

    # sanity
    assert emb_all.shape[0] == len(df_all)
    assert emb_all.shape[1] == 4096
    print(f"Total: {emb_all.shape[0]} windows, emb_dim={emb_all.shape[1]}")
    print(f"Phyla: {df_all['phylum'].value_counts().to_dict()}")
    print(f"Per-type pos counts:")
    for c in ["Polyketide", "NRP", "Terpene", "RiPP",
              "Saccharide", "Alkaloid", "Other"]:
        print(f"  {c}: {int(df_all[c].sum())}")

    np.save(out / "embeddings.npy", emb_all)
    df_all.to_csv(out / "labels.csv", index=False)
    print(f"Wrote {out/'embeddings.npy'} and {out/'labels.csv'}")


if __name__ == "__main__":
    main()
