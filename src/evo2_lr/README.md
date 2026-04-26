# evo2_lr

Library code for the active `evo2_lr_multiscale` mainline.

Pipeline: FASTA windows -> frozen Evo2 mean-pooled embeddings -> LR scores -> region decoding.

Active config: `configs/evo2_lr_multiscale.yaml`.

Main CLIs live in:

- `scripts/extract/extract_evo2_embeddings.py`
- `scripts/inference/apply_lr.py`
- `scripts/decode/evo2_lr_multiscale.py`
- `scripts/evaluate/evo2_lr_multiscale.py`
