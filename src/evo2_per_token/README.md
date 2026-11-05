# per_token/

Per-nucleotide BGC prediction on Evo2 per-token hidden states. An architecture
experiment competing with the v4.1 mean-pool+LR pipeline.

Not yet promoted — see `experiments/evo2_per_token_unet/OER_UNET_READOUT.md` for
the current E10/E11 readouts. E10 is close to v4.1 on 9-genome and improves
same-script OER antiSMASH recall, but the checkpoint+threshold selection rule
is not yet fully pre-registered.

## Layout

- `labels.py` — construct dense per-nucleotide labels (`-1`/`0`/`1`) from
  antiSMASH region coordinates + training window metadata.
- `model.py` — `PerTokenHead`: 3× dilated Conv1D + 1× readout, ~50 K params;
  `UNetHead`: 1D U-Net with skip connections, ~321 K params at hidden=64.
  The current best exploratory checkpoint is E10 (`UNetHead`, hidden=64,
  lr=1e-3, batch=32, Dice+BCE).
- `decode.py` — per-window token probabilities → per-nt track → regions via
  threshold + gap-fill. No `min_support_windows` hack.

## Usage

```python
import sys; sys.path.insert(0, "src")
from evo2_per_token.labels import load_windows, load_bgc_annotations, per_nt_labels
from evo2_per_token.model import PerTokenHead
from evo2_per_token.decode import decode_all

# Dense labels
windows = load_windows("data/windows/edge_padding/windows_metadata_edgepad_filtered.csv")
bgc_map = load_bgc_annotations("data/bgc_annotations.csv")
y = per_nt_labels(windows, bgc_map)   # (N, T) int8, values in {-1, 0, 1}

# Training loop — see scripts/train/per_token/train.py

# Inference — see scripts/inference/evo2_per_token/infer.py
# Decoding
regions = decode_all(window_probs, window_coords, threshold=0.5, min_len_bp=2000)
```

## Assumptions

- Evo2 byte-level tokenizer: `len(tokens) == len(cleaned_sequence)`.
- Features stored per-window, fp16, shape `(T_kept, D_reduced)`.
- `T_kept = (end - start) / K` where K is the extraction-time pool factor
  (currently K=8; set at extraction time, baked into storage).
- `D_reduced = 128` via fixed Gaussian random projection from Evo2's native
  `blocks.20.mlp.l3` (4096-d).
- Training windows at w=16384 bp; inference windows at w=8192 bp.
  Head is length-agnostic (pure Conv1D).

## Metrics

Standard evaluation against the unified `experiments/eval_harness/baseline_table.md`:

- 9g block: `bench` mode F1 from `src/benchmark/nine_genome.py`.
- OER block: `n_regions`, AS-overlap rate, per-type count, against
  `/data/muskliu/oer_bgc_scan/results/antismash/OER*.ge10000/` (≥10 kb filtered).

Current E10 snapshot:
- 9g, OER-preselected threshold 0.3: `dedup_F1=0.4561`.
- 9g, OER-preselected threshold 0.4: `dedup_F1=0.4664`.
- 9g, default threshold 0.5: `dedup_F1=0.4751`.
- OER, threshold 0.3: AS-recall `0.3117` (616 / 1,976), 1,377 regions.

Pass/fail gates in `experiments/eval_harness/baseline_table.md` are intentionally
stricter than the current exploratory rows. Before promotion, freeze the model
family and threshold rule without inspecting 9-genome, then run the final 9g
readout once.
