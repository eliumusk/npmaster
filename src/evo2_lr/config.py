"""Frozen operating points and scale configs.

All OPs were chosen BEFORE any 9-genome readout (see
`docs/superpowers/plans/2026-04-21-evo2-first-principles-v4.md` §§2, 13).
They must never be tuned on any evaluation corpus — if someone later adds
a new OP here, they must document why it is prior-fixed.

Two public recipes:
  - v4 single-scale (PRIMARY_OP / ALT_OP) — baseline
  - v4.1 multi-scale ensemble (V4_1_SCALES) — mainline (+0.011 strict over v4)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# --- v4 baseline (single-scale window-LR) -------------------------------------

PRIMARY_OP = {
    "start_threshold": 0.90,
    "extend_threshold": 0.90,
    "min_support_windows": 1,
    "max_gap": 0,
    "name": "primary",
    "rationale": "LR 0.9 calibrated cutoff, single threshold (hysteresis off)",
}

ALT_OP = {
    "start_threshold": 0.95,
    "extend_threshold": 0.80,
    "min_support_windows": 3,
    "max_gap": 0,
    "name": "alt",
    "rationale": "conservative seeds + relaxed extend + 3-window spike filter",
}

DEFAULT_GEOMETRY = {"window": 8192, "stride": 2048}
DEFAULT_LAYER = "blocks.20.mlp.l3"


# --- v4.1 multi-scale ensemble (mainline) -------------------------------------
# Primary scale = v4 alt OP at w=8192. Short rescue scale = w=4096 with ms=5
# (natural min region = 8 kb) + length cap 10 kb (biological cutoff).
# See plan §13 and `configs/experiments/evo2_first_principles_v4_1_frozen.yaml`.

V4_1_SCALES = {
    "primary": {
        "window": 8192, "stride": 2048,
        "operating_point": dict(ALT_OP),
        "length_filter_max": None,
    },
    "short_rescue": {
        "window": 4096, "stride": 1024,
        "operating_point": {
            "start_threshold": 0.95, "extend_threshold": 0.80,
            "min_support_windows": 5, "max_gap": 0,
        },
        "length_filter_max": 10000,
    },
}
V4_1_IOU_DEDUP = 0.3


# --- FrozenConfig dataclass ---------------------------------------------------

@dataclass(frozen=True)
class ScaleConfig:
    window: int
    stride: int
    operating_point: dict
    length_filter_max: int | None = None


@dataclass(frozen=True)
class FrozenConfig:
    """Loads `configs/experiments/evo2_first_principles_v4_1_frozen.yaml`."""
    lr_weights_path: Path
    layer: str
    scales: dict  # {"primary": ScaleConfig, "short_rescue": ScaleConfig}
    iou_dedup: float = V4_1_IOU_DEDUP

    # Convenience: single-scale fallback to primary when a caller just wants
    # the v4-style (window, stride, primary_op) tuple.
    @property
    def window(self) -> int:
        return self.scales["primary"].window

    @property
    def stride(self) -> int:
        return self.scales["primary"].stride

    @property
    def primary_op(self) -> dict:
        return dict(self.scales["primary"].operating_point)

    @property
    def alt_op(self) -> dict:
        # v4.1's "primary" == v4's "alt" — keep alias for back-compat.
        return dict(self.scales["primary"].operating_point)

    @classmethod
    def from_yaml(cls, path: Path) -> "FrozenConfig":
        d = yaml.safe_load(path.read_text())
        model = d["model"]
        scales_yaml = d.get("scales") or {}
        scales = {}
        for name, s in scales_yaml.items():
            scales[name] = ScaleConfig(
                window=int(s["window"]),
                stride=int(s["stride"]),
                operating_point=dict(s["operating_point"]),
                length_filter_max=s.get("length_filter_max"),
            )
        feat = model.get("feature", "")
        layer = DEFAULT_LAYER
        if "blocks." in feat:
            try:
                layer = "blocks." + feat.split("_blocks.")[-1].split("_")[0]
            except Exception:
                layer = DEFAULT_LAYER
        return cls(
            lr_weights_path=Path(model["weights"]),
            layer=layer,
            scales=scales,
            iou_dedup=float(d.get("fusion", {}).get("iou_threshold", V4_1_IOU_DEDUP)),
        )


def load_frozen_config(path: Path | str | None = None) -> FrozenConfig:
    """Load the v4.1 frozen config yaml. Defaults to the repo-relative file."""
    if path is None:
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "configs" / "experiments" / "evo2_first_principles_v4_1_frozen.yaml"
    return FrozenConfig.from_yaml(Path(path))
