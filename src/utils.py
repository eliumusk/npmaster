from __future__ import annotations
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(base_path: str, ablation_path: str | None = None) -> dict:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if ablation_path:
        with open(ablation_path) as f:
            over = yaml.safe_load(f)
        _deep_update(cfg, over)
    return cfg


def _deep_update(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def dann_lambda(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, epoch / warmup_epochs))


def autodetect_device(pref: str) -> str:
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_json(obj: Any, path: str | Path):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)


def _to_builtin(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
