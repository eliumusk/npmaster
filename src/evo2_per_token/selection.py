"""Selection helpers that keep 9-genome out of model/OP choice."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

NINE_GENOME_ALIASES = {"9g", "9genome", "ninegenome", "ng"}


def _source_name(candidate: dict) -> str:
    return str(candidate.get("source_dataset", "")).strip().lower()


def _normalise_source_name(source: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", source.lower())


def _validate_non_nine_genome(candidates: Iterable[dict]) -> None:
    bad: list[str] = []
    missing: list[str] = []
    for candidate in candidates:
        source = _source_name(candidate)
        name = str(candidate.get("name", "<unnamed>"))
        if not source:
            missing.append(name)
            continue
        normalised = _normalise_source_name(source)
        if any(alias in normalised for alias in NINE_GENOME_ALIASES):
            bad.append(name)
    if bad:
        raise ValueError(f"9-genome candidates are not allowed during selection: {bad}")
    if missing:
        raise ValueError(f"candidates must declare source_dataset: {missing}")


def select_operating_point(
    candidates: list[dict],
    max_regions: int,
    min_as_recall: float = 0.0,
) -> dict:
    """Choose the highest-recall non-9g operating point under a region budget."""
    _validate_non_nine_genome(candidates)
    feasible = [
        c for c in candidates
        if int(c.get("metrics", {}).get("n_pred", 0)) <= max_regions
        and float(c.get("metrics", {}).get("as_recall", 0.0)) >= min_as_recall
    ]
    if not feasible:
        raise ValueError("no candidate satisfies the non-9g selection constraints")
    return max(
        feasible,
        key=lambda c: (
            float(c.get("metrics", {}).get("as_recall", 0.0)),
            -int(c.get("metrics", {}).get("n_pred", 0)),
            -int(c.get("metrics", {}).get("novel", 0)),
        ),
    )


def write_frozen_protocol(
    out_json: str | Path,
    checkpoint: str,
    operating_points: list[dict],
    selection_sources: list[str],
) -> None:
    """Write the frozen recipe that must exist before a final 9g readout."""
    normalised_sources = {_normalise_source_name(source) for source in selection_sources}
    if any(any(alias in source for alias in NINE_GENOME_ALIASES) for source in normalised_sources):
        raise ValueError("selection_sources must not include 9-genome")
    payload = {
        "checkpoint": checkpoint,
        "operating_points": operating_points,
        "selection_sources": selection_sources,
        "nine_genome_allowed_final_readout_only": True,
    }
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_protocol_operating_point(protocol_json: str | Path, name: str) -> dict:
    """Load a frozen operating point from a protocol JSON file."""
    protocol = json.loads(Path(protocol_json).read_text())
    if not protocol.get("nine_genome_allowed_final_readout_only"):
        raise ValueError("protocol does not declare 9-genome as final-readout-only")
    for op in protocol.get("operating_points", []):
        if op.get("name") == name:
            return op
    raise ValueError(f"operating point {name!r} not found in protocol")


def validate_final_readout_protocol(protocol_json: str | Path) -> dict:
    """Validate that a final-readout protocol was frozen from non-9g sources."""
    protocol = json.loads(Path(protocol_json).read_text())
    sources = protocol.get("selection_sources", [])
    normalised_sources = {_normalise_source_name(str(source)) for source in sources}
    if any(any(alias in source for alias in NINE_GENOME_ALIASES) for source in normalised_sources):
        raise ValueError("protocol selection_sources include 9-genome")
    if not protocol.get("nine_genome_allowed_final_readout_only"):
        raise ValueError("protocol does not mark 9-genome as final-readout-only")
    return protocol
