from __future__ import annotations

import json
from pathlib import Path


REQUIRED_MANIFEST_FIELDS = (
    "experiment_id",
    "git_commit",
    "server",
    "command",
)


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def validate_split(split_data: dict) -> dict:
    if not isinstance(split_data, dict):
        raise ValueError("split_data must be an object")

    version = _require_non_empty_string(split_data.get("version"), "version")
    dev = split_data.get("dev")
    lockbox = split_data.get("lockbox")

    if not isinstance(dev, list) or not isinstance(lockbox, list):
        raise ValueError("dev and lockbox must be lists")
    if len(dev) != 5:
        raise ValueError("dev must contain exactly 5 genomes")
    if len(lockbox) != 4:
        raise ValueError("lockbox must contain exactly 4 genomes")

    combined = dev + lockbox
    validated = [_require_non_empty_string(genome_id, "genome_id") for genome_id in combined]
    if len(validated) != len(set(validated)):
        raise ValueError("each genome must appear exactly once")

    return {
        "version": version,
        "dev": validated[: len(dev)],
        "lockbox": validated[len(dev):],
    }


def load_split(path: str | Path) -> dict:
    split_path = Path(path)
    with split_path.open(encoding="utf-8") as handle:
        split_data = json.load(handle)
    return validate_split(split_data)


def validate_manifest(manifest_dict: dict) -> dict:
    if not isinstance(manifest_dict, dict):
        raise ValueError("manifest_dict must be an object")

    validated = dict(manifest_dict)
    for field_name in REQUIRED_MANIFEST_FIELDS:
        if field_name not in manifest_dict:
            raise ValueError(f"missing required manifest field: {field_name}")
        validated[field_name] = _require_non_empty_string(
            manifest_dict[field_name],
            field_name,
        )
    return validated


def write_manifest(path: str | Path, manifest_dict: dict) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_manifest(manifest_dict)
    manifest_path.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
