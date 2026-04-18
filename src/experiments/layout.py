from __future__ import annotations

from pathlib import Path

import yaml


def create_experiment_layout(root_dir: str | Path, experiment_id: str) -> dict[str, Path]:
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string")

    base_dir = Path(root_dir)
    if base_dir.name == "experiments":
        root = base_dir / experiment_id
    else:
        root = base_dir / "experiments" / experiment_id
    paths = {
        "root": root,
        "manifest": root / "manifest.json",
        "config": root / "config.resolved.yaml",
        "env": root / "env.txt",
        "command": root / "command.txt",
        "logs": root / "logs",
        "metrics": root / "metrics",
        "artifacts": root / "artifacts",
        "checkpoints": root / "artifacts" / "checkpoints",
        "predictions": root / "artifacts" / "predictions",
        "plots": root / "artifacts" / "plots",
        "tables": root / "artifacts" / "tables",
        "inputs": root / "inputs",
        "input_manifest": root / "inputs" / "manifest.json",
    }

    for key in (
        "logs",
        "metrics",
        "checkpoints",
        "predictions",
        "plots",
        "tables",
        "inputs",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)

    for key in ("manifest", "config", "env", "command", "input_manifest"):
        paths[key].touch(exist_ok=True)

    return paths


def write_command(path: str | Path, command: str) -> Path:
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")

    command_path = Path(path)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(command.rstrip() + "\n", encoding="utf-8")
    return command_path


def write_resolved_config(path: str | Path, config_dict: dict) -> Path:
    if not isinstance(config_dict, dict):
        raise ValueError("config_dict must be an object")

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config_dict, sort_keys=True),
        encoding="utf-8",
    )
    return config_path
