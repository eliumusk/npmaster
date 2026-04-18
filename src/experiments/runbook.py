from __future__ import annotations

from pathlib import Path


def build_runbook_entry(
    *,
    experiment_id: str,
    server: str,
    command: str,
    manifest_path: str | Path,
) -> dict[str, str]:
    for field_name, value in (
        ("experiment_id", experiment_id),
        ("server", server),
        ("command", command),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

    return {
        "experiment_id": experiment_id,
        "server": server,
        "command": command,
        "manifest_path": str(Path(manifest_path)),
    }
