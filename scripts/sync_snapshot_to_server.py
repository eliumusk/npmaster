#!/usr/bin/env python3
"""Prepare immutable local experiment bundles for execution servers."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_FIELDS = ("experiment_id", "git_commit", "command")
REQUIRED_BUNDLE_FILES = (
    "manifest.json",
    "config.resolved.yaml",
    "command.txt",
    "inputs/manifest.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_run_bundle_spec(experiment_dir: Path, repo_root: Path) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    repo_root = repo_root.resolve()

    try:
        relative_experiment_path = experiment_dir.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError("experiment_dir must live inside repo_root") from exc

    manifest_path = experiment_dir / "manifest.json"
    manifest = _read_json(manifest_path)

    for field in REQUIRED_MANIFEST_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"manifest.json missing required field: {field}")

    missing_files = [
        rel_path
        for rel_path in REQUIRED_BUNDLE_FILES
        if not (experiment_dir / rel_path).is_file()
    ]
    if missing_files:
        raise ValueError(
            f"experiment bundle missing required files: {', '.join(missing_files)}"
        )

    command = (experiment_dir / "command.txt").read_text(encoding="utf-8").strip()
    if not command:
        raise ValueError("command.txt must contain an execution command")

    return {
        "experiment_id": manifest["experiment_id"].strip(),
        "git_commit": manifest["git_commit"].strip(),
        "snapshot_id": (
            f"{manifest['git_commit'].strip()}-dirty"
            if manifest.get("git_dirty")
            else manifest["git_commit"].strip()
        ),
        "command": command,
        "relative_experiment_path": relative_experiment_path,
        "required_files": list(REQUIRED_BUNDLE_FILES),
    }


def build_sync_plan(
    bundle_spec: dict[str, Any],
    host_config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    workspace_root = host_config["workspace_root"].rstrip("/")
    snapshot_dirname = host_config["snapshot_dirname"].strip("/")
    snapshot_id = bundle_spec.get("snapshot_id", bundle_spec["git_commit"])
    snapshot_root = f"{workspace_root}/{snapshot_dirname}/{snapshot_id}"

    snapshot_cmd = [
        "rsync",
        "-az",
        "--delete",
        f"{repo_root}/",
        f"{host_config['ssh_target']}:{snapshot_root}/repo/",
    ]

    return {
        "server": host_config["name"],
        "ssh_target": host_config["ssh_target"],
        "remote_snapshot_root": snapshot_root,
        "commands": [
            " ".join(shlex.quote(part) for part in snapshot_cmd),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a local experiment bundle and print the sync plan."
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    bundle_spec = build_run_bundle_spec(args.experiment_dir, args.repo_root)
    print(json.dumps(bundle_spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
