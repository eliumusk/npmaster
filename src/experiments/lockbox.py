from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class LockboxGateError(Exception):
    """Raised when the lockbox gate refuses to open."""


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def is_lockbox_split(split_part: str) -> bool:
    return split_part == "lockbox"


def assert_candidate_frozen(candidate_manifest_path: Path) -> dict:
    manifest_path = Path(candidate_manifest_path)
    if not manifest_path.is_file():
        raise LockboxGateError(
            f"candidate manifest not found: {manifest_path}"
        )

    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LockboxGateError(
            f"could not read candidate manifest {manifest_path}: {exc}"
        ) from exc

    if not isinstance(manifest, dict):
        raise LockboxGateError("candidate manifest must be a JSON object")

    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise LockboxGateError("candidate manifest missing non-empty experiment_id")

    git_commit = manifest.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise LockboxGateError("candidate manifest missing non-empty git_commit")

    if manifest.get("git_dirty") is not False:
        raise LockboxGateError("candidate manifest must have git_dirty == False")

    if manifest.get("frozen") is not True:
        raise LockboxGateError("candidate manifest must have frozen == True")

    return manifest


def record_lockbox_open(
    audit_log_path: Path,
    *,
    experiment_id: str,
    git_commit: str,
    candidate_manifest_path: Path,
    reason: str,
) -> dict:
    experiment_id = _require_non_empty_string(experiment_id, "experiment_id")
    git_commit = _require_non_empty_string(git_commit, "git_commit")
    reason = _require_non_empty_string(reason, "reason")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "git_commit": git_commit,
        "candidate_manifest_path": str(candidate_manifest_path),
        "reason": reason,
    }

    log_path = Path(audit_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    return entry


def gate_lockbox_run(
    *,
    split_part: str,
    candidate_manifest_path: Path,
    audit_log_path: Path,
    reason: str,
) -> dict | None:
    reason = _require_non_empty_string(reason, "reason")

    if not is_lockbox_split(split_part):
        return None

    manifest = assert_candidate_frozen(candidate_manifest_path)
    return record_lockbox_open(
        audit_log_path,
        experiment_id=manifest["experiment_id"],
        git_commit=manifest["git_commit"],
        candidate_manifest_path=candidate_manifest_path,
        reason=reason,
    )
