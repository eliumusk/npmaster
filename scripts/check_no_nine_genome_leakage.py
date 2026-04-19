#!/usr/bin/env python3
"""Stage 5 nine-genome embargo leakage checker.

Scans text-like files under one or more paths for occurrences of
embargoed 9-genome assembly IDs. Exits non-zero if any hit is found
outside the optional allowlist.

Intended use: pre-flight guard before launching any Stage 5 training,
mining, or selection job. Final 9-genome *report* artifacts legitimately
contain these IDs and should be allowlisted explicitly.

Example:
    python scripts/check_no_nine_genome_leakage.py \\
        --embargo configs/splits/nine_genome_embargo_v1.json \\
        --paths configs scripts src docs
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Iterable


# Directories that are never scanned (build / VCS noise).
_NOISE_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "node_modules",
        ".tox",
    }
)

# Extensions that are always skipped (binary / large).
_BINARY_EXTENSIONS = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".bin",
        ".npy",
        ".npz",
        ".pkl",
        ".parquet",
        ".feather",
        ".h5",
        ".hdf5",
        ".tar",
        ".tgz",
        ".gz",
        ".zip",
        ".xz",
        ".bz2",
        ".zst",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".so",
        ".dylib",
        ".o",
    }
)

# Hard cap so we never read a massive CSV into memory line by line.
_MAX_FILE_BYTES = 64 * 1024 * 1024  # 64 MiB


def load_embargo_ids(path: Path) -> set[str]:
    """Load embargoed genome IDs from a JSON embargo-config file."""
    if not path.is_file():
        raise FileNotFoundError(f"Embargo config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed embargo JSON at {path}: {exc}") from exc
    ids = data.get("embargoed_genome_ids")
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError(
            f"{path} must contain list[str] at 'embargoed_genome_ids'"
        )
    return set(ids)


def _iter_candidate_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield files to scan, skipping noise dirs and binary extensions."""
    for root in paths:
        if root.is_file():
            if root.suffix.lower() not in _BINARY_EXTENSIONS:
                yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if any(part in _NOISE_DIR_NAMES for part in path.parts):
                continue
            if path.suffix.lower() in _BINARY_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path


def _is_allowed(path: Path, allowlist_globs: list[str]) -> bool:
    if not allowlist_globs:
        return False
    posix = path.as_posix()
    name = path.name
    return any(
        fnmatch.fnmatch(posix, glob) or fnmatch.fnmatch(name, glob)
        for glob in allowlist_globs
    )


def scan_paths_for_embargo_ids(
    paths: list[Path],
    embargo_ids: set[str],
    allowlist_globs: list[str] | None = None,
) -> list[dict]:
    """Scan every text file under ``paths`` for embargoed IDs.

    Returns a list of hit dicts with keys: ``path``, ``line``, ``genome_id``.
    Hits inside allowlisted files are omitted.
    """
    allowlist_globs = list(allowlist_globs or [])
    hits: list[dict] = []
    for path in _iter_candidate_files(paths):
        if _is_allowed(path, allowlist_globs):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not any(bad in text for bad in embargo_ids):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for bad in embargo_ids:
                if bad in line:
                    hits.append(
                        {
                            "path": str(path),
                            "line": line_no,
                            "genome_id": bad,
                        }
                    )
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embargo",
        type=Path,
        default=Path("configs/splits/nine_genome_embargo_v1.json"),
        help="Path to the embargo config JSON.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        type=Path,
        help="Files or directories to scan recursively.",
    )
    parser.add_argument(
        "--allow",
        nargs="*",
        default=None,
        help=(
            "Glob(s) for files legitimately allowed to contain embargo IDs "
            "(e.g. final 9-genome report artifacts)."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="If set, write the hit list as JSON to this path.",
    )
    args = parser.parse_args()

    embargo_ids = load_embargo_ids(args.embargo)
    hits = scan_paths_for_embargo_ids(
        args.paths, embargo_ids, allowlist_globs=args.allow
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(hits, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for hit in hits:
        print(f"{hit['path']}:{hit['line']}:{hit['genome_id']}")

    if hits:
        print(
            f"\nFAIL: {len(hits)} embargo-ID hit(s) across "
            f"{len({h['path'] for h in hits})} file(s).",
            file=sys.stderr,
        )
        return 1
    print("OK: no embargoed 9-genome IDs found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
