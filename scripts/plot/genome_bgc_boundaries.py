#!/usr/bin/env python3
"""Boundary plot for one genome's longest contig (CLI).

Thin wrapper around :func:`plotting.boundaries.main`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from plotting.boundaries import main  # noqa: E402

if __name__ == "__main__":
    main()
