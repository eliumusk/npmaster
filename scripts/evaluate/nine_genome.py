#!/usr/bin/env python3
"""9-genome region-level evaluation under a frozen operating point (CLI).

Thin wrapper around :func:`benchmark.nine_genome.main`. See that module for
the merge/evaluate logic and the four overlap rules (bench / loose / strict / iou).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmark.nine_genome import main  # noqa: E402

if __name__ == "__main__":
    main()
