#!/usr/bin/env python3
"""v4.1 multi-scale ensemble honest readout on 9-genome (CLI).

Thin wrapper around :func:`decoding.v4_1.main`. See that module for the
ensemble logic (primary w=8192 + short rescue w=4096 with L<10kb filter,
union-dedup at IoU=0.3).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from decoding.v4_1 import main  # noqa: E402

if __name__ == "__main__":
    main()
