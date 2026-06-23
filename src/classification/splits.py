#!/usr/bin/env python3
"""Family-disjoint train/val/test splitting to prevent BGC homology leakage.

BGCs from the same gene-cluster family are near-identical; a random split leaks
them across train/test and inflates metrics (and would hide exactly the PKS/NRP
confusion we care about). We split by *family group* so no family straddles
splits.

`family_from_mmseqs` builds families from an mmseqs2 `easy-cluster` TSV (the
light, dependency-free way to get families); if no clustering is available,
`identity_families` falls back to one-cluster-per-id with a loud caller-side
warning that metrics will be leakage-optimistic.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path


def family_from_mmseqs(cluster_tsv: Path) -> dict[str, str]:
    """Parse mmseqs2 easy-cluster output (TSV: representative<TAB>member) →
    member_id → family_id (the representative)."""
    fam: dict[str, str] = {}
    for line in Path(cluster_tsv).read_text().splitlines():
        if not line.strip():
            continue
        rep, member = line.split("\t")[:2]
        fam[member] = rep
    return fam


def identity_families(ids: Sequence[str]) -> dict[str, str]:
    """Degenerate mapping: every id is its own family (NO leakage protection)."""
    return {i: i for i in ids}


def group_split(
    ids: Sequence[str],
    family_of: dict[str, str],
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, list[str]]:
    """Split ids into train/val/test with whole families kept together.

    Families are shuffled (deterministically by `seed`) and greedily assigned to
    the split that is most under its target id-count quota, so the *id* counts
    land near `fractions` even when family sizes are uneven.
    """
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1"
    members: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        members[family_of.get(i, i)].append(i)

    fams = list(members)
    random.Random(seed).shuffle(fams)

    n_total = len(ids)
    targets = {
        "train": fractions[0] * n_total,
        "val": fractions[1] * n_total,
        "test": fractions[2] * n_total,
    }
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for fam in fams:
        # assign to the split furthest below its target (by remaining quota)
        split = max(out, key=lambda s: targets[s] - len(out[s]))
        out[split].extend(members[fam])
    return out
