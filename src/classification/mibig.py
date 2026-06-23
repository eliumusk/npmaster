#!/usr/bin/env python3
"""MIBiG parsing + canonical 7-class biosynthetic labels for the domain-based type head.

The product-type classifier is trained on MIBiG (curated, experimentally-supported
labels). Each MIBiG entry carries one or more biosynthetic classes; clusters are
genuinely multi-label (hybrid PKS-NRP → both Polyketide and NRP), so labels are
multi-hot, not single argmax.

This module is deliberately tolerant about MIBiG's JSON schema (it changed between
v2/v3 and v4): `extract_classes` walks a few known shapes and falls back gracefully.

Canonical class order matches the repo's existing 7-class taxonomy so the R^8
evaluation contract stays coherent across the codebase.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Canonical 7 classes (order fixed; persisted with each trained model).
CANONICAL_CLASSES = [
    "Alkaloid", "Terpene", "NRP", "Polyketide", "RiPP", "Saccharide", "Other",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# Raw MIBiG / antiSMASH class & product strings → canonical. Keys are lowercased.
# MIBiG biosyn_class values are already coarse; the PKS subtype aliases make the
# map robust if labels ever arrive as antiSMASH product strings instead.
_ALIASES = {
    "nrp": "NRP", "nrps": "NRP", "nrps-like": "NRP",
    "non-ribosomal peptide": "NRP", "nonribosomal peptide": "NRP",
    "polyketide": "Polyketide", "pks": "Polyketide",
    "t1pks": "Polyketide", "t2pks": "Polyketide", "t3pks": "Polyketide",
    "transatpks": "Polyketide", "transat-pks": "Polyketide",
    "hr-t1pks": "Polyketide", "hglks": "Polyketide", "pks-like": "Polyketide",
    "ripp": "RiPP", "ribosomal": "RiPP", "ripp-like": "RiPP",
    "terpene": "Terpene",
    "saccharide": "Saccharide", "oligosaccharide": "Saccharide",
    "alkaloid": "Alkaloid",
    "other": "Other",
}


def normalize_class(raw: str) -> str:
    """Map one raw class/product string to a canonical class ('Other' if unknown)."""
    key = str(raw).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # tolerate substrings like 'NRP;Polyketide' fragments or 'Type I PKS'
    if "pks" in key or "polyketide" in key:
        return "Polyketide"
    if "nrp" in key:
        return "NRP"
    if "ripp" in key or "ribosom" in key:
        return "RiPP"
    if "terpene" in key:
        return "Terpene"
    if "saccharide" in key:
        return "Saccharide"
    if "alkaloid" in key:
        return "Alkaloid"
    return "Other"


@dataclass
class MibigEntry:
    accession: str
    classes: list[str]          # canonical, de-duplicated, order-stable
    organism: str
    taxid: str | None


def _extract_raw_classes(data: dict) -> list[str]:
    """Pull biosynthetic-class strings from a MIBiG JSON, schema-tolerant.

    MIBiG 4.0 puts them at `biosynthesis.classes[].class` (values PKS / NRPS /
    ribosomal / other / terpene / saccharide). Older v2/v3 used a top-level
    `biosyn_class` list. We deliberately IGNORE `compounds[].classes` (those are
    chemical-structure classes, not biosynthetic classes).
    """
    raw: list[str] = []
    bios = data.get("biosynthesis")
    if isinstance(bios, dict):
        for c in bios.get("classes", []) or []:
            if isinstance(c, dict) and isinstance(c.get("class"), str):
                raw.append(c["class"])
    if raw:
        return raw
    # legacy fallback: explicit biosyn_class key anywhere
    legacy = data.get("biosyn_class") or data.get("cluster", {}).get("biosyn_class")
    if isinstance(legacy, list):
        raw.extend(s for s in legacy if isinstance(s, str))
    return raw


def _first_str(obj, *keys: str) -> str | None:
    """Find the first string value under any of `keys` anywhere in the JSON."""
    out: list[str] = []

    def rec(node):
        if out:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if k.lower() in keys and isinstance(v, str):
                    out.append(v)
                    return
                rec(v)
        elif isinstance(node, list):
            for v in node:
                rec(v)

    rec(obj)
    return out[0] if out else None


def parse_mibig_metadata(json_path: Path) -> MibigEntry | None:
    """Parse one MIBiG metadata JSON into a MibigEntry, or None if no class found."""
    data = json.loads(Path(json_path).read_text())
    raw_classes = _extract_raw_classes(data)
    if not raw_classes:
        return None
    classes: list[str] = []
    for rc in raw_classes:
        c = normalize_class(rc)
        if c not in classes:
            classes.append(c)
    # MIBiG files are named {accession}.json — the stem is the most reliable id
    # (nested 'accession' keys in the loci point to NCBI sequence ids, not the BGC).
    accession = Path(json_path).stem or _first_str(data, "mibig_accession", "accession")
    organism = _first_str(data, "organism_name", "organism") or ""
    taxid = _first_str(data, "ncbi_tax_id", "taxid")
    return MibigEntry(accession=accession, classes=classes, organism=organism, taxid=taxid)


def multihot(classes: Iterable[str]) -> list[int]:
    """Canonical multi-hot label vector (len == len(CANONICAL_CLASSES))."""
    vec = [0] * len(CANONICAL_CLASSES)
    for c in classes:
        if c in CLASS_TO_IDX:
            vec[CLASS_TO_IDX[c]] = 1
    return vec
