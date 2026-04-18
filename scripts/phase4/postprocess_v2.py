#!/usr/bin/env python3
"""Second-stage region decoder for 9-genome postprocessing experiments.

This module keeps the original threshold+merge evaluator stable while adding a
more explicit region decoder with:

- dual-threshold hysteresis (`start_threshold`, `extend_threshold`)
- minimum region support in windows (`min_support_windows`)
- optional valley-based splitting inside a region (`valley_threshold`)
- optional hard length cap (`length_cap`)

The design goal is traceability rather than maximal cleverness. Each region is
still derived from concrete window spans, so the resulting calls can be audited
against the original per-window CSVs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowScore:
    contig: str
    start: int
    end: int
    score: float


def split_regions_by_length_cap(
    regions: list[tuple[str, list[WindowScore]]],
    length_cap: int | None,
) -> list[tuple[str, int, int]]:
    if not length_cap:
        return [
            (contig, windows[0].start, windows[-1].end)
            for contig, windows in regions
            if windows
        ]

    out: list[tuple[str, int, int]] = []
    for contig, windows in regions:
        if not windows:
            continue
        current_chunk = [windows[0]]
        for window in windows[1:]:
            chunk_start = current_chunk[0].start
            proposed_end = window.end
            if proposed_end - chunk_start > length_cap:
                out.append((contig, current_chunk[0].start, current_chunk[-1].end))
                current_chunk = [window]
            else:
                current_chunk.append(window)
        if current_chunk:
            out.append((contig, current_chunk[0].start, current_chunk[-1].end))
    return out


def split_windows_on_valleys(
    windows: list[WindowScore],
    valley_threshold: float | None,
    valley_min_windows: int,
) -> list[list[WindowScore]]:
    """Split a seeded region on sustained weak-score valleys.

    Valley windows are removed from the emitted segments. This is deliberate:
    they represent low-confidence bridges that we do not want to force into a
    final BGC call just because strong windows exist on both sides.
    """
    if not windows:
        return []
    if valley_threshold is None:
        return [windows]

    segments: list[list[WindowScore]] = []
    current: list[WindowScore] = []
    index = 0
    while index < len(windows):
        if windows[index].score >= valley_threshold:
            current.append(windows[index])
            index += 1
            continue

        run_end = index
        while run_end < len(windows) and windows[run_end].score < valley_threshold:
            run_end += 1
        valley_run = windows[index:run_end]
        if len(valley_run) >= valley_min_windows:
            if current:
                segments.append(current)
                current = []
        else:
            current.extend(valley_run)
        index = run_end

    if current:
        segments.append(current)
    return segments


def decode_regions_hysteresis(
    windows: list[WindowScore],
    start_threshold: float,
    extend_threshold: float,
    max_gap: int,
    min_support_windows: int = 1,
    length_cap: int | None = None,
    valley_threshold: float | None = None,
    valley_min_windows: int = 1,
) -> list[tuple[str, int, int]]:
    """Decode final regions with a dual-threshold hysteresis policy."""
    if extend_threshold > start_threshold:
        raise ValueError("extend_threshold must be <= start_threshold")
    if min_support_windows < 1:
        raise ValueError("min_support_windows must be >= 1")
    if valley_min_windows < 1:
        raise ValueError("valley_min_windows must be >= 1")

    windows = sorted(windows, key=lambda row: (row.contig, row.start))
    seeded_regions: list[tuple[str, list[WindowScore]]] = []
    current: list[WindowScore] = []
    seeded = False

    for window in windows:
        if not current:
            if window.score >= start_threshold:
                current = [window]
                seeded = True
            continue

        last = current[-1]
        contiguous = (
            window.contig == last.contig and
            window.start <= last.end + max_gap
        )
        if not contiguous:
            if seeded and len(current) >= min_support_windows:
                seeded_regions.append((current[0].contig, current))
            current = [window] if window.score >= start_threshold else []
            seeded = bool(current)
            continue

        if window.score >= extend_threshold:
            current.append(window)
            if window.score >= start_threshold:
                seeded = True
            continue

        if seeded and len(current) >= min_support_windows:
            seeded_regions.append((current[0].contig, current))
        current = [window] if window.score >= start_threshold else []
        seeded = bool(current)

    if current and seeded and len(current) >= min_support_windows:
        seeded_regions.append((current[0].contig, current))

    split_regions: list[tuple[str, list[WindowScore]]] = []
    for contig, region_windows in seeded_regions:
        for segment in split_windows_on_valleys(
            region_windows,
            valley_threshold=valley_threshold,
            valley_min_windows=valley_min_windows,
        ):
            if len(segment) >= min_support_windows:
                split_regions.append((contig, segment))

    return split_regions_by_length_cap(split_regions, length_cap)
