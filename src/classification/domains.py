#!/usr/bin/env python3
"""Protein-domain feature extraction for the BGC product-type classifier.

The biosynthetic class of a cluster is determined by which catalytic/carrier
domains its proteins carry and how they are arranged (KS/AT+ACP → polyketide;
C/A+PCP → NRP). This module turns a cluster into a fixed-length Pfam-domain
feature vector — the same representation GECCO and SanntiS feed to a RandomForest.

Two entry points produce the per-cluster ordered list of domain accessions:

  - `proteins_from_genbank`  : training, from a MIBiG .gbk (uses curated CDS).
  - `proteins_from_nucleotide`: inference, from a detected region's DNA
                                 (pyrodigal gene calls → same downstream path).

`scan_domains` (pyhmmer over a Pfam-A HMM DB) maps proteins → domain hits, and
`domain_vector` projects hits onto a fixed vocabulary. The vocabulary/vector
logic is pure (no bio deps) so it is unit-testable without HMM databases.

Heavy deps (pyhmmer, pyrodigal, Bio) are imported lazily inside the functions
that need them; importing this module is cheap.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Pure vocabulary / vector logic (testable without HMM/Pfam)
# ---------------------------------------------------------------------------


def build_vocab(all_hits: Iterable[Sequence[str]], min_count: int = 3) -> list[str]:
    """Build a sorted domain vocabulary from training domain hits.

    `all_hits` is one sequence of domain accessions per cluster. A domain is
    kept if it appears in at least `min_count` clusters (document-frequency),
    which controls dimensionality and drops singletons that can't generalize.
    """
    df: Counter[str] = Counter()
    for hits in all_hits:
        for dom in set(hits):
            df[dom] += 1
    vocab = sorted(d for d, n in df.items() if n >= min_count)
    return vocab


def domain_vector(
    hits: Sequence[str], vocab_index: dict[str, int], *, binary: bool = False
) -> np.ndarray:
    """Project one cluster's domain hits onto the fixed vocab.

    binary=False → counts (how many copies of each domain, captures e.g. the
    multiple KS modules of a large PKS); binary=True → presence/absence.
    Domains outside the vocab are ignored.
    """
    vec = np.zeros(len(vocab_index), dtype=np.float32)
    for dom in hits:
        j = vocab_index.get(dom)
        if j is not None:
            vec[j] += 1.0
    if binary:
        vec = (vec > 0).astype(np.float32)
    return vec


def stack_vectors(
    hits_per_cluster: Iterable[Sequence[str]],
    vocab: Sequence[str],
    *,
    binary: bool = False,
) -> np.ndarray:
    """(N_clusters, |vocab|) feature matrix."""
    index = {d: i for i, d in enumerate(vocab)}
    rows = [domain_vector(h, index, binary=binary) for h in hits_per_cluster]
    return np.stack(rows, axis=0) if rows else np.zeros((0, len(vocab)), np.float32)


# ---------------------------------------------------------------------------
# Protein extraction (lazy bio deps)
# ---------------------------------------------------------------------------


def proteins_from_genbank(gbk_path: Path) -> list[str]:
    """Extract CDS amino-acid translations from a (MIBiG) GenBank file.

    Falls back to translating the CDS nucleotides when no /translation qualifier
    is present. Returns proteins in record/genomic order.
    """
    from Bio import SeqIO  # lazy

    proteins: list[str] = []
    try:
        records = list(SeqIO.parse(str(gbk_path), "genbank"))
    except Exception:
        # malformed/truncated GenBank (e.g. a bad CDS join) → skip the file
        return proteins
    for rec in records:
        for feat in rec.features:
            if feat.type != "CDS":
                continue
            tr = feat.qualifiers.get("translation")
            if tr:
                proteins.append(str(tr[0]).replace("*", "").strip())
            else:
                try:
                    nt = feat.extract(rec.seq)
                    aa = str(nt.translate(table=11, to_stop=True))
                    if aa:
                        proteins.append(aa)
                except Exception:
                    continue
    return [p for p in proteins if p]


def proteins_from_nucleotide(seq: str, *, meta: bool = True) -> list[str]:
    """Call genes on a nucleotide region (pyrodigal) → protein translations.

    Used at inference time on detected regions so train/inference share the same
    gene-calling convention. `meta=True` uses metagenomic mode (no per-genome
    training), appropriate for arbitrary contigs/regions.
    """
    import pyrodigal  # lazy

    orf_finder = pyrodigal.GeneFinder(meta=meta)
    proteins: list[str] = []
    for gene in orf_finder.find_genes(seq.encode() if isinstance(seq, str) else seq):
        proteins.append(gene.translate().replace("*", ""))
    return proteins


# ---------------------------------------------------------------------------
# HMM scanning (lazy pyhmmer)
# ---------------------------------------------------------------------------


def scan_domains(
    proteins: Sequence[str],
    hmm_db: Path,
    *,
    bit_cutoffs: str | None = "gathering",
    e_value: float = 1e-5,
) -> list[str]:
    """Run pyhmmer hmmsearch of `proteins` against an HMM DB → domain accessions.

    Returns the list of domain accessions hit across all proteins (with repeats,
    so `domain_vector` can count copies). `bit_cutoffs="gathering"` uses each
    profile's curated GA threshold (recommended for Pfam-A); set to None to use
    the `e_value` cutoff instead.
    """
    import pyhmmer  # lazy

    alphabet = pyhmmer.easel.Alphabet.amino()
    seqs = [
        pyhmmer.easel.TextSequence(name=f"p{i}".encode(), sequence=p).digitize(alphabet)
        for i, p in enumerate(proteins)
        if p
    ]
    if not seqs:
        return []

    hits_out: list[str] = []
    with pyhmmer.plan7.HMMFile(str(hmm_db)) as hmm_file:
        kwargs = {"bit_cutoffs": bit_cutoffs} if bit_cutoffs else {"E": e_value}
        for top_hits in pyhmmer.hmmsearch(hmm_file, seqs, **kwargs):
            acc = _query_acc(top_hits.query)
            for hit in top_hits:
                if hit.included:
                    hits_out.append(acc)
    return hits_out


def _query_acc(query) -> str:
    acc = getattr(query, "accession", None) or getattr(query, "name", None)
    acc = acc.decode() if isinstance(acc, bytes) else str(acc)
    return acc.split(".")[0]  # strip Pfam version (PF00109.27 → PF00109)


def scan_proteins_batch(
    named_proteins: Sequence[tuple[str, str]],
    hmm_db: Path,
    *,
    bit_cutoffs: str | None = "gathering",
    e_value: float = 1e-5,
) -> dict[str, list[str]]:
    """Scan MANY proteins against a (large) HMM DB in ONE streaming pass.

    `scan_domains` re-reads the whole HMM file per call, which is fine for a tiny
    DB but catastrophic for full Pfam-A (~20k profiles) over thousands of BGCs.
    This loads each HMM once and searches the entire in-memory protein database,
    returning seq_name → list of domain accessions hit.

    Names must be unique and non-empty. Use `cpus=0` to let pyhmmer use all cores.
    """
    import pyhmmer  # lazy

    alphabet = pyhmmer.easel.Alphabet.amino()
    seqs = [
        pyhmmer.easel.TextSequence(name=name.encode(), sequence=p).digitize(alphabet)
        for name, p in named_proteins
        if p
    ]
    hits_by_name: dict[str, list[str]] = defaultdict(list)
    if not seqs:
        return hits_by_name
    with pyhmmer.plan7.HMMFile(str(hmm_db)) as hmm_file:
        kwargs = {"bit_cutoffs": bit_cutoffs} if bit_cutoffs else {"E": e_value}
        for top_hits in pyhmmer.hmmsearch(hmm_file, seqs, cpus=0, **kwargs):
            acc = _query_acc(top_hits.query)
            for hit in top_hits:
                if hit.included:
                    name = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
                    hits_by_name[name].append(acc)
    return hits_by_name
