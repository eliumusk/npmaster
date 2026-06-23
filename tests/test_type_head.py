"""Tests for the domain-based product-type head.

The end-to-end test uses a *synthetic* domain dataset (no Pfam/MIBiG needed)
where each class carries signature domains — PKS gets KS/AT, NRP gets C/A, both
share a carrier domain (as in reality). It asserts the RandomForest head
separates the classes and, specifically, does NOT confuse PKS with NRP — the
exact failure mode that motivated this rebuild. Real MIBiG numbers come from the
server runbook; this validates the machinery + the modeling premise locally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classification.domains import build_vocab, domain_vector, stack_vectors
from classification.mibig import multihot, normalize_class, CANONICAL_CLASSES
from classification.metrics import multilabel_report, single_label_confusion
from classification.splits import group_split, identity_families
from classification.type_head import TypeHead


# ---- signature domains per class (PKS vs NRP share the carrier PF00550) -----
SIG = {
    "Alkaloid":   ["ALK1", "ALK2"],
    "Terpene":    ["PF03936", "TERP2"],          # terpene synthase
    "NRP":        ["PF00668", "PF00501", "PF00550"],  # C, A, PCP(carrier)
    "Polyketide": ["PF00109", "PF00698", "PF00550"],  # KS, AT, ACP(carrier)
    "RiPP":       ["RIPP1", "RIPP2"],
    "Saccharide": ["PF00201", "SUG2"],           # glycosyltransferase
    "Other":      ["OTH1", "OTH2"],
}
BACKGROUND = [f"HK{i}" for i in range(8)]  # housekeeping noise shared by all


def _make_cluster(rng, classes):
    """Build one cluster's domain-hit list for the given (multi-label) classes."""
    hits = []
    for c in classes:
        for d in SIG[c]:
            # PKS/NRP megasynthases have multiple catalytic modules → repeat
            reps = rng.integers(1, 5) if d.startswith("PF00") else 1
            hits += [d] * int(reps)
    hits += list(rng.choice(BACKGROUND, size=int(rng.integers(2, 5)), replace=True))
    rng.shuffle(hits)
    return hits


def _make_dataset(n_per_class=70, n_hybrid=40, seed=0):
    rng = np.random.default_rng(seed)
    entries = []
    fam = {}
    fid = 0
    single = ["Alkaloid", "Terpene", "NRP", "Polyketide", "RiPP", "Saccharide", "Other"]
    for c in single:
        for k in range(n_per_class):
            cid = f"{c}_{k}"
            entries.append({"id": cid, "classes": [c],
                            "label": multihot([c]), "hits": _make_cluster(rng, [c])})
            # families of ~5 near-identical clusters (homology groups)
            fam[cid] = f"fam{fid // 5}"
            fid += 1
    for k in range(n_hybrid):  # hybrid PKS-NRP, multi-label
        cid = f"hybrid_{k}"
        entries.append({"id": cid, "classes": ["Polyketide", "NRP"],
                        "label": multihot(["Polyketide", "NRP"]),
                        "hits": _make_cluster(rng, ["Polyketide", "NRP"])})
        fam[cid] = f"fam{fid // 5}"
        fid += 1
    return entries, fam


# --------------------------------------------------------------------------- #
# unit tests
# --------------------------------------------------------------------------- #

def test_normalize_class_pks_is_polyketide():
    assert normalize_class("T1PKS") == "Polyketide"
    assert normalize_class("transAT-PKS") == "Polyketide"
    assert normalize_class("NRPS") == "NRP"
    assert normalize_class("RiPP-like") == "RiPP"
    assert normalize_class("something-weird") == "Other"


def test_domain_vector_counts_and_binary():
    vocab = ["A", "B", "C"]
    idx = {d: i for i, d in enumerate(vocab)}
    v = domain_vector(["A", "A", "C", "Z"], idx)
    assert list(v) == [2.0, 0.0, 1.0]            # Z ignored, A counted twice
    vb = domain_vector(["A", "A", "C"], idx, binary=True)
    assert list(vb) == [1.0, 0.0, 1.0]


def test_build_vocab_min_count():
    hits = [["A", "B"], ["A", "C"], ["A"], ["B"]]
    assert build_vocab(hits, min_count=2) == ["A", "B"]  # C appears once → dropped


def test_group_split_is_family_disjoint():
    ids = [f"x{i}" for i in range(100)]
    fam = {i: f"f{int(i[1:]) // 4}" for i in ids}
    sp = group_split(ids, fam, fractions=(0.6, 0.2, 0.2), seed=1)
    fam_of = lambda split: {fam[i] for i in sp[split]}
    assert fam_of("train") & fam_of("val") == set()
    assert fam_of("train") & fam_of("test") == set()
    assert fam_of("val") & fam_of("test") == set()
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == 100


# --------------------------------------------------------------------------- #
# end-to-end: does it separate classes and avoid PKS→NRP?
# --------------------------------------------------------------------------- #

def _train_eval(seed=0):
    entries, fam = _make_dataset(seed=seed)
    by_id = {e["id"]: e for e in entries}
    ids = list(by_id)
    sp = group_split(ids, fam, fractions=(0.7, 0.15, 0.15), seed=seed)

    vocab = build_vocab([by_id[i]["hits"] for i in sp["train"]], min_count=2)

    def xy(split_ids):
        X = stack_vectors([by_id[i]["hits"] for i in split_ids], vocab)
        Y = np.array([by_id[i]["label"] for i in split_ids], dtype=int)
        return X, Y

    X_tr, Y_tr = xy(sp["train"])
    X_te, Y_te = xy(sp["test"])
    head = TypeHead(classes=CANONICAL_CLASSES, vocab=vocab,
                    n_estimators=200, seed=seed).fit(X_tr, Y_tr)
    P = head.predict_proba(X_te)
    return Y_te, P


def test_end_to_end_macro_f1_high():
    Y, P = _train_eval()
    rep = multilabel_report(Y, P, CANONICAL_CLASSES)
    assert rep["macro_f1"] >= 0.8, rep


def test_pks_not_confused_with_nrp():
    Y, P = _train_eval()
    conf = single_label_confusion(Y, P, CANONICAL_CLASSES)
    classes = conf["classes"]
    mat = np.array(conf["matrix"])
    pks, nrp = classes.index("Polyketide"), classes.index("NRP")
    # diagonal dominates its row for both PKS and NRP
    assert mat[pks, pks] >= mat[pks].sum() - mat[pks, pks], conf
    assert mat[nrp, nrp] >= mat[nrp].sum() - mat[nrp, nrp], conf
    # the specific failure mode: PKS predicted as NRP must be a small minority
    if mat[pks].sum() > 0:
        assert mat[pks, nrp] <= 0.2 * mat[pks].sum(), conf


def test_save_load_roundtrip(tmp_path):
    entries, fam = _make_dataset(seed=2)
    by_id = {e["id"]: e for e in entries}
    ids = list(by_id)[:200]
    vocab = build_vocab([by_id[i]["hits"] for i in ids], min_count=2)
    X = stack_vectors([by_id[i]["hits"] for i in ids], vocab)
    Y = np.array([by_id[i]["label"] for i in ids], dtype=int)
    head = TypeHead(classes=CANONICAL_CLASSES, vocab=vocab, n_estimators=50).fit(X, Y)
    p = tmp_path / "m.joblib"
    head.save(p)
    head2 = TypeHead.load(p)
    assert head2.classes == CANONICAL_CLASSES
    assert head2.vocab == vocab
    np.testing.assert_allclose(head.predict_proba(X), head2.predict_proba(X))
