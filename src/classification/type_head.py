#!/usr/bin/env python3
"""Multi-label product-type head: RandomForest over Pfam-domain feature vectors.

This is the classifier itself — the same family of model GECCO and SanntiS use
downstream of their domain features. Multi-label (one sigmoid-like probability
per class) so hybrid PKS-NRP clusters can be both Polyketide and NRP.

Default backend is sklearn RandomForest (CPU, no extra deps, robust on sparse
high-dim domain vectors). LightGBM is an optional drop-in if installed.

The model bundles its `classes` and feature `vocab` so inference is fully
self-describing — load the .joblib and feed a region's domain hits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .domains import stack_vectors


@dataclass
class TypeHead:
    classes: list[str]
    vocab: list[str]
    binary_features: bool = False
    n_estimators: int = 400
    max_depth: int | None = None
    seed: int = 0
    _model: object = field(default=None, repr=False)

    # -- training -----------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TypeHead":
        """X: (N, |vocab|) domain features. Y: (N, |classes|) multi-hot labels."""
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.seed,
        )
        clf.fit(X, Y)
        self._model = clf
        return self

    # -- inference ----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, |classes|) probability of each class being present.

        sklearn's multilabel predict_proba yields a per-label list; we extract
        P(label==1) and guard labels that were constant in training.
        """
        if self._model is None:
            raise RuntimeError("TypeHead not fitted/loaded")
        raw = self._model.predict_proba(X)
        n = X.shape[0]
        out = np.zeros((n, len(self.classes)), dtype=np.float32)
        # raw is a list (one (N, k) array per label) for multi-output RF
        per_label = raw if isinstance(raw, list) else [raw]
        # For a multi-output RF, model.classes_ is a list of 1-D class arrays,
        # one per label (NOT estimators_, which are the trees).
        model_classes = self._model.classes_
        if not isinstance(model_classes, list):
            model_classes = [model_classes]
        for j, probs in enumerate(per_label):
            classes_j = [int(c) for c in model_classes[j]]
            if 1 in classes_j:
                out[:, j] = probs[:, classes_j.index(1)]
            else:
                # label was all-zeros in training → probability of presence is 0
                out[:, j] = 0.0
        return out

    def predict_from_hits(self, hits_per_cluster, *, binary=None) -> np.ndarray:
        """Convenience: raw domain-hit lists → (N, |classes|) probabilities."""
        b = self.binary_features if binary is None else binary
        X = stack_vectors(hits_per_cluster, self.vocab, binary=b)
        return self.predict_proba(X)

    # -- persistence --------------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classes": self.classes,
                "vocab": self.vocab,
                "binary_features": self.binary_features,
                "model": self._model,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TypeHead":
        import joblib

        d = joblib.load(path)
        head = cls(
            classes=d["classes"],
            vocab=d["vocab"],
            binary_features=d["binary_features"],
        )
        head._model = d["model"]
        return head
