"""Learned state classifier using sklearn LogisticRegression.

Trains on (StateFeatures -> gold_state) using 5-fold CV.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

from ..types import StateFeatures, STATE_LABELS


class LearnedClassifier:
    """Wraps sklearn LogisticRegression for state classification."""

    def __init__(self, backend: str = "logistic", random_state: int = 42):
        self.backend = backend
        self.random_state = random_state
        self.model = None
        self.label_names: list[str] = list(STATE_LABELS)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
            C=1.0,
        )
        self.model.fit(X, y)

    def predict(self, features: StateFeatures) -> str:
        """Predict the state from a single 8-dim feature vector."""
        if self.model is None:
            raise RuntimeError("Model not trained; call fit() first")
        x = np.array(features.as_vector()).reshape(1, -1)
        pred_idx = self.model.predict(x)[0]
        return self.label_names[int(pred_idx)]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained; call fit() first")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump({"model": self.model, "labels": self.label_names, "backend": self.backend}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.label_names = data["labels"]
        self.backend = data["backend"]


def train_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    backend: str = "logistic",
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train with k-fold CV and return per-fold metrics + the final model.

    Returns:
        {
            "model": fitted LearnedClassifier,
            "cv_predictions": array of predictions (same order as input),
            "report": classification report dict,
            "per_fold_acc": list of per-fold accuracies,
            "mean_acc": float,
        }
    """
    clf = LearnedClassifier(backend=backend, random_state=random_state)
    clf.label_names = label_names

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Collect per-fold accuracy
    fold_accs = []
    y_pred_cv = np.zeros_like(y)

    for train_idx, test_idx in skf.split(X, y):
        fold_clf = LearnedClassifier(backend=backend, random_state=random_state)
        fold_clf.label_names = label_names
        fold_clf.fit(X[train_idx], y[train_idx])
        preds = fold_clf.predict_batch(X[test_idx])
        y_pred_cv[test_idx] = preds
        fold_acc = float(np.mean(preds == y[test_idx]))
        fold_accs.append(fold_acc)

    # Final model trained on ALL data
    clf.fit(X, y)

    report = classification_report(
        y, y_pred_cv, target_names=label_names, output_dict=True, zero_division=0
    )

    return {
        "model": clf,
        "cv_predictions": y_pred_cv,
        "report": report,
        "per_fold_acc": fold_accs,
        "mean_acc": float(np.mean(fold_accs)),
    }
