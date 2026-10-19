"""Point and point+hull state policies for the shared-executor comparison.

Both policies are logistic regressions over stored or live diagnostic features
and are trained on development traces only.  They differ in exactly one thing:
whether the five diagnostic generators' credal geometry (bounds, diameter and
bound gaps) is appended to the evidence signature.

The policies return an operational state; the orchestrator maps it to the same
action space and runs the same executor, so any difference in task success is
attributable to the decision rule rather than to the execution machinery.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..types import STATE_LABELS, StateFeatures
from .credal_controller import extreme_distributions


def point_vector(features: StateFeatures) -> list[float]:
    return list(features.as_vector())


def hull_vector(features: StateFeatures, time_sensitive: bool) -> list[float]:
    gens = extreme_distributions(features, time_sensitive=time_sensitive)
    lower, upper = gens.min(axis=0), gens.max(axis=0)
    mean = gens.mean(axis=0)
    diameter = float(np.abs(gens[:, None, :] - gens[None, :, :]).sum(-1).max() / 2)
    return (
        point_vector(features)
        + [float(x) for x in lower]
        + [float(x) for x in upper]
        + [diameter]
        + [float(x) for x in mean - lower]
        + [float(x) for x in upper - mean]
    )


class PointPolicy:
    """Logistic state policy over the evidence signature, optionally with hull."""

    def __init__(self, use_hull: bool = False, seed: int = 42, class_weight=None) -> None:
        self.use_hull = use_hull
        self.class_weight = class_weight
        self.seed = seed
        self.model = None
        self.labels = list(STATE_LABELS)

    def _x(self, features: StateFeatures, time_sensitive: bool) -> list[float]:
        return hull_vector(features, time_sensitive) if self.use_hull else point_vector(features)

    def fit(self, rows: list[tuple[StateFeatures, bool, str]]) -> None:
        X = np.array([self._x(f, ts) for f, ts, _ in rows])
        y = np.array([self.labels.index(state) for _, _, state in rows])
        self.model = make_pipeline(
            StandardScaler(), LogisticRegression(
                max_iter=3000, C=1.0, random_state=self.seed, class_weight=self.class_weight
            )
        )
        self.model.fit(X, y)

    def predict(
        self,
        features: StateFeatures,
        time_sensitive: bool = False,
        **_ignored,
    ) -> str:
        if self.model is None:
            raise RuntimeError("PointPolicy is not fitted")
        x = np.array(self._x(features, time_sensitive)).reshape(1, -1)
        return self.labels[int(self.model.predict(x)[0])]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as handle:
            pickle.dump({"model": self.model, "use_hull": self.use_hull}, handle)

    @classmethod
    def load(cls, path: str) -> "PointPolicy":
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        policy = cls(use_hull=bool(data["use_hull"]))
        policy.model = data["model"]
        return policy
