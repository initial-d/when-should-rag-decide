"""Credal evidence-state controller.

This module turns the OESC evidence signature into a finitely generated
credal set over the six evidence states. Each diagnostic group contributes
one extreme distribution. The controller then makes a cautious decision from
lower/upper state probabilities plus the credal-set diameter, rather than
training a point classifier over the feature vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import STATE_LABELS, StateFeatures
from .rule_classifier import classify_state


# TV diameters lie in [0, 1], but their round-to-round ratio is formally
# unbounded when the previous diameter is nearly zero.  Bound only the value
# exposed to statistical calibrators; the raw diagnostic remains available.
# Current ESC-Bench/OESC-Real logs max out below 2.44, so this guard does not
# alter any reported example.
MAX_CALIBRATOR_CONTRACTION_RATIO = 4.0


_STATE_TO_IDX = {s: i for i, s in enumerate(STATE_LABELS)}


def bounded_contraction_ratio(value: Optional[float]) -> float:
    """Return a finite contraction-ratio feature for downstream calibrators."""

    ratio = 1.0 if value is None else float(value)
    if not np.isfinite(ratio):
        ratio = MAX_CALIBRATOR_CONTRACTION_RATIO
    return min(max(ratio, 0.0), MAX_CALIBRATOR_CONTRACTION_RATIO)


@dataclass(frozen=True)
class CredalDiagnostics:
    """Geometry of the score-induced diagnostic credal set.

    ``center`` is retained as a serialized compatibility name for the
    arithmetic mean of the listed generators. It is not an invariant centroid
    of the convex hull: duplicate or interior generators can change it.
    """

    diameter: float
    contraction_ratio: Optional[float]
    center_shift: Optional[float]
    lower: dict[str, float]
    upper: dict[str, float]
    center: dict[str, float]
    triggered_shift: bool
    fallback_state: str

    def to_dict(self) -> dict:
        return {
            "diameter": self.diameter,
            "contraction_ratio": self.contraction_ratio,
            "center_shift": self.center_shift,
            "lower": self.lower,
            "upper": self.upper,
            "center": self.center,
            "triggered_shift": self.triggered_shift,
            "fallback_state": self.fallback_state,
        }


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _normalise(scores: dict[str, float], floor: float = 1e-4) -> np.ndarray:
    vals = np.array([max(floor, float(scores.get(s, 0.0))) for s in STATE_LABELS], dtype=float)
    total = float(vals.sum())
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(STATE_LABELS), dtype=float) / len(STATE_LABELS)
    return vals / total


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance on the state simplex."""

    return float(0.5 * np.abs(p - q).sum())


def _closedness(features: StateFeatures) -> float:
    return _clamp01(
        0.34 * features.need_coverage
        + 0.32 * features.answer_support
        + 0.14 * features.critical_need_recall
        + 0.20 * getattr(features, "strict_sufficiency", 1.0)
    )


def _openness(features: StateFeatures) -> float:
    return _clamp01(1.0 - _closedness(features))


def extreme_distributions(features: StateFeatures, time_sensitive: bool = False) -> np.ndarray:
    """Build extreme distributions from the OESC diagnostic groups.

    The groups mirror the evidence diagnostics: alignment, disagreement,
    temporal freshness, ambiguity, and refinement/exhaustion.
    """

    c = _clamp01(features.need_coverage)
    s = _clamp01(features.answer_support)
    ccrit = _clamp01(features.critical_need_recall)
    strict = _clamp01(getattr(features, "strict_sufficiency", 1.0))
    ds = _clamp01(features.source_conflict)
    da = _clamp01(features.answer_disagreement)
    dc = _clamp01(features.claim_disagreement)
    f = _clamp01(features.freshness)
    amb = max(_clamp01(features.intent_ambiguity), _clamp01(getattr(features, "criterion_ambiguity", 0.0)))
    nov = _clamp01(features.retrieval_novelty)

    closed = _closedness(features)
    open_ = _openness(features)
    gap = _clamp01(1.0 - min(c, s, ccrit, strict))
    exhaustion = _clamp01((1.0 - nov) * (1.0 - max(c, s)))

    alignment = _normalise(
        {
            "sufficient": closed,
            "missing": 0.75 * gap + 0.20 * open_,
            "unanswerable": 0.55 * (1.0 - c) * (1.0 - s),
            "conflicting": 0.10 * ds,
            "underspecified": 0.10 * amb,
            "outdated": 0.10 * (1.0 - f if time_sensitive else 0.0),
        }
    )

    disagreement = _normalise(
        {
            "conflicting": 0.85 * ds * (1.0 - 0.45 * amb) + 0.35 * dc,
            "unanswerable": 0.45 * da * (1.0 - closed),
            "missing": 0.25 * open_,
            "sufficient": 0.40 * closed * (1.0 - max(ds, dc)),
            "underspecified": 0.45 * amb,
            "outdated": 0.05 * (1.0 - f if time_sensitive else 0.0),
        }
    )

    temporal = _normalise(
        {
            "outdated": (1.0 - f) if time_sensitive else 0.02,
            "sufficient": closed * (f if time_sensitive else 0.75),
            "missing": gap,
            "unanswerable": exhaustion,
            "conflicting": 0.15 * ds,
            "underspecified": 0.10 * amb,
        }
    )

    ambiguity = _normalise(
        {
            "underspecified": 0.95 * amb + 0.25 * da,
            "missing": 0.35 * open_,
            "sufficient": 0.45 * closed * (1.0 - amb),
            "conflicting": 0.20 * ds,
            "unanswerable": 0.20 * exhaustion,
            "outdated": 0.05 * (1.0 - f if time_sensitive else 0.0),
        }
    )

    refinement = _normalise(
        {
            "missing": 0.75 * gap * max(0.2, nov),
            "unanswerable": 0.80 * exhaustion + 0.25 * (1.0 - c) * (1.0 - s),
            "sufficient": 0.55 * closed * (1.0 - exhaustion),
            "conflicting": 0.20 * ds,
            "underspecified": 0.15 * amb,
            "outdated": 0.10 * (1.0 - f if time_sensitive else 0.0),
        }
    )

    return np.vstack([alignment, disagreement, temporal, ambiguity, refinement])


def separated_extreme_distributions(features: StateFeatures, time_sensitive: bool = False) -> np.ndarray:
    """Build view-separated evidence experts.

    Unlike ``extreme_distributions``, which mirrors the original aggregate OESC
    signature, these experts intentionally reuse fewer features across views:
    sufficiency, missingness, conflict, ambiguity, freshness, and exhaustion
    each form a full posterior from a narrower diagnostic family.
    """

    c = _clamp01(features.need_coverage)
    s = _clamp01(features.answer_support)
    ccrit = _clamp01(features.critical_need_recall)
    strict = _clamp01(getattr(features, "strict_sufficiency", 1.0))
    source_conf = _clamp01(features.source_conflict)
    ans_dis = _clamp01(features.answer_disagreement)
    claim_dis = _clamp01(features.claim_disagreement)
    fresh = _clamp01(features.freshness)
    amb = max(_clamp01(features.intent_ambiguity), _clamp01(getattr(features, "criterion_ambiguity", 0.0)))
    novelty = _clamp01(features.retrieval_novelty)

    closure = (c * s * ccrit * strict) ** 0.25
    missing = 1.0 - min(c, s, ccrit, strict)
    conflict = max(source_conf, ans_dis, claim_dis)
    outdated = (1.0 - fresh) if time_sensitive else 0.02 * (1.0 - fresh)
    exhausted = (1.0 - novelty) * max(0.0, 1.0 - max(c, s))

    sufficiency = _normalise(
        {
            "sufficient": 1.50 * closure,
            "missing": 0.60 * (1.0 - min(c, ccrit)),
            "conflicting": 0.05,
            "unanswerable": 0.25 * (1.0 - s),
            "underspecified": 0.05,
            "outdated": 0.05,
        }
    )
    missing_expert = _normalise(
        {
            "missing": 1.35 * missing,
            "sufficient": 0.75 * min(c, s, ccrit),
            "unanswerable": 0.35 * (1.0 - max(c, s)),
            "conflicting": 0.05,
            "underspecified": 0.05,
            "outdated": 0.05,
        }
    )
    conflict_expert = _normalise(
        {
            "conflicting": 1.50 * conflict,
            "sufficient": 0.90 * closure * (1.0 - conflict),
            "unanswerable": 0.35 * ans_dis * (1.0 - closure),
            "missing": 0.10,
            "underspecified": 0.10 * amb,
            "outdated": 0.05,
        }
    )
    ambiguity_expert = _normalise(
        {
            "underspecified": 1.50 * amb,
            "sufficient": 0.85 * closure * (1.0 - amb),
            "missing": 0.30 * missing,
            "conflicting": 0.10 * conflict,
            "unanswerable": 0.15,
            "outdated": 0.05,
        }
    )
    freshness_expert = _normalise(
        {
            "outdated": 1.60 * outdated,
            "sufficient": 0.90 * closure * (fresh if time_sensitive else 0.85),
            "missing": 0.25 * missing,
            "conflicting": 0.05,
            "unanswerable": 0.15,
            "underspecified": 0.05,
        }
    )
    exhaustion_expert = _normalise(
        {
            "unanswerable": 1.20 * exhausted + 0.40 * (1.0 - max(c, s)),
            "missing": 0.90 * missing * max(0.15, novelty),
            "sufficient": 0.75 * closure * (1.0 - exhausted),
            "conflicting": 0.05,
            "underspecified": 0.05,
            "outdated": 0.05,
        }
    )

    return np.vstack(
        [
            sufficiency,
            missing_expert,
            conflict_expert,
            ambiguity_expert,
            freshness_expert,
            exhaustion_expert,
        ]
    )


def get_extreme_distributions(
    features: StateFeatures,
    time_sensitive: bool = False,
    expert_mode: str = "controller",
) -> np.ndarray:
    if expert_mode == "controller":
        return extreme_distributions(features, time_sensitive=time_sensitive)
    if expert_mode == "separated":
        return separated_extreme_distributions(features, time_sensitive=time_sensitive)
    raise ValueError(f"unknown credal expert_mode: {expert_mode}")


def credal_diagnostics(
    features: StateFeatures,
    previous_features: Optional[StateFeatures] = None,
    time_sensitive: bool = False,
    eps: float = 1e-6,
    ratio_limit: float = 1.35,
    center_shift_limit: float = 0.25,
    thresholds: Optional[dict] = None,
    num_claims: Optional[int] = None,
    round_idx: int = 0,
    initial_only: bool = False,
    expert_mode: str = "controller",
) -> CredalDiagnostics:
    """Compute generator-mean and convex-hull diagnostic coordinates.

    The normalized generator scores are not asserted to be calibrated or to
    contain the unknown true conditional distribution.
    """

    dists = get_extreme_distributions(features, time_sensitive=time_sensitive, expert_mode=expert_mode)
    pairwise = [
        _tv_distance(dists[i], dists[j])
        for i in range(len(dists))
        for j in range(i + 1, len(dists))
    ]
    diameter = max(pairwise) if pairwise else 0.0
    center = dists.mean(axis=0)

    prev_diameter = None
    center_shift = None
    contraction_ratio = None
    if previous_features is not None:
        prev_dists = get_extreme_distributions(previous_features, time_sensitive=time_sensitive, expert_mode=expert_mode)
        prev_pairwise = [
            _tv_distance(prev_dists[i], prev_dists[j])
            for i in range(len(prev_dists))
            for j in range(i + 1, len(prev_dists))
        ]
        prev_diameter = max(prev_pairwise) if prev_pairwise else 0.0
        contraction_ratio = float(diameter / (prev_diameter + eps))
        center_shift = _tv_distance(center, prev_dists.mean(axis=0))

    lower_arr = dists.min(axis=0)
    upper_arr = dists.max(axis=0)
    fallback_state = classify_state(
        features,
        thresholds,
        num_claims=num_claims,
        time_sensitive=time_sensitive,
        round_idx=round_idx,
        initial_only=initial_only,
    )
    triggered_shift = bool(
        previous_features is not None
        and (
            (contraction_ratio is not None and contraction_ratio > ratio_limit)
            or (center_shift is not None and center_shift > center_shift_limit)
        )
    )

    return CredalDiagnostics(
        diameter=diameter,
        contraction_ratio=contraction_ratio,
        center_shift=center_shift,
        lower={s: float(lower_arr[i]) for s, i in _STATE_TO_IDX.items()},
        upper={s: float(upper_arr[i]) for s, i in _STATE_TO_IDX.items()},
        center={s: float(center[i]) for s, i in _STATE_TO_IDX.items()},
        triggered_shift=triggered_shift,
        fallback_state=fallback_state,
    )


class CredalEvidenceController:
    """Set-valued controller for OESC information actions."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.ratio_limit = float(cfg.get("ratio_limit", 1.35))
        self.center_shift_limit = float(cfg.get("center_shift_limit", 0.25))
        self.stable_diameter = float(cfg.get("stable_diameter", 0.34))
        self.accept_lower = float(cfg.get("accept_lower", 0.22))
        self.alarm_upper = float(cfg.get("alarm_upper", 0.45))
        self.diameter_penalty = float(cfg.get("diameter_penalty", 0.0))
        self.lower_weight = float(cfg.get("lower_weight", 0.35))
        self.upper_weight = float(cfg.get("upper_weight", 0.20))
        self.shift_bonus = float(cfg.get("shift_bonus", 0.08))
        self.sufficient_bias = float(cfg.get("sufficient_bias", 0.0))
        self.missing_bias = float(cfg.get("missing_bias", 0.0))
        self.conflicting_bias = float(cfg.get("conflicting_bias", 0.0))
        self.unanswerable_bias = float(cfg.get("unanswerable_bias", 0.0))
        self.underspecified_bias = float(cfg.get("underspecified_bias", 0.0))
        self.outdated_bias = float(cfg.get("outdated_bias", 0.0))
        self.expert_mode = str(cfg.get("expert_mode", "controller"))

    def _bias(self, state: str) -> float:
        return float(
            {
                "sufficient": self.sufficient_bias,
                "missing": self.missing_bias,
                "conflicting": self.conflicting_bias,
                "unanswerable": self.unanswerable_bias,
                "underspecified": self.underspecified_bias,
                "outdated": self.outdated_bias,
            }.get(state, 0.0)
        )

    def diagnose(
        self,
        features: StateFeatures,
        previous_features: Optional[StateFeatures] = None,
        time_sensitive: bool = False,
        thresholds: Optional[dict] = None,
        num_claims: Optional[int] = None,
        round_idx: int = 0,
        initial_only: bool = False,
        expert_mode: Optional[str] = None,
    ) -> CredalDiagnostics:
        return credal_diagnostics(
            features=features,
            previous_features=previous_features,
            time_sensitive=time_sensitive,
            ratio_limit=self.ratio_limit,
            center_shift_limit=self.center_shift_limit,
            thresholds=thresholds,
            num_claims=num_claims,
            round_idx=round_idx,
            initial_only=initial_only,
            expert_mode=expert_mode or self.expert_mode,
        )

    def predict(
        self,
        features: StateFeatures,
        previous_features: Optional[StateFeatures] = None,
        time_sensitive: bool = False,
        thresholds: Optional[dict] = None,
        num_claims: Optional[int] = None,
        round_idx: int = 0,
        initial_only: bool = False,
        expert_mode: Optional[str] = None,
    ) -> str:
        diag = self.diagnose(
            features=features,
            previous_features=previous_features,
            time_sensitive=time_sensitive,
            thresholds=thresholds,
            num_claims=num_claims,
            round_idx=round_idx,
            initial_only=initial_only,
            expert_mode=expert_mode,
        )

        if num_claims is not None and num_claims == 0:
            return "missing" if (initial_only and round_idx == 0) else "unanswerable"

        lower = diag.lower
        upper = diag.upper

        # High-precision terminal guards stay deterministic.
        if time_sensitive and features.freshness < (thresholds or {}).get("freshness", 0.15):
            return "outdated"
        if upper["conflicting"] >= self.alarm_upper and features.source_conflict >= 0.20:
            return "conflicting"
        criterion_ambiguity = _clamp01(getattr(features, "criterion_ambiguity", 0.0))
        if upper["underspecified"] >= self.alarm_upper and max(features.intent_ambiguity, criterion_ambiguity) >= 0.15:
            return "underspecified"

        # Diameter expansion is the credal analogue of a knowledge shift.
        if diag.triggered_shift:
            if features.source_conflict >= 0.20:
                return "conflicting"
            if time_sensitive and features.freshness < 0.35:
                return "outdated"
            if features.critical_need_recall < 0.65 or features.need_coverage < 0.45:
                return "missing"
            return diag.fallback_state

        scores: dict[str, float] = {}
        for state in STATE_LABELS:
            score = (
                diag.center[state]
                + self.lower_weight * lower[state]
                + self.upper_weight * upper[state]
                + self._bias(state)
                - self.diameter_penalty * diag.diameter
            )
            if diag.triggered_shift and state in {"missing", "conflicting", "outdated"}:
                score += self.shift_bonus
            scores[state] = score

        # Preserve the closure condition: sufficiency must be supported by all
        # extreme views unless the configured lower threshold is intentionally
        # relaxed by tuning.
        if (
            lower["sufficient"] < self.accept_lower
            or features.critical_need_recall < 0.60
            or getattr(features, "strict_sufficiency", 1.0) < 0.45
        ):
            scores["sufficient"] -= 1.0
        if diag.diameter > self.stable_diameter:
            scores["sufficient"] -= self.diameter_penalty + 0.10
        if features.retrieval_novelty > 0.25:
            scores["unanswerable"] -= 0.05

        return max(scores, key=scores.get)


def credal_vector(
    features: StateFeatures,
    previous_features: Optional[StateFeatures] = None,
    time_sensitive: bool = False,
    expert_mode: str = "controller",
) -> list[float]:
    """Vectorize a round as a hybrid point--set diagnostic summary.

    The vector includes raw features, the listed-generator mean, coordinate
    lower/upper masses, and diameter dynamics.  It is not a complete encoding
    of the convex hull: distinct hulls can share the compact set functionals,
    and the listed mean is not invariant to duplicate or interior generators.
    """

    diag = credal_diagnostics(
        features=features,
        previous_features=previous_features,
        time_sensitive=time_sensitive,
        expert_mode=expert_mode,
    )
    contraction_ratio = bounded_contraction_ratio(diag.contraction_ratio)

    vector = (
        features.as_vector()
        + [diag.center[s] for s in STATE_LABELS]
        + [diag.lower[s] for s in STATE_LABELS]
        + [diag.upper[s] for s in STATE_LABELS]
        + [
            diag.diameter,
            contraction_ratio,
            diag.center_shift if diag.center_shift is not None else 0.0,
            1.0 if diag.triggered_shift else 0.0,
        ]
    )
    if not np.all(np.isfinite(np.asarray(vector, dtype=float))):
        raise ValueError("credal_vector produced a non-finite feature")
    return vector


class CredalMemoryController:
    """Nearest-neighbour controller over credal evidence geometry.

    The memory is transparent: each exemplar is a normalized credal vector and
    a state label. It replaces opaque feature routers with a case-based
    controller whose distance space is defined by credal-set geometry.
    """

    def __init__(self, k: int = 1):
        self.k = int(k)
        self.labels: list[str] = []
        self.mean: list[float] = []
        self.scale: list[float] = []
        self.vectors: list[list[float]] = []

    def fit(self, vectors: list[list[float]], labels: list[str]) -> None:
        if not vectors:
            raise ValueError("Cannot fit CredalMemoryController with no vectors")
        arr = np.asarray(vectors, dtype=float)
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale < 1e-8] = 1.0
        norm = (arr - mean) / scale
        self.mean = mean.tolist()
        self.scale = scale.tolist()
        self.vectors = norm.tolist()
        self.labels = list(labels)

    def _normalise_one(self, vector: list[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=float)
        return (arr - np.asarray(self.mean, dtype=float)) / np.asarray(self.scale, dtype=float)

    def predict_vector(self, vector: list[float]) -> str:
        if not self.vectors:
            raise RuntimeError("Credal memory is empty; call fit() or load() first")
        x = self._normalise_one(vector)
        mem = np.asarray(self.vectors, dtype=float)
        dists = np.linalg.norm(mem - x, axis=1)
        order = np.argsort(dists)[: max(1, min(self.k, len(dists)))]
        votes: dict[str, float] = {}
        for idx in order:
            label = self.labels[int(idx)]
            votes[label] = votes.get(label, 0.0) + 1.0 / (float(dists[int(idx)]) + 1e-6)
        return max(votes, key=votes.get)

    def predict(
        self,
        features: StateFeatures,
        previous_features: Optional[StateFeatures] = None,
        time_sensitive: bool = False,
        num_claims: Optional[int] = None,
        round_idx: int = 0,
        initial_only: bool = False,
        expert_mode: str = "controller",
    ) -> str:
        if num_claims is not None and num_claims == 0:
            return "missing" if (initial_only and round_idx == 0) else "unanswerable"
        return self.predict_vector(
            credal_vector(
                features=features,
                previous_features=previous_features,
                time_sensitive=time_sensitive,
                expert_mode=expert_mode,
            )
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "k": self.k,
                    "labels": self.labels,
                    "mean": self.mean,
                    "scale": self.scale,
                    "vectors": self.vectors,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: str | Path) -> None:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.k = int(data.get("k", 1))
        self.labels = list(data["labels"])
        self.mean = [float(x) for x in data["mean"]]
        self.scale = [float(x) if math.isfinite(float(x)) and float(x) > 0 else 1.0 for x in data["scale"]]
        self.vectors = [[float(x) for x in row] for row in data["vectors"]]


class CredalCalibratorController:
    """Train/test calibrated controller over a point--set diagnostic vector.

    The serialized model is produced by ``scripts/25_train_credal_memory.py``.
    It receives the same credal vector used by the memory controller: raw OESC
    signature, lower/upper/center probabilities, diameter, contraction ratio,
    and center shift.
    """

    def __init__(self) -> None:
        self.model = None
        self.label_names = list(STATE_LABELS)
        self.metadata: dict = {}

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.label_names = list(data.get("label_names") or STATE_LABELS)
        self.metadata = {k: v for k, v in data.items() if k not in {"model"}}

    def predict(
        self,
        features: StateFeatures,
        previous_features: Optional[StateFeatures] = None,
        time_sensitive: bool = False,
        num_claims: Optional[int] = None,
        round_idx: int = 0,
        initial_only: bool = False,
        expert_mode: str = "controller",
    ) -> str:
        if self.model is None:
            raise RuntimeError("Credal calibrator is empty; call load() first")
        vec = np.asarray(
            credal_vector(
                features=features,
                previous_features=previous_features,
                time_sensitive=time_sensitive,
                expert_mode=expert_mode,
            ),
            dtype=float,
        ).reshape(1, -1)
        pred = int(self.model.predict(vec)[0])
        return self.label_names[pred]
