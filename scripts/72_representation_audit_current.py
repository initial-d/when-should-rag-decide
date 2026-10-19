"""Re-run the representation comparison on the current backbones, offline.

Every quantity is rebuilt from stored ``StateFeatures``: the diagnostic
generators are deterministic functions of the feature vector, so point pools,
hull coordinates and canonical vertices can be compared without new model
calls.  Folds are grouped by source question, and the classifier is fixed
(balanced logistic) so that differences come from the representation.

Reports action accuracy (state prediction mapped through the action table) and
the gold action loss of a mean-risk versus a robust-hull rule under one
asymmetric loss.  No API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oesc_rag.controller.credal_controller import (  # noqa: E402
    extreme_distributions,
    separated_extreme_distributions,
)
from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.evaluation.serialization import load_results  # noqa: E402
from oesc_rag.types import STATE_LABELS, STATE_TO_ACTION  # noqa: E402

STATES = list(STATE_LABELS)
IDX = {s: i for i, s in enumerate(STATES)}

TRACES = {
    "DeepSeek-V4.1-Flash": "data/results/valid_deepseek41flash/oesc_results.json",
    "Qwen3.8-Flash": "data/results/valid_qwen38flash/oesc_results.json",
}


def generators(features, time_sensitive, mode):
    """mode 'separated' -> six experts (paper's audit); 'aggregate' -> five views."""
    if mode == "separated":
        return separated_extreme_distributions(features, time_sensitive=time_sensitive)
    return extreme_distributions(features, time_sensitive=time_sensitive)


def representations(features, ts, mode):
    """All representations derived from one observation."""
    dists = generators(features, ts, mode)
    n_gen = dists.shape[0]
    flat = features.as_vector()
    lower, upper = dists.min(axis=0), dists.max(axis=0)
    mean = dists.mean(axis=0)
    diameter = float(np.abs(dists[:, None, :] - dists[None, :, :]).sum(-1).max() / 2)
    return {
        "signature": np.array(flat, dtype=float),
        "uniform_linear": mean,
        "hull": np.concatenate([lower, upper, [diameter]]),
        "mean_hull": np.concatenate([mean, lower, upper, [diameter]]),
        "vertices": dists.reshape(-1),
        "dists": dists,
    }


def fit_predict(X, y, groups, seed=17):
    n_splits = min(5, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in cv.split(X, y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed),
        )
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def asymmetric_loss():
    """loss[gold, predicted]; answering a non-answerable case is worst."""
    loss = np.full((len(STATES), len(STATES)), 1.0)
    for i, gold in enumerate(STATES):
        for j, pred in enumerate(STATES):
            if gold == pred:
                loss[i, j] = 0.0
            elif pred == "answer":
                loss[i, j] = 3.0
            elif pred == "abstain":
                loss[i, j] = 2.0
            elif pred in ("unanswerable", "outdated"):
                loss[i, j] = 0.2
    return loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="data/benchmark/valid_test.json")
    p.add_argument("--mode", choices=["separated", "aggregate"], default="separated")
    p.add_argument("--out", default="data/results/representation_audit_current.json")
    args = p.parse_args()

    gold = {s.id: s for s in load_benchmark(args.benchmark)}
    report = {}
    for backbone, path in TRACES.items():
        rows, seen = [], set()
        for result in load_results(path):
            sample = gold.get(result.sample_id)
            if sample is None or not result.rounds or result.sample_id in seen:
                continue
            seen.add(result.sample_id)
            rows.append((
                result.rounds[0].features,
                bool((result.metadata or {}).get("time_sensitive")),
                sample.gold_state,
                sample.source_qid,
            ))
        y = np.array([IDX[r[2]] for r in rows])
        groups = np.array([r[3] for r in rows])
        reps = [representations(f, ts, args.mode) for f, ts, _, _ in rows]
        result = {"n": len(rows), "states": dict(Counter(r[2] for r in rows))}
        print(f"\n{backbone}: n={len(rows)}  states={result['states']}")

        for name in ("signature", "uniform_linear", "hull", "mean_hull", "vertices"):
            X = np.array([r[name] for r in reps])
            pred = fit_predict(X, y, groups)
            acc = float(np.mean(pred == y))
            action_acc = float(np.mean([STATE_TO_ACTION[STATES[p]] == STATE_TO_ACTION[STATES[t]]
                                        for p, t in zip(pred, y)]))
            result[name] = {"state_accuracy": acc, "action_accuracy": action_acc,
                            "dim": int(X.shape[1])}
            print(f"  {name:15} dim={X.shape[1]:3}  state-acc={acc:.3f}  action-acc={action_acc:.3f}")

        # Mean risk vs worst-case risk over the same generators.
        loss = asymmetric_loss()
        dists = np.array([r["dists"] for r in reps])
        mean_risk = dists.mean(axis=1) @ loss
        lower, upper = dists.min(axis=1), dists.max(axis=1)
        robust_risk = np.maximum(lower @ loss, upper @ loss)
        mean_action = [int(np.argmin(r)) for r in mean_risk]
        robust_action = [int(np.argmin(r)) for r in robust_risk]
        gold_loss = {
            "mean": float(np.mean([loss[t, a] for t, a in zip(y, mean_action)])),
            "robust": float(np.mean([loss[t, a] for t, a in zip(y, robust_action)])),
            "action_disagreement": int(np.sum(np.array(mean_action) != np.array(robust_action))),
            "mean_actions": dict(Counter(STATES[i] for i in mean_action)),
            "robust_actions": dict(Counter(STATES[i] for i in robust_action)),
        }
        result["risk"] = gold_loss
        print(f"  mean-risk gold loss={gold_loss['mean']:.3f}  "
              f"robust={gold_loss['robust']:.3f}  delta={gold_loss['robust']-gold_loss['mean']:+.3f}"
              f"  disagreeing actions={gold_loss['action_disagreement']}")
        report[backbone] = result

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
