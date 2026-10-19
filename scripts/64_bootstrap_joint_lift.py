"""Paired question-level bootstrap CI for task-success differences between two runs.

Each run is given as ``results_dir:method``; the difference reported is B - A.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.evaluation.policy_metrics import per_state_task_success  # noqa: E402
from oesc_rag.evaluation.serialization import load_results  # noqa: E402


def _per_sample_success(spec: str, gold: dict) -> dict[str, float]:
    results_dir, method = spec.rsplit(":", 1)
    path = Path(results_dir) / f"{method}_results.json"
    out = {}
    for result in load_results(str(path)):
        if result.sample_id not in gold:
            continue
        per_state = per_state_task_success([result], gold)
        out[result.sample_id] = next(iter(per_state.values()))["task_success"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmark/samples_full.json")
    parser.add_argument("--a", required=True, help="results_dir:method (reference)")
    parser.add_argument("--b", required=True, help="results_dir:method (compared)")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    gold = {sample.id: sample for sample in load_benchmark(args.benchmark)}
    a = _per_sample_success(args.a, gold)
    b = _per_sample_success(args.b, gold)
    ids = sorted(set(a) & set(b))
    if not ids:
        sys.exit("[error] no common sample IDs")

    diff = np.array([b[i] - a[i] for i in ids])
    rng = np.random.default_rng(args.seed)
    boots = diff[rng.integers(0, len(ids), size=(args.resamples, len(ids)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mean_a = np.mean([a[i] for i in ids])
    mean_b = np.mean([b[i] for i in ids])
    print(
        f"n={len(ids)}  A={mean_a:.3f}  B={mean_b:.3f}  "
        f"B-A={diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
    )


if __name__ == "__main__":
    main()
