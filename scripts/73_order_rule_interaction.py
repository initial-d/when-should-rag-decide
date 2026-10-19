"""Paired bootstrap for decision-rule, execution-order and interaction effects.

Reads the confirmation-set result directories and reports, per backbone, the
task-success of each (order, rule) cell, every pairwise rule contrast within an
order, the order effect within each rule, and the order x rule interaction
``(rf_b - df_b) - (rf_a - df_a)``.  Resampling is over questions, jointly for
all cells, so contrasts share resamples.  Optional extra runs (e.g. adaptive
gates) are compared against retrieve-first point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.evaluation.policy_metrics import per_state_task_success  # noqa: E402
from oesc_rag.evaluation.serialization import load_results  # noqa: E402

CELLS = {
    ("df", "rule"): ("confirm_{b}", "oesc"),
    ("df", "point"): ("confirm_{b}", "point"),
    ("df", "hull"): ("confirm_{b}", "point_hull"),
    ("rf", "rule"): ("confirm_{b}_rf", "oesc"),
    ("rf", "point"): ("confirm_{b}_rf", "point"),
    ("rf", "hull"): ("confirm_{b}_rf", "point_hull"),
    ("base", "always_retrieve"): ("confirm_{b}", "always_retrieve"),
}
EXTRA = {
    ("adaptive", "confidence"): ("confirm_{b}_adaptive", "adaptive_confidence"),
    ("adaptive", "complexity"): ("confirm_{b}_adaptive", "adaptive_complexity"),
    ("adaptive", "oracle"): ("confirm_{b}_adaptive", "adaptive_oracle"),
    # Rule thresholds re-selected on post-retrieval development observations.
    ("rf", "rule_tuned"): ("confirm_{b}_rf_tuned", "oesc"),
}


def success_and_calls(path: Path, gold: dict) -> tuple[dict, dict, dict]:
    ok, calls, gate = {}, {}, {}
    for result in load_results(str(path)):
        if result.sample_id not in gold:
            continue
        per_state = per_state_task_success([result], gold)
        ok[result.sample_id] = float(next(iter(per_state.values()))["task_success"])
        calls[result.sample_id] = float(result.cost.get("llm_calls", 0.0))
        if "gate_retrieve" in (result.metadata or {}):
            gate[result.sample_id] = bool(result.metadata["gate_retrieve"])
    return ok, calls, gate


def ci(values: np.ndarray, idx: np.ndarray) -> list[float]:
    boots = values[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [float(values.mean()), float(lo), float(hi)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="data/benchmark/valid_confirm.json")
    p.add_argument("--results-root", default="data/results")
    p.add_argument("--backbones", nargs="+", default=["deepseek41flash", "qwen38flash"])
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--no-extra", action="store_true", help="Skip adaptive runs")
    p.add_argument("--out", default="data/results/order_rule_interaction.json")
    args = p.parse_args()

    samples = load_benchmark(args.benchmark)
    gold = {s.id: s for s in samples}
    report = {}
    for backbone in args.backbones:
        runs, calls, gates = {}, {}, {}
        for key, (folder, method) in {**CELLS, **({} if args.no_extra else EXTRA)}.items():
            path = Path(args.results_root) / folder.format(b=backbone) / f"{method}_results.json"
            if not path.exists():
                continue
            runs[key], calls[key], gates[key] = success_and_calls(path, gold)
        ids = sorted(set.intersection(*(set(v) for v in runs.values())))
        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, len(ids), size=(args.resamples, len(ids)))
        vec = {k: np.array([v[i] for i in ids]) for k, v in runs.items()}
        out = {"n": len(ids), "cells": {}, "contrasts": {}}
        for key, v in vec.items():
            out["cells"]["/".join(key)] = {
                "task_success": float(v.mean()),
                "calls": float(np.mean([calls[key][i] for i in ids])),
                "gate_retrieve_rate": (
                    float(np.mean([gates[key][i] for i in ids])) if gates[key] else None
                ),
            }

        def add(name: str, values: np.ndarray) -> None:
            out["contrasts"][name] = ci(values, idx)

        for order in ("df", "rf"):
            add(f"{order}: point-rule", vec[(order, "point")] - vec[(order, "rule")])
            add(f"{order}: hull-rule", vec[(order, "hull")] - vec[(order, "rule")])
            add(f"{order}: hull-point", vec[(order, "hull")] - vec[(order, "point")])
        for rule in ("rule", "point", "hull"):
            add(f"order effect ({rule}): rf-df", vec[("rf", rule)] - vec[("df", rule)])
        add(
            "interaction: (rf-df)_point - (rf-df)_rule",
            (vec[("rf", "point")] - vec[("df", "point")])
            - (vec[("rf", "rule")] - vec[("df", "rule")]),
        )
        add("rf point - always_retrieve", vec[("rf", "point")] - vec[("base", "always_retrieve")])
        if ("rf", "rule_tuned") in vec:
            tuned = vec[("rf", "rule_tuned")]
            add("rf point - rf rule_tuned", vec[("rf", "point")] - tuned)
            add("rf rule_tuned - rf rule", tuned - vec[("rf", "rule")])
            add("order effect (rule_tuned): rf-df", tuned - vec[("df", "rule")])
            add(
                "interaction (tuned): (rf-df)_point - (rf_tuned-df)_rule",
                (vec[("rf", "point")] - vec[("df", "point")]) - (tuned - vec[("df", "rule")]),
            )
        for key in EXTRA:
            if key[0] != "adaptive":
                continue
            if key in vec:
                add(f"rf point - adaptive {key[1]}", vec[("rf", "point")] - vec[key])
                add(f"adaptive {key[1]} - df point", vec[key] - vec[("df", "point")])
        cost = {}
        for key in vec:
            cost["/".join(key)] = float(np.mean([calls[key][i] for i in ids]))
        out["calls"] = cost
        report[backbone] = out
        print(f"== {backbone} (n={len(ids)})")
        for name, cell in out["cells"].items():
            print(f"  {name:22s} TS={cell['task_success']:.3f} calls={cell['calls']:.1f}"
                  + (f" gate={cell['gate_retrieve_rate']:.2f}" if cell["gate_retrieve_rate"] is not None else ""))
        for name, (m, lo, hi) in out["contrasts"].items():
            print(f"  {name:45s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
