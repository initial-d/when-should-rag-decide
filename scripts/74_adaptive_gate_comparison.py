#!/usr/bin/env python3
"""Compare adaptive retrieval gates with the two unconditional schedules.

Reads stored confirmation-set traces only (no model calls).  For each
backbone it reports task success, per-state success, logical calls, the
share of questions routed to retrieve-first, and paired bootstrap contrasts
against the retrieve-first point policy.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.evaluation.policy_metrics import (  # noqa: E402
    first_action,
    per_state_task_success,
)
from oesc_rag.evaluation.serialization import load_results  # noqa: E402

BACKBONES = ["deepseek41flash", "qwen38flash"]
GATES = ["confidence", "complexity", "oracle"]


def success_vector(results, gold, ids):
    by_id = {r.sample_id: r for r in results}
    out = []
    for i in ids:
        ps = per_state_task_success([by_id[i]], gold)
        out.append(next(iter(ps.values()))["task_success"])
    return np.array(out, dtype=float)


def summarize(name, results, gold, ids, routed):
    by_id = {r.sample_id: r for r in results}
    res = [by_id[i] for i in ids]
    ps = per_state_task_success(res, gold)
    n_calls = np.array([float(r.cost.get("llm_calls", 0.0)) for r in res])
    gate = np.array([float(r.cost.get("gate_calls", 0.0)) for r in res])
    return {
        "method": name,
        "n": len(res),
        "task_success": float(success_vector(res, gold, ids).mean()),
        "per_state": {k: v["task_success"] for k, v in ps.items()},
        "llm_calls": float(n_calls.mean()),
        "gate_calls": float(gate.mean()),
        "retrieve_first_rate": float(np.mean([routed(r) for r in res])),
    }


def paired_ci(diff, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    s = diff[idx].mean(1)
    return float(diff.mean()), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark/valid_confirm.json")
    ap.add_argument("--results-root", default="data/results")
    ap.add_argument("--out", default="data/results/adaptive_gate_comparison.json")
    args = ap.parse_args()

    gold = {s.id: s for s in load_benchmark(args.benchmark)}
    ids = sorted(gold)
    root = Path(args.results_root)
    report = {}
    for bb in BACKBONES:
        runs = {
            "diagnose-first": (root / f"confirm_{bb}/point_results.json",
                               lambda r: first_action(r) == "retrieve-more" and False),
            "retrieve-first": (root / f"confirm_{bb}_rf/point_results.json",
                               lambda r: True),
        }
        for g in GATES:
            runs[f"gate-{g}"] = (root / f"confirm_{bb}_adaptive/adaptive_{g}_results.json",
                                 lambda r: bool((r.metadata or {}).get("gate_retrieve")))
        loaded = {k: list(load_results(p)) for k, (p, _) in runs.items()}
        missing = {k: len(set(ids) - {r.sample_id for r in v}) for k, v in loaded.items()}
        if any(missing.values()):
            raise SystemExit(f"{bb}: incomplete runs {missing}")
        rows = [summarize(k, loaded[k], gold, ids, runs[k][1]) for k in runs]
        ref = success_vector(loaded["retrieve-first"], gold, ids)
        for row in rows:
            vec = success_vector(loaded[row["method"]], gold, ids)
            row["vs_retrieve_first"] = paired_ci(vec - ref)
        report[bb] = rows

    Path(args.out).write_text(json.dumps(report, indent=2))
    for bb, rows in report.items():
        print(bb)
        for r in rows:
            d, lo, hi = r["vs_retrieve_first"]
            ps = " ".join(f"{k[:4]}={v:.3f}" for k, v in r["per_state"].items())
            print(f"  {r['method']:16s} TS={r['task_success']:.3f} {ps} calls={r['llm_calls']:.2f}"
                  f" gate={r['gate_calls']:.2f} rf_rate={r['retrieve_first_rate']:.3f}"
                  f" d={d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
