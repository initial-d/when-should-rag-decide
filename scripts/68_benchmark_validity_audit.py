"""Validity audit for evidence-state benchmarks (no model calls).

Applies ``oesc_rag.data.validity.sample_issues`` to every sample and reports
the fraction of each state whose label is contradicted by where the answer
actually appears.  The builder in ``69_build_valid_benchmark.py`` uses the
same checks, so audit and construction share one definition of validity.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.data.validity import contains_answer, sample_issues  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "data/benchmark/samples_full.json",
            "data/benchmark/samples_e2e_extension.json",
        ],
    )
    parser.add_argument("--out", default="data/results/benchmark_validity_audit.json")
    args = parser.parse_args()

    report = {}
    for path in args.benchmarks:
        samples = load_benchmark(path)
        rows = [
            {
                "id": s.id,
                "gold_state": s.gold_state,
                "answer_in_initial": contains_answer(
                    " ".join(d.text for d in s.initial_evidence), s.gold_answer
                ),
                "issues": sample_issues(s),
            }
            for s in samples
        ]
        per_state = {}
        for state in sorted({r["gold_state"] for r in rows}):
            sub = [r for r in rows if r["gold_state"] == state]
            per_state[state] = {
                "n": len(sub),
                "invalid": sum(1 for r in sub if r["issues"]),
                "answer_in_initial": sum(1 for r in sub if r["answer_in_initial"]),
                "issues": dict(Counter(i for r in sub for i in r["issues"])),
            }
        report[path] = {"n": len(rows), "per_state": per_state, "rows": rows}

        print(f"\n=== {path}  (n={len(rows)}) ===")
        for state, stats in per_state.items():
            rate = stats["invalid"] / stats["n"] if stats["n"] else 0.0
            print(
                f"  {state:14} n={stats['n']:3}  invalid={stats['invalid']:3} ({rate:5.1%})"
                f"  answer-in-initial={stats['answer_in_initial']:3}"
            )
            if stats["issues"]:
                print(f"                 {stats['issues']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
