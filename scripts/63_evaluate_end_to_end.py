"""Score end-to-end policy results produced by ``62_run_end_to_end_eval.py``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.evaluation.policy_metrics import summarize_policy  # noqa: E402
from oesc_rag.evaluation.serialization import load_results  # noqa: E402

DEFAULT_METHODS = ("vanilla", "always_retrieve", "verifier_only", "oesc")


def table_row(method: str, summary: dict) -> dict:
    actions = summary["actions"]
    answers = summary["answer_quality"]
    per_state = summary["task_success"]["per_state"]
    row = {
        "method": method,
        "n": summary["n"],
        "task_success": summary["task_success"]["task_success"],
        "first_act": actions["first_action_accuracy"],
        "term_act": actions["terminal_action_accuracy"],
        "FAR": actions["false_answer_rate"],
        "unneeded_retr": actions["unnecessary_retrieval_rate"],
        "suff_F1": answers["answer_f1"],
        "suff_supported": answers["supported_answer_rate"],
        "miss_recovery": summary["missing_recovery"]["recovery_success"],
        "calls": summary["avg_llm_calls"],
    }
    for state, values in per_state.items():
        row[f"ts_{state}"] = values["task_success"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmark/samples_full.json")
    parser.add_argument("--results-dir", default="data/results/end_to_end")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--out", default="data/results/end_to_end/report.json")
    args = parser.parse_args()

    gold = {sample.id: sample for sample in load_benchmark(args.benchmark)}
    report: dict[str, dict] = {}
    rows = []
    for method in args.methods:
        path = Path(args.results_dir) / f"{method}_results.json"
        if not path.exists():
            print(f"[skip] {method}: {path} not found")
            continue
        results = [result for result in load_results(str(path)) if result.sample_id in gold]
        if not results:
            print(f"[skip] {method}: no matching results")
            continue
        report[method] = summarize_policy(results, gold)
        rows.append(table_row(method, report[method]))

    if rows:
        id_sets = {method: set(summary["sample_ids"]) for method, summary in report.items()}
        if len({frozenset(ids) for ids in id_sets.values()}) > 1:
            print("[warn] methods were scored on different sample sets:")
            for method, ids in id_sets.items():
                print(f"  {method}: n={len(ids)}")
        columns = list(dict.fromkeys(key for row in rows for key in row))
        print(" | ".join(columns))
        for row in rows:
            print(" | ".join(
                f"{row.get(col, ''):.3f}" if isinstance(row.get(col), float) else str(row.get(col, ""))
                for col in columns
            ))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report -> {out_path}")


if __name__ == "__main__":
    main()
