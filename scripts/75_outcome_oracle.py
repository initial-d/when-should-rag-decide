#!/usr/bin/env python3
"""Per-instance outcome oracle over the two unconditional point schedules.

Reads stored confirmation-set traces only (no model calls).  For each
backbone it pairs the diagnose-first and retrieve-first point-policy runs
question by question and reports, overall and per gold state:

  both succeed / only diagnose-first / only retrieve-first / both fail,
  the outcome oracle (at least one branch succeeds),
  and the mean calls of the oracle when it takes the cheaper successful
  branch (diagnose-first calls when neither succeeds).

Pairing is checked first: both runs must cover exactly the benchmark ids,
once each, with no recorded execution error.  If the check fails, the
oracle is not computed.  As a consistency check, the gold-state gate run is
compared with the gate composed from the two stored runs.

The outcome oracle is not deployable: it chooses after seeing both
outcomes, and its call count excludes the cost of running both branches.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.evaluation.policy_metrics import per_state_task_success  # noqa: E402
from oesc_rag.evaluation.serialization import load_results  # noqa: E402

BACKBONES = ["deepseek41flash", "qwen38flash"]
STATES = ["sufficient", "missing", "conflicting", "unanswerable"]


def success(result, gold):
    ps = per_state_task_success([result], gold)
    return bool(next(iter(ps.values()))["task_success"])


def has_error(result):
    meta = result.metadata or {}
    return any(k in meta for k in ("error", "exception", "failed")) and bool(
        meta.get("error") or meta.get("exception") or meta.get("failed")
    )


def check_pairing(name, results, ids):
    problems = []
    counts = Counter(r.sample_id for r in results)
    dup = [i for i, c in counts.items() if c > 1]
    missing = sorted(set(ids) - set(counts))
    extra = sorted(set(counts) - set(ids))
    errors = [r.sample_id for r in results if has_error(r)]
    if dup:
        problems.append(f"{name}: {len(dup)} duplicated ids")
    if missing:
        problems.append(f"{name}: {len(missing)} ids missing")
    if extra:
        problems.append(f"{name}: {len(extra)} ids not in benchmark")
    if errors:
        problems.append(f"{name}: {len(errors)} records with execution errors")
    return problems


def paired_ci(diff, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    s = diff[idx].mean(1)
    return [float(diff.mean()), float(np.quantile(s, 0.025)),
            float(np.quantile(s, 0.975))]


def calls(result):
    return float(result.cost.get("llm_calls", 0.0))


def tabulate(ids, gold, df, rf):
    rows = {}
    groups = {"all": ids}
    for s in STATES:
        groups[s] = [i for i in ids if gold[i].gold_state == s]
    for g, members in groups.items():
        c = Counter()
        oracle_calls = []
        for i in members:
            a, b = success(df[i], gold), success(rf[i], gold)
            c[(a, b)] += 1
            if a and b:
                oracle_calls.append(min(calls(df[i]), calls(rf[i])))
            elif b:
                oracle_calls.append(calls(rf[i]))
            else:
                oracle_calls.append(calls(df[i]))
        n = len(members)
        rows[g] = {
            "n": n,
            "both_succeed": c[(True, True)] / n,
            "only_diagnose_first": c[(True, False)] / n,
            "only_retrieve_first": c[(False, True)] / n,
            "both_fail": c[(False, False)] / n,
            "diagnose_first": (c[(True, True)] + c[(True, False)]) / n,
            "retrieve_first": (c[(True, True)] + c[(False, True)]) / n,
            "outcome_oracle": 1 - c[(False, False)] / n,
            "oracle_calls": float(np.mean(oracle_calls)),
            "retrieve_first_calls": float(np.mean([calls(rf[i]) for i in members])),
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark/valid_confirm.json")
    ap.add_argument("--results-root", default="data/results")
    ap.add_argument("--out", default="data/results/outcome_oracle.json")
    args = ap.parse_args()

    gold = {s.id: s for s in load_benchmark(args.benchmark)}
    ids = sorted(gold)
    root = Path(args.results_root)
    report = {}
    for bb in BACKBONES:
        df_list = list(load_results(root / f"confirm_{bb}" / "point_results.json"))
        rf_list = list(load_results(root / f"confirm_{bb}_rf" / "point_results.json"))
        problems = check_pairing("diagnose-first", df_list, ids)
        problems += check_pairing("retrieve-first", rf_list, ids)
        if problems:
            report[bb] = {"pairing_ok": False, "problems": problems}
            print(bb, "PAIRING FAILED:", problems)
            continue
        df = {r.sample_id: r for r in df_list}
        rf = {r.sample_id: r for r in rf_list}
        rows = tabulate(ids, gold, df, rf)

        # Consistency: stored gold-state gate vs composition of the two runs.
        gate_path = root / f"confirm_{bb}_adaptive" / "adaptive_oracle_results.json"
        consistency = None
        if gate_path.exists():
            gate = {r.sample_id: r for r in load_results(gate_path)}
            agree = 0
            for i in ids:
                src = rf[i] if gold[i].gold_state == "missing" else df[i]
                agree += success(gate[i], gold) == success(src, gold)
            composed = np.mean([
                success(rf[i] if gold[i].gold_state == "missing" else df[i], gold)
                for i in ids
            ])
            consistency = {
                "success_agreement": agree / len(ids),
                "composed_gold_state_gate": float(composed),
                "stored_gold_state_gate": float(np.mean([success(gate[i], gold) for i in ids])),
            }
        # Repeatability control.  The gold-state gate re-executes the same
        # configuration and policy as one of the two stored runs (retrieve-
        # first on gold missing, diagnose-first otherwise).  Taking the best
        # of two executions of the *same* schedule measures how much an
        # outcome oracle gains from run-to-run variation alone.
        noise = None
        if gate_path.exists():
            problems = check_pairing("gold-state gate", list(gate.values()), ids)
            if problems:
                noise = {"pairing_ok": False, "problems": problems}
            else:
                s_df = np.array([success(df[i], gold) for i in ids], float)
                s_rf = np.array([success(rf[i], gold) for i in ids], float)
                s_g = np.array([success(gate[i], gold) for i in ids], float)
                same = np.array([
                    s_rf[k] if gold[i].gold_state == "missing" else s_df[k]
                    for k, i in enumerate(ids)
                ])
                cross = np.maximum(s_df, s_rf)
                repeat = np.maximum(same, s_g)
                noise = {
                    "pairing_ok": True,
                    "rerun_disagreement": float(np.mean(same != s_g)),
                    "repeat_oracle": float(repeat.mean()),
                    "repeat_gain_over_single_run": paired_ci(repeat - same),
                    "cross_oracle_minus_rf": paired_ci(cross - s_rf),
                    "cross_oracle_minus_repeat_oracle": paired_ci(cross - repeat),
                    "per_state": {},
                }
                for s in STATES:
                    m = np.array([gold[i].gold_state == s for i in ids])
                    noise["per_state"][s] = {
                        "rerun_disagreement": float(np.mean(same[m] != s_g[m])),
                        "repeat_oracle": float(repeat[m].mean()),
                        "cross_oracle": float(cross[m].mean()),
                    }
        report[bb] = {"pairing_ok": True, "rows": rows,
                      "gold_state_consistency": consistency, "repeatability": noise}

        print(bb)
        for g, r in rows.items():
            print(f"  {g:13s} n={r['n']:3d} both={r['both_succeed']:.3f}"
                  f" onlyDF={r['only_diagnose_first']:.3f} onlyRF={r['only_retrieve_first']:.3f}"
                  f" fail={r['both_fail']:.3f} DF={r['diagnose_first']:.3f}"
                  f" RF={r['retrieve_first']:.3f} oracle={r['outcome_oracle']:.3f}"
                  f" calls(oracle/RF)={r['oracle_calls']:.1f}/{r['retrieve_first_calls']:.1f}")
        print("  gold-state consistency:", consistency)
        print("  repeatability:", json.dumps(noise, indent=None))

    Path(args.out).write_text(json.dumps(report, indent=2))
    print("->", args.out)


if __name__ == "__main__":
    main()
