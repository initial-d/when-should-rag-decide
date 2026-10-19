"""Train post-retrieval point / point+hull policies for retrieve-first control.

Under retrieve-first, the first diagnosis happens after one retrieval, so the
training observation is the round-1 feature vector.  The label is the state of
the evidence actually observed at that point: a ``missing`` question whose
hidden supporting document has been retrieved is now ``sufficient``; otherwise
the gold state is kept.  Only development traces are used.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oesc_rag.controller.point_policy import PointPolicy  # noqa: E402
from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.evaluation.serialization import load_results  # noqa: E402


def observed_state(sample, round_record) -> str:
    if sample.gold_state != "missing":
        return sample.gold_state
    hidden = set(sample.supporting_doc_ids) - {d.doc_id for d in sample.initial_evidence}
    return "sufficient" if hidden & set(round_record.new_doc_ids) else "missing"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="data/benchmark/valid_dev.json")
    p.add_argument("--backbones", nargs="+", default=["deepseek41flash", "qwen38flash"])
    p.add_argument("--out-dir", default="data/models")
    p.add_argument("--gold-labels", action="store_true",
                   help="Keep the gold state as label (no post-retrieval relabelling); "
                        "used to separate label semantics from feature distribution")
    args = p.parse_args()

    gold = {s.id: s for s in load_benchmark(args.benchmark)}
    for backbone in args.backbones:
        rows = []
        for result in load_results(f"data/results/rfdev_{backbone}/oesc_results.json"):
            sample = gold.get(result.sample_id)
            if sample is None or len(result.rounds) < 2:
                continue
            rnd = result.rounds[1]
            rows.append((
                rnd.features,
                bool((result.metadata or {}).get("time_sensitive")),
                sample.gold_state if args.gold_labels else observed_state(sample, rnd),
            ))
        print(f"{backbone}: {len(rows)} post-retrieval dev rows", dict(Counter(r[2] for r in rows)))
        variants = ((False, "rf_goldlabel_point"),) if args.gold_labels else (
            (False, "rf_point"), (True, "rf_point_hull"))
        for use_hull, name in variants:
            policy = PointPolicy(use_hull=use_hull, class_weight="balanced")
            policy.fit(rows)
            out = Path(args.out_dir) / f"{backbone}_{name}_policy.pkl"
            policy.save(str(out))
            print(f"  saved {out}")


if __name__ == "__main__":
    main()
