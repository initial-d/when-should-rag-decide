"""Train point / point+hull policies on development traces only.

Round-0 features are computed before any decision, so they are identical under
every controller; traces from any earlier run on a development sample can be
reused as training rows.  One pair of policies is trained per backbone.
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

TRACES = {
    "deepseek41flash": [
        "data/results/end_to_end_deepseek41flash_v6_dev/oesc_results.json",
        "data/results/end_to_end_deepseek41flash_ext_v6/oesc_results.json",
    ],
    "qwen38flash": [
        "data/results/end_to_end_qwen38flash_v6_dev/oesc_results.json",
        "data/results/end_to_end_qwen38flash_ext_v6/oesc_results.json",
    ],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="data/benchmark/valid_dev.json")
    p.add_argument("--backbones", nargs="+", default=list(TRACES))
    p.add_argument("--out-dir", default="data/models")
    args = p.parse_args()

    gold = {s.id: s for s in load_benchmark(args.benchmark)}
    for backbone in args.backbones:
        rows, seen = [], set()
        for path in TRACES[backbone]:
            for result in load_results(path):
                sample = gold.get(result.sample_id)
                if sample is None or not result.rounds or result.sample_id in seen:
                    continue
                seen.add(result.sample_id)
                rows.append((
                    result.rounds[0].features,
                    bool((result.metadata or {}).get("time_sensitive")),
                    sample.gold_state,
                ))
        print(f"{backbone}: {len(rows)} dev rows", dict(Counter(r[2] for r in rows)))
        for use_hull, name in ((False, "point"), (True, "point_hull")):
            policy = PointPolicy(use_hull=use_hull, class_weight="balanced")
            policy.fit(rows)
            out = Path(args.out_dir) / f"{backbone}_{name}_policy.pkl"
            policy.save(str(out))
            print(f"  saved {out}")


if __name__ == "__main__":
    main()
