#!/usr/bin/env python3
"""Separate observation timing from policy re-training in the retrieve-first gain.

Reads stored confirmation-set traces only (no model calls).  Per backbone it
scores four point-policy cells on the same questions:

  DF            diagnose-first policy, initial observation (E0 -> E0)
  RF-fixed      diagnose-first policy applied after the fixed retrieval (E0 -> E1)
  RF-gold       retrieve-first policy trained on E1 with gold labels (E1 -> E1)
  RF            retrieve-first policy trained on E1 with relabelled missing (E1 -> E1)

and paired contrasts with 95% bootstrap intervals (5,000 resamples, one draw
shared by all contrasts of a backbone):

  timing      RF-fixed - DF   (earlier evidence, policy unchanged)
  retraining  RF-gold - RF-fixed   (fit to E1, labels unchanged)
  relabel     RF - RF-gold    (missing-but-retrieved questions labelled sufficient)

RF-fixed sees inputs outside its training distribution and RF-gold keeps a
label semantics that no longer matches E1, so the split is descriptive and
not a unique causal decomposition.  Every run must cover the benchmark once
with no execution errors, otherwise nothing is computed.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.evaluation.serialization import load_results  # noqa: E402

_spec = importlib.util.spec_from_file_location("oracle", ROOT / "scripts/75_outcome_oracle.py")
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)

CELLS = {
    "DF": "confirm_{b}",
    "RF-fixed": "confirm_{b}_rf_fixed",
    "RF-gold": "confirm_{b}_rf_goldlabel",
    "RF": "confirm_{b}_rf",
}
CONTRASTS = {
    "timing": ("RF-fixed", "DF"),
    "retraining": ("RF-gold", "RF-fixed"),
    "relabel": ("RF", "RF-gold"),
    "total": ("RF", "DF"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark/valid_confirm.json")
    ap.add_argument("--results-root", default="data/results")
    ap.add_argument("--out", default="data/results/adaptation_audit.json")
    args = ap.parse_args()

    samples = load_benchmark(args.benchmark)
    gold = {s.id: s for s in samples}
    ids = [s.id for s in samples]
    report = {}
    for b in oracle.BACKBONES:
        runs, problems = {}, []
        for cell, tmpl in CELLS.items():
            path = Path(args.results_root) / tmpl.format(b=b) / "point_results.json"
            res = list(load_results(path))
            problems += oracle.check_pairing(f"{b}/{cell}", res, ids)
            runs[cell] = {r.sample_id: r for r in res}
        if problems:
            report[b] = {"pairing_ok": False, "problems": problems}
            print(b, "PAIRING FAILED:", problems)
            continue

        succ = {c: np.array([oracle.success(runs[c][i], gold) for i in ids], float) for c in CELLS}
        calls = {c: float(np.mean([oracle.calls(runs[c][i]) for i in ids])) for c in CELLS}
        state = np.array([gold[i].gold_state for i in ids])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(ids), size=(5000, len(ids)))
        row = {"pairing_ok": True, "cells": {}, "contrasts": {}}
        for c in CELLS:
            row["cells"][c] = {
                "task_success": float(succ[c].mean()),
                "calls": calls[c],
                "per_state": {s: float(succ[c][state == s].mean()) for s in oracle.STATES},
            }
        for name, (a, z) in CONTRASTS.items():
            d = succ[a] - succ[z]
            boot = d[idx].mean(1)
            row["contrasts"][name] = {
                "diff": float(d.mean()),
                "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                "per_state": {s: float(d[state == s].mean()) for s in oracle.STATES},
            }
        report[b] = row
        print(b)
        for c, v in row["cells"].items():
            ps = " ".join(f"{s[:4]}={x:.2f}" for s, x in v["per_state"].items())
            print(f"  {c:9s} {v['task_success']:.3f} calls={v['calls']:.1f}  {ps}")
        for n, v in row["contrasts"].items():
            ps = " ".join(f"{s[:4]}={x:+.2f}" for s, x in v["per_state"].items())
            print(f"  {n:10s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  {ps}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
