"""Build a structurally valid four-state evidence benchmark.

Two outputs:

* ``--dev-out``  -- samples from existing benchmarks that pass every validity
  check.  These were used during development, so they serve only for training
  and tuning.
* ``--test-out`` -- freshly constructed samples from HotpotQA questions that
  appear in no existing benchmark.  Each state is built so its label is true
  by construction and then re-checked with ``sample_issues``:

  sufficient    all supporting paragraphs are initial evidence, and the answer
                appears in them;
  missing       the supporting paragraph(s) containing the answer are hidden
                in the pool; the initial evidence (remaining supporting
                paragraphs plus answer-free distractors) does not contain it;
  conflicting   supporting paragraphs plus a generated paragraph asserting a
                different answer, which must not restate the gold answer;
  unanswerable  only answer-free distractors, in the initial evidence and pool.

Questions with yes/no answers, or whose answer appears in the question, are
excluded because string checks cannot certify where such answers live.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from oesc_rag.data.build_benchmark import (  # noqa: E402
    _build_conflicting,
    _paragraph_to_doc,
    load_benchmark,
    save_benchmark,
)
from oesc_rag.data.load_hotpotqa import (  # noqa: E402
    load_hotpotqa_samples,
    non_supporting_paragraph_indices,
    paragraph_text,
    supporting_paragraph_indices,
)
from oesc_rag.data.validity import checkable_answer, contains_answer, sample_issues  # noqa: E402
from oesc_rag.types import BenchmarkSample  # noqa: E402

STATES = ("sufficient", "missing", "conflicting", "unanswerable")


def _answer_free(record, indices):
    return [i for i in indices if not contains_answer(paragraph_text(record, i), record["answer"])]


def build_sufficient(record, sid):
    sup = supporting_paragraph_indices(record)
    if len(sup) < 2:
        return None
    init = [_paragraph_to_doc(record, i, sid) for i in sup]
    extra = [_paragraph_to_doc(record, i, sid) for i in non_supporting_paragraph_indices(record)[:2]]
    return BenchmarkSample(
        id=sid, question=record["question"], gold_state="sufficient", gold_action="answer",
        gold_answer=record["answer"], initial_evidence=init, evidence_pool=init + extra,
        supporting_doc_ids=[d.doc_id for d in init], source_qid=record["_id"],
        perturbation_notes="all supporting paragraphs in initial evidence",
    )


def build_missing(record, sid):
    sup = supporting_paragraph_indices(record)
    if len(sup) < 2:
        return None
    hidden = [i for i in sup if contains_answer(paragraph_text(record, i), record["answer"])]
    kept = [i for i in sup if i not in hidden]
    if not hidden or not kept:
        return None
    distract = _answer_free(record, non_supporting_paragraph_indices(record))[:2]
    init = [_paragraph_to_doc(record, i, sid) for i in kept + distract]
    hidden_docs = [_paragraph_to_doc(record, i, sid) for i in hidden]
    return BenchmarkSample(
        id=sid, question=record["question"], gold_state="missing", gold_action="retrieve-more",
        gold_answer=record["answer"], initial_evidence=init, evidence_pool=init + hidden_docs,
        supporting_doc_ids=[_paragraph_to_doc(record, i, sid).doc_id for i in sup],
        source_qid=record["_id"],
        perturbation_notes=f"hid {len(hidden)} answer-bearing supporting paragraph(s)",
    )


def build_unanswerable(record, sid):
    distract = _answer_free(record, non_supporting_paragraph_indices(record))[:3]
    if len(distract) < 3:
        return None
    init = [_paragraph_to_doc(record, i, sid) for i in distract]
    return BenchmarkSample(
        id=sid, question=record["question"], gold_state="unanswerable", gold_action="abstain",
        gold_answer=record["answer"], initial_evidence=init, evidence_pool=list(init),
        supporting_doc_ids=[], source_qid=record["_id"],
        perturbation_notes="answer-free distractors only",
    )


def build_conflicting(record, sid, client, model, attempts):
    for _ in range(attempts):
        sample = _build_conflicting(record, sid, client, model)
        if sample is not None and not sample_issues(sample):
            return sample
    return None


def retain_valid(paths):
    kept = []
    for path in paths:
        for sample in load_benchmark(path):
            if sample.gold_state in STATES and not sample_issues(sample):
                kept.append(sample)
    return kept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/e2e_deepseek41flash.yaml")
    p.add_argument("--existing", nargs="+", default=[
        "data/benchmark/samples_full.json",
        "data/benchmark/samples_e2e_extension.json",
    ])
    p.add_argument("--dev-out", default="data/benchmark/valid_dev.json")
    p.add_argument("--test-out", default="data/benchmark/valid_test.json")
    p.add_argument("--per-state", type=int, default=100)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--conflict-attempts", type=int, default=3)
    p.add_argument("--key-file", type=Path)
    p.add_argument("--id-prefix", default="val")
    p.add_argument("--retain-only", action="store_true", help="Only write the retained dev set")
    args = p.parse_args()

    dev = retain_valid(args.existing)
    save_benchmark(dev, args.dev_out)
    print(f"retained valid existing samples -> {args.dev_out}: {len(dev)}",
          dict(Counter(s.gold_state for s in dev)))
    if args.retain_only:
        return

    if args.key_file:
        os.environ["OPENAI_API_KEY"] = args.key_file.read_text().strip()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from oesc_rag.llm import make_client

    client = make_client(cache_dir="data/cache/llm", config_path=args.config)
    model = cfg["models"]["claim_extractor"]

    used = {s.source_qid for path in args.existing for s in load_benchmark(path)}
    raw = load_hotpotqa_samples(n=10**6, seed=args.seed, cache_dir="data/raw")
    pool = [r for r in raw if r["_id"] not in used and checkable_answer(r["question"], r["answer"])]
    random.Random(args.seed).shuffle(pool)
    print(f"candidate source questions: {len(pool)} (unused, string-checkable)")

    out = {s: [] for s in STATES}
    rejected = Counter()
    builders = {
        "sufficient": build_sufficient,
        "missing": build_missing,
        "unanswerable": build_unanswerable,
    }
    idx = 0
    for record in pool:
        if all(len(out[s]) >= args.per_state for s in STATES):
            break
        state = min((s for s in STATES if len(out[s]) < args.per_state), key=lambda s: len(out[s]))
        sid = f"{args.id_prefix}_{state[:4]}_{idx:04d}"
        idx += 1
        if state == "conflicting":
            sample = build_conflicting(record, sid, client, model, args.conflict_attempts)
        else:
            sample = builders[state](record, sid)
        if sample is None:
            rejected[f"{state}:construction"] += 1
            continue
        issues = sample_issues(sample)
        if issues:
            rejected[f"{state}:{issues[0]}"] += 1
            continue
        out[state].append(sample)

    test = [s for st in STATES for s in out[st]]
    overlap = {s.source_qid for s in test} & used
    assert not overlap, f"source overlap with existing benchmarks: {len(overlap)}"
    assert len({s.source_qid for s in test}) == len(test), "duplicate source question"
    save_benchmark(test, args.test_out)
    print(f"built test -> {args.test_out}: {len(test)}", dict(Counter(s.gold_state for s in test)))
    print("rejections:", dict(rejected))
    Path(args.test_out).with_suffix(".build.json").write_text(json.dumps({
        "seed": args.seed, "per_state": args.per_state, "conflict_model": model,
        "rejections": dict(rejected), "n_candidates": len(pool),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
