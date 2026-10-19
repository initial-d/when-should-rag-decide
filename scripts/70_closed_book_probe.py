"""Closed-book probe: can each backbone answer a question with no evidence?

Records, per sample and backbone, whether the model's evidence-free answer is
correct.  This measures parametric-knowledge leakage, so results can be
reported on the full test set and on the subset where retrieval is actually
needed.  It does not filter the benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.evaluation.metrics import _answer_correct  # noqa: E402

PROMPT = (
    "Answer the question from your own knowledge in as few words as possible. "
    'Return JSON only: {{"answer": "..."}}\n\nQuestion: {question}'
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="data/benchmark/valid_test.json")
    p.add_argument("--configs", nargs="+", default=[
        "configs/e2e_deepseek41flash.yaml",
        "configs/e2e_qwen38flash.yaml",
    ])
    p.add_argument("--out", default="data/results/closed_book_probe.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--key-file", type=Path)
    args = p.parse_args()

    if args.key_file:
        os.environ["OPENAI_API_KEY"] = args.key_file.read_text().strip()
    from oesc_rag.llm import make_client

    samples = load_benchmark(args.benchmark)
    report: dict[str, dict] = {}
    for config in args.configs:
        cfg = yaml.safe_load(open(config, encoding="utf-8"))
        model = cfg["models"]["answer_generator"]
        client = make_client(cache_dir="data/cache/llm", config_path=config)

        def ask(sample):
            try:
                data = client.chat_json(
                    model=model,
                    messages=[{"role": "user", "content": PROMPT.format(question=sample.question)}],
                    temperature=0.0,
                )
                answer = str(data.get("answer", "")).strip()
            except Exception:
                answer = ""
            return sample.id, answer, _answer_correct(answer, sample.gold_answer)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(ask, samples))
        report[model] = {sid: {"answer": ans, "correct": ok} for sid, ans, ok in rows}

        print(f"\n{model}")
        for state in ("sufficient", "missing", "conflicting", "unanswerable"):
            ids = [s.id for s in samples if s.gold_state == state]
            hit = sum(report[model][i]["correct"] for i in ids)
            print(f"  {state:13} closed-book correct {hit:3}/{len(ids)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
