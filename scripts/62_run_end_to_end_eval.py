"""Run the initial-evidence-only end-to-end policy comparison.

Every method sees only ``initial_evidence`` before its first decision; the full
``evidence_pool`` becomes reachable only through a retrieval action. Results are
written as ``{out_dir}/{method}_results.json`` and scored by
``63_evaluate_end_to_end.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from oesc_rag.baselines.always_retrieve import run_always_retrieve  # noqa: E402
from oesc_rag.baselines.vanilla_rag import run_vanilla  # noqa: E402
from oesc_rag.baselines.verifier_only import run_verifier_only  # noqa: E402
from oesc_rag.data.build_benchmark import load_benchmark  # noqa: E402
from oesc_rag.orchestrator import OESCPipeline  # noqa: E402

METHODS = ("vanilla", "always_retrieve", "verifier_only", "oesc", "point", "point_hull",
           "adaptive_confidence", "adaptive_complexity", "adaptive_oracle")
MVP_STATES = ("sufficient", "missing", "conflicting", "unanswerable")


def build_runner(name: str, client, cfg: dict, embedder=None) -> Callable:
    if name == "vanilla":
        return lambda sample: run_vanilla(client, sample, cfg)
    if name == "always_retrieve":
        return lambda sample: run_always_retrieve(client, sample, cfg)
    if name == "verifier_only":
        return lambda sample: run_verifier_only(client, sample, cfg)
    if name == "oesc":
        pipeline = OESCPipeline(client=client, config=cfg, embedder=embedder)
        return pipeline.run
    if name in {"point", "point_hull"}:
        from oesc_rag.controller.point_policy import PointPolicy

        policy_path = cfg["controller"].get(
            "point_policy_path", f"data/models/{name}_policy.pkl"
        )
        policy = PointPolicy.load(policy_path)
        point_cfg = copy.deepcopy(cfg)
        point_cfg.setdefault("controller", {})["type"] = "point"
        pipeline = OESCPipeline(
            client=client, config=point_cfg, embedder=embedder, point_policy=policy
        )
        return pipeline.run
    if name.startswith("adaptive_"):
        from oesc_rag.baselines.adaptive_action import build_gated_runner

        return build_gated_runner(client, cfg, name[len("adaptive_"):], embedder)
    raise ValueError(f"unknown method: {name}")


def prepare_config(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("intervention", {})["initial_only_observation"] = True
    return out


def select_samples(samples: list, states: tuple[str, ...], limit: int | None) -> list:
    chosen = [sample for sample in samples if sample.gold_state in states]
    return chosen[:limit] if limit else chosen


def run_method(name: str, runner: Callable, samples: list, out_path: Path, workers: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)
    done = {row["sample_id"] for row in results}
    todo = [sample for sample in samples if sample.id not in done]
    print(f"[{name}] {len(done)} done, {len(todo)} to run")
    lock = threading.Lock()

    def run_one(sample):
        try:
            return runner(sample)
        except Exception as exc:
            print(f"[error] {name} on {sample.id}: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, sample): sample for sample in todo}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            with lock:
                results.append(result.to_dict())
                with out_path.open("w", encoding="utf-8") as handle:
                    json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"[{name}] saved {len(results)} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--benchmark", default="data/benchmark/samples_full.json")
    parser.add_argument("--out-dir", default="data/results/end_to_end")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--states", nargs="+", default=list(MVP_STATES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without model calls")
    parser.add_argument("--controller", choices=["rule", "credal"], default=None)
    parser.add_argument("--key-file", type=Path, help="File holding only the API key")
    parser.add_argument("--base-url")
    args = parser.parse_args()

    if args.key_file:
        credential = args.key_file.read_text().strip()
        if not credential or len(credential.splitlines()) != 1:
            parser.error("--key-file must contain one nonempty credential value")
        os.environ["OPENAI_API_KEY"] = credential
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = prepare_config(yaml.safe_load(handle))
    if args.controller:
        cfg.setdefault("controller", {})["type"] = args.controller
    samples = select_samples(load_benchmark(args.benchmark), tuple(args.states), args.limit)
    print(f"{len(samples)} samples; states={args.states}; initial_only_observation=True")
    if args.dry_run:
        for method in args.methods:
            print(f"would run {method} -> {Path(args.out_dir) / f'{method}_results.json'}")
        return

    from oesc_rag.llm import make_client

    client = make_client(cache_dir="data/cache/llm", config_path=args.config)
    embedder = None
    if any(m in ("oesc", "point", "point_hull") or m.startswith("adaptive_") for m in args.methods):
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(cfg["embedding"]["model"])
        except Exception as exc:
            print(f"[warn] embedder unavailable ({exc}); using fallback")

    for method in args.methods:
        method_cfg = copy.deepcopy(cfg)
        runner = build_runner(method, client, method_cfg, embedder)
        run_method(method, runner, samples, Path(args.out_dir) / f"{method}_results.json", args.workers)


if __name__ == "__main__":
    main()
