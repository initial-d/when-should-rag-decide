"""Adaptive retrieval-timing baselines that share the full OESC executor.

These baselines answer the question a reviewer will ask: is retrieve-first plus
a learned decision just a standard adaptive-retrieval policy?  Each one reads
the question and the *initial* evidence, decides whether to retrieve before the
first diagnosis, and then dispatches to one of two pipelines that are otherwise
identical to the main method:

* no retrieval  -> diagnose-first pipeline with the diagnose-first point policy;
* retrieval     -> retrieve-first pipeline with the post-retrieval point policy.

Both pipelines share the action space, evidence pool, BM25 settings, query
variants, gap queries, round ceiling, call budget and terminal resolution, so a
difference from always-retrieve-first isolates *when* retrieval happens.

Gates:

* ``confidence`` -- self-reported sufficiency confidence (Self-RAG / FLARE
  style retrieve-on-demand); retrieve when confidence < threshold.
* ``complexity`` -- Adaptive-RAG style three-way complexity router; retrieve
  unless the router answers ``A`` (no retrieval needed).
* ``oracle``     -- retrieve exactly when the gold state is ``missing``.  It
  reads the label, so it is a headroom analysis, not a baseline.
"""
from __future__ import annotations

import copy

from ..llm import LLMClient
from ..types import BenchmarkSample, EvidenceDoc, PipelineResult


CONFIDENCE_PROMPT = """You are calibrating a retrieval-augmented QA system.

Read the question and the currently available evidence, then report how
confident you are that the evidence is sufficient to answer correctly.

Output JSON only:
{{"confidence": 0.0-1.0, "missing": "<what evidence is missing, or empty>"}}

Question: {question}

Available evidence:
{evidence}
"""

COMPLEXITY_PROMPT = """You are a query-complexity router.

Decide how much retrieval this question needs before an answer can be trusted:
- A: the given evidence or common knowledge is enough; do not retrieve.
- B: one targeted retrieval is needed.
- C: several retrieval hops are needed.

Output JSON only:
{{"complexity": "A|B|C", "rationale": "<one sentence>"}}

Question: {question}

Available evidence:
{evidence}
"""

GATE_KINDS = ("confidence", "complexity", "oracle")


def _format(docs: list[EvidenceDoc], max_chars: int = 400) -> str:
    if not docs:
        return "(none)"
    return "\n\n".join(
        f"[{d.doc_id} | {d.title}] {(d.text or '')[:max_chars]}" for d in docs
    )


def gate_decision(
    kind: str, client: LLMClient, sample: BenchmarkSample, config: dict
) -> tuple[bool, dict, int]:
    """Return (retrieve_first, gate record, LLM calls spent by the gate)."""
    if kind == "oracle":
        return sample.gold_state == "missing", {"oracle": True}, 0
    model = config["models"]["query_rewriter"]
    prompt = CONFIDENCE_PROMPT if kind == "confidence" else COMPLEXITY_PROMPT
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt.format(
                question=sample.question, evidence=_format(sample.initial_evidence))}],
            temperature=0.0,
        )
    except Exception as exc:  # a failed gate falls back to retrieval
        return True, {"error": str(exc)[:200]}, 1
    if not isinstance(data, dict):
        return True, {"error": "non-object gate output"}, 1
    if kind == "confidence":
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        threshold = float(config.get("baselines", {}).get("confidence_threshold", 0.7))
        return conf < threshold, {"confidence": conf, "threshold": threshold}, 1
    label = str(data.get("complexity", "B")).strip().upper()[:1] or "B"
    return label != "A", {"complexity": label}, 1


def build_gated_runner(client, cfg: dict, kind: str, embedder=None):
    """Build a runner that gates retrieval and dispatches to a shared pipeline.

    ``cfg`` is the retrieve-first configuration; the diagnose-first branch is
    the same configuration with ``retrieve_first`` off and the policy path
    taken from ``controller.diagnose_first_policy_path``.
    """
    if kind not in GATE_KINDS:
        raise ValueError(f"unknown gate: {kind}")
    from ..controller.point_policy import PointPolicy
    from ..orchestrator import OESCPipeline

    rf_cfg = copy.deepcopy(cfg)
    rf_cfg.setdefault("controller", {}).update(type="point", retrieve_first=True)
    df_cfg = copy.deepcopy(cfg)
    df_cfg["controller"]["type"] = "point"
    df_cfg["controller"]["retrieve_first"] = False
    df_cfg["controller"]["point_policy_path"] = cfg["controller"]["diagnose_first_policy_path"]
    rf_pipeline = OESCPipeline(
        client=client, config=rf_cfg, embedder=embedder,
        point_policy=PointPolicy.load(rf_cfg["controller"]["point_policy_path"]),
    )
    df_pipeline = OESCPipeline(
        client=client, config=df_cfg, embedder=embedder,
        point_policy=PointPolicy.load(df_cfg["controller"]["point_policy_path"]),
    )

    def run(sample: BenchmarkSample) -> PipelineResult:
        retrieve, record, gate_calls = gate_decision(kind, client, sample, cfg)
        result = (rf_pipeline if retrieve else df_pipeline).run(sample)
        result.cost["llm_calls"] = float(result.cost.get("llm_calls", 0.0)) + gate_calls
        result.cost["gate_calls"] = float(gate_calls)
        result.method = f"adaptive-{kind}"
        result.metadata = dict(result.metadata or {})
        result.metadata.update(gate=kind, gate_retrieve=retrieve, gate_record=record)
        return result

    return run
