"""Adaptive-RAG baseline (Jeong et al., NAACL 2024).

Adaptive-RAG routes each query, *before* answering, through a complexity
classifier into one of three strategies:

  A (straightforward) -> no retrieval; answer from the given context only.
  B (simple)          -> single-step retrieval, then answer.
  C (complex)         -> multi-step iterative retrieve-read, then answer.

The original method trains a small T5 complexity classifier. That trained
router is not available here, so we use a zero-shot *LLM* complexity router
on the same backbone. This is a task-adapted scope control, not an exact
reproduction of the original trained router.
Everything else matches the shared baseline protocol: same evidence pool,
same backbone LLM, same BM25 top-k, same round budget, and the
``intervention.initial_only_observation`` gate is honoured exactly as in
``agentic_rag.py``.

Key contrast with OESC-RAG: Adaptive-RAG adapts *how much to retrieve* and
then *always answers*. It never abstains, reports a conflict, or asks for
clarification. On the six-state action benchmark it therefore behaves like a
cost-adaptive vanilla RAG: its ``predicted_action`` is always ``answer``.
This is the intended comparison -- retrieval-complexity routing is not
evidence-state control -- not a strawman.
"""

from __future__ import annotations

from typing import Optional

from ..llm import LLMClient
from ..pipeline.answer_generator import generate_candidate_answer
from ..retrieval.bm25_retriever import BM25Index
from ..types import (
    STATE_TO_ACTION,
    BenchmarkSample,
    CandidateAnswer,
    EvidenceDoc,
    PipelineResult,
    RetrievalRound,
    StateFeatures,
)


COMPLEXITY_PROMPT = """You are a query-complexity router for a retrieval-augmented QA system.

Classify how much retrieval the question needs:
- A: straightforward; answerable directly from the already-provided context
  or common knowledge, no extra retrieval needed.
- B: simple; a single retrieval step over the corpus is enough.
- C: complex; needs multiple retrieval hops (e.g. multi-entity, multi-fact,
  comparison, or chained reasoning) before it can be answered.

Output JSON only:
{{"complexity": "A|B|C", "rationale": "<one sentence>"}}

Question: {question}

Already-provided context:
{evidence}
"""


REFINE_QUERY_PROMPT = """Given the question and a note on what is still missing,
write ONE concise search query (<= 15 words) to retrieve the missing facts.

Question: {question}
Missing info: {missing_info}

Output JSON only: {{"query": "..."}}"""


_NO_RETRIEVAL_BUDGET = "A"
_SINGLE_STEP_BUDGET = "B"
_MULTI_STEP_BUDGET = "C"


def _format_evidence(docs: list[EvidenceDoc]) -> str:
    if not docs:
        return "(no evidence)"
    return "\n\n".join(f"[{d.doc_id} | {d.title}]\n{d.text}" for d in docs)


def _classify_complexity(client: LLMClient, sample: BenchmarkSample, config: dict) -> str:
    prompt = COMPLEXITY_PROMPT.format(
        question=sample.question,
        evidence=_format_evidence(sample.initial_evidence),
    )
    try:
        out = client.chat_json(
            model=config["models"]["answer_generator"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        c = str(out.get("complexity", "C")).strip().upper()[:1]
        if c not in (_NO_RETRIEVAL_BUDGET, _SINGLE_STEP_BUDGET, _MULTI_STEP_BUDGET):
            c = _MULTI_STEP_BUDGET
        return c
    except Exception:
        return _MULTI_STEP_BUDGET


def run_adaptive(
    client: LLMClient,
    sample: BenchmarkSample,
    config: dict,
) -> PipelineResult:
    k = config["retrieval"]["top_k"]
    max_rounds = int(config.get("loop", {}).get("max_rounds", 3))
    budget = float(config.get("loop", {}).get("budget_calls", 30))

    intervention = config.get("intervention", {})
    initial_only = bool(intervention.get("initial_only_observation", False))
    pool_index: Optional[BM25Index] = None

    def _pool() -> BM25Index:
        nonlocal pool_index
        if pool_index is None:
            pool_index = BM25Index(sample.evidence_pool)
        return pool_index

    accumulated: list[EvidenceDoc] = list(sample.initial_evidence)
    seen: set[str] = {d.doc_id for d in accumulated}
    rounds: list[RetrievalRound] = []
    feats = StateFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    # 1. Complexity routing (1 LLM call)
    complexity = _classify_complexity(client, sample, config)
    llm_calls = 1

    # A: 1 read, no retrieval. B: 1 retrieve-read. C: up to max_rounds hops.
    hops = max_rounds if complexity == _MULTI_STEP_BUDGET else 1
    retrieves = complexity != _NO_RETRIEVAL_BUDGET

    current_query = sample.question
    last_candidate: Optional[CandidateAnswer] = None

    for step in range(hops):
        added_ids: list[str] = []
        if retrieves:
            # initial_only locks the pool until a retrieval is actually issued
            # (step >= 1); for step 0 we still retrieve from initial evidence.
            if initial_only and step == 0:
                index = BM25Index(sample.initial_evidence)
            else:
                index = _pool()
            for d in index.topk(current_query, k=k):
                if d.doc_id not in seen:
                    accumulated.append(d)
                    seen.add(d.doc_id)
                    added_ids.append(d.doc_id)

        cand = generate_candidate_answer(
            client=client,
            question=sample.question,
            evidence=accumulated,
            evidence_view_idx=step,
            model=config["models"]["answer_generator"],
            temperature=config["openai"]["temperature_answer"],
        )
        llm_calls += 1
        last_candidate = cand

        rounds.append(RetrievalRound(
            round_idx=step,
            queries=[current_query],
            new_doc_ids=added_ids,
            state="sufficient",
            action="answer",
            features=feats,
            llm_calls=llm_calls,
        ))

        # Only multi-step (C) continues; refine the query for the next hop.
        if step >= hops - 1:
            break
        missing_info = (getattr(cand, "missing_info", "") or "").strip()
        if not missing_info:
            break  # reader is satisfied; stop early (Adaptive-RAG behaviour)
        refine_prompt = REFINE_QUERY_PROMPT.format(
            question=sample.question,
            missing_info=missing_info,
        )
        try:
            refine = client.chat_json(
                model=config["models"]["query_rewriter"],
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.0,
            )
            llm_calls += 1
            next_q = str(refine.get("query", "")).strip() or missing_info[:120]
        except Exception:
            next_q = missing_info[:120]
        if next_q == current_query:
            break
        current_query = next_q

    state = "sufficient"
    return PipelineResult(
        sample_id=sample.id,
        predicted_state=state,
        predicted_action=STATE_TO_ACTION[state],
        features=feats,
        candidate_answers=[last_candidate] if last_candidate else [],
        verifications=[],
        information_needs=[],
        final_answer=last_candidate.answer_text if last_candidate else None,
        rounds=rounds,
        cost={
            "llm_calls": float(llm_calls),
            "rounds": float(len(rounds)),
            "budget": budget,
        },
        method="adaptive-rag",
        metadata={"complexity": complexity, "initial_answer": last_candidate.answer_text if last_candidate else None},
    )
