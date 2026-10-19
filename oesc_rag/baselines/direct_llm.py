"""Direct 6-action LLM baseline.

A single LLM call: given the question and retrieved evidence, the LLM
classifies the evidence state into one of the six states defined in OESC
and produces the action-specific output. No claim extraction, no
verification, no feature vector --- this isolates the question of whether
a strong LLM with the right action menu can match the structured controller.
"""

from __future__ import annotations

from typing import Optional

from ..llm import LLMClient
from ..retrieval.bm25_retriever import BM25Index
from ..types import (
    ACTION_LABELS,
    STATE_LABELS,
    STATE_TO_ACTION,
    BenchmarkSample,
    CandidateAnswer,
    EvidenceDoc,
    PipelineResult,
    StateFeatures,
)


DIRECT_PROMPT = """You are a careful RAG controller.

Given the user question and retrieved evidence, classify the evidence state
into ONE of:
- sufficient: evidence supports a unique, verifiable answer.
- missing: evidence is on-topic but lacks key facts needed to answer.
- conflicting: sources disagree on a key fact.
- underspecified: the question itself is ambiguous (multiple defensible
  answers depending on a hidden criterion the user did not specify).
- unanswerable: the evidence cannot answer the question and more retrieval
  is unlikely to help.
- outdated: the question is time-sensitive and the evidence is stale.

Each state maps to one action:
  sufficient -> answer
  missing -> retrieve-more
  conflicting -> report-conflict
  underspecified -> ask-clarification
  unanswerable -> abstain
  outdated -> retrieve-fresh-source

Output JSON only:
{{
  "state": "<one of: sufficient|missing|conflicting|underspecified|unanswerable|outdated>",
  "action": "<corresponding action>",
  "answer": "<the answer text if action==answer, else ''>",
  "clarification": "<the clarifying question if action==ask-clarification, else ''>",
  "conflict_summary": "<short summary if action==report-conflict, else ''>",
  "abstain_reason": "<brief reason if action==abstain, else ''>",
  "rationale": "<one sentence>"
}}

Question: {question}

Evidence:
{evidence}
"""


def _format_evidence(docs: list[EvidenceDoc]) -> str:
    if not docs:
        return "(no evidence)"
    return "\n\n".join(f"[{d.doc_id} | {d.title}]\n{d.text}" for d in docs)


def run_direct_llm(
    client: LLMClient,
    sample: BenchmarkSample,
    config: dict,
) -> PipelineResult:
    k = config["retrieval"]["top_k"]
    pool_index = BM25Index(sample.evidence_pool)
    docs = pool_index.topk(sample.question, k=k)
    seen = {d.doc_id for d in docs}
    for d in sample.initial_evidence:
        if d.doc_id not in seen:
            docs.append(d)
            seen.add(d.doc_id)

    prompt = DIRECT_PROMPT.format(
        question=sample.question,
        evidence=_format_evidence(docs),
    )
    try:
        data = client.chat_json(
            model=config["models"]["answer_generator"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception:
        data = {}

    state = str(data.get("state", "unanswerable")).strip().lower()
    if state not in STATE_LABELS:
        state = "unanswerable"
    action = STATE_TO_ACTION[state]
    explicit_action = str(data.get("action", "")).strip().lower()
    if explicit_action in ACTION_LABELS:
        action = explicit_action

    final_answer: Optional[str]
    if action == "answer":
        final_answer = str(data.get("answer", "")).strip() or None
    elif action == "ask-clarification":
        final_answer = str(data.get("clarification", "")).strip() or None
    elif action == "report-conflict":
        final_answer = str(data.get("conflict_summary", "")).strip() or None
    elif action == "abstain":
        final_answer = str(data.get("abstain_reason", "")).strip() or None
    else:
        final_answer = None

    cand = CandidateAnswer(
        answer_text=str(data.get("answer", "")).strip(),
        claims=[],
        confidence=0.5,
        evidence_view_idx=0,
        missing_info=None,
    )
    budget = float(config.get("loop", {}).get("budget_calls", 30))

    return PipelineResult(
        sample_id=sample.id,
        predicted_state=state,
        predicted_action=action,
        features=StateFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        candidate_answers=[cand],
        verifications=[],
        information_needs=[],
        final_answer=final_answer,
        cost={"llm_calls": 1.0, "rounds": 1.0, "budget": budget},
        method="direct-llm",
    )
