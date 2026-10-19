"""Agentic / FLARE-style iterative baseline.

Iterative loop (matches OESC's 3-round budget):
  Round 0: retrieve, generate answer
  Self-assess: ANSWER | RETRIEVE | CONFLICT | CLARIFY | ABSTAIN
    ANSWER  -> terminate, return answer.
    CONFLICT -> terminate, report conflict.
    CLARIFY  -> terminate, return clarification question.
    ABSTAIN  -> terminate, abstain.
    RETRIEVE -> generate a refined query from `missing_info`, retrieve
                additional docs into the accumulated pool, next round.
  After max_rounds with no terminal decision -> abstain.

No multi-view retrieval, no claim graph, no feature vector. This is a
task-adapted Self-RAG / FLARE-style comparator with the same evidence pool,
backbone LLM, and round ceiling. It exposes five decisions and has no dedicated
OUTDATED / retrieve-fresh-source action, so it is not a fully interface-matched
six-action reproduction.
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


SELF_ASSESS_PROMPT = """You are evaluating whether a candidate answer can be reliably emitted.

Given the question, the current accumulated evidence, and the candidate
answer, output ONE of:
- ANSWER: the candidate is fully supported and the question is unambiguous.
- RETRIEVE: more evidence is needed (specify what is missing in `missing_info`).
- CONFLICT: the evidence sources disagree on a key fact.
- CLARIFY: the question itself is ambiguous and a clarifying question is
  required from the user.
- ABSTAIN: the evidence cannot answer the question and more retrieval will
  not help.

Output JSON only:
{{
  "decision": "ANSWER|RETRIEVE|CONFLICT|CLARIFY|ABSTAIN",
  "missing_info": "<what to retrieve if decision==RETRIEVE, else ''>",
  "conflict_summary": "<summary if decision==CONFLICT, else ''>",
  "clarification": "<clarifying question if decision==CLARIFY, else ''>",
  "rationale": "<one sentence>"
}}

Question: {question}

Current evidence:
{evidence}

Candidate answer: {answer}
"""


REFINE_QUERY_PROMPT = """Given a missing-information note, write ONE concise search
query (<= 15 words) that would retrieve the missing facts.

Question: {question}
Missing info: {missing_info}

Output JSON only: {{"query": "..."}}"""


_DECISION_TO_STATE = {
    "ANSWER": "sufficient",
    "RETRIEVE": "missing",
    "CONFLICT": "conflicting",
    "CLARIFY": "underspecified",
    "ABSTAIN": "unanswerable",
}


def _format_evidence(docs: list[EvidenceDoc]) -> str:
    if not docs:
        return "(no evidence)"
    return "\n\n".join(f"[{d.doc_id} | {d.title}]\n{d.text}" for d in docs)


def run_agentic(
    client: LLMClient,
    sample: BenchmarkSample,
    config: dict,
) -> PipelineResult:
    k = config["retrieval"]["top_k"]
    max_rounds = int(config.get("loop", {}).get("max_rounds", 3))
    budget = float(config.get("loop", {}).get("budget_calls", 30))

    intervention = config.get("intervention", {})
    initial_only = bool(intervention.get("initial_only_observation", False))
    if initial_only:
        pool_index = BM25Index(sample.initial_evidence)
        full_pool_index: Optional[BM25Index] = None
    else:
        pool_index = BM25Index(sample.evidence_pool)
        full_pool_index = pool_index
    accumulated: list[EvidenceDoc] = list(sample.initial_evidence)
    seen: set[str] = {d.doc_id for d in accumulated}
    current_query = sample.question

    rounds: list[RetrievalRound] = []
    llm_calls = 0
    last_decision = "ABSTAIN"
    last_candidate: Optional[CandidateAnswer] = None
    final_answer: Optional[str] = None
    state = "unanswerable"

    for round_idx in range(max_rounds):
        if initial_only and round_idx >= 1:
            if full_pool_index is None:
                full_pool_index = BM25Index(sample.evidence_pool)
            current_index = full_pool_index
        else:
            current_index = pool_index
        new_docs = current_index.topk(current_query, k=k)
        added_ids: list[str] = []
        for d in new_docs:
            if d.doc_id not in seen:
                accumulated.append(d)
                seen.add(d.doc_id)
                added_ids.append(d.doc_id)

        cand = generate_candidate_answer(
            client=client,
            question=sample.question,
            evidence=accumulated,
            evidence_view_idx=round_idx,
            model=config["models"]["answer_generator"],
            temperature=config["openai"]["temperature_answer"],
        )
        llm_calls += 1
        last_candidate = cand

        assess_prompt = SELF_ASSESS_PROMPT.format(
            question=sample.question,
            evidence=_format_evidence(accumulated),
            answer=cand.answer_text,
        )
        try:
            assess = client.chat_json(
                model=config["models"]["answer_generator"],
                messages=[{"role": "user", "content": assess_prompt}],
                temperature=0.0,
            )
            llm_calls += 1
        except Exception:
            assess = {"decision": "ABSTAIN"}

        decision = str(assess.get("decision", "ABSTAIN")).strip().upper()
        if decision not in _DECISION_TO_STATE:
            decision = "ABSTAIN"
        last_decision = decision
        state = _DECISION_TO_STATE[decision]

        rounds.append(RetrievalRound(
            round_idx=round_idx,
            queries=[current_query],
            new_doc_ids=added_ids,
            state=state,
            action=STATE_TO_ACTION[state],
            features=StateFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            llm_calls=llm_calls,
        ))

        if decision == "ANSWER":
            final_answer = cand.answer_text
            break
        if decision == "CONFLICT":
            final_answer = str(assess.get("conflict_summary", "")).strip() or None
            break
        if decision == "CLARIFY":
            final_answer = str(assess.get("clarification", "")).strip() or None
            break
        if decision == "ABSTAIN":
            final_answer = None
            break

        # RETRIEVE: build a refined query for the next round
        missing_info = str(assess.get("missing_info", "")).strip()
        if not missing_info or round_idx == max_rounds - 1:
            state = "unanswerable"
            final_answer = None
            break
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
            state = "unanswerable"
            final_answer = None
            break
        current_query = next_q

    # If we ran out of rounds while still in RETRIEVE mode, demote to abstain
    if last_decision == "RETRIEVE":
        state = "unanswerable"
        final_answer = None

    return PipelineResult(
        sample_id=sample.id,
        predicted_state=state,
        predicted_action=STATE_TO_ACTION[state],
        features=StateFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        candidate_answers=[last_candidate] if last_candidate else [],
        verifications=[],
        information_needs=[],
        final_answer=final_answer,
        rounds=rounds,
        cost={
            "llm_calls": float(llm_calls),
            "rounds": float(len(rounds)),
            "budget": budget,
        },
        method="agentic-rag",
    )
