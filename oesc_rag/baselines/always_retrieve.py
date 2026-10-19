"""Always-Retrieve baseline: force the controller's decision to be retrieve-more.

Round 0: BM25 over `initial_evidence`, generate `initial_answer` (stored in
         metadata).
Round 1: BM25 over the full `evidence_pool`, generate `final_answer`.

Predicted action is always `answer`. Cost is 2 LLM calls + 2 BM25 lookups
per sample.

This is the "no-controller" lower bound on what a permissive retrieval policy
can recover. If OESC's Recovery Rate matches this at lower call cost on
sufficient samples, the controller's selectivity is the contribution.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..pipeline.answer_generator import generate_candidate_answer
from ..retrieval.bm25_retriever import BM25Index
from ..types import (
    BenchmarkSample,
    PipelineResult,
    RetrievalRound,
    StateFeatures,
)


def run_always_retrieve(
    client: LLMClient,
    sample: BenchmarkSample,
    config: dict,
) -> PipelineResult:
    k = config["retrieval"]["top_k"]
    answer_temp = config["openai"]["temperature_answer"]
    answer_model = config["models"]["answer_generator"]

    # Round 0: initial-evidence-only answer ("what vanilla would say")
    init_index = BM25Index(sample.initial_evidence)
    init_docs = init_index.topk(sample.question, k=k)
    init_seen = {d.doc_id for d in init_docs}
    for d in sample.initial_evidence:
        if d.doc_id not in init_seen:
            init_docs.append(d)
            init_seen.add(d.doc_id)

    cand_initial = generate_candidate_answer(
        client=client,
        question=sample.question,
        evidence=init_docs,
        evidence_view_idx=0,
        model=answer_model,
        temperature=answer_temp,
    )

    # Round 1: full-pool expanded answer
    pool_index = BM25Index(sample.evidence_pool)
    pool_docs = pool_index.topk(sample.question, k=k)
    accumulated = list(init_docs)
    seen = set(init_seen)
    for d in pool_docs:
        if d.doc_id not in seen:
            accumulated.append(d)
            seen.add(d.doc_id)
    new_doc_ids = [d.doc_id for d in pool_docs if d.doc_id not in init_seen]

    cand_final = generate_candidate_answer(
        client=client,
        question=sample.question,
        evidence=accumulated,
        evidence_view_idx=1,
        model=answer_model,
        temperature=answer_temp,
    )

    feat = StateFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    rounds = [
        RetrievalRound(
            round_idx=0,
            queries=[sample.question],
            new_doc_ids=[d.doc_id for d in init_docs],
            state="sufficient",
            action="answer",
            features=feat,
            llm_calls=1,
        ),
        RetrievalRound(
            round_idx=1,
            queries=[sample.question],
            new_doc_ids=new_doc_ids,
            state="sufficient",
            action="answer",
            features=feat,
            llm_calls=2,
        ),
    ]

    return PipelineResult(
        sample_id=sample.id,
        predicted_state="sufficient",
        predicted_action="answer",
        features=feat,
        candidate_answers=[cand_initial, cand_final],
        verifications=[],
        information_needs=[],
        final_answer=cand_final.answer_text,
        rounds=rounds,
        cost={"llm_calls": 2.0, "rounds": 2.0, "budget": 0.0},
        method="always-retrieve",
        metadata={"initial_answer": cand_initial.answer_text, "initial_action": "retrieve-more"},
    )
