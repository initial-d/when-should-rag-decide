"""Vanilla RAG: single retrieval, single answer, always predicts action=answer."""

from __future__ import annotations

from ..llm import LLMClient
from ..pipeline.answer_generator import generate_candidate_answer
from ..retrieval.bm25_retriever import BM25Index
from ..types import (
    BenchmarkSample,
    PipelineResult,
    StateFeatures,
)


def run_vanilla(
    client: LLMClient,
    sample: BenchmarkSample,
    config: dict,
) -> PipelineResult:
    k = config["retrieval"]["top_k"]
    index = BM25Index(sample.initial_evidence)
    docs = index.topk(sample.question, k=k)
    # Always include the initial evidence
    seen = {d.doc_id for d in docs}
    for d in sample.initial_evidence:
        if d.doc_id not in seen:
            docs.append(d)
            seen.add(d.doc_id)

    cand = generate_candidate_answer(
        client=client,
        question=sample.question,
        evidence=docs,
        evidence_view_idx=0,
        model=config["models"]["answer_generator"],
        temperature=config["openai"]["temperature_answer"],
    )

    return PipelineResult(
        sample_id=sample.id,
        predicted_state="sufficient",
        predicted_action="answer",
        features=StateFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        candidate_answers=[cand],
        verifications=[],
        information_needs=[],
        final_answer=cand.answer_text,
        cost={"llm_calls": 1.0, "rounds": 1.0, "budget": 0.0},
        method="vanilla-rag",
        metadata={"initial_answer": cand.answer_text},
    )
