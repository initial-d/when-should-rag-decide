"""Self-consistency baseline: generate N answers; answer or abstain by agreement."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from ..llm import LLMClient
from ..pipeline.answer_generator import generate_candidate_answer
from ..retrieval.bm25_retriever import BM25Index
from ..types import (
    BenchmarkSample,
    CandidateAnswer,
    PipelineResult,
    StateFeatures,
)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def run_self_consistency(
    client: LLMClient,
    sample: BenchmarkSample,
    config: dict,
    n: int = 5,
    agreement_threshold: float = 0.6,
) -> PipelineResult:
    k = config["retrieval"]["top_k"]
    index = BM25Index(sample.initial_evidence)
    docs = index.topk(sample.question, k=k)
    seen = {d.doc_id for d in docs}
    for d in sample.initial_evidence:
        if d.doc_id not in seen:
            docs.append(d)

    candidates: list[CandidateAnswer] = []
    for i in range(n):
        cand = generate_candidate_answer(
            client=client,
            question=sample.question,
            evidence=docs,
            evidence_view_idx=i,
            model=config["models"]["answer_generator"],
            temperature=0.7,  # higher temp for diverse samples
        )
        candidates.append(cand)

    # Majority vote on normalized answer text
    counts = Counter(_normalize(c.answer_text) for c in candidates if c.answer_text)
    if not counts:
        action = "abstain"
        final_answer = None
    else:
        top_norm, top_count = counts.most_common(1)[0]
        agreement = top_count / n
        if agreement >= agreement_threshold and top_norm and not top_norm.startswith("insufficient"):
            action = "answer"
            # Find the first candidate whose answer normalizes to top
            final_answer = next(
                (c.answer_text for c in candidates if _normalize(c.answer_text) == top_norm),
                None,
            )
        else:
            action = "abstain"
            final_answer = None

    return PipelineResult(
        sample_id=sample.id,
        predicted_state="sufficient" if action == "answer" else "unanswerable",
        predicted_action=action,
        features=StateFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        candidate_answers=candidates,
        verifications=[],
        final_answer=final_answer,
        method="self-consistency",
    )
