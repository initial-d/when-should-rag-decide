"""Verifier-only baseline: generate one answer, verify claims, answer or abstain."""

from __future__ import annotations

import numpy as np

from ..llm import LLMClient
from ..pipeline.answer_generator import generate_candidate_answer
from ..pipeline.claim_extractor import extract_claims
from ..pipeline.claim_verifier import verify_claims
from ..pipeline.feature_extractor import _max_support_per_claim
from ..retrieval.bm25_retriever import BM25Index
from ..types import (
    BenchmarkSample,
    PipelineResult,
    StateFeatures,
)


def run_verifier_only(
    client: LLMClient,
    sample: BenchmarkSample,
    config: dict,
    support_threshold: float = 0.5,
) -> PipelineResult:
    k = config["retrieval"]["top_k"]
    # MVP: retrieve only from initial_evidence (single-pass setting).
    index = BM25Index(sample.initial_evidence)
    docs = index.topk(sample.question, k=k)
    seen = {d.doc_id for d in docs}
    for d in sample.initial_evidence:
        if d.doc_id not in seen:
            docs.append(d)

    cand = generate_candidate_answer(
        client=client,
        question=sample.question,
        evidence=docs,
        evidence_view_idx=0,
        model=config["models"]["answer_generator"],
        temperature=config["openai"]["temperature_answer"],
    )
    cand.claims = extract_claims(
        client=client,
        answer_text=cand.answer_text,
        question=sample.question,
        source_answer_idx=0,
        model=config["models"]["claim_extractor"],
        max_claims=config["pipeline"]["max_claims_per_answer"],
    )

    verifications = verify_claims(
        client=client,
        claims=cand.claims,
        evidence=docs,
        model=config["models"]["claim_verifier"],
        max_docs=config["pipeline"]["max_evidence_docs_for_verification"],
        joint=bool(config["pipeline"].get("joint_verification", False)),
    )

    claim_ids = [c.claim_id for c in cand.claims]
    sup = _max_support_per_claim(verifications, claim_ids)
    mean_support = float(np.mean(list(sup.values()))) if sup else 0.0

    if cand.claims and mean_support >= support_threshold:
        action = "answer"
        final_answer = cand.answer_text
        state = "sufficient"
    else:
        action = "abstain"
        final_answer = None
        state = "unanswerable"

    return PipelineResult(
        sample_id=sample.id,
        predicted_state=state,
        predicted_action=action,
        features=StateFeatures(0.0, mean_support, 0.0, 0.0, 0.0, 1.0),
        candidate_answers=[cand],
        verifications=verifications,
        final_answer=final_answer,
        cost={
            "llm_calls": 2.0
            + float(bool(cand.claims and docs))
            * (2.0 if config["pipeline"].get("joint_verification") else 1.0),
            "rounds": 1.0,
            "budget": 0.0,
        },
        method="verifier-only",
        metadata={"initial_answer": cand.answer_text, "initial_action": action},
    )
