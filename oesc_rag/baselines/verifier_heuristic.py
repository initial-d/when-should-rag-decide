"""Verifier + 6-action heuristic baseline.

Extends the verifier-only baseline from {answer, abstain} to all six
actions using simple rules on top of verifier scores plus a handful of
cheap signals (time-sensitive question keywords, intent-ambiguity
keywords, doc timestamps). No multi-view retrieval, no claim graph, no
intent LLM call --- this isolates whether the OESC controller's value
comes from the modular pipeline or just from having multi-state output.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from ..llm import LLMClient
from ..pipeline.answer_generator import generate_candidate_answer
from ..pipeline.claim_extractor import extract_claims
from ..pipeline.claim_verifier import verify_claims
from ..pipeline.feature_extractor import _max_support_per_claim
from ..pipeline.freshness_scorer import aggregate_freshness, is_time_sensitive
from ..retrieval.bm25_retriever import BM25Index
from ..types import (
    STATE_TO_ACTION,
    BenchmarkSample,
    ClaimVerification,
    PipelineResult,
    StateFeatures,
)


_AMBIGUITY_PATTERNS = [
    r"\b(which|what)\s+is\s+(the\s+)?(best|better|right|correct|optimal|recommended)\b",
    r"\bshould\s+(i|we)\b",
    r"\b(recommend|prefer|favorite)\b",
    r"\b(how\s+good|how\s+effective)\b",
]


def _is_ambiguous(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _AMBIGUITY_PATTERNS)


def _max_refute_per_claim(
    verifications: Iterable[ClaimVerification],
    claim_ids: list[str],
) -> dict[str, float]:
    out = {cid: 0.0 for cid in claim_ids}
    for v in verifications:
        if v.relation == "refute" and v.claim_id in out:
            out[v.claim_id] = max(out[v.claim_id], v.confidence)
    return out


def run_verifier_heuristic(
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
    )

    claim_ids = [c.claim_id for c in cand.claims]
    sup = _max_support_per_claim(verifications, claim_ids)
    ref = _max_refute_per_claim(verifications, claim_ids)
    mean_support = float(np.mean(list(sup.values()))) if sup else 0.0
    has_per_claim_conflict = any(
        sup.get(cid, 0.0) >= 0.4 and ref.get(cid, 0.0) >= 0.3 for cid in claim_ids
    )

    time_sensitive = is_time_sensitive(sample.question)
    freshness = aggregate_freshness(docs)
    ambiguous = _is_ambiguous(sample.question)

    # Heuristic priority tree (mirrors the OESC rule order)
    if time_sensitive and freshness < 0.3:
        state = "outdated"
    elif has_per_claim_conflict:
        state = "conflicting"
    elif ambiguous and mean_support < 0.7:
        state = "underspecified"
    elif not cand.claims or mean_support < 0.2:
        state = "unanswerable"
    elif mean_support >= 0.5:
        state = "sufficient"
    else:
        state = "missing"

    action = STATE_TO_ACTION.get(state, "abstain")
    final_answer = cand.answer_text if action == "answer" else None
    budget = float(config.get("loop", {}).get("budget_calls", 30))

    return PipelineResult(
        sample_id=sample.id,
        predicted_state=state,
        predicted_action=action,
        features=StateFeatures(
            need_coverage=0.0,
            answer_support=mean_support,
            source_conflict=1.0 if has_per_claim_conflict else 0.0,
            answer_disagreement=0.0,
            claim_disagreement=0.0,
            retrieval_novelty=1.0,
            intent_ambiguity=1.0 if ambiguous else 0.0,
            freshness=freshness,
        ),
        candidate_answers=[cand],
        verifications=verifications,
        information_needs=[],
        final_answer=final_answer,
        cost={"llm_calls": 3.0, "rounds": 1.0, "budget": budget},
        method="verifier-heuristic",
    )
