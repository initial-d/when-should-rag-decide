"""Serialization helpers for saved OESC-RAG pipeline results."""

from __future__ import annotations

import json
from typing import Any

from ..types import (
    CandidateAnswer,
    Claim,
    ClaimVerification,
    ClarificationQuestion,
    ConflictReport,
    InformationNeed,
    PipelineResult,
    RetrievalRound,
    StateFeatures,
)


def result_from_dict(data: dict[str, Any]) -> PipelineResult:
    """Reconstruct a PipelineResult from its JSON representation."""
    features = StateFeatures(**data["features"])
    candidates = [
        CandidateAnswer(
            answer_text=item["answer_text"],
            claims=[Claim(**claim) for claim in item["claims"]],
            confidence=item["confidence"],
            evidence_view_idx=item["evidence_view_idx"],
            missing_info=item.get("missing_info"),
        )
        for item in data.get("candidate_answers", [])
    ]
    verifications = [ClaimVerification(**item) for item in data.get("verifications", [])]
    needs = [InformationNeed(**item) for item in data.get("information_needs", [])]
    rounds = []
    for item in data.get("rounds") or []:
        round_features = StateFeatures(**item["features"]) if item.get("features") else features
        rounds.append(
            RetrievalRound(
                round_idx=item["round_idx"],
                queries=item.get("queries", []),
                new_doc_ids=item.get("new_doc_ids", []),
                state=item.get("state", ""),
                action=item.get("action", ""),
                features=round_features,
                llm_calls=item.get("llm_calls", 0),
            )
        )

    clarification = (
        ClarificationQuestion(**data["clarification"])
        if data.get("clarification")
        else None
    )
    conflict_report = (
        ConflictReport(**data["conflict_report"])
        if data.get("conflict_report")
        else None
    )
    return PipelineResult(
        sample_id=data["sample_id"],
        predicted_state=data["predicted_state"],
        predicted_action=data["predicted_action"],
        features=features,
        candidate_answers=candidates,
        verifications=verifications,
        information_needs=needs,
        final_answer=data.get("final_answer"),
        clarification=clarification,
        conflict_report=conflict_report,
        rounds=rounds,
        cost=data.get("cost") or {},
        method=data.get("method", "unknown"),
        metadata=data.get("metadata", {}),
    )


def load_results(path: str) -> list[PipelineResult]:
    """Load valid saved results, skipping explicit errors and malformed rows."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    results = []
    for item in data:
        if "error" in item or "features" not in item:
            continue
        try:
            results.append(result_from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return results
