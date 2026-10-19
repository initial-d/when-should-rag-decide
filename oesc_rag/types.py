"""Shared dataclasses defining the contract between OESC-RAG modules.

Full version: 6 evidence states, 6 actions, 8-dim feature vector, iterative loop tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


STATE_LABELS = (
    "sufficient",
    "missing",
    "conflicting",
    "unanswerable",
    "underspecified",
    "outdated",
)
ACTION_LABELS = (
    "answer",
    "retrieve-more",
    "report-conflict",
    "abstain",
    "ask-clarification",
    "retrieve-fresh-source",
)

STATE_TO_ACTION = {
    "sufficient": "answer",
    "missing": "retrieve-more",
    "conflicting": "report-conflict",
    "unanswerable": "abstain",
    "underspecified": "ask-clarification",
    "outdated": "retrieve-fresh-source",
}


@dataclass
class EvidenceDoc:
    doc_id: str
    title: str
    text: str
    source: str = "wiki"
    timestamp: Optional[str] = None
    authority: str = "neutral"
    freshness_score: float = 0.5
    authority_score: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InformationNeed:
    need_id: str
    description: str
    need_type: str
    is_essential: bool = True
    suggested_query: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    claim_id: str
    text: str
    source_answer_idx: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimVerification:
    claim_id: str
    evidence_doc_id: str
    relation: str  # support | refute | nei | irrelevant
    confidence: float
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateAnswer:
    answer_text: str
    claims: list[Claim]
    confidence: float
    evidence_view_idx: int
    missing_info: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "claims": [c.to_dict() for c in self.claims],
            "confidence": self.confidence,
            "evidence_view_idx": self.evidence_view_idx,
            "missing_info": self.missing_info,
        }


@dataclass
class ClarificationQuestion:
    """System-generated clarification for underspecified questions."""
    text: str
    latent_criteria: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConflictReport:
    """Structured report for conflicting evidence states."""
    summary: str
    conflicting_passages: list[dict[str, str]] = field(default_factory=list)  # [{doc_id, claim, ...}]
    resolution_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StateFeatures:
    need_coverage: float
    answer_support: float
    source_conflict: float
    answer_disagreement: float
    claim_disagreement: float
    retrieval_novelty: float = 1.0
    intent_ambiguity: float = 0.0
    freshness: float = 1.0
    # 9th feature (P3): fraction of essential needs *directly* addressed by raw
    # evidence, judged against chunk text rather than supported claims. Default
    # 1.0 = "no signal" so legacy results load unchanged.
    critical_need_recall: float = 1.0
    # Boundary auditors. Defaults keep legacy logs unchanged.
    criterion_ambiguity: float = 0.0
    strict_sufficiency: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def as_vector(self) -> list[float]:
        return [
            self.need_coverage,
            self.answer_support,
            self.source_conflict,
            self.answer_disagreement,
            self.claim_disagreement,
            self.retrieval_novelty,
            self.intent_ambiguity,
            self.freshness,
            self.critical_need_recall,
            self.criterion_ambiguity,
            self.strict_sufficiency,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "need_coverage",
            "answer_support",
            "source_conflict",
            "answer_disagreement",
            "claim_disagreement",
            "retrieval_novelty",
            "intent_ambiguity",
            "freshness",
            "critical_need_recall",
            "criterion_ambiguity",
            "strict_sufficiency",
        ]


@dataclass
class RetrievalRound:
    """One iteration of the information-refinement loop."""
    round_idx: int
    queries: list[str]
    new_doc_ids: list[str]
    state: str
    action: str
    features: StateFeatures
    llm_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "queries": self.queries,
            "new_doc_ids": self.new_doc_ids,
            "state": self.state,
            "action": self.action,
            "features": self.features.to_dict(),
            "llm_calls": self.llm_calls,
        }


@dataclass
class BenchmarkSample:
    id: str
    question: str
    gold_state: str
    gold_action: str
    gold_answer: str
    initial_evidence: list[EvidenceDoc]
    evidence_pool: list[EvidenceDoc]
    supporting_doc_ids: list[str]
    conflict_doc: Optional[EvidenceDoc] = None
    source_qid: Optional[str] = None
    perturbation_notes: str = ""
    # New fields for full version
    latent_criteria: list[str] = field(default_factory=list)  # for underspecified samples
    original_question: Optional[str] = None  # for underspecified samples (rewritten q)
    fresh_doc_ids: list[str] = field(default_factory=list)  # for outdated samples
    time_sensitive: bool = False  # whether q expects recent info

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "gold_state": self.gold_state,
            "gold_action": self.gold_action,
            "gold_answer": self.gold_answer,
            "initial_evidence": [e.to_dict() for e in self.initial_evidence],
            "evidence_pool": [e.to_dict() for e in self.evidence_pool],
            "supporting_doc_ids": self.supporting_doc_ids,
            "conflict_doc": self.conflict_doc.to_dict() if self.conflict_doc else None,
            "source_qid": self.source_qid,
            "perturbation_notes": self.perturbation_notes,
            "latent_criteria": self.latent_criteria,
            "original_question": self.original_question,
            "fresh_doc_ids": self.fresh_doc_ids,
            "time_sensitive": self.time_sensitive,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchmarkSample":
        return cls(
            id=d["id"],
            question=d["question"],
            gold_state=d["gold_state"],
            gold_action=d["gold_action"],
            gold_answer=d["gold_answer"],
            initial_evidence=[EvidenceDoc(**e) for e in d["initial_evidence"]],
            evidence_pool=[EvidenceDoc(**e) for e in d["evidence_pool"]],
            supporting_doc_ids=d["supporting_doc_ids"],
            conflict_doc=EvidenceDoc(**d["conflict_doc"]) if d.get("conflict_doc") else None,
            source_qid=d.get("source_qid"),
            perturbation_notes=d.get("perturbation_notes", ""),
            latent_criteria=d.get("latent_criteria") or [],
            original_question=d.get("original_question"),
            fresh_doc_ids=d.get("fresh_doc_ids") or [],
            time_sensitive=bool(d.get("time_sensitive", False)),
        )


@dataclass
class PipelineResult:
    sample_id: str
    predicted_state: str
    predicted_action: str
    features: StateFeatures
    candidate_answers: list[CandidateAnswer]
    verifications: list[ClaimVerification]
    information_needs: list[InformationNeed] = field(default_factory=list)
    final_answer: Optional[str] = None
    clarification: Optional[ClarificationQuestion] = None
    conflict_report: Optional[ConflictReport] = None
    rounds: list[RetrievalRound] = field(default_factory=list)
    cost: dict[str, float] = field(default_factory=dict)
    method: str = "oesc-rag"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "predicted_state": self.predicted_state,
            "predicted_action": self.predicted_action,
            "features": self.features.to_dict(),
            "candidate_answers": [a.to_dict() for a in self.candidate_answers],
            "verifications": [v.to_dict() for v in self.verifications],
            "information_needs": [n.to_dict() for n in self.information_needs],
            "final_answer": self.final_answer,
            "clarification": self.clarification.to_dict() if self.clarification else None,
            "conflict_report": self.conflict_report.to_dict() if self.conflict_report else None,
            "rounds": [r.to_dict() for r in self.rounds],
            "cost": self.cost,
            "method": self.method,
            "metadata": self.metadata,
        }
