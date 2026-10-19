"""Auditors for the sufficient/underspecified boundary.

These functions deliberately use only runtime-visible objects: the question,
retrieved evidence text, generated candidate answers, and verifier relations.
They do not inspect benchmark labels or construction metadata.
"""

from __future__ import annotations

import re

from ..types import CandidateAnswer, ClaimVerification, EvidenceDoc, InformationNeed


_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "with",
}


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in _STOP
    }


def _answer_spans(text: str) -> list[str]:
    spans = []
    for match in re.finditer(r"\bthe answer is\s+([^.;\n]+)", text or "", flags=re.I):
        val = match.group(1).strip()
        if val:
            spans.append(val)
    return spans


def criterion_ambiguity_score(
    question: str,
    evidence_docs: list[EvidenceDoc],
    answers: list[CandidateAnswer],
    verifications: list[ClaimVerification],
) -> float:
    """Estimate whether evidence supports multiple interpretations.

    High values indicate multiple evidence chunks answer neighbouring but
    distinct interpretations of the question without direct support/refute
    contradiction. This is the ``underspecified`` side of the boundary.
    """

    if len(evidence_docs) < 2:
        return 0.0
    q_toks = _tokens(question)
    doc_signals = []
    answer_values = set()
    for doc in evidence_docs[:8]:
        text = f"{doc.title} {doc.text}"
        doc_toks = _tokens(text)
        overlap = len(q_toks & doc_toks) / max(1, len(q_toks))
        spans = _answer_spans(doc.text)
        for span in spans:
            answer_values.add(span.lower())
        has_interpretive_question = "?" in doc.text[:180] or re.search(r"\b(which|what|when|who|where|how)\b", doc.text[:120], re.I)
        if overlap >= 0.35 and (spans or has_interpretive_question):
            doc_signals.append((overlap, bool(spans), has_interpretive_question))

    if len(doc_signals) < 2:
        return 0.0

    refutes = sum(1 for v in verifications if str(v.relation).lower() == "refute" and float(v.confidence) >= 0.5)
    supports = sum(1 for v in verifications if str(v.relation).lower() == "support" and float(v.confidence) >= 0.5)
    contradiction_pressure = min(1.0, refutes / max(1, supports))

    multiplicity = min(1.0, len(doc_signals) / 3.0)
    distinct_answer_bonus = 0.25 if len(answer_values) >= 2 else 0.0
    score = 0.70 * multiplicity + distinct_answer_bonus - 0.60 * contradiction_pressure
    return max(0.0, min(1.0, score))


def strict_sufficiency_score(
    question: str,
    needs: list[InformationNeed],
    evidence_docs: list[EvidenceDoc],
    answers: list[CandidateAnswer],
    verifications: list[ClaimVerification],
) -> float:
    """Estimate whether the current answer is exactly grounded.

    This complements ``critical_need_recall`` by penalising cases where only a
    subset of interpretive evidence chunks supports the chosen answer, or where
    support is mostly topically adjacent.
    """

    if not answers:
        return 0.0
    q_toks = _tokens(question)
    evidence_text = " ".join(f"{d.title} {d.text}" for d in evidence_docs[:8])
    evidence_toks = _tokens(evidence_text)
    q_evidence_overlap = len(q_toks & evidence_toks) / max(1, len(q_toks))

    support = sum(1 for v in verifications if str(v.relation).lower() == "support" and float(v.confidence) >= 0.5)
    refute = sum(1 for v in verifications if str(v.relation).lower() == "refute" and float(v.confidence) >= 0.5)
    nei = sum(1 for v in verifications if str(v.relation).lower() in {"nei", "not_enough_info"} and float(v.confidence) >= 0.5)
    total = max(1, support + refute + nei)
    support_ratio = support / total
    refute_penalty = refute / total
    nei_penalty = min(0.5, nei / total)

    multi_interp = criterion_ambiguity_score(question, evidence_docs, answers, verifications)
    need_count = len([n for n in needs if n.is_essential] or needs)
    need_bonus = 0.10 if need_count <= 1 else 0.0

    score = 0.45 * support_ratio + 0.35 * q_evidence_overlap + need_bonus
    score -= 0.40 * refute_penalty + 0.20 * nei_penalty + 0.30 * multi_interp
    return max(0.0, min(1.0, score))
