"""Grounded answer generator. Produces answer + claims + missing-info note."""

from __future__ import annotations

from ..llm import LLMClient
from ..types import CandidateAnswer, Claim, EvidenceDoc


GENERATE_PROMPT = """You are a grounded QA model.

Answer the question using ONLY the provided evidence. You MUST always provide a \
direct answer based on the evidence, even if evidence is partial or conflicting. \
If sources disagree, pick the answer supported by the most evidence and note the \
conflict in missing_information. Do NOT use outside knowledge. \
Only say 'INSUFFICIENT EVIDENCE' if NO evidence is remotely related to the question.

Output JSON only with this schema:
{{
  "answer": "concise direct answer (always attempt one)",
  "atomic_claims": ["minimal verifiable claim 1", "..."],
  "missing_information": "conflicting info or gaps, or '' if none",
  "confidence": 0.0-1.0
}}

Question: {question}

Evidence:
{evidence}
"""


def _format_evidence(docs: list[EvidenceDoc]) -> str:
    if not docs:
        return "(no evidence)"
    parts = []
    for d in docs:
        parts.append(f"[{d.doc_id} | {d.title}]\n{d.text}")
    return "\n\n".join(parts)


def generate_candidate_answer(
    client: LLMClient,
    question: str,
    evidence: list[EvidenceDoc],
    evidence_view_idx: int = 0,
    model: str = "default-chat-model",
    temperature: float = 0.3,
) -> CandidateAnswer:
    prompt = GENERATE_PROMPT.format(question=question, evidence=_format_evidence(evidence))
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    except Exception:
        data = {
            "answer": "INSUFFICIENT EVIDENCE",
            "atomic_claims": [],
            "missing_information": "LLM call failed",
            "confidence": 0.0,
        }

    answer_text = str(data.get("answer", "")).strip()
    raw_claims = data.get("atomic_claims") or []
    claims = [
        Claim(
            claim_id=f"v{evidence_view_idx}_c{i}",
            text=str(c).strip(),
            source_answer_idx=evidence_view_idx,
        )
        for i, c in enumerate(raw_claims)
        if str(c).strip()
    ]
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return CandidateAnswer(
        answer_text=answer_text,
        claims=claims,
        confidence=confidence,
        evidence_view_idx=evidence_view_idx,
        missing_info=str(data.get("missing_information", "")).strip() or None,
    )
