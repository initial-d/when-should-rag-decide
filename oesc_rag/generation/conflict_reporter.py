"""Produce a structured conflict report when evidence contains contradictions.

Rather than picking a side, the report surfaces both positions, names the sources,
and (if possible) suggests how the conflict could be resolved (e.g. authoritative
source, more recent source).
"""

from __future__ import annotations

from ..llm import LLMClient
from ..types import (
    CandidateAnswer,
    Claim,
    ClaimVerification,
    ConflictReport,
    EvidenceDoc,
)


CONFLICT_REPORT_PROMPT = """You produce a short, structured conflict report when evidence disagrees.

Do NOT pick a side. Surface both positions, attribute each to the supporting evidence, \
and suggest how the conflict could be resolved.

Output JSON only:
{{
  "summary": "One sentence describing the disagreement.",
  "conflicting_passages": [
    {{"doc_id": "...", "position": "what this source claims"}}
  ],
  "resolution_note": "What additional info would resolve this (e.g. official source, latest data)."
}}

Question: {question}

Candidate answers from different evidence views:
{answers}

Conflicting claims:
{conflicts}

Evidence excerpts:
{evidence}
"""


def _format_evidence_excerpts(docs: list[EvidenceDoc], max_chars: int = 200) -> str:
    lines = []
    for d in docs[:8]:
        snippet = (d.text or "")[:max_chars].replace("\n", " ").strip()
        lines.append(f"- [{d.doc_id}] {d.title}: {snippet}")
    return "\n".join(lines)


def _identify_conflicting_claims(
    claims: list[Claim],
    verifications: list[ClaimVerification],
    sup_thr: float = 0.5,
    ref_thr: float = 0.5,
) -> list[Claim]:
    """Return claims that have both support and refute evidence above threshold."""
    sup_by: dict[str, float] = {}
    ref_by: dict[str, float] = {}
    for v in verifications:
        if v.relation == "support":
            sup_by[v.claim_id] = max(sup_by.get(v.claim_id, 0.0), v.confidence)
        elif v.relation == "refute":
            ref_by[v.claim_id] = max(ref_by.get(v.claim_id, 0.0), v.confidence)
    out: list[Claim] = []
    for c in claims:
        if sup_by.get(c.claim_id, 0.0) >= sup_thr and ref_by.get(c.claim_id, 0.0) >= ref_thr:
            out.append(c)
    return out


def generate_conflict_report(
    client: LLMClient,
    question: str,
    candidate_answers: list[CandidateAnswer],
    verifications: list[ClaimVerification],
    evidence: list[EvidenceDoc],
    model: str = "default-chat-model",
) -> ConflictReport:
    """Build a ConflictReport from the pipeline state."""
    all_claims: list[Claim] = []
    for a in candidate_answers:
        all_claims.extend(a.claims)
    conflicting = _identify_conflicting_claims(all_claims, verifications)

    answers_text = "\n".join(
        f"- View {a.evidence_view_idx}: {a.answer_text}"
        for a in candidate_answers
        if a.answer_text
    ) or "(no candidate answers)"

    conflicts_text = (
        "\n".join(f"- [{c.claim_id}] {c.text}" for c in conflicting) or "(no claim-level conflicts)"
    )

    prompt = CONFLICT_REPORT_PROMPT.format(
        question=question,
        answers=answers_text,
        conflicts=conflicts_text,
        evidence=_format_evidence_excerpts(evidence),
    )

    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception:
        data = {}

    summary = str(data.get("summary", "")).strip()
    if not summary:
        summary = "The available evidence contains conflicting claims; no single answer is fully supported."

    passages_raw = data.get("conflicting_passages") or []
    passages: list[dict[str, str]] = []
    for p in passages_raw:
        if isinstance(p, dict):
            passages.append(
                {
                    "doc_id": str(p.get("doc_id", "")).strip(),
                    "position": str(p.get("position", "")).strip(),
                }
            )

    resolution = str(data.get("resolution_note", "")).strip()

    return ConflictReport(
        summary=summary,
        conflicting_passages=passages,
        resolution_note=resolution,
    )
