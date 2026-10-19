"""Verify claims against evidence (support / refute / NEI / irrelevant)."""

from __future__ import annotations

from ..llm import LLMClient
from ..types import Claim, ClaimVerification, EvidenceDoc


VERIFY_PROMPT = """You are a strict evidence verifier.

You MUST output exactly one verification result for EACH (claim, evidence) pair listed below. \
There are {n_claims} claims and {n_docs} evidence documents, so output exactly {n_pairs} results.

For each pair, classify the relation:
- SUPPORT: evidence entails the claim
- REFUTE: evidence contradicts the claim (e.g. evidence says X did Y but claim says X did Z)
- NEI: evidence is about the same topic but doesn't entail or contradict
- IRRELEVANT: evidence is about a different topic

A claim like "Tommy wrote Cool Runnings" is REFUTED by evidence saying "Tommy wrote Little Giants" \
when the question asks which film Tommy wrote (the two are competing answers).

Use ONLY the evidence text; do not use outside knowledge.

Output JSON only:
{{
  "results": [
    {{"claim_id": "...", "evidence_doc_id": "...", "relation": "support|refute|nei|irrelevant", "confidence": 0.0-1.0, "rationale": "one short sentence"}}
  ]
}}

REQUIRED PAIRS (output exactly these {n_pairs} results, one per pair):
{required_pairs}

Claims:
{claims}

Evidence:
{evidence}
"""


JOINT_DOC_ID = "__joint__"

JOINT_VERIFY_PROMPT = """You are a strict evidence verifier for multi-hop questions.

Judge each claim against ALL evidence documents taken TOGETHER. A claim is SUPPORT \
if the documents jointly entail it, even when no single document does (e.g. one \
document links A to B and another gives B's property).

- SUPPORT: the combined evidence entails the claim
- REFUTE: the combined evidence contradicts the claim
- NEI: the combined evidence neither entails nor contradicts the claim

Use ONLY the evidence text; do not use outside knowledge.

Output JSON only:
{{
  "results": [
    {{"claim_id": "...", "relation": "support|refute|nei", "confidence": 0.0-1.0, "rationale": "one short sentence"}}
  ]
}}

Claims:
{claims}

Evidence:
{evidence}
"""


def _format_claims(claims: list[Claim]) -> str:
    return "\n".join(f"- {c.claim_id}: {c.text}" for c in claims)


def _format_evidence(docs: list[EvidenceDoc]) -> str:
    return "\n\n".join(f"[{d.doc_id} | {d.title}]\n{d.text}" for d in docs)


_RELATION_MAP = {
    "support": "support",
    "supports": "support",
    "entail": "support",
    "entailment": "support",
    "refute": "refute",
    "refutes": "refute",
    "contradict": "refute",
    "contradicts": "refute",
    "nei": "nei",
    "not_enough_information": "nei",
    "not enough information": "nei",
    "neutral": "nei",
    "irrelevant": "irrelevant",
}


def _normalize_relation(r: str) -> str:
    return _RELATION_MAP.get(r.strip().lower(), "nei")


def verify_claims(
    client: LLMClient,
    claims: list[Claim],
    evidence: list[EvidenceDoc],
    model: str = "default-chat-model",
    max_docs: int = 10,
    joint: bool = False,
) -> list[ClaimVerification]:
    if not claims or not evidence:
        return []
    evidence = evidence[:max_docs]
    required_pairs = []
    for c in claims:
        for d in evidence:
            required_pairs.append(f"  ({c.claim_id}, {d.doc_id})")
    n_pairs = len(required_pairs)
    prompt = VERIFY_PROMPT.format(
        claims=_format_claims(claims),
        evidence=_format_evidence(evidence),
        n_claims=len(claims),
        n_docs=len(evidence),
        n_pairs=n_pairs,
        required_pairs="\n".join(required_pairs),
    )
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        items = data.get("results", [])
    except Exception:
        items = []

    out: list[ClaimVerification] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        cid = str(item.get("claim_id", "")).strip()
        did = str(item.get("evidence_doc_id", "")).strip()
        if not cid or not did:
            continue
        if (cid, did) in seen:
            continue
        seen.add((cid, did))
        rel = _normalize_relation(str(item.get("relation", "nei")))
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        out.append(
            ClaimVerification(
                claim_id=cid,
                evidence_doc_id=did,
                relation=rel,
                confidence=max(0.0, min(1.0, conf)),
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    if joint:
        out.extend(
            verify_claims_joint(client=client, claims=claims, evidence=evidence, model=model)
        )
    return out


def verify_claims_joint(
    client: LLMClient,
    claims: list[Claim],
    evidence: list[EvidenceDoc],
    model: str = "default-chat-model",
) -> list[ClaimVerification]:
    if not claims or not evidence:
        return []
    prompt = JOINT_VERIFY_PROMPT.format(
        claims=_format_claims(claims), evidence=_format_evidence(evidence)
    )
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        items = data.get("results", [])
    except Exception:
        items = []
    valid_ids = {c.claim_id for c in claims}
    out: list[ClaimVerification] = []
    seen: set[str] = set()
    for item in items:
        cid = str(item.get("claim_id", "")).strip()
        if cid not in valid_ids or cid in seen:
            continue
        seen.add(cid)
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        out.append(
            ClaimVerification(
                claim_id=cid,
                evidence_doc_id=JOINT_DOC_ID,
                relation=_normalize_relation(str(item.get("relation", "nei"))),
                confidence=max(0.0, min(1.0, conf)),
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    return out
