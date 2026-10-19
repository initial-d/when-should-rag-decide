"""Compute the 9-dim state feature vector from pipeline outputs.

Full version: 6 original features + intent_ambiguity + freshness +
critical_need_recall (P3). Also includes LLM-based need coverage and
cross-answer contradiction check.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from ..types import (
    CandidateAnswer,
    Claim,
    ClaimVerification,
    EvidenceDoc,
    InformationNeed,
    StateFeatures,
)
from .boundary_auditor import criterion_ambiguity_score, strict_sufficiency_score


def _max_support_per_claim(
    verifications: list[ClaimVerification], claim_ids: list[str]
) -> dict[str, float]:
    best: dict[str, float] = {cid: 0.0 for cid in claim_ids}
    for v in verifications:
        if v.claim_id in best and v.relation == "support":
            if v.confidence > best[v.claim_id]:
                best[v.claim_id] = v.confidence
    return best


def _max_refute_per_claim(
    verifications: list[ClaimVerification], claim_ids: list[str]
) -> dict[str, float]:
    best: dict[str, float] = {cid: 0.0 for cid in claim_ids}
    for v in verifications:
        if v.claim_id in best and v.relation == "refute":
            if v.confidence > best[v.claim_id]:
                best[v.claim_id] = v.confidence
    return best


def _tokens(text: str) -> list[str]:
    import re
    return [t.lower() for t in re.findall(r"\w+", text or "")]


# ---------------------------------------------------------------------------
# Need Coverage (token-overlap version for backward compat / no-LLM ablation)
# ---------------------------------------------------------------------------

def _need_coverage_token(
    needs: list[InformationNeed],
    claims: list[Claim],
    sup_per_claim: dict[str, float],
    threshold: float = 0.5,
) -> float:
    essential = [n for n in needs if n.is_essential] or needs
    if not essential:
        return 1.0
    supported_claims = [c for c in claims if sup_per_claim.get(c.claim_id, 0.0) >= threshold]
    if not supported_claims:
        return 0.0
    covered = 0
    for n in essential:
        n_tokens = set(_tokens(n.description))
        if not n_tokens:
            continue
        if any(
            len(n_tokens & set(_tokens(c.text))) / max(1, len(n_tokens)) >= 0.3
            for c in supported_claims
        ):
            covered += 1
    return covered / len(essential)


# ---------------------------------------------------------------------------
# Need Coverage (LLM-based: more accurate, fixes missing→sufficient leak)
# ---------------------------------------------------------------------------

_NEED_COVERAGE_PROMPT = """You judge whether each information need is covered by the evidence claims.

For each need, output:
- "covered" if a claim directly addresses this need with evidence support
- "partial" if a claim touches on the need but doesn't fully answer it
- "missing" if no claim addresses this need

Output JSON only:
{{
  "coverage": [
    {{"need_id": "...", "status": "covered|partial|missing", "reason": "short"}}
  ]
}}

Information needs:
{needs}

Supported claims (claims with evidence support):
{claims}
"""


def _need_coverage_llm(
    needs: list[InformationNeed],
    claims: list[Claim],
    sup_per_claim: dict[str, float],
    client,
    model: str = "default-chat-model",
    threshold: float = 0.5,
) -> float:
    """LLM-based need-claim coverage assessment."""
    essential = [n for n in needs if n.is_essential] or needs
    if not essential:
        return 1.0
    supported_claims = [c for c in claims if sup_per_claim.get(c.claim_id, 0.0) >= threshold]
    if not supported_claims:
        return 0.0

    needs_text = "\n".join(f"- [{n.need_id}] {n.description}" for n in essential)
    claims_text = "\n".join(f"- [{c.claim_id}] {c.text}" for c in supported_claims)
    prompt = _NEED_COVERAGE_PROMPT.format(needs=needs_text, claims=claims_text)

    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        items = data.get("coverage") or []
        covered = sum(1 for item in items if str(item.get("status", "")).strip().lower() == "covered")
        return covered / len(essential)
    except Exception:
        # Fallback to token-based
        return _need_coverage_token(essential, claims, sup_per_claim, threshold)


# ---------------------------------------------------------------------------
# Critical-Need Recall (P3): checks essential needs against RAW evidence chunks,
# not supported claims. Designed to catch MISSING samples where the answer
# generator confidently fabricates from topically-adjacent evidence.
# ---------------------------------------------------------------------------

_CRITICAL_NEED_PROMPT = """You are a strict evidence auditor. For each essential information need, judge whether ANY single evidence chunk DIRECTLY addresses it.

Definitions:
- DIRECTLY_ADDRESSED = a chunk contains the specific entity / time / value / relation the need asks for, not just the topic.
- TOPICALLY_RELATED = a chunk discusses the same topic, but does NOT supply the specific fact the need requests.
- MISSING = no chunk discusses the need at all.

Be strict: if the chunks only mention the topic without giving the requested specific fact, that is TOPICALLY_RELATED, not DIRECTLY_ADDRESSED.

Output JSON only:
{{
  "needs": [
    {{"need_id": "...", "status": "directly_addressed|topically_related|missing", "reason": "short"}}
  ]
}}

Question:
{question}

Essential information needs:
{needs}

Evidence chunks (raw retrieved text):
{evidence}
"""


def _critical_need_recall_llm(
    question: str,
    needs: list[InformationNeed],
    evidence_docs: list[EvidenceDoc],
    client,
    model: str = "default-chat-model",
    max_chunks: int = 10,
    max_chars_per_chunk: int = 600,
) -> float:
    """Fraction of essential needs DIRECTLY addressed by raw evidence chunks.

    Unlike `_need_coverage_llm`, this function inspects raw evidence text rather
    than supported claims, which prevents the leak where an over-confident
    answer generator fabricates from topically-adjacent evidence and the
    coverage LLM rubber-stamps it.
    """
    essential = [n for n in needs if n.is_essential] or needs
    if not essential:
        return 1.0
    if not evidence_docs:
        return 0.0

    needs_text = "\n".join(f"- [{n.need_id}] {n.description}" for n in essential)
    chunks = evidence_docs[:max_chunks]
    evidence_text = "\n\n".join(
        f"[{e.doc_id}] {(e.title + ': ') if e.title else ''}{(e.text or '')[:max_chars_per_chunk]}"
        for e in chunks
    )
    prompt = _CRITICAL_NEED_PROMPT.format(
        question=question,
        needs=needs_text,
        evidence=evidence_text,
    )

    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        items = data.get("needs") or []
        directly = sum(
            1 for item in items
            if str(item.get("status", "")).strip().lower() == "directly_addressed"
        )
        return directly / len(essential)
    except Exception:
        # Fail open (1.0) so a transient LLM error doesn't flip a sample to MISSING.
        return 1.0


# ---------------------------------------------------------------------------
# Answer Support
# ---------------------------------------------------------------------------

def _answer_support(claims: list[Claim], sup_per_claim: dict[str, float]) -> float:
    if not claims:
        return 0.0
    vals = [sup_per_claim.get(c.claim_id, 0.0) for c in claims]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Source Conflict
# ---------------------------------------------------------------------------

def _source_conflict(
    claims: list[Claim],
    verifications: list[ClaimVerification],
    answers: list[CandidateAnswer] | None = None,
    sup_thr: float = 0.5,
    ref_thr: float = 0.5,
    group_claims: bool = True,
    cross_answer_conflict: float = 0.0,
) -> float:
    """Fraction of claims with both support and refute evidence.

    Also folds in:
    - answer-level token disagreement (from MVP)
    - cross_answer_conflict score from LLM contradiction check (new)
    """
    if not claims:
        return max(0.0, cross_answer_conflict)

    tokens_by_id: dict[str, set[str]] = {c.claim_id: set(_tokens(c.text)) for c in claims}
    groups: list[list[str]] = []
    if not group_claims:
        groups = [[c.claim_id] for c in claims]
    else:
        for c in claims:
            placed = False
            for g in groups:
                ref_toks = tokens_by_id[g[0]]
                cur_toks = tokens_by_id[c.claim_id]
                if not ref_toks and not cur_toks:
                    continue
                jacc = len(ref_toks & cur_toks) / max(1, len(ref_toks | cur_toks))
                if jacc >= 0.6:
                    g.append(c.claim_id)
                    placed = True
                    break
            if not placed:
                groups.append([c.claim_id])

    conflicting_groups = 0
    for g in groups:
        g_set = set(g)
        max_sup = 0.0
        max_ref = 0.0
        for v in verifications:
            if v.claim_id in g_set:
                if v.relation == "support":
                    max_sup = max(max_sup, v.confidence)
                elif v.relation == "refute":
                    max_ref = max(max_ref, v.confidence)
        if max_sup >= sup_thr and max_ref >= ref_thr:
            conflicting_groups += 1

    claim_conflict = conflicting_groups / len(groups) if groups else 0.0

    # Fallback: token-level answer disagreement
    answer_conflict = 0.0
    if answers and len(answers) >= 2:
        ans_texts = [set(_tokens(a.answer_text)) for a in answers if a.answer_text]
        if len(ans_texts) >= 2:
            pairwise_jacc = []
            for i in range(len(ans_texts)):
                for j in range(i + 1, len(ans_texts)):
                    a, b = ans_texts[i], ans_texts[j]
                    jacc = len(a & b) / max(1, len(a | b))
                    pairwise_jacc.append(jacc)
            mean_jacc = float(np.mean(pairwise_jacc))
            if mean_jacc < 0.3:
                answer_conflict = 0.5

    return max(claim_conflict, answer_conflict, cross_answer_conflict)


# ---------------------------------------------------------------------------
# Answer Disagreement
# ---------------------------------------------------------------------------

def _answer_disagreement(
    answers: list[CandidateAnswer],
    embedder: Optional[object] = None,
) -> float:
    if len(answers) < 2:
        return 0.0
    texts = [a.answer_text for a in answers]
    if not all(t.strip() for t in texts):
        return 0.0
    if embedder is None:
        sims = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a = set(_tokens(texts[i]))
                b = set(_tokens(texts[j]))
                if not a and not b:
                    sims.append(1.0)
                    continue
                sims.append(len(a & b) / max(1, len(a | b)))
        return float(1.0 - np.mean(sims)) if sims else 0.0
    try:
        embs = embedder.encode(texts, normalize_embeddings=True)
        embs = np.asarray(embs)
        sims = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(float(embs[i] @ embs[j]))
        return float(1.0 - np.mean(sims)) if sims else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Claim Disagreement
# ---------------------------------------------------------------------------

def _claim_disagreement(answers: list[CandidateAnswer]) -> float:
    if len(answers) < 2:
        return 0.0
    all_claims: list[tuple[int, set[str]]] = []
    for a in answers:
        for c in a.claims:
            all_claims.append((a.evidence_view_idx, set(_tokens(c.text))))
    if not all_claims:
        return 0.0
    groups: list[list[tuple[int, set[str]]]] = []
    for view_idx, toks in all_claims:
        placed = False
        for g in groups:
            ref = g[0][1]
            if not toks and not ref:
                continue
            jacc = len(toks & ref) / max(1, len(toks | ref))
            if jacc >= 0.5:
                g.append((view_idx, toks))
                placed = True
                break
        if not placed:
            groups.append([(view_idx, toks)])
    unique_view_groups = sum(1 for g in groups if len({v for v, _ in g}) == 1)
    return unique_view_groups / max(1, len(groups))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_features(
    needs: list[InformationNeed],
    answers: list[CandidateAnswer],
    verifications: list[ClaimVerification],
    embedder: Optional[object] = None,
    retrieval_novelty: float = 1.0,
    intent_ambiguity: float = 0.0,
    freshness: float = 1.0,
    cross_answer_conflict: float = 0.0,
    client=None,
    model: str = "default-chat-model",
    ablation: Optional[dict] = None,
    question: str = "",
    evidence_docs: Optional[list[EvidenceDoc]] = None,
) -> StateFeatures:
    """Extract 9-dim feature vector.

    New parameters vs MVP:
      - intent_ambiguity: from intent_analyzer (0–1)
      - freshness: from freshness_scorer (0–1)
      - cross_answer_conflict: from cross_answer_check (0–1)
      - client + model: for LLM-based need coverage
      - question + evidence_docs: for critical-need recall (P3)

    ablation dict can disable components:
      - no_multiview: collapse to single view
      - no_claim_graph: set source_conflict=0, claim_disagreement=0
      - no_need_coverage: set need_coverage=1.0
      - no_source_conflict: set source_conflict=0
      - no_verifier: set answer_support=0, source_conflict=0
      - no_intent_ambiguity: set intent_ambiguity=0
      - no_freshness: set freshness=1.0
      - no_llm_coverage: use token-overlap instead of LLM-based coverage
      - no_critical_need: skip critical-need recall (default 1.0)
    """
    abl = ablation or {}
    all_claims: list[Claim] = []
    for a in answers:
        all_claims.extend(a.claims)
    claim_ids = [c.claim_id for c in all_claims]

    if abl.get("no_verifier"):
        sup = {cid: 0.0 for cid in claim_ids}
        ref = {cid: 0.0 for cid in claim_ids}
    else:
        sup = _max_support_per_claim(verifications, claim_ids)
        ref = _max_refute_per_claim(verifications, claim_ids)

    # Need coverage
    if abl.get("no_need_coverage"):
        need_cov = 1.0
    elif client is not None and not abl.get("no_llm_coverage"):
        need_cov = _need_coverage_llm(needs, all_claims, sup, client, model)
    else:
        need_cov = _need_coverage_token(needs, all_claims, sup)

    ans_sup = 0.0 if abl.get("no_verifier") else _answer_support(all_claims, sup)

    if abl.get("no_claim_graph") or abl.get("no_source_conflict") or abl.get("no_verifier"):
        src_conf = 0.0
    else:
        src_conf = _source_conflict(
            all_claims, verifications, answers,
            cross_answer_conflict=0.0 if abl.get("no_source_conflict") else cross_answer_conflict,
        )

    if abl.get("no_multiview"):
        ans_dis = 0.0
        clm_dis = 0.0
    else:
        ans_dis = _answer_disagreement(answers, embedder)
        clm_dis = 0.0 if abl.get("no_claim_graph") else _claim_disagreement(answers)

    ia = 0.0 if abl.get("no_intent_ambiguity") else intent_ambiguity
    fr = 1.0 if abl.get("no_freshness") else freshness

    # Critical-need recall (P3)
    if abl.get("no_critical_need") or client is None or not evidence_docs:
        cnr = 1.0
    else:
        cnr = _critical_need_recall_llm(question, needs, evidence_docs, client, model)

    boundary_docs = evidence_docs or []
    criterion_amb = 0.0 if abl.get("no_intent_ambiguity") else criterion_ambiguity_score(
        question,
        boundary_docs,
        answers,
        verifications,
    )
    strict_suff = strict_sufficiency_score(
        question,
        needs,
        boundary_docs,
        answers,
        verifications,
    )

    return StateFeatures(
        need_coverage=need_cov,
        answer_support=ans_sup,
        source_conflict=src_conf,
        answer_disagreement=ans_dis,
        claim_disagreement=clm_dis,
        retrieval_novelty=retrieval_novelty,
        intent_ambiguity=ia,
        freshness=fr,
        critical_need_recall=cnr,
        criterion_ambiguity=criterion_amb,
        strict_sufficiency=strict_suff,
    )
