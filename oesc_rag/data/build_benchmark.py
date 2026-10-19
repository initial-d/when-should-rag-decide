"""Construct the OpenEvidenceStateQA benchmark from HotpotQA.

Supports all 6 evidence states: sufficient / missing / conflicting / unanswerable /
underspecified / outdated.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from ..types import BenchmarkSample, EvidenceDoc
from .inject_conflict import generate_conflict_paragraph
from .inject_outdated import generate_outdated_versions
from .inject_underspecified import generate_underspecified_question
from .load_hotpotqa import (
    load_hotpotqa_samples,
    non_supporting_paragraph_indices,
    paragraph_text,
    supporting_paragraph_indices,
)
from ..llm import LLMClient


DEFAULT_STATES = (
    "sufficient",
    "missing",
    "conflicting",
    "unanswerable",
    "underspecified",
    "outdated",
)
STATE_TO_ACTION = {
    "sufficient": "answer",
    "missing": "retrieve-more",
    "conflicting": "report-conflict",
    "unanswerable": "abstain",
    "underspecified": "ask-clarification",
    "outdated": "retrieve-fresh-source",
}


def _paragraph_to_doc(
    record: dict, idx: int, prefix: str, freshness: float = 0.5
) -> EvidenceDoc:
    return EvidenceDoc(
        doc_id=f"{prefix}_{idx}",
        title=record["context"]["title"][idx],
        text=paragraph_text(record, idx),
        source="hotpotqa",
        freshness_score=freshness,
    )


def _build_sufficient(record: dict, sid: str) -> Optional[BenchmarkSample]:
    sup_idx = supporting_paragraph_indices(record)
    non_idx = non_supporting_paragraph_indices(record)
    if len(sup_idx) < 2:
        return None
    init_docs = [_paragraph_to_doc(record, i, sid) for i in sup_idx]
    extra = [_paragraph_to_doc(record, i, sid) for i in non_idx[:2]]
    pool = init_docs + extra
    return BenchmarkSample(
        id=sid,
        question=record["question"],
        gold_state="sufficient",
        gold_action="answer",
        gold_answer=record["answer"],
        initial_evidence=init_docs,
        evidence_pool=pool,
        supporting_doc_ids=[d.doc_id for d in init_docs],
        source_qid=record["_id"],
        perturbation_notes="all supporting paragraphs kept",
    )


def _build_missing(record: dict, sid: str) -> Optional[BenchmarkSample]:
    sup_idx = supporting_paragraph_indices(record)
    non_idx = non_supporting_paragraph_indices(record)
    if len(sup_idx) < 2:
        return None
    keep = sup_idx[:1]
    hidden = sup_idx[1:]
    init_docs = [_paragraph_to_doc(record, i, sid) for i in keep]
    init_docs += [_paragraph_to_doc(record, i, sid) for i in non_idx[:2]]
    hidden_docs = [_paragraph_to_doc(record, i, sid) for i in hidden]
    pool = init_docs + hidden_docs
    return BenchmarkSample(
        id=sid,
        question=record["question"],
        gold_state="missing",
        gold_action="retrieve-more",
        gold_answer=record["answer"],
        initial_evidence=init_docs,
        evidence_pool=pool,
        supporting_doc_ids=[d.doc_id for d in init_docs + hidden_docs
                            if d.title in record["supporting_facts"]["title"]],
        source_qid=record["_id"],
        perturbation_notes=f"hidden {len(hidden)} supporting paragraph(s) from initial evidence",
    )


def _build_unanswerable(record: dict, sid: str) -> Optional[BenchmarkSample]:
    non_idx = non_supporting_paragraph_indices(record)
    if len(non_idx) < 2:
        return None
    init_docs = [_paragraph_to_doc(record, i, sid) for i in non_idx[:3]]
    pool = list(init_docs)
    return BenchmarkSample(
        id=sid,
        question=record["question"],
        gold_state="unanswerable",
        gold_action="abstain",
        gold_answer=record["answer"],
        initial_evidence=init_docs,
        evidence_pool=pool,
        supporting_doc_ids=[],
        source_qid=record["_id"],
        perturbation_notes="all supporting paragraphs removed from pool",
    )


def _build_conflicting(record: dict, sid: str, client: LLMClient, model: str) -> Optional[BenchmarkSample]:
    sup_idx = supporting_paragraph_indices(record)
    if len(sup_idx) < 2:
        return None
    init_docs = [_paragraph_to_doc(record, i, sid) for i in sup_idx]
    supporting_text = "\n\n".join(d.text for d in init_docs)
    conflict = generate_conflict_paragraph(
        client=client,
        question=record["question"],
        gold_answer=record["answer"],
        supporting_text=supporting_text,
        model=model,
    )
    if not conflict["paragraph"]:
        return None
    conflict_doc = EvidenceDoc(
        doc_id=f"{sid}_conflict",
        title=f"Alt source on {record['context']['title'][sup_idx[0]]}",
        text=conflict["paragraph"],
        source="generated_conflict",
        authority="unverified",
        freshness_score=0.5,
    )
    pool = init_docs + [conflict_doc]
    init = init_docs + [conflict_doc]
    return BenchmarkSample(
        id=sid,
        question=record["question"],
        gold_state="conflicting",
        gold_action="report-conflict",
        gold_answer=record["answer"],
        initial_evidence=init,
        evidence_pool=pool,
        supporting_doc_ids=[d.doc_id for d in init_docs],
        conflict_doc=conflict_doc,
        source_qid=record["_id"],
        perturbation_notes=f"injected conflict; alt={conflict['alternative_answer']}",
    )


def _build_underspecified(
    record: dict, sid: str, client: LLMClient, model: str
) -> Optional[BenchmarkSample]:
    """Rewrite the question into an ambiguous form; keep evidence intact."""
    sup_idx = supporting_paragraph_indices(record)
    non_idx = non_supporting_paragraph_indices(record)
    if len(sup_idx) < 2:
        return None
    out = generate_underspecified_question(
        client=client,
        question=record["question"],
        gold_answer=record["answer"],
        model=model,
    )
    if not out["rewritten_question"]:
        return None
    init_docs = [_paragraph_to_doc(record, i, sid) for i in sup_idx]
    extra = [_paragraph_to_doc(record, i, sid) for i in non_idx[:2]]
    pool = init_docs + extra
    return BenchmarkSample(
        id=sid,
        question=out["rewritten_question"],
        gold_state="underspecified",
        gold_action="ask-clarification",
        gold_answer=record["answer"],
        initial_evidence=init_docs,
        evidence_pool=pool,
        supporting_doc_ids=[d.doc_id for d in init_docs],
        source_qid=record["_id"],
        perturbation_notes="question rewritten to hide criterion",
        latent_criteria=out["latent_criteria"],
        original_question=record["question"],
    )


def _build_outdated(
    record: dict, sid: str, client: LLMClient, model: str
) -> Optional[BenchmarkSample]:
    """Replace supporting paragraph with old version; pool also contains fresh version."""
    sup_idx = supporting_paragraph_indices(record)
    non_idx = non_supporting_paragraph_indices(record)
    if len(sup_idx) < 1:
        return None
    # Use the first supporting paragraph as the basis for old/new versions
    first_sup_idx = sup_idx[0]
    supporting_text = paragraph_text(record, first_sup_idx)
    out = generate_outdated_versions(
        client=client,
        question=record["question"],
        gold_answer=record["answer"],
        supporting_text=supporting_text,
        model=model,
    )
    if not out["old_paragraph"]:
        return None

    base_title = record["context"]["title"][first_sup_idx]

    old_doc = EvidenceDoc(
        doc_id=f"{sid}_old",
        title=f"{base_title} ({out['old_year']})",
        text=out["old_paragraph"],
        source="generated_outdated",
        timestamp=str(out["old_year"]),
        authority="neutral",
        freshness_score=0.1,
    )
    new_doc = EvidenceDoc(
        doc_id=f"{sid}_new",
        title=f"{base_title} ({out['new_year']})",
        text=out["new_paragraph"],
        source="generated_fresh",
        timestamp=str(out["new_year"]),
        authority="neutral",
        freshness_score=0.95,
    )

    # Distractors (non-supporting) for realism
    extras = [_paragraph_to_doc(record, i, sid) for i in non_idx[:2]]
    init_docs = [old_doc] + extras  # initial sees only the OLD version
    pool = [old_doc, new_doc] + extras

    return BenchmarkSample(
        id=sid,
        question=record["question"],
        gold_state="outdated",
        gold_action="retrieve-fresh-source",
        gold_answer=str(out["updated_answer"]),
        initial_evidence=init_docs,
        evidence_pool=pool,
        supporting_doc_ids=[new_doc.doc_id],
        source_qid=record["_id"],
        perturbation_notes=(
            f"initial=old({out['old_year']})={out['old_answer']}; "
            f"fresh({out['new_year']})={out['updated_answer']}"
        ),
        fresh_doc_ids=[new_doc.doc_id],
        time_sensitive=bool(out["time_sensitive"]),
    )


def build_benchmark(
    n_source: int = 600,
    per_state: int = 40,
    seed: int = 42,
    cache_dir: str = "data/raw",
    client: Optional[LLMClient] = None,
    models: Optional[dict] = None,
    states: tuple[str, ...] = DEFAULT_STATES,
) -> list[BenchmarkSample]:
    """Build a balanced benchmark of `len(states) * per_state` samples."""
    raw = load_hotpotqa_samples(n=n_source, seed=seed, cache_dir=cache_dir)
    rng = random.Random(seed)
    rng.shuffle(raw)

    by_state: dict[str, list[BenchmarkSample]] = {k: [] for k in states}
    used_qids: set[str] = set()
    raw_iter = iter(raw)

    while any(len(by_state[s]) < per_state for s in states):
        try:
            record = next(raw_iter)
        except StopIteration:
            break
        if record["_id"] in used_qids:
            continue
        state = _pick_next_state(by_state, per_state, list(states))
        if state is None:
            break
        sample = _build_for_state(
            record, state, len(used_qids), client, models or {}
        )
        if sample is None:
            continue
        by_state[state].append(sample)
        used_qids.add(record["_id"])

    out: list[BenchmarkSample] = []
    for s in states:
        out.extend(by_state[s])
    return out


def _pick_next_state(
    by_state: dict, per_state: int, targets: list[str]
) -> Optional[str]:
    for s in targets:
        if len(by_state[s]) < per_state:
            return s
    return None


def _build_for_state(
    record: dict,
    state: str,
    idx: int,
    client: Optional[LLMClient],
    models: dict,
) -> Optional[BenchmarkSample]:
    sid = f"{state[:4]}_{idx:04d}"
    if state == "sufficient":
        return _build_sufficient(record, sid)
    if state == "missing":
        return _build_missing(record, sid)
    if state == "unanswerable":
        return _build_unanswerable(record, sid)
    if state == "conflicting":
        if client is None:
            raise ValueError("LLMClient required for 'conflicting' state")
        return _build_conflicting(
            record, sid, client, models.get("claim_extractor", "default-chat-model")
        )
    if state == "underspecified":
        if client is None:
            raise ValueError("LLMClient required for 'underspecified' state")
        return _build_underspecified(
            record, sid, client,
            models.get("underspecified_injector", "default-chat-model"),
        )
    if state == "outdated":
        if client is None:
            raise ValueError("LLMClient required for 'outdated' state")
        return _build_outdated(
            record, sid, client,
            models.get("outdated_injector", "default-chat-model"),
        )
    raise ValueError(f"unknown state: {state}")


def save_benchmark(samples: list[BenchmarkSample], path: str) -> None:
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in samples], f, ensure_ascii=False, indent=2)


def load_benchmark(path: str) -> list[BenchmarkSample]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [BenchmarkSample.from_dict(d) for d in data]
