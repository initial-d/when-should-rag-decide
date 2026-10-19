"""Structural validity checks for evidence-state samples.

Each state label makes an implicit claim about where the answer is.  These
checks verify those claims with normalised string containment, so they are
only applied to answers that string matching can certify: yes/no answers and
answers that already appear in the question are excluded upstream.
"""

from __future__ import annotations

import re

STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "for", "by", "with"}


def norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def answer_tokens(answer: str) -> list[str]:
    return [t for t in norm(answer).split() if t not in STOP]


def contains_answer(text: str, answer: str) -> bool:
    """Normalised containment; short answers also match as a token subset."""
    needle = norm(answer)
    if not needle:
        return False
    hay = f" {norm(text)} "
    if f" {needle} " in hay:
        return True
    toks = answer_tokens(answer)
    if not toks or len(toks) > 4:
        return False
    return all(f" {t} " in hay for t in toks)


def checkable_answer(question: str, answer: str) -> bool:
    """True when string containment can certify where the answer lives."""
    if norm(answer) in {"yes", "no"} or not answer_tokens(answer):
        return False
    return not contains_answer(question, answer)


def alternative_from_notes(notes: str) -> str:
    match = re.search(r"alt=(.*)", notes or "")
    return match.group(1).strip() if match else ""


def sample_issues(sample) -> list[str]:
    """Return the validity violations of one BenchmarkSample (empty = valid)."""
    gold = sample.gold_answer
    initial_text = " ".join(d.text for d in sample.initial_evidence)
    pool_text = " ".join(d.text for d in sample.evidence_pool)
    initial_ids = {d.doc_id for d in sample.initial_evidence}
    hidden = [
        d for d in sample.evidence_pool
        if d.doc_id in set(sample.supporting_doc_ids) - initial_ids
    ]

    issues: list[str] = []
    if not checkable_answer(sample.question, gold):
        issues.append("answer_not_checkable_by_string")

    state = sample.gold_state
    if state == "sufficient":
        if not contains_answer(initial_text, gold):
            issues.append("sufficient_answer_not_in_initial")
    elif state == "missing":
        if not hidden:
            issues.append("missing_has_no_hidden_doc")
        if contains_answer(initial_text, gold):
            issues.append("missing_answer_already_in_initial")
        if not contains_answer(pool_text, gold):
            issues.append("missing_answer_absent_from_pool")
    elif state == "conflicting":
        conflict = sample.conflict_doc
        alt = alternative_from_notes(sample.perturbation_notes)
        if conflict is None:
            issues.append("conflict_doc_absent")
        else:
            if not alt or not contains_answer(conflict.text, alt):
                issues.append("conflict_doc_lacks_alternative")
            if contains_answer(conflict.text, gold):
                issues.append("conflict_doc_also_states_gold")
            if alt and norm(alt) == norm(gold):
                issues.append("alternative_equals_gold")
        support_text = " ".join(
            d.text for d in sample.initial_evidence
            if conflict is None or d.doc_id != conflict.doc_id
        )
        if not contains_answer(support_text, gold):
            issues.append("conflict_gold_not_supported")
    elif state == "unanswerable":
        if contains_answer(pool_text, gold):
            issues.append("unanswerable_answer_present_in_pool")
    return issues
