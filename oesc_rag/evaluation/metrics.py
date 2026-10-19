"""Action-level and state-level evaluation metrics for OESC-RAG."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..types import BenchmarkSample, PipelineResult


# ------------------------------------------------------------------ helpers

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_f1(predicted: str | None, gold: str) -> float:
    """Standard SQuAD/HotpotQA token-F1 in [0, 1] after normalization."""
    if not predicted or not gold:
        return 0.0
    p = _norm(predicted)
    g = _norm(gold)
    if not p or not g:
        return 0.0
    p_tokens = p.split()
    g_tokens = g.split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p_tokens)
    rec = overlap / len(g_tokens)
    return 2 * prec * rec / (prec + rec)


def _exact_match(predicted: str | None, gold: str) -> bool:
    if not predicted or not gold:
        return False
    return _norm(predicted) == _norm(gold)


def _answer_correct(predicted: str | None, gold: str) -> bool:
    """Binary correctness: substring match OR token-F1 >= 0.6."""
    if not predicted or not gold:
        return False
    p = _norm(predicted)
    g = _norm(gold)
    if not p or not g:
        return False
    if g in p or p in g:
        return True
    return _token_f1(predicted, gold) >= 0.6


# ------------------------------------------------------------------ metrics

def action_accuracy(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.predicted_action == gold[r.sample_id].gold_action)
    return correct / len(results)


def state_accuracy(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.predicted_state == gold[r.sample_id].gold_state)
    return correct / len(results)


def false_answer_rate(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> float:
    """P(predict=answer | gold != answer)."""
    non_answer = [r for r in results if gold[r.sample_id].gold_action != "answer"]
    if not non_answer:
        return 0.0
    wrong = sum(1 for r in non_answer if r.predicted_action == "answer")
    return wrong / len(non_answer)


def over_abstention_rate(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> float:
    """P(predict=abstain | gold=answer)."""
    answer_gold = [r for r in results if gold[r.sample_id].gold_action == "answer"]
    if not answer_gold:
        return 0.0
    wrong = sum(1 for r in answer_gold if r.predicted_action == "abstain")
    return wrong / len(answer_gold)


def unsupported_claim_rate(results: list[PipelineResult]) -> float:
    """Fraction of all generated claims with no supporting evidence."""
    total_claims = 0
    unsupported = 0
    for r in results:
        sup_ids: set[str] = set()
        for v in r.verifications:
            if v.relation == "support":
                sup_ids.add(v.claim_id)
        for c in (cl for a in r.candidate_answers for cl in a.claims):
            total_claims += 1
            if c.claim_id not in sup_ids:
                unsupported += 1
    if total_claims == 0:
        return 0.0
    return unsupported / total_claims


def closure_recall(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> float:
    """Among gold-sufficient samples, fraction that system correctly answers.

    ClosureRecall = P(predict=answer | gold=answer).
    Penalises over-conservative systems that refuse to answer even when
    evidence is sufficient.
    """
    answer_gold = [r for r in results if gold[r.sample_id].gold_action == "answer"]
    if not answer_gold:
        return 0.0
    correct = sum(1 for r in answer_gold if r.predicted_action == "answer")
    return correct / len(answer_gold)


def unnecessary_nonanswer_rate(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> float:
    """Among gold-sufficient samples, fraction that system routes to any non-answer action.

    UNR = P(predict != answer | gold=answer).
    Broader than OAR (which counts only abstain); includes retrieve-more,
    ask-clarification, report-conflict on samples that should have been
    answered directly.
    """
    answer_gold = [r for r in results if gold[r.sample_id].gold_action == "answer"]
    if not answer_gold:
        return 0.0
    wrong = sum(1 for r in answer_gold if r.predicted_action != "answer")
    return wrong / len(answer_gold)


def correct_supported_answer_rate(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> float:
    """Dataset-level joint rate of correct, fully supported answers.

    This metric divides by all evaluated samples, not only answered samples.
    It therefore rewards useful supported coverage while remaining zero for
    systems that never answer.
    """
    answered = [r for r in results if r.predicted_action == "answer"]
    if not answered:
        return 0.0
    hits = 0
    for r in answered:
        sample = gold[r.sample_id]
        if not _answer_correct(r.final_answer, sample.gold_answer):
            continue
        sup_ids = {v.claim_id for v in r.verifications if v.relation == "support"}
        claims = [c for a in r.candidate_answers for c in a.claims]
        if claims and not all(c.claim_id in sup_ids for c in claims):
            continue
        hits += 1
    return hits / len(results)


def per_class_metrics(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
    field: str = "action",
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, F1 for action or state."""
    if field == "action":
        pred = {r.sample_id: r.predicted_action for r in results}
        true = {sid: g.gold_action for sid, g in gold.items()}
    else:
        pred = {r.sample_id: r.predicted_state for r in results}
        true = {sid: g.gold_state for sid, g in gold.items()}

    classes = sorted(set(true.values()) | set(pred.values()))
    out: dict[str, dict[str, float]] = {}
    for cls in classes:
        tp = sum(1 for sid in pred if pred[sid] == cls and true.get(sid) == cls)
        fp = sum(1 for sid in pred if pred[sid] == cls and true.get(sid) != cls)
        fn = sum(1 for sid in true if true[sid] == cls and pred.get(sid) != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def confusion_matrix(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
    field: str = "state",
) -> dict[str, dict[str, int]]:
    if field == "action":
        pred = {r.sample_id: r.predicted_action for r in results}
        true = {sid: g.gold_action for sid, g in gold.items()}
    else:
        pred = {r.sample_id: r.predicted_state for r in results}
        true = {sid: g.gold_state for sid, g in gold.items()}
    classes = sorted(set(true.values()) | set(pred.values()))
    cm: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    for sid in pred:
        if sid in true:
            cm[true[sid]][pred[sid]] += 1
    return cm


def clarification_trigger_accuracy(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> float:
    """Among underspecified samples, what fraction correctly triggered ask-clarification?"""
    under = [r for r in results if gold[r.sample_id].gold_state == "underspecified"]
    if not under:
        return 0.0
    correct = sum(1 for r in under if r.predicted_action == "ask-clarification")
    return correct / len(under)


def freshness_detection_accuracy(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> float:
    """Among outdated samples, what fraction correctly triggered retrieve-fresh-source?"""
    outdated = [r for r in results if gold[r.sample_id].gold_state == "outdated"]
    if not outdated:
        return 0.0
    correct = sum(1 for r in outdated if r.predicted_action == "retrieve-fresh-source")
    return correct / len(outdated)


def targeted_retrieval_success_rate(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> float:
    """Among missing samples with multi-round results, what fraction retrieved a hidden
    supporting doc by round 2?"""
    missing = [r for r in results if gold[r.sample_id].gold_state == "missing"]
    if not missing:
        return 0.0
    hits = 0
    for r in missing:
        sample = gold[r.sample_id]
        hidden_ids = set(sample.supporting_doc_ids) - {d.doc_id for d in sample.initial_evidence}
        if not hidden_ids:
            hits += 1  # nothing to retrieve — count as success
            continue
        all_retrieved = set()
        for rnd in r.rounds:
            all_retrieved.update(rnd.new_doc_ids)
        if hidden_ids & all_retrieved:
            hits += 1
    return hits / len(missing)


def cost_adjusted_utility(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
    alpha: float = 1.0,  # FAR penalty
    beta: float = 0.5,   # UCR penalty
    gamma: float = 0.01,  # cost penalty per LLM call
    delta: float = 0.3,  # over-abstention penalty
) -> float:
    """U = CSAR - alpha*FAR - beta*UCR - gamma*Cost - delta*OAR."""
    csar = correct_supported_answer_rate(results, gold)
    far = false_answer_rate(results, gold)
    oar = over_abstention_rate(results, gold)
    ucr = unsupported_claim_rate(results)
    avg_cost = 0.0
    if results:
        total_calls = sum(r.cost.get("llm_calls", 0.0) for r in results)
        avg_cost = total_calls / len(results)
    return csar - alpha * far - beta * ucr - gamma * avg_cost - delta * oar


def summarize(
    results: list[PipelineResult],
    gold: dict[str, BenchmarkSample],
) -> dict[str, Any]:
    return {
        "n": len(results),
        "action_accuracy": action_accuracy(results, gold),
        "state_accuracy": state_accuracy(results, gold),
        "false_answer_rate": false_answer_rate(results, gold),
        "over_abstention_rate": over_abstention_rate(results, gold),
        "unsupported_claim_rate": unsupported_claim_rate(results),
        "correct_supported_answer_rate": correct_supported_answer_rate(results, gold),
        "closure_recall": closure_recall(results, gold),
        "unnecessary_nonanswer_rate": unnecessary_nonanswer_rate(results, gold),
        "clarification_trigger_accuracy": clarification_trigger_accuracy(results, gold),
        "freshness_detection_accuracy": freshness_detection_accuracy(results, gold),
        "targeted_retrieval_success_rate": targeted_retrieval_success_rate(results, gold),
        "cost_adjusted_utility": cost_adjusted_utility(results, gold),
        "per_class_action": per_class_metrics(results, gold, field="action"),
        "per_class_state": per_class_metrics(results, gold, field="state"),
        "confusion_state": confusion_matrix(results, gold, field="state"),
        "confusion_action": confusion_matrix(results, gold, field="action"),
    }
