"""Policy-level end-to-end metrics for OESC intervention experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..types import BenchmarkSample, PipelineResult
from .metrics import _answer_correct, _exact_match, _token_f1


def first_action(result: PipelineResult) -> str:
    """Return the action selected on the initial observation."""
    metadata_action = (result.metadata or {}).get("initial_action")
    if metadata_action:
        return str(metadata_action)
    if result.rounds:
        return result.rounds[0].action or result.predicted_action
    return result.predicted_action


def terminal_action(result: PipelineResult) -> str:
    """Return the action associated with the final emitted output."""
    return result.predicted_action


def _post_action_doc_ids(result: PipelineResult) -> set[str]:
    return {
        doc_id
        for round_record in result.rounds
        if round_record.round_idx >= 1
        for doc_id in round_record.new_doc_ids
    }


def _hidden_supporting_ids(sample: BenchmarkSample) -> set[str]:
    initial_ids = {doc.doc_id for doc in sample.initial_evidence}
    return set(sample.supporting_doc_ids) - initial_ids


def _terminal_claims(result: PipelineResult) -> list:
    for candidate in result.candidate_answers:
        if candidate.answer_text == result.final_answer:
            return candidate.claims
    return result.candidate_answers[-1].claims if result.candidate_answers else []


def _terminal_answer_supported(result: PipelineResult) -> bool:
    """True only when the emitted answer has claims and all are verified."""
    claims = _terminal_claims(result)
    if not claims:
        return False
    supported = {
        verification.claim_id
        for verification in result.verifications
        if verification.relation == "support" and verification.confidence >= 0.5
    }
    return all(claim.claim_id in supported for claim in claims)


def _answerable_rows(
    results: list[PipelineResult], gold: dict[str, BenchmarkSample]
) -> list[tuple[PipelineResult, BenchmarkSample]]:
    return [
        (result, gold[result.sample_id])
        for result in results
        if result.sample_id in gold and gold[result.sample_id].gold_action == "answer"
    ]


def answer_quality(
    results: list[PipelineResult], gold: dict[str, BenchmarkSample]
) -> dict[str, float]:
    """Score factual answers only on gold answer-action samples."""
    rows = _answerable_rows(results, gold)
    if not rows:
        return {
            "n": 0,
            "answer_rate": 0.0,
            "answer_em": 0.0,
            "answer_f1": 0.0,
            "conditional_em": 0.0,
            "conditional_f1": 0.0,
            "supported_answer_rate": 0.0,
        }
    emitted = [(result, sample) for result, sample in rows if result.predicted_action == "answer"]
    answered_text = {
        result.sample_id: result.final_answer if result.predicted_action == "answer" else None
        for result, _ in rows
    }
    em = [_exact_match(answered_text[result.sample_id], sample.gold_answer) for result, sample in rows]
    f1 = [_token_f1(answered_text[result.sample_id], sample.gold_answer) for result, sample in rows]
    supported = [
        result.predicted_action == "answer"
        and _answer_correct(result.final_answer, sample.gold_answer)
        and _terminal_answer_supported(result)
        for result, sample in rows
    ]
    conditional_em = [
        _exact_match(result.final_answer, sample.gold_answer) for result, sample in emitted
    ]
    conditional_f1 = [
        _token_f1(result.final_answer, sample.gold_answer) for result, sample in emitted
    ]
    return {
        "n": float(len(rows)),
        "answer_rate": len(emitted) / len(rows),
        "answer_em": sum(em) / len(em),
        "answer_f1": sum(f1) / len(f1),
        "conditional_em": sum(conditional_em) / len(conditional_em) if emitted else 0.0,
        "conditional_f1": sum(conditional_f1) / len(conditional_f1) if emitted else 0.0,
        "supported_answer_rate": sum(supported) / len(rows),
    }


def action_metrics(
    results: list[PipelineResult], gold: dict[str, BenchmarkSample]
) -> dict[str, float]:
    rows = [(result, gold[result.sample_id]) for result in results if result.sample_id in gold]
    if not rows:
        return {"n": 0, "first_action_accuracy": 0.0, "terminal_action_accuracy": 0.0}
    return {
        "n": float(len(rows)),
        "first_action_accuracy": sum(
            first_action(result) == sample.gold_action for result, sample in rows
        ) / len(rows),
        "terminal_action_accuracy": sum(
            terminal_action(result) == sample.gold_action for result, sample in rows
        ) / len(rows),
        "false_answer_rate": sum(
            terminal_action(result) == "answer" and sample.gold_action != "answer"
            for result, sample in rows
            if sample.gold_action != "answer"
        ) / max(1, sum(sample.gold_action != "answer" for _, sample in rows)),
        "non_answer_action_accuracy": sum(
            terminal_action(result) == sample.gold_action
            for result, sample in rows
            if sample.gold_action != "answer"
        ) / max(1, sum(sample.gold_action != "answer" for _, sample in rows)),
        "unnecessary_retrieval_rate": sum(
            first_action(result) == "retrieve-more"
            for result, sample in rows
            if sample.gold_state != "missing"
        ) / max(1, sum(sample.gold_state != "missing" for _, sample in rows)),
    }


def missing_recovery(
    results: list[PipelineResult], gold: dict[str, BenchmarkSample]
) -> dict[str, float]:
    rows = [
        (result, gold[result.sample_id])
        for result in results
        if result.sample_id in gold and gold[result.sample_id].gold_state == "missing"
    ]
    if not rows:
        return {"n": 0, "hidden_evidence_recall": 0.0, "recovery_success": 0.0}
    evidence_hits = []
    recovered = []
    for result, sample in rows:
        hidden = _hidden_supporting_ids(sample)
        post_action = _post_action_doc_ids(result)
        hit = not hidden or bool(hidden & post_action)
        evidence_hits.append(hit)
        recovered.append(
            hit
            and first_action(result) == "retrieve-more"
            and terminal_action(result) == "answer"
            and _answer_correct(result.final_answer, sample.gold_answer)
        )
    return {
        "n": float(len(rows)),
        "hidden_evidence_recall": sum(evidence_hits) / len(rows),
        "recovery_success": sum(recovered) / len(rows),
    }


def per_state_task_success(
    results: list[PipelineResult], gold: dict[str, BenchmarkSample]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        sample = gold.get(result.sample_id)
        if sample is None:
            continue
        if sample.gold_action == "answer":
            success = terminal_action(result) == "answer" and _answer_correct(
                result.final_answer, sample.gold_answer
            )
        elif sample.gold_state == "missing":
            hidden = _hidden_supporting_ids(sample)
            hit = not hidden or bool(hidden & _post_action_doc_ids(result))
            success = (
                first_action(result) == "retrieve-more"
                and terminal_action(result) == "answer"
                and hit
                and _answer_correct(result.final_answer, sample.gold_answer)
            )
        else:
            success = terminal_action(result) == sample.gold_action
        grouped[sample.gold_state].append(success)

    return {
        state: {"n": float(len(values)), "task_success": sum(values) / len(values)}
        for state, values in sorted(grouped.items())
        if values
    }


def task_success(
    results: list[PipelineResult], gold: dict[str, BenchmarkSample]
) -> dict[str, Any]:
    per_state = per_state_task_success(results, gold)
    values = [
        row["task_success"]
        for row in per_state.values()
        for _ in range(int(row["n"]))
    ]
    return {
        "n": float(len(values)),
        "task_success": sum(values) / len(values) if values else 0.0,
        "per_state": per_state,
    }


def summarize_policy(
    results: list[PipelineResult], gold: dict[str, BenchmarkSample]
) -> dict[str, Any]:
    calls = [result.cost.get("llm_calls", 0.0) for result in results]
    rounds = [result.cost.get("rounds", 0.0) for result in results]
    return {
        "n": len(results),
        "sample_ids": sorted(result.sample_id for result in results),
        "actions": action_metrics(results, gold),
        "answer_quality": answer_quality(results, gold),
        "missing_recovery": missing_recovery(results, gold),
        "task_success": task_success(results, gold),
        "avg_llm_calls": sum(calls) / len(calls) if calls else 0.0,
        "avg_rounds": sum(rounds) / len(rounds) if rounds else 0.0,
    }
