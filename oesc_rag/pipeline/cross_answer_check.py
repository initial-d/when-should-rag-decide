"""LLM-based cross-answer contradiction check.

Given multiple candidate answers from different evidence views, asks the LLM whether
any pair of answers makes contradictory factual claims. This is a fallback signal for
source_conflict when the claim-level verifier misses contradictions.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..types import CandidateAnswer


CONTRADICTION_PROMPT = """You compare two answers to the same question and check for \
factual contradictions.

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Do answers A and B make contradictory factual claims? A paraphrase or a difference \
in wording does NOT count. Only genuine factual disagreements count \
(different names, dates, numbers, events, or conclusions).

Output JSON only:
{{
  "contradicts": true|false,
  "contradictory_claims": ["claim in A vs claim in B"],
  "rationale": "one sentence"
}}
"""


def check_answer_contradiction(
    client: LLMClient,
    question: str,
    answer_a: str,
    answer_b: str,
    model: str = "default-chat-model",
) -> dict:
    """Return {contradicts: bool, contradictory_claims: list, rationale: str}."""
    prompt = CONTRADICTION_PROMPT.format(
        question=question,
        answer_a=answer_a.strip()[:1000],
        answer_b=answer_b.strip()[:1000],
    )
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception:
        return {"contradicts": False, "contradictory_claims": [], "rationale": ""}
    return {
        "contradicts": bool(data.get("contradicts", False)),
        "contradictory_claims": data.get("contradictory_claims") or [],
        "rationale": str(data.get("rationale", "")).strip(),
    }


def check_pairwise_contradictions(
    client: LLMClient,
    question: str,
    answers: list[CandidateAnswer],
    model: str = "default-chat-model",
) -> float:
    """Return a conflict score in [0, 1] based on pairwise contradiction checks.

    Fraction of answer pairs that contradict each other.
    """
    if len(answers) < 2:
        return 0.0
    pairs = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            pairs.append((answers[i], answers[j]))
    if not pairs:
        return 0.0
    contradictions = 0
    for a, b in pairs:
        result = check_answer_contradiction(
            client=client,
            question=question,
            answer_a=a.answer_text,
            answer_b=b.answer_text,
            model=model,
        )
        if result["contradicts"]:
            contradictions += 1
    return contradictions / len(pairs)
