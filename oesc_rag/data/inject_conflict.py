"""Generate a contradicting evidence paragraph for the conflicting state.

We ask the configured chat model to write a plausible but contradictory passage that supports
a different answer than the gold one.
"""

from __future__ import annotations

from typing import Optional

from ..llm import LLMClient


CONFLICT_PROMPT = """You are constructing test data for a RAG benchmark.

Given the question and the correct answer (with its supporting evidence), write \
ONE short paragraph (3-5 sentences) that looks like a plausible Wikipedia-style \
source but **contradicts** the correct answer by supporting a *different* answer.

Constraints:
- The paragraph must be self-consistent and read like a real source.
- It must contradict the correct answer on the central fact, not on trivia.
- Invent a plausible alternative answer (e.g., a different name, date, place, or value).
- Do NOT mention that this is fictional.
- Output JSON only with keys: "alternative_answer" and "paragraph".

Question:
{question}

Correct answer:
{gold_answer}

Supporting evidence:
{supporting_text}
"""


def generate_conflict_paragraph(
    client: LLMClient,
    question: str,
    gold_answer: str,
    supporting_text: str,
    model: str = "default-chat-model",
) -> dict[str, str]:
    """Return {"alternative_answer": ..., "paragraph": ...}."""
    prompt = CONFLICT_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        supporting_text=supporting_text[:1500],
    )
    messages = [{"role": "user", "content": prompt}]
    data = client.chat_json(model=model, messages=messages, temperature=0.7)
    return {
        "alternative_answer": str(data.get("alternative_answer", "")).strip(),
        "paragraph": str(data.get("paragraph", "")).strip(),
    }
