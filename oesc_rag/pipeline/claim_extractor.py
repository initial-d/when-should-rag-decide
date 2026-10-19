"""Extract atomic factual claims from an answer string."""

from __future__ import annotations

from ..llm import LLMClient
from ..types import Claim


EXTRACT_PROMPT = """Extract atomic factual claims that the answer makes in response to the question.

Each claim must be:
- minimal (one fact per claim)
- independently verifiable against evidence
- not a compound sentence
- tied to a specific entity, time, value, or relation if applicable
- a complete proposition that combines the question's subject with the answer
  (e.g., for Q="Who wrote X?" A="Alice", produce "Alice wrote X", not "Alice is a person")

Output JSON only:
{{"claims": ["...", "..."]}}

If the answer is "INSUFFICIENT EVIDENCE" or expresses uncertainty, return an empty list.

Question: {question}
Answer: {answer}
"""


def extract_claims(
    client: LLMClient,
    answer_text: str,
    question: str = "",
    source_answer_idx: int = 0,
    model: str = "default-chat-model",
    max_claims: int = 8,
) -> list[Claim]:
    if not answer_text or answer_text.strip().upper().startswith("INSUFFICIENT"):
        return []
    try:
        data = client.chat_json(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACT_PROMPT.format(
                        question=question or "(no question provided)",
                        answer=answer_text,
                    ),
                }
            ],
            temperature=0.0,
        )
        raw = data.get("claims", [])
    except Exception:
        raw = []
    claims: list[Claim] = []
    for i, c in enumerate(raw[:max_claims]):
        text = str(c).strip()
        if text:
            claims.append(
                Claim(
                    claim_id=f"v{source_answer_idx}_c{i}",
                    text=text,
                    source_answer_idx=source_answer_idx,
                )
            )
    return claims
