"""Generate an abstention message citing the missing information needs.

Used when the open-state classifier decides the question is unanswerable
(either from the start, or after the retrieval budget is exhausted).
"""

from __future__ import annotations

from ..llm import LLMClient
from ..types import InformationNeed


ABSTENTION_PROMPT = """You produce a short, honest abstention for a question that cannot be reliably answered \
with the available evidence.

Constraints:
- Two short sentences max.
- Name the specific missing information.
- Do NOT speculate or guess an answer.
- Suggest what kind of source could help (official, recent, primary, etc.).

Output JSON only:
{{
  "abstention": "Two-sentence abstention text"
}}

Question: {question}

Missing information needs:
{missing}

Reason (one of: missing-evidence, budget-exhausted, no-supporting-claims, no-public-information): {reason}
"""


def generate_abstention(
    client: LLMClient,
    question: str,
    missing_needs: list[InformationNeed],
    reason: str = "missing-evidence",
    model: str = "default-chat-model",
) -> str:
    """Return a short abstention string."""
    if missing_needs:
        missing_text = "\n".join(f"- {n.description}" for n in missing_needs[:5])
    else:
        missing_text = "(no specific needs decomposed)"

    prompt = ABSTENTION_PROMPT.format(
        question=question,
        missing=missing_text,
        reason=reason,
    )

    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = str(data.get("abstention", "")).strip()
    except Exception:
        text = ""

    if not text:
        # Deterministic fallback
        if missing_needs:
            needs_short = "; ".join(n.description for n in missing_needs[:3])
            text = (
                f"I cannot reliably answer this question from the available evidence. "
                f"Missing information: {needs_short}."
            )
        else:
            text = (
                "I cannot reliably answer this question from the available evidence. "
                "Additional authoritative sources would be needed."
            )

    return text
