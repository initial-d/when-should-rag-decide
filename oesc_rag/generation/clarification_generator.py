"""Generate a clarification question for underspecified user questions.

Given the latent criteria detected by intent_analyzer, produce a single, minimal
clarification question that surfaces the missing criterion in a user-friendly way.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..types import ClarificationQuestion


CLARIFICATION_PROMPT = """You generate a minimal clarification question for an underspecified user query.

The user asked a question that admits multiple defensible answers because they did not specify a key criterion. \
Ask ONE short clarification question. Offer 2–5 concrete options the user can pick from.

Constraints:
- Do not assume any option for the user.
- Keep the question under 30 words.
- The options should reflect the actual latent criteria.
- Do not answer the original question.

Output JSON only:
{{
  "question": "Your one-sentence clarification question",
  "options": ["option 1", "option 2", ...]
}}

User question: {question}

Latent criteria detected: {criteria}
"""


def generate_clarification(
    client: LLMClient,
    question: str,
    latent_criteria: list[str],
    model: str = "default-chat-model",
) -> ClarificationQuestion:
    """Produce a ClarificationQuestion grounded in the detected criteria."""
    criteria_text = ", ".join(latent_criteria) if latent_criteria else "(none detected)"
    prompt = CLARIFICATION_PROMPT.format(question=question, criteria=criteria_text)

    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except Exception:
        data = {}

    text = str(data.get("question", "")).strip()
    options = [str(o).strip() for o in (data.get("options") or []) if str(o).strip()]

    if not text:
        # Fallback to a generic clarification using the criteria
        if latent_criteria:
            text = (
                "Which of these criteria do you care about most: "
                + ", ".join(latent_criteria[:5])
                + "?"
            )
            options = list(latent_criteria[:5])
        else:
            text = "Could you clarify what criterion or scope you have in mind?"

    return ClarificationQuestion(
        text=text,
        latent_criteria=latent_criteria,
        options=options,
    )
