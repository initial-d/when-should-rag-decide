"""Detect latent criteria and intent ambiguity in a question."""

from __future__ import annotations

from ..llm import LLMClient


INTENT_PROMPT = """You analyze user questions for intent ambiguity.

A question is UNDERSPECIFIED if it admits multiple defensible answers depending on a \
hidden criterion the user did not specify (e.g. "Which model is best?" — best by speed, \
price, accuracy, ease of use?). A question is SPECIFIC if it points to a unique \
defensible answer.

Identify:
1. Latent criteria the user might mean (if any).
2. Whether the question is underspecified (true/false).
3. An ambiguity score in [0, 1] where:
   - 0.0–0.2 = clearly specific (one obvious criterion)
   - 0.3–0.5 = mildly ambiguous (multiple criteria but one dominant)
   - 0.6–1.0 = highly underspecified (multiple equally defensible criteria)

Output JSON only:
{{
  "is_underspecified": true|false,
  "ambiguity_score": 0.0-1.0,
  "latent_criteria": ["criterion 1", "criterion 2"],
  "rationale": "one short sentence"
}}

Question: {question}
"""


def analyze_intent(
    client: LLMClient,
    question: str,
    model: str = "default-chat-model",
) -> dict:
    """Return {ambiguity_score, latent_criteria, is_underspecified, rationale}."""
    prompt = INTENT_PROMPT.format(question=question)
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception:
        return {
            "ambiguity_score": 0.0,
            "latent_criteria": [],
            "is_underspecified": False,
            "rationale": "",
        }
    try:
        score = float(data.get("ambiguity_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    criteria = [str(c).strip() for c in (data.get("latent_criteria") or []) if str(c).strip()]
    return {
        "ambiguity_score": score,
        "latent_criteria": criteria,
        "is_underspecified": bool(data.get("is_underspecified", False)),
        "rationale": str(data.get("rationale", "")).strip(),
    }
