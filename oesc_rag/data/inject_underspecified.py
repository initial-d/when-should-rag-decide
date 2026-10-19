"""Rewrite a question into an underspecified form that hides a key criterion.

The resulting question admits multiple defensible answers depending on which
latent criterion the user actually cares about.
"""

from __future__ import annotations

from ..llm import LLMClient


UNDERSPECIFIED_PROMPT = """You are constructing test data for an open-world QA benchmark.

Given a specific question with a clear correct answer, rewrite it into an \
UNDERSPECIFIED version that omits a key criterion needed to identify a unique answer. \
The rewritten question must:
- Be grammatical and natural-sounding.
- Admit at least TWO defensible answers depending on the hidden criterion.
- Stay on the same topic as the original.

Examples:
- Original: "Which film directed by Christopher Nolan has the highest box office gross?"
  Rewritten: "Which film by Christopher Nolan is the best?"
  Criteria: [box office, critical acclaim, audience score, awards]

- Original: "Which programming language has the fastest execution speed for numerical code?"
  Rewritten: "Which programming language is the best?"
  Criteria: [speed, ease of use, ecosystem, popularity, paradigm]

Output JSON only:
{{
  "rewritten_question": "...",
  "latent_criteria": ["...", "...", "..."],
  "defensible_answers": ["answer assuming criterion A", "answer assuming criterion B"]
}}

Original question:
{question}

Correct answer:
{gold_answer}
"""


def generate_underspecified_question(
    client: LLMClient,
    question: str,
    gold_answer: str,
    model: str = "default-chat-model",
) -> dict[str, object]:
    """Return {"rewritten_question", "latent_criteria", "defensible_answers"} or empty on failure."""
    prompt = UNDERSPECIFIED_PROMPT.format(question=question, gold_answer=gold_answer)
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
    except Exception:
        return {"rewritten_question": "", "latent_criteria": [], "defensible_answers": []}
    rewritten = str(data.get("rewritten_question", "")).strip()
    criteria = [str(c).strip() for c in (data.get("latent_criteria") or []) if str(c).strip()]
    answers = [str(a).strip() for a in (data.get("defensible_answers") or []) if str(a).strip()]
    # Validate: need >=2 criteria and >=2 answers to qualify as underspecified
    if not rewritten or len(criteria) < 2 or len(answers) < 2:
        return {"rewritten_question": "", "latent_criteria": [], "defensible_answers": []}
    return {
        "rewritten_question": rewritten,
        "latent_criteria": criteria,
        "defensible_answers": answers,
    }
