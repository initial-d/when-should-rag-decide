"""Generate an updated (fresh) version of a supporting paragraph for the outdated state.

We ask the LLM to:
1. Rewrite the original paragraph as if it were an OLD version (with an old timestamp)
   reporting an OUTDATED value.
2. Generate a NEW version with a more recent timestamp reporting an UPDATED value.

We also ask whether the question is time-sensitive (would the user expect the latest info?).
"""

from __future__ import annotations

from ..llm import LLMClient


OUTDATED_PROMPT = """You are constructing test data for a temporal RAG benchmark.

Given a question and a supporting paragraph, create two versions of the SAME fact:

1. OLD version: the paragraph rephrased to report an OUTDATED fact (e.g. an earlier \
   value, an earlier record holder, an earlier policy). Add an explicit timestamp \
   reading like "As of [OLD_YEAR], ...".
2. NEW version: a paragraph reporting the UPDATED fact (e.g. a current/new value). \
   Add an explicit timestamp like "As of [NEW_YEAR], ...". NEW_YEAR must be strictly \
   later than OLD_YEAR.

Constraints:
- Both versions must be self-consistent and read like real sources.
- The factual claim must differ between OLD and NEW on the specific value the \
  question asks about.
- The "updated_answer" is the answer derivable from the NEW version.
- Also state whether the question is time-sensitive (does the user likely want the \
  current/latest fact?). Examples of time-sensitive: "current", "latest", "as of now". \
  Examples of NOT time-sensitive: "in 2010", "during World War II".

Output JSON only:
{{
  "old_year": "YYYY",
  "old_paragraph": "...",
  "old_answer": "outdated answer",
  "new_year": "YYYY",
  "new_paragraph": "...",
  "updated_answer": "...",
  "time_sensitive": true
}}

Question:
{question}

Original supporting paragraph:
{supporting_text}

Original answer (still valid as the old fact):
{gold_answer}
"""


def generate_outdated_versions(
    client: LLMClient,
    question: str,
    gold_answer: str,
    supporting_text: str,
    model: str = "default-chat-model",
) -> dict[str, object]:
    """Return dict with old/new paragraphs + timestamps + updated_answer + time_sensitive flag.

    Returns empty fields on validation failure.
    """
    prompt = OUTDATED_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        supporting_text=supporting_text[:1500],
    )
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
    except Exception:
        return _empty()

    old_year = str(data.get("old_year", "")).strip()
    new_year = str(data.get("new_year", "")).strip()
    old_para = str(data.get("old_paragraph", "")).strip()
    new_para = str(data.get("new_paragraph", "")).strip()
    old_ans = str(data.get("old_answer", "")).strip()
    new_ans = str(data.get("updated_answer", "")).strip()
    time_sensitive = bool(data.get("time_sensitive", True))

    # Validate
    if not (old_para and new_para and old_ans and new_ans):
        return _empty()
    if not (old_year and new_year):
        return _empty()
    try:
        if int(new_year) <= int(old_year):
            return _empty()
    except ValueError:
        return _empty()
    if old_ans.lower() == new_ans.lower():
        return _empty()

    return {
        "old_year": old_year,
        "old_paragraph": old_para,
        "old_answer": old_ans,
        "new_year": new_year,
        "new_paragraph": new_para,
        "updated_answer": new_ans,
        "time_sensitive": time_sensitive,
    }


def _empty() -> dict[str, object]:
    return {
        "old_year": "",
        "old_paragraph": "",
        "old_answer": "",
        "new_year": "",
        "new_paragraph": "",
        "updated_answer": "",
        "time_sensitive": False,
    }
