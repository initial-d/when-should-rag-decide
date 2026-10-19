"""Generate query variants for multi-view retrieval."""

from __future__ import annotations

from typing import Optional

from ..llm import LLMClient


REWRITE_PROMPT = """You are a query rewriting module for a RAG system.

Produce {n} diverse search queries for the user question below. Each query \
should retrieve DIFFERENT but relevant evidence. Use these strategies, mixing as \
appropriate:

- Strategy A: rephrase the question with synonyms
- Strategy B: extract key entities and search for them explicitly
- Strategy C: break the question into a sub-question covering ONE step
- Strategy D: search for contradictions / alternative answers
- Strategy E: search for the most specific value or fact required

Output JSON only:
{{"queries": ["...", "...", "..."]}}

Question: {question}
"""


def generate_query_variants(
    client: LLMClient,
    question: str,
    n: int = 3,
    model: str = "default-chat-model",
) -> list[str]:
    """Return n diverse query strings (always includes the original as q[0])."""
    prompt = REWRITE_PROMPT.format(n=n, question=question)
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        queries = data.get("queries") or []
    except Exception:
        queries = []
    out: list[str] = [question]
    for q in queries:
        s = str(q).strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= n:
            break
    while len(out) < n:
        out.append(question)
    return out[:n]
