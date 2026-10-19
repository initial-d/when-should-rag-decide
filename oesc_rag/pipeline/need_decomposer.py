"""Decompose a question into atomic information needs."""

from __future__ import annotations

from ..llm import LLMClient
from ..types import InformationNeed


DECOMPOSE_PROMPT = """You are an evidence planning module.

Given a user question, decompose it into the minimal information needs required to \
answer reliably. Each need should be a single atomic fact, entity, or relation.

Output JSON only:
{{
  "needs": [
    {{
      "need_id": "n1",
      "description": "...",
      "type": "entity|time|event|metric|comparison|causality|preference|source",
      "essential": true,
      "query": "short search query for this need"
    }}
  ]
}}

Limit to at most 5 needs. Mark only the truly essential ones as essential=true.

Question: {question}
"""


def decompose_needs(
    client: LLMClient,
    question: str,
    model: str = "default-chat-model",
) -> list[InformationNeed]:
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}],
            temperature=0.0,
        )
        items = data.get("needs", [])
    except Exception:
        items = []

    out: list[InformationNeed] = []
    for i, item in enumerate(items[:5]):
        out.append(
            InformationNeed(
                need_id=str(item.get("need_id") or f"n{i+1}"),
                description=str(item.get("description", "")).strip(),
                need_type=str(item.get("type", "entity")),
                is_essential=bool(item.get("essential", True)),
                suggested_query=item.get("query"),
            )
        )
    if not out:  # fallback: treat the whole question as one need
        out.append(
            InformationNeed(
                need_id="n1",
                description=question,
                need_type="entity",
                is_essential=True,
                suggested_query=question,
            )
        )
    return out
