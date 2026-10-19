"""Generate targeted retrieval queries from uncovered information needs.

When the state is 'missing', we generate specific queries aimed at retrieving
the evidence that would satisfy the uncovered needs.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..types import InformationNeed


GAP_QUERY_PROMPT = """You are a search-query generation module.

Given a user question and a list of information needs that are NOT YET covered \
by the available evidence, generate a targeted search query for EACH uncovered need.

Each query should be specific, concise, and likely to retrieve the missing information.

Output JSON only:
{{
  "queries": [
    {{
      "need_id": "...",
      "query": "specific search query",
      "rationale": "why this query would help"
    }}
  ]
}}

Question:
{question}

Uncovered needs:
{needs}
"""


FRESHNESS_QUERY_PROMPT = """You are a search-query generation module.

The current evidence for this question is OUTDATED. Generate 2-3 search queries \
that would retrieve the MOST RECENT version of the relevant information. Include \
time modifiers like "latest", "2025", "current", or "updated" in the queries.

Output JSON only:
{{
  "queries": ["query 1", "query 2"]
}}

Question:
{question}
"""


def generate_gap_queries(
    client: LLMClient,
    question: str,
    uncovered_needs: list[InformationNeed],
    model: str = "default-chat-model",
) -> list[str]:
    """Generate targeted queries for missing evidence."""
    if not uncovered_needs:
        return [question]
    needs_text = "\n".join(
        f"- [{n.need_id}] {n.description} (type: {n.need_type})"
        for n in uncovered_needs
    )
    prompt = GAP_QUERY_PROMPT.format(question=question, needs=needs_text)
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        items = data.get("queries") or []
        queries = [str(item.get("query", "")).strip() for item in items if item.get("query")]
    except Exception:
        queries = []
    # Fallback: use the needs' suggested queries or the original question
    if not queries:
        queries = [n.suggested_query or n.description for n in uncovered_needs[:3]]
    return queries[:5]


def generate_freshness_queries(
    client: LLMClient,
    question: str,
    model: str = "default-chat-model",
) -> list[str]:
    """Generate queries aimed at retrieving the most recent evidence."""
    prompt = FRESHNESS_QUERY_PROMPT.format(question=question)
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
    except Exception:
        queries = []
    if not queries:
        queries = [f"latest {question}", f"current {question} 2025"]
    return queries[:3]
