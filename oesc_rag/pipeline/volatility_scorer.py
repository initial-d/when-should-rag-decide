"""Volatility-aware freshness: estimate how fast a question's answer goes stale.

Optional component (default OFF). When enabled, the deployed fixed 10-year
freshness decay is replaced by a per-question *staleness horizon* estimated by
the LLM from the QUESTION TEXT ALONE. It never reads any dataset volatility
label (e.g. FreshQA's fast/slow tag), never sees the evidence freshness score,
and tunes no threshold on the test set -- so any correlation with a gold
volatility label is a genuine capability, not leakage.

A question whose answer turns over yearly (current office-holders, rankings,
"latest" anything) gets a short horizon, so even a 2-year-old document reads as
stale; a settled historical fact gets a long horizon, so an old document still
reads as fresh.
"""

from __future__ import annotations

from ..llm import LLMClient

_DEFAULT_HORIZON = 10.0
_MIN_HORIZON = 0.5
_MAX_HORIZON = 100.0

VOLATILITY_PROMPT = """You estimate how quickly the correct answer to a question changes over time.

Reason about the answer's natural update cycle, independent of any specific document:
- Some answers change yearly or faster: current office-holders, league leaders, \
prices, populations, "latest"/"current"/"newest" anything.
- Some change every few years: champions of periodic events, standing records.
- Some effectively never change: historical dates, definitions, settled facts.

Estimate the typical number of YEARS before the answer is likely to become \
outdated (its "staleness horizon"). Use a small number for fast-changing answers \
and a large number for stable ones.

Output JSON only:
{{
  "staleness_horizon_years": <a number greater than 0>,
  "rationale": "one short sentence"
}}

Question: {question}
"""


def estimate_staleness_horizon(
    client: LLMClient,
    question: str,
    model: str = "default-chat-model",
) -> float:
    """Estimate the answer's staleness horizon in years from the question alone.

    Returns a value in ``[_MIN_HORIZON, _MAX_HORIZON]``. On any failure returns
    ``_DEFAULT_HORIZON`` (10.0), which reproduces the fixed-decay behaviour, so a
    failed estimate never changes the deployed result.
    """
    prompt = VOLATILITY_PROMPT.format(question=question)
    try:
        data = client.chat_json(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception:
        return _DEFAULT_HORIZON
    try:
        horizon = float(data.get("staleness_horizon_years", _DEFAULT_HORIZON))
    except (TypeError, ValueError):
        return _DEFAULT_HORIZON
    if horizon <= 0:
        return _DEFAULT_HORIZON
    return max(_MIN_HORIZON, min(_MAX_HORIZON, horizon))
