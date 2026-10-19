"""Compute freshness score for evidence documents and time-sensitivity of questions."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from ..types import EvidenceDoc


# Keywords indicating a question expects current/recent information
_TIME_SENSITIVE_PATTERNS = [
    r"\bcurrent(ly)?\b",
    r"\b(latest|newest|most recent)\b",
    r"\b(as of )?now\b",
    r"\btoday\b",
    r"\bpresent(ly)?\b",
    r"\b20(2[3-9]|[3-9]\d)\b",  # mentions a recent year (2023+)
]


def is_time_sensitive(question: str) -> bool:
    """Return True if the question likely expects the latest information."""
    q_lower = question.lower()
    for pat in _TIME_SENSITIVE_PATTERNS:
        if re.search(pat, q_lower):
            return True
    return False


def doc_freshness(
    doc: EvidenceDoc,
    reference_year: int = 2025,
    horizon_years: float = 10.0,
) -> float:
    """Compute freshness in [0, 1] from the document's timestamp field.

    Strategy:
    - If doc.freshness_score is explicitly set (not the 0.5 default from construction),
      trust it (these are set by inject_outdated).
    - If timestamp is a parseable year, map [old_year, reference_year] to [0, 1]
      using a linear decay over ``horizon_years`` (a document ``horizon_years``
      old reads as ~0). ``horizon_years`` defaults to 10, reproducing the fixed
      decay; the optional volatility-aware component passes a per-question
      horizon instead.
    - Fallback to 0.5 (unknown freshness).
    """
    # Explicitly set by benchmark construction
    if doc.freshness_score not in (0.5, None):
        return doc.freshness_score

    if not doc.timestamp:
        return 0.5  # unknown

    year = _extract_year(doc.timestamp)
    if year is None:
        return 0.5
    horizon = horizon_years if horizon_years and horizon_years > 0 else 10.0
    # Map: anything >= reference_year is 1.0, anything `horizon` years old is ~0
    age = max(0, reference_year - year)
    freshness = max(0.0, 1.0 - age / horizon)
    return freshness


def aggregate_freshness(
    docs: list[EvidenceDoc],
    reference_year: int = 2025,
    horizon_years: float = 10.0,
) -> float:
    """Minimum freshness across all documents (conservative: one stale doc pulls it down)."""
    if not docs:
        return 0.5
    scores = [doc_freshness(d, reference_year, horizon_years) for d in docs]
    return min(scores)


def _extract_year(timestamp: str) -> Optional[int]:
    """Try to extract a 4-digit year from a timestamp string."""
    if not timestamp:
        return None
    # Try YYYY-MM-DD or just YYYY
    m = re.search(r"\b(19|20)\d{2}\b", timestamp)
    if m:
        return int(m.group())
    return None
