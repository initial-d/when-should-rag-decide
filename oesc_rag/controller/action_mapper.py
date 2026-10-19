"""Map predicted state to the information action."""

from __future__ import annotations

from ..types import STATE_TO_ACTION


def map_action(state: str) -> str:
    """state -> action; falls back to abstain for unknown states."""
    return STATE_TO_ACTION.get(state, "abstain")
