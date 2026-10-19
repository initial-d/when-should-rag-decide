"""Generation modules for non-answer information actions.

When the open-state classifier decides the system should NOT directly answer,
these modules produce the appropriate output:

- clarification_generator: ask the user for missing intent
- conflict_reporter: present the conflicting evidence without picking a side
- abstention_generator: refuse with reasons (missing needs)
"""

from .abstention_generator import generate_abstention
from .clarification_generator import generate_clarification
from .conflict_reporter import generate_conflict_report

__all__ = [
    "generate_abstention",
    "generate_clarification",
    "generate_conflict_report",
]
