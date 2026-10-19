"""Rule-based open-state classifier — Full version (6 states).

The default behaviour is the full priority tree (round >= 1 branch below): this
is what the main experiment and reporting scripts use, since they call
``classify_state`` without ``initial_only``. The round-0 special-casing is
*opt-in* and only fires when ``initial_only=True`` — set exclusively by the
recovery experiment via ``intervention.initial_only_observation``. This keeps
the main-experiment classifier behaviour frozen while letting the recovery
pipeline defer judgement on round 0.

When ``initial_only`` is active, the semantics differ between round 0 (the
system has only seen ``initial_evidence``) and round >= 1 (a retrieve-more /
retrieve-fresh-source has been attempted):

  Round 0:
    - default to ``missing`` unless evidence is plainly closed (sufficient),
      plainly underspecified given enough coverage, plainly outdated, or
      plainly conflicting at a very strong threshold (~2x normal). Low
      coverage/support cannot distinguish "evidence doesn't exist" from
      "evidence is locked behind retrieve-more" so ``unanswerable`` is deferred.

  Round >= 1:
    - the original priority tree fires: very low coverage AND support →
      unanswerable; outdated; conflicting; underspecified; sufficient (with
      critical_need_recall check); missing fallback.

This shape matches the controller premise: reliable RAG begins with knowing
what to do before answering. On round 0 with restricted observation, the only
defensible answer is "try retrieving" unless we can prove otherwise.
"""

from __future__ import annotations

from typing import Optional

from ..types import StateFeatures


DEFAULT_THRESHOLDS = {
    "coverage": 0.4,
    "support": 0.4,
    "conflict": 0.25,
    "disagreement": 0.5,
    "intent_ambiguity": 0.15,
    "freshness": 0.15,
    "unanswerable_coverage": 0.1,
    "unanswerable_support": 0.3,
    "critical_need": 0.6,
    "round0_conflict": 0.5,  # 2x normal: very strong signal required on round 0
}


def classify_state(
    features: StateFeatures,
    thresholds: Optional[dict] = None,
    num_claims: Optional[int] = None,
    time_sensitive: bool = False,
    round_idx: int = 0,
    initial_only: bool = False,
    disabled_states: frozenset = frozenset(),
) -> str:
    """Return one of: sufficient | missing | conflicting | unanswerable | underspecified | outdated.

    ``initial_only`` gates the round-0 special-casing (recovery experiment only).
    When False (the main-experiment default), the full priority tree is
    always used regardless of ``round_idx``. States in ``disabled_states`` are
    skipped, for benchmarks whose action space excludes them.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    # No claims at all
    if num_claims is not None and num_claims == 0:
        # Under initial-only observation on round 0, defer to retrieval;
        # otherwise (incl. all main-experiment callers) genuine unanswerable.
        return "missing" if (initial_only and round_idx == 0) else "unanswerable"

    # ---------- Round 0 (initial observation only) ----------
    if initial_only and round_idx == 0:
        if "outdated" not in disabled_states and time_sensitive and features.freshness < t["freshness"]:
            return "outdated"
        if features.source_conflict >= t.get("round0_conflict", 0.5):
            return "conflicting"
        # Bias toward answer when coverage and support are both reasonable.
        # The critical-need gate is opt-in via ``round0_critical_need`` because
        # it is only meaningful when critical_need_recall is actually computed.
        if (
            features.need_coverage >= t["coverage"]
            and features.answer_support >= t["support"]
            and features.critical_need_recall >= t.get("round0_critical_need", 0.0)
        ):
            return "sufficient"
        # Default: try retrieving more
        return "missing"

    # ---------- Round >= 1 (retrieval has been attempted) ----------
    unanswerable_cov = t.get("unanswerable_coverage", 0.1)
    unanswerable_sup = t.get("unanswerable_support", 0.3)
    if features.need_coverage < unanswerable_cov and features.answer_support < unanswerable_sup:
        return "unanswerable"

    if "outdated" not in disabled_states and time_sensitive and features.freshness < t["freshness"]:
        return "outdated"

    if features.source_conflict >= t["conflict"]:
        return "conflicting"

    if "underspecified" not in disabled_states and features.intent_ambiguity >= t["intent_ambiguity"]:
        return "underspecified"

    if (
        features.need_coverage >= t["coverage"]
        and features.answer_support >= t["support"]
    ):
        if features.critical_need_recall < t["critical_need"]:
            return "missing"
        return "sufficient"

    if features.need_coverage < t["coverage"] and features.answer_support < t["support"]:
        if features.answer_disagreement >= t["disagreement"]:
            return "unanswerable"
        return "missing"

    return "missing"


def classify_with_reason(
    features: StateFeatures,
    thresholds: Optional[dict] = None,
    num_claims: Optional[int] = None,
    time_sensitive: bool = False,
    round_idx: int = 0,
    initial_only: bool = False,
) -> tuple[str, str]:
    """Return (state, reason_string) for debugging."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    if num_claims is not None and num_claims == 0:
        if initial_only and round_idx == 0:
            return "missing", "no claims on round 0; defer to retrieval"
        return "unanswerable", "no claims after retrieval"

    if initial_only and round_idx == 0:
        if time_sensitive and features.freshness < t["freshness"]:
            return "outdated", f"time_sensitive + freshness={features.freshness:.2f}<{t['freshness']}"
        round0_conflict = t.get("round0_conflict", 0.5)
        if features.source_conflict >= round0_conflict:
            return "conflicting", (
                f"round0 source_conflict={features.source_conflict:.2f}>={round0_conflict}"
            )
        if (
            features.need_coverage >= t["coverage"]
            and features.answer_support >= t["support"]
        ):
            return "sufficient", (
                f"round0 cov={features.need_coverage:.2f} sup={features.answer_support:.2f} "
                "(critical_need/intent gates dropped at round 0)"
            )
        return "missing", (
            f"round0 default: cov={features.need_coverage:.2f} sup={features.answer_support:.2f}"
        )

    unanswerable_cov = t.get("unanswerable_coverage", 0.1)
    unanswerable_sup = t.get("unanswerable_support", 0.3)
    if features.need_coverage < unanswerable_cov and features.answer_support < unanswerable_sup:
        return "unanswerable", (
            f"very low cov={features.need_coverage:.2f}<{unanswerable_cov} "
            f"& sup={features.answer_support:.2f}<{unanswerable_sup}"
        )

    if time_sensitive and features.freshness < t["freshness"]:
        return "outdated", f"time_sensitive + freshness={features.freshness:.2f}<{t['freshness']}"

    if features.source_conflict >= t["conflict"]:
        return "conflicting", f"source_conflict={features.source_conflict:.2f}>={t['conflict']}"

    if features.intent_ambiguity >= t["intent_ambiguity"]:
        return "underspecified", (
            f"intent_ambiguity={features.intent_ambiguity:.2f}>={t['intent_ambiguity']}"
        )

    if features.need_coverage >= t["coverage"] and features.answer_support >= t["support"]:
        if features.critical_need_recall < t["critical_need"]:
            return "missing", (
                f"coverage={features.need_coverage:.2f}>={t['coverage']} "
                f"but critical_need_recall={features.critical_need_recall:.2f}<{t['critical_need']}"
            )
        return "sufficient", (
            f"coverage={features.need_coverage:.2f}>={t['coverage']} "
            f"and support={features.answer_support:.2f}>={t['support']}"
        )

    if features.need_coverage < t["coverage"] and features.answer_support < t["support"]:
        if features.answer_disagreement >= t["disagreement"]:
            return "unanswerable", (
                f"low cov+sup, high disagree={features.answer_disagreement:.2f}"
            )
        return "missing", (
            f"low coverage={features.need_coverage:.2f}, support={features.answer_support:.2f}"
        )
    return "missing", "partial coverage/support fallback"
