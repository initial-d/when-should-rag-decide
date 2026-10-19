"""LLM-as-state-classifier baseline.

Uses the full OESC feature-extraction pipeline (multi-view retrieval,
claim graph, verification, intent analysis, freshness scoring) but
replaces the rule / learned classifier with an LLM that sees the
8-dimensional feature vector + brief descriptions of each feature and
each target state.

This isolates the classifier from the feature extractor: does the LLM
do better, the same, or worse than a hand-tuned rule tree / 100-tree
the structured controller when both see the same features?
"""

from __future__ import annotations

from ..llm import LLMClient
from ..orchestrator import OESCPipeline
from ..types import (
    STATE_LABELS,
    BenchmarkSample,
    PipelineResult,
    StateFeatures,
)


CLASSIFIER_PROMPT = """You are an evidence-state classifier for retrieval-augmented QA.

Given an 8-dimensional feature vector describing the current evidence
state, classify it into ONE of:
sufficient | missing | conflicting | unanswerable | underspecified | outdated.

Feature semantics (all in [0,1] unless noted):
- need_coverage: fraction of decomposed information needs covered by
  supported claims.
- answer_support: mean per-claim support score across all candidate
  answers.
- source_conflict: fraction of key claims with BOTH strong support AND
  strong refute.
- answer_disagreement: semantic spread across candidate answers.
- claim_disagreement: claim-level variance across candidate answers.
- retrieval_novelty: distance between this round's features and the
  previous round's.
- intent_ambiguity: question-level ambiguity (0 = specific, 1 = highly
  underspecified).
- freshness: minimum freshness across evidence docs (lower = staler).
- time_sensitive: whether the question expects current information.
- num_candidate_claims: integer; 0 implies unanswerable.

Decision guidance:
- 0 claims -> unanswerable.
- time_sensitive AND very low freshness -> outdated.
- high source_conflict -> conflicting.
- high intent_ambiguity -> underspecified.
- high coverage AND high support -> sufficient.
- otherwise -> missing or unanswerable depending on overall quality.

Output JSON only:
{{
  "state": "<one of: sufficient|missing|conflicting|unanswerable|underspecified|outdated>",
  "rationale": "<one short sentence>"
}}

Features:
- need_coverage: {C:.3f}
- answer_support: {S:.3f}
- source_conflict: {DS:.3f}
- answer_disagreement: {DA:.3f}
- claim_disagreement: {DC:.3f}
- retrieval_novelty: {N:.3f}
- intent_ambiguity: {I:.3f}
- freshness: {F:.3f}
- time_sensitive: {ts}
- num_candidate_claims: {nc}
"""


class _LLMClassifierPipeline(OESCPipeline):
    """Override _classify to call an LLM instead of the rule/learned classifier."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_llm_calls = 0

    def _classify(
        self,
        features: StateFeatures,
        num_claims: int,
        time_sensitive: bool,
        round_idx: int = 0,
    ) -> str:
        if num_claims == 0:
            return "unanswerable"
        prompt = CLASSIFIER_PROMPT.format(
            C=features.need_coverage,
            S=features.answer_support,
            DS=features.source_conflict,
            DA=features.answer_disagreement,
            DC=features.claim_disagreement,
            N=features.retrieval_novelty,
            I=features.intent_ambiguity,
            F=features.freshness,
            ts=str(time_sensitive).lower(),
            nc=num_claims,
        )
        try:
            data = self.client.chat_json(
                model=self.cfg["models"].get(
                    "state_classifier",
                    self.cfg["models"]["claim_verifier"],
                ),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            self.extra_llm_calls += 1
        except Exception:
            data = {}
        state = str(data.get("state", "missing")).strip().lower()
        if state not in STATE_LABELS:
            state = "missing"
        return state

    def run(self, sample: BenchmarkSample) -> PipelineResult:
        self.extra_llm_calls = 0
        result = super().run(sample)
        result.cost["llm_calls"] = result.cost.get("llm_calls", 0.0) + float(self.extra_llm_calls)
        result.method = "llm-classifier"
        return result


def run_llm_classifier(
    client: LLMClient,
    sample: BenchmarkSample,
    config: dict,
) -> PipelineResult:
    pipeline = _LLMClassifierPipeline(
        client=client,
        config={**config, "controller": {"type": "llm"}},
        embedder=None,
    )
    return pipeline.run(sample)
