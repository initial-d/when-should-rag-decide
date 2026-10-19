"""End-to-end OESC-RAG pipeline — Full version with iterative retrieval loop.

Pseudocode:
    for round in range(max_rounds):
        retrieve views with current queries
        generate candidate answers, extract claims, verify
        compute 8-dim features (intent + freshness on round 0)
        classify state
        if closed/conflicting/underspecified/unanswerable: terminate
        if missing:    queries = gap queries (uncovered needs)
        if outdated:   queries = freshness queries

When budget is exhausted, abstain.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

from .controller.action_mapper import map_action
from .controller.credal_controller import (
    CredalCalibratorController,
    CredalEvidenceController,
    CredalMemoryController,
)
from .controller.learned_classifier import LearnedClassifier
from .controller.rule_classifier import classify_state
from .generation.abstention_generator import generate_abstention
from .generation.clarification_generator import generate_clarification
from .generation.conflict_reporter import generate_conflict_report
from .llm import LLMClient
from .pipeline.answer_generator import generate_candidate_answer
from .pipeline.claim_extractor import extract_claims
from .pipeline.claim_verifier import verify_claims
from .pipeline.cross_answer_check import check_pairwise_contradictions
from .pipeline.feature_extractor import extract_features
from .pipeline.freshness_scorer import aggregate_freshness, is_time_sensitive
from .pipeline.volatility_scorer import estimate_staleness_horizon
from .pipeline.gap_query_gen import generate_freshness_queries, generate_gap_queries
from .pipeline.intent_analyzer import analyze_intent
from .pipeline.need_decomposer import decompose_needs
from .retrieval.bm25_retriever import BM25Index, merge_unique
from .retrieval.query_rewriter import generate_query_variants
from .types import (
    BenchmarkSample,
    CandidateAnswer,
    Claim,
    EvidenceDoc,
    InformationNeed,
    PipelineResult,
    RetrievalRound,
    StateFeatures,
)


class _BudgetExceeded(BaseException):
    """Raised internally when the per-sample LLM-call budget is exhausted.

    Derives from ``BaseException`` so the many ``except Exception`` guards in the
    per-step modules cannot swallow it: a truncated round must abort cleanly and
    leave the last complete round intact, rather than degrading into empty
    evidence and no claims.
    """


class _CallCounter:
    """Wraps an LLMClient and counts chat_json calls; raises when budget hit."""

    def __init__(self, client: LLMClient, budget: int):
        self._client = client
        self._budget = budget
        self.calls = 0
        # delegate everything else
        # we only intercept chat_json which is the only place LLM calls happen

    def chat_json(self, *args, **kwargs):
        if self._budget > 0 and self.calls >= self._budget:
            raise _BudgetExceeded()
        self.calls += 1
        return self._client.chat_json(*args, **kwargs)

    def __getattr__(self, name):
        # forward any other attribute access (e.g. chat_text, cache_path)
        return getattr(self._client, name)


class OESCPipeline:
    def __init__(
        self,
        client: LLMClient,
        config: dict,
        embedder: Optional[Any] = None,
        ablation: Optional[dict] = None,
        learned_classifier: Optional[LearnedClassifier] = None,
        point_policy: Optional[Any] = None,
    ) -> None:
        self.client = client
        self.cfg = config
        self.embedder = embedder
        self.ablation = ablation or {}
        self.learned_classifier = learned_classifier
        self.point_policy = point_policy
        self.credal_controller = CredalEvidenceController(
            self.cfg.get("credal_controller", {})
        )
        self.credal_memory: Optional[CredalMemoryController | CredalCalibratorController] = None
        memory_path = self.cfg.get("credal_controller", {}).get("memory_path")
        if memory_path:
            try:
                if str(memory_path).lower().endswith(".pkl"):
                    self.credal_memory = CredalCalibratorController()
                else:
                    self.credal_memory = CredalMemoryController()
                self.credal_memory.load(memory_path)
            except Exception:
                self.credal_memory = None

    # ------------------------------------------------------------------
    def _models(self) -> dict[str, str]:
        return self.cfg["models"]

    def _model_or(self, key: str, default_key: str = "answer_generator") -> str:
        models = self._models()
        return models.get(key, models[default_key])

    # ------------------------------------------------------------------
    def _retrieve_views(
        self,
        pool_index: BM25Index,
        queries: list[str],
        initial_evidence: list[EvidenceDoc],
        k: int,
        include_initial: bool,
        memory: Optional[list[EvidenceDoc]] = None,
    ) -> list[list[EvidenceDoc]]:
        views = pool_index.retrieve_multi(queries, k=k)
        carry = list(memory or [])
        if include_initial and initial_evidence:
            carry = list(initial_evidence) + carry
        if carry:
            # Evidence already observed stays visible; a retrieval action adds
            # to the evidence memory rather than replacing it.
            seen = {d.doc_id for d in views[0]}
            views[0] = views[0] + [d for d in carry if d.doc_id not in seen]
        return views

    def _generate_queries_round0(self, question: str, n_variants: int) -> list[str]:
        if self.ablation.get("no_multiview"):
            return [question]
        return generate_query_variants(
            client=self.client,
            question=question,
            n=n_variants,
            model=self._models()["query_rewriter"],
        )

    # ------------------------------------------------------------------
    def _classify(
        self,
        features: StateFeatures,
        num_claims: int,
        time_sensitive: bool,
        round_idx: int = 0,
        previous_features: Optional[StateFeatures] = None,
    ) -> str:
        controller_type = self.cfg.get("controller", {}).get("type", "rule")
        initial_only = bool(
            self.cfg.get("intervention", {}).get("initial_only_observation", False)
        )
        if controller_type == "credal":
            if self.credal_memory is not None:
                return self.credal_memory.predict(
                    features=features,
                    previous_features=previous_features,
                    time_sensitive=time_sensitive,
                    num_claims=num_claims,
                    round_idx=round_idx,
                    initial_only=initial_only,
                )
            return self.credal_controller.predict(
                features=features,
                previous_features=previous_features,
                time_sensitive=time_sensitive,
                thresholds=self.cfg["thresholds"],
                num_claims=num_claims,
                round_idx=round_idx,
                initial_only=initial_only,
            )
        if controller_type == "point" and self.point_policy is not None:
            if num_claims == 0:
                return "missing" if (initial_only and round_idx == 0) else "unanswerable"
            return self.point_policy.predict(
                features=features,
                previous_features=previous_features,
                time_sensitive=time_sensitive,
                thresholds=self.cfg.get("thresholds"),
                num_claims=num_claims,
                round_idx=round_idx,
                initial_only=initial_only,
            )
        if self.learned_classifier is not None and controller_type == "learned":            # The learned model only sees the feature vector; fall back to rule
            # for the no_claims edge case to preserve unanswerable detection.
            if num_claims == 0:
                return "unanswerable"
            try:
                return self.learned_classifier.predict(features)
            except Exception:
                pass
        return classify_state(
            features,
            self.cfg["thresholds"],
            num_claims=num_claims,
            time_sensitive=time_sensitive,
            round_idx=round_idx,
            initial_only=initial_only,
            disabled_states=frozenset(self.cfg.get("controller", {}).get("disabled_states", [])),
        )

    # ------------------------------------------------------------------
    def _pick_best_answer(
        self,
        candidates: list[CandidateAnswer],
        verifications: list,
    ) -> Optional[str]:
        if not candidates:
            return None
        return candidates[self._best_answer_index(candidates, verifications)].answer_text

    def _best_answer_support(
        self,
        candidates: list[CandidateAnswer],
        verifications: list,
    ) -> float:
        if not candidates:
            return 0.0
        sup_by_claim: dict[str, float] = {}
        for v in verifications:
            if v.relation == "support":
                sup_by_claim[v.claim_id] = max(sup_by_claim.get(v.claim_id, 0.0), v.confidence)
        scores = [
            0.0
            if not cand.claims
            else sum(sup_by_claim.get(c.claim_id, 0.0) for c in cand.claims) / len(cand.claims)
            for cand in candidates
        ]
        return max(scores)

    @staticmethod
    def _best_answer_index(candidates: list[CandidateAnswer], verifications: list) -> int:
        sup_by_claim: dict[str, float] = {}
        for v in verifications:
            if v.relation == "support":
                sup_by_claim[v.claim_id] = max(sup_by_claim.get(v.claim_id, 0.0), v.confidence)
        best_idx = 0
        best_score = -1.0
        for i, cand in enumerate(candidates):
            if not cand.claims:
                score = cand.confidence * 0.1
            else:
                score = sum(sup_by_claim.get(c.claim_id, 0.0) for c in cand.claims) / len(cand.claims)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    def _round(
        self,
        round_idx: int,
        question: str,
        queries: list[str],
        pool_index: BM25Index,
        initial_evidence: list[EvidenceDoc],
        needs: list[InformationNeed],
        prev_doc_ids: set[str],
        prev_features: Optional[StateFeatures],
        intent_ambiguity: float,
        time_sensitive: bool,
        counter: _CallCounter,
        staleness_horizon: float = 10.0,
        memory: Optional[list[EvidenceDoc]] = None,
    ) -> tuple[RetrievalRound, list[CandidateAnswer], list, list[EvidenceDoc], StateFeatures, str]:
        """Execute one round of the refinement loop.

        Returns: (round_record, candidates, verifications, merged_evidence, features, state)
        """
        models = self._models()
        k = self.cfg["retrieval"]["top_k"]
        max_verify_docs = self.cfg["pipeline"]["max_evidence_docs_for_verification"]
        abl = self.ablation

        include_initial = (round_idx == 0) and not abl.get("no_initial_evidence")
        views = self._retrieve_views(
            pool_index=pool_index,
            queries=queries,
            initial_evidence=initial_evidence,
            k=k,
            include_initial=include_initial,
            memory=memory,
        )

        # Generate candidate answers + extract claims for each view
        candidates: list[CandidateAnswer] = []
        for i, view in enumerate(views):
            cand = generate_candidate_answer(
                client=counter,
                question=question,
                evidence=view,
                evidence_view_idx=i,
                model=models["answer_generator"],
                temperature=self.cfg["openai"]["temperature_answer"],
            )
            cand.claims = extract_claims(
                client=counter,
                answer_text=cand.answer_text,
                question=question,
                source_answer_idx=i,
                model=models["claim_extractor"],
                max_claims=self.cfg["pipeline"]["max_claims_per_answer"],
            )
            candidates.append(cand)

        all_claims: list[Claim] = []
        for c in candidates:
            all_claims.extend(c.claims)

        merged_evidence = merge_unique(
            ([list(memory)] if memory else []) + views
        )[:max_verify_docs]
        verifications = verify_claims(
            client=counter,
            claims=all_claims,
            evidence=merged_evidence,
            model=models["claim_verifier"],
            max_docs=max_verify_docs,
            joint=bool(self.cfg["pipeline"].get("joint_verification", False)),
        )

        # Cross-answer contradiction signal (only when 2+ answers and not ablated)
        cross_conf = 0.0
        if not abl.get("no_source_conflict") and len(candidates) >= 2:
            try:
                cross_conf = check_pairwise_contradictions(
                    client=counter,
                    question=question,
                    answers=candidates,
                    model=models.get("cross_answer_check", models["claim_verifier"]),
                )
            except _BudgetExceeded:
                raise
            except Exception:
                cross_conf = 0.0

        # Freshness from current merged evidence
        if abl.get("no_freshness"):
            freshness = 1.0
        else:
            freshness = aggregate_freshness(merged_evidence, horizon_years=staleness_horizon)

        # Compute features
        critical_check = bool(self.cfg["pipeline"].get("critical_need_check", False))
        features = extract_features(
            needs=needs,
            answers=candidates,
            verifications=verifications,
            embedder=self.embedder,
            intent_ambiguity=intent_ambiguity,
            freshness=freshness,
            cross_answer_conflict=cross_conf,
            client=counter,
            model=models.get("need_coverage", models["claim_verifier"]),
            ablation=abl,
            question=question if critical_check else "",
            evidence_docs=merged_evidence if critical_check else None,
        )

        # Retrieval novelty: distance between this round's vector and the previous
        if prev_features is not None:
            try:
                v1 = np.array(prev_features.as_vector())
                v2 = np.array(features.as_vector())
                features.retrieval_novelty = float(np.linalg.norm(v1 - v2))
            except Exception:
                features.retrieval_novelty = 1.0
        else:
            features.retrieval_novelty = 1.0

        state = self._classify(
            features,
            num_claims=len(all_claims),
            time_sensitive=time_sensitive,
            round_idx=round_idx,
            previous_features=prev_features,
        )
        action = map_action(state)

        new_doc_ids = [d.doc_id for d in merged_evidence if d.doc_id not in prev_doc_ids]

        round_record = RetrievalRound(
            round_idx=round_idx,
            queries=list(queries),
            new_doc_ids=new_doc_ids,
            state=state,
            action=action,
            features=features,
            llm_calls=counter.calls,
        )

        return round_record, candidates, verifications, merged_evidence, features, state

    # ------------------------------------------------------------------
    def run(self, sample: BenchmarkSample) -> PipelineResult:
        models = self._models()
        n_variants = self.cfg["retrieval"]["num_query_variants"]
        max_rounds = int(self.cfg.get("loop", {}).get("max_rounds", 1))
        budget = int(self.cfg.get("loop", {}).get("budget_calls", 0))
        early_stop = bool(self.cfg.get("loop", {}).get("early_stop", True))
        abl = self.ablation

        # Single-pass override (ablation)
        if abl.get("no_iterative_loop"):
            max_rounds = 1

        counter = _CallCounter(self.client, budget=budget)

        # One-time pre-loop computations
        try:
            needs = decompose_needs(counter, sample.question, model=models["decomposer"])
        except _BudgetExceeded:
            needs = []

        # Intent ambiguity is a property of the question, computed once
        intent_ambiguity = 0.0
        intent_criteria: list[str] = []
        if not abl.get("no_intent_ambiguity"):
            try:
                intent = analyze_intent(
                    client=counter,
                    question=sample.question,
                    model=models.get("intent_analyzer", models["claim_verifier"]),
                )
                intent_ambiguity = float(intent.get("ambiguity_score", 0.0))
                intent_criteria = intent.get("latent_criteria", []) or []
            except _BudgetExceeded:
                pass
            except Exception:
                pass

        time_sensitive = is_time_sensitive(sample.question)

        # Volatility-aware freshness (optional, default OFF). When enabled, replace
        # the fixed 10-year decay horizon with a per-question horizon the LLM
        # estimates from the question text alone. Off by default so legacy runs
        # are bit-for-bit unchanged.
        staleness_horizon = 10.0
        if self.cfg.get("freshness", {}).get("volatility_aware", False):
            try:
                staleness_horizon = estimate_staleness_horizon(
                    client=counter,
                    question=sample.question,
                    model=models.get("volatility_scorer", models["claim_verifier"]),
                )
            except _BudgetExceeded:
                pass
            except Exception:
                pass

        intervention = self.cfg.get("intervention", {})
        initial_only = bool(intervention.get("initial_only_observation", False))
        # Round 0 honesty: when `initial_only_observation` is on, the controller
        # only sees BM25 over `initial_evidence` until it picks an action that
        # legitimately expands the pool (retrieve-more / retrieve-fresh-source,
        # which happen on round >= 1). Off by default so legacy runs are
        # unchanged.
        if initial_only:
            pool_index = BM25Index(sample.initial_evidence)
            full_pool_index: Optional[BM25Index] = None
        else:
            pool_index = BM25Index(sample.evidence_pool)
            full_pool_index = pool_index
        queries = self._generate_queries_round0(sample.question, n_variants)

        rounds: list[RetrievalRound] = []
        prev_doc_ids: set[str] = set()
        prev_features: Optional[StateFeatures] = None
        last_candidates: list[CandidateAnswer] = []
        last_verifications: list = []
        last_evidence: list[EvidenceDoc] = []
        last_features: Optional[StateFeatures] = None
        last_state: str = "unanswerable"
        round0_best_answer: Optional[str] = None
        # When enabled, evidence observed in earlier rounds stays in the pool for
        # later rounds instead of being replaced by a fresh BM25 top-k.
        memory_retrieval = bool(
            self.cfg.get("pipeline", {}).get("memory_retrieval", False)
        )
        evidence_memory: list[EvidenceDoc] = []
        budget_exhausted = False

        terminal_states = {"sufficient", "conflicting", "underspecified", "unanswerable"}

        # Retrieve-first (opt-in): the first action is a fixed retrieval taken
        # before any diagnosis, so diagnostics only ever run on expanded
        # evidence.  It is logged as round 0 with action retrieve-more, and the
        # initial evidence stays visible through the evidence memory.
        retrieve_first = bool(self.cfg.get("controller", {}).get("retrieve_first", False))
        start_round = 0
        if retrieve_first:
            rounds.append(
                RetrievalRound(
                    round_idx=0,
                    queries=list(queries),
                    new_doc_ids=[d.doc_id for d in sample.initial_evidence],
                    state="missing",
                    action="retrieve-more",
                    features=StateFeatures(0.0, 0.0, 0.0, 0.0, 0.0),
                    llm_calls=counter.calls,
                )
            )
            prev_doc_ids.update(d.doc_id for d in sample.initial_evidence)
            evidence_memory = list(sample.initial_evidence)
            memory_retrieval = True
            start_round = 1

        try:
            for round_idx in range(start_round, max_rounds):
                if initial_only and round_idx >= 1:
                    if full_pool_index is None:
                        full_pool_index = BM25Index(sample.evidence_pool)
                    current_index = full_pool_index
                else:
                    current_index = pool_index
                round_record, candidates, verifications, evidence, features, state = self._round(
                    round_idx=round_idx,
                    question=sample.question,
                    queries=queries,
                    pool_index=current_index,
                    initial_evidence=sample.initial_evidence,
                    needs=needs,
                    prev_doc_ids=prev_doc_ids,
                    prev_features=prev_features,
                    intent_ambiguity=intent_ambiguity,
                    time_sensitive=time_sensitive,
                    counter=counter,
                    staleness_horizon=staleness_horizon,
                    memory=evidence_memory if memory_retrieval else None,
                )
                rounds.append(round_record)
                last_candidates = candidates
                last_verifications = verifications
                last_evidence = evidence
                last_features = features
                last_state = state
                prev_features = features
                prev_doc_ids.update(d.doc_id for d in evidence)
                if memory_retrieval:
                    seen_ids = {d.doc_id for d in evidence_memory}
                    evidence_memory.extend(d for d in evidence if d.doc_id not in seen_ids)
                if round_idx == 0:
                    round0_best_answer = self._pick_best_answer(candidates, verifications)

                # Early stop on terminal states
                if early_stop and state in terminal_states:
                    break

                # Open states with an action that can change the evidence
                if state == "missing":
                    uncovered = self._uncovered_needs(needs, candidates, verifications)
                    try:
                        new_queries = generate_gap_queries(
                            client=counter,
                            question=sample.question,
                            uncovered_needs=uncovered,
                            model=models.get("gap_query_gen", models["query_rewriter"]),
                        )
                    except _BudgetExceeded:
                        break
                    if not new_queries or set(new_queries) == set(queries):
                        # No new direction → stop, will fall through to abstain below
                        break
                    # Always include the original question so we match the recall
                    # of a "blind retrieval" baseline; gap queries add targeted lift.
                    merged = [sample.question] + [q for q in new_queries if q != sample.question]
                    queries = merged[:n_variants]
                elif state == "outdated":
                    try:
                        new_queries = generate_freshness_queries(
                            client=counter,
                            question=sample.question,
                            model=models.get("gap_query_gen", models["query_rewriter"]),
                        )
                    except _BudgetExceeded:
                        break
                    if not new_queries or set(new_queries) == set(queries):
                        break
                    merged = [sample.question] + [q for q in new_queries if q != sample.question]
                    queries = merged[:n_variants]
                else:
                    # Unknown open state — terminate
                    break

        except _BudgetExceeded:
            # Budget hit; treat as terminal. Everything assigned after the last
            # successful _round() call is the last complete observation, so the
            # state and candidates from that round stay in force.
            budget_exhausted = True
            pass

        # If we never ran a single round, build an empty fallback
        if last_features is None:
            last_features = StateFeatures(
                need_coverage=0.0,
                answer_support=0.0,
                source_conflict=0.0,
                answer_disagreement=0.0,
                claim_disagreement=0.0,
                retrieval_novelty=0.0,
                intent_ambiguity=intent_ambiguity,
                freshness=1.0,
            )
            last_state = "unanswerable"

        predicted_state = last_state
        predicted_action = map_action(predicted_state)

        # No-harm guard (opt-in, default OFF): when enabled, if the controller
        # wants to give up (unanswerable / conflicting / underspecified) but
        # round 0 already produced a candidate answer with non-trivial verifier
        # support, prefer answering. On \bench{} this *lowers* action accuracy
        # (it suppresses genuine conflicting/underspecified detection, the very
        # behaviour the system is meant to win on), so it defaults to OFF and
        # the headline results are produced with the guard disabled. Kept
        # behind a flag for ablation only.
        no_harm_guard = bool(self.cfg.get("controller", {}).get("no_harm_guard", False))
        give_up_states = {"unanswerable", "conflicting", "underspecified"}
        round0_features = rounds[0].features if rounds else None
        if (
            no_harm_guard
            and predicted_state in give_up_states
            and round0_best_answer
            and round0_features is not None
            and round0_features.answer_support >= 0.4
        ):
            predicted_state = "sufficient"
            predicted_action = "answer"

        # Build the final output for the predicted action
        final_answer: Optional[str] = None
        clarification = None
        conflict_report = None

        try:
            if predicted_state == "sufficient":
                # Prefer round-0 best answer when we rescued, otherwise pick
                # from the last round's candidates.
                if round0_best_answer and last_state in give_up_states:
                    final_answer = round0_best_answer
                else:
                    final_answer = self._pick_best_answer(last_candidates, last_verifications)
            elif predicted_state == "conflicting":
                conflict_report = generate_conflict_report(
                    client=counter,
                    question=sample.question,
                    candidate_answers=last_candidates,
                    verifications=last_verifications,
                    evidence=last_evidence,
                    model=models.get("conflict_report", models["answer_generator"]),
                )
                final_answer = conflict_report.summary

            elif predicted_state == "underspecified":
                clarification = generate_clarification(
                    client=counter,
                    question=sample.question,
                    latent_criteria=intent_criteria,
                    model=models.get("clarification", models["answer_generator"]),
                )
                final_answer = clarification.text

            elif predicted_state == "unanswerable":
                missing_needs = self._uncovered_needs(needs, last_candidates, last_verifications) or needs
                reason = "budget-exhausted" if counter.calls >= budget and budget > 0 else "missing-evidence"
                final_answer = generate_abstention(
                    client=counter,
                    question=sample.question,
                    missing_needs=missing_needs,
                    reason=reason,
                    model=models.get("abstention", models["answer_generator"]),
                )

            else:
                # missing / outdated reached only when budget exhausted before
                # close. With the no-harm guard ENABLED, fall back to answering
                # (high round-0 support) or abstaining. With the guard DISABLED
                # (default), preserve the open state (missing -> retrieve-more,
                # outdated -> retrieve-fresh) and emit a best-effort answer
                # string; this reproduces the headline runs, which retain
                # `missing`/`outdated` as the final predicted state.
                if no_harm_guard and (
                    round0_best_answer
                    and round0_features is not None
                    and round0_features.answer_support >= 0.4
                ):
                    predicted_state = "sufficient"
                    predicted_action = "answer"
                    final_answer = round0_best_answer
                elif no_harm_guard:
                    predicted_state = "unanswerable"
                    predicted_action = "abstain"
                    missing_needs = self._uncovered_needs(needs, last_candidates, last_verifications) or needs
                    final_answer = generate_abstention(
                        client=counter,
                        question=sample.question,
                        missing_needs=missing_needs,
                        reason="budget-exhausted",
                        model=models.get("abstention", models["answer_generator"]),
                    )
                else:
                    final_answer = round0_best_answer or self._pick_best_answer(
                        last_candidates, last_verifications
                    )

        except _BudgetExceeded:
            # Budget exhausted during generation; produce a deterministic fallback
            if predicted_state == "sufficient" and last_candidates:
                final_answer = last_candidates[0].answer_text
            elif final_answer is None:
                final_answer = (
                    "I cannot reliably answer this question from the available evidence."
                )

        # Terminal resolution of an open state (opt-in): when the loop ends in
        # missing/outdated because no further retrieval is possible (round cap,
        # repeated queries, or an exhausted call budget), "retrieve-more" is no
        # longer an available action. Resolve to the best supported answer, or
        # to abstention when no candidate is supported.
        resolve_open = bool(
            self.cfg.get("controller", {}).get("resolve_open_terminal", False)
        )
        if resolve_open and predicted_state in {"missing", "outdated"}:
            support = self._best_answer_support(last_candidates, last_verifications)
            if last_candidates and support >= 0.5:
                predicted_state = "sufficient"
                predicted_action = "answer"
                final_answer = self._pick_best_answer(last_candidates, last_verifications)
            else:
                predicted_state = "unanswerable"
                predicted_action = "abstain"
                missing_needs = self._uncovered_needs(needs, last_candidates, last_verifications) or needs
                try:
                    final_answer = generate_abstention(
                        client=counter,
                        question=sample.question,
                        missing_needs=missing_needs,
                        reason="budget-exhausted" if budget_exhausted else "missing-evidence",
                        model=models.get("abstention", models["answer_generator"]),
                    )
                except _BudgetExceeded:
                    final_answer = (
                        "I cannot reliably answer this question from the available evidence."
                    )

        usage = {}
        if hasattr(self.client, "thread_usage"):
            candidate = self.client.thread_usage()
            if isinstance(candidate, dict):
                usage = candidate
        cost = {
            "llm_calls": float(counter.calls),
            "rounds": float(len(rounds)),
            "budget": float(budget),
            **usage,
        }

        return PipelineResult(
            sample_id=sample.id,
            predicted_state=predicted_state,
            predicted_action=predicted_action,
            features=last_features,
            candidate_answers=last_candidates,
            verifications=last_verifications,
            information_needs=needs,
            final_answer=final_answer,
            clarification=clarification,
            conflict_report=conflict_report,
            rounds=rounds,
            cost=cost,
            method="oesc-rag",
            metadata={
                "intent_criteria": intent_criteria,
                "time_sensitive": time_sensitive,
                "queries_round0": rounds[0].queries if rounds else [],
                "controller": self.cfg.get("controller", {}).get("type", "rule"),
                "initial_only_observation": initial_only,
                "initial_answer": round0_best_answer,
                "initial_action": rounds[0].action if rounds else predicted_action,
                "initial_observation_doc_ids": [d.doc_id for d in sample.initial_evidence]
                if initial_only
                else [d.doc_id for d in sample.evidence_pool],
                "budget_exhausted": budget_exhausted
                or bool(budget > 0 and counter.calls >= budget),
            },
        )

    # ------------------------------------------------------------------
    def _uncovered_needs(
        self,
        needs: list[InformationNeed],
        candidates: list[CandidateAnswer],
        verifications: list,
    ) -> list[InformationNeed]:
        """Heuristic: a need is uncovered if no supported claim contains overlapping tokens."""
        import re

        def toks(t: str) -> set[str]:
            return {w.lower() for w in re.findall(r"\w+", t or "")}

        sup_claim_ids = {v.claim_id for v in verifications if v.relation == "support" and v.confidence >= 0.5}
        sup_claim_texts: list[set[str]] = []
        for cand in candidates:
            for c in cand.claims:
                if c.claim_id in sup_claim_ids:
                    sup_claim_texts.append(toks(c.text))

        uncovered: list[InformationNeed] = []
        for n in needs:
            n_t = toks(n.description)
            if not n_t:
                continue
            overlap_any = any(
                (len(n_t & ct) / max(1, len(n_t))) >= 0.3 for ct in sup_claim_texts
            )
            if not overlap_any:
                uncovered.append(n)
        return uncovered
