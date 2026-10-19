"""Baseline RAG pipelines for comparison."""

from .adaptive_rag import run_adaptive
from .agentic_rag import run_agentic
from .always_retrieve import run_always_retrieve
from .direct_llm import run_direct_llm
from .llm_classifier import run_llm_classifier
from .self_consistency import run_self_consistency
from .vanilla_rag import run_vanilla
from .verifier_heuristic import run_verifier_heuristic
from .verifier_only import run_verifier_only

__all__ = [
    "run_adaptive",
    "run_agentic",
    "run_always_retrieve",
    "run_direct_llm",
    "run_llm_classifier",
    "run_self_consistency",
    "run_vanilla",
    "run_verifier_heuristic",
    "run_verifier_only",
]
