"""BM25 retrieval over a closed evidence pool (per-sample index)."""

from __future__ import annotations

import re
from typing import Iterable

from rank_bm25 import BM25Okapi

from ..types import EvidenceDoc


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, docs: list[EvidenceDoc]):
        self.docs = docs
        corpus = [_tokenize(f"{d.title} {d.text}") for d in docs]
        # rank_bm25 fails on empty corpora; guard.
        if not corpus or all(len(c) == 0 for c in corpus):
            self.bm25 = None
        else:
            self.bm25 = BM25Okapi(corpus)

    def topk(self, query: str, k: int = 5) -> list[EvidenceDoc]:
        if self.bm25 is None or not self.docs:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return self.docs[:k]
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in order[:k]]

    def retrieve_multi(self, queries: Iterable[str], k: int = 5) -> list[list[EvidenceDoc]]:
        return [self.topk(q, k=k) for q in queries]


def merge_unique(views: list[list[EvidenceDoc]]) -> list[EvidenceDoc]:
    """Union of retrieved docs across views, preserving first-seen order."""
    seen: set[str] = set()
    out: list[EvidenceDoc] = []
    for view in views:
        for d in view:
            if d.doc_id not in seen:
                seen.add(d.doc_id)
                out.append(d)
    return out
