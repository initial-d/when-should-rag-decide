"""Load and cache HotpotQA for benchmark construction.

HotpotQA distractor split records have:
- _id: question id
- question: str
- answer: str (gold)
- supporting_facts: {title: list[str], sent_id: list[int]}
- context: {title: list[str], sentences: list[list[str]]}
- type: bridge | comparison
- level: easy | medium | hard
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Optional


def load_hotpotqa_samples(
    split: str = "validation",
    n: int = 200,
    seed: int = 42,
    cache_dir: str = "data/raw",
    require_levels: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Load n HotpotQA samples (caches the raw download)."""
    cache_path = Path(cache_dir) / f"hotpotqa_{split}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        from datasets import load_dataset

        ds = load_dataset("hotpot_qa", "distractor", split=split)
        raw = [_normalize_record(r) for r in ds]
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)

    if require_levels:
        raw = [r for r in raw if r.get("level") in require_levels]

    # Filter for multi-hop questions with >=2 supporting paragraphs
    filtered = []
    for r in raw:
        sf_titles = set(r["supporting_facts"]["title"])
        if len(sf_titles) >= 2:
            filtered.append(r)

    rng = random.Random(seed)
    rng.shuffle(filtered)
    return filtered[:n]


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Coerce HF Datasets row into plain dict structure used downstream."""
    return {
        "_id": record["id"],
        "question": record["question"],
        "answer": record["answer"],
        "supporting_facts": {
            "title": list(record["supporting_facts"]["title"]),
            "sent_id": list(record["supporting_facts"]["sent_id"]),
        },
        "context": {
            "title": list(record["context"]["title"]),
            "sentences": [list(s) for s in record["context"]["sentences"]],
        },
        "type": record.get("type", "bridge"),
        "level": record.get("level", "medium"),
    }


def paragraph_text(record: dict[str, Any], idx: int) -> str:
    """Concatenate sentences for paragraph at the given context index."""
    return " ".join(record["context"]["sentences"][idx]).strip()


def supporting_paragraph_indices(record: dict[str, Any]) -> list[int]:
    """Return context indices whose titles appear in supporting_facts."""
    sf_titles = set(record["supporting_facts"]["title"])
    return [
        i for i, t in enumerate(record["context"]["title"])
        if t in sf_titles
    ]


def non_supporting_paragraph_indices(record: dict[str, Any]) -> list[int]:
    sf_titles = set(record["supporting_facts"]["title"])
    return [
        i for i, t in enumerate(record["context"]["title"])
        if t not in sf_titles
    ]
