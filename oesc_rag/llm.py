"""OpenAI client wrapper with retry, JSON parsing, and optional caching."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - openai is a runtime requirement
    OpenAI = None  # type: ignore

from tenacity import retry, stop_after_attempt, wait_exponential


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _estimate_tokens(text: str) -> float:
    """Character-based token estimate (~4 chars/token) for cached responses."""
    return float(max(1, len(text or "") // 4))


@dataclass
class LLMConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout: int = 60


class LLMClient:
    """Thin wrapper over the OpenAI chat completions API.

    - Supports structured JSON responses (with parsing fallback).
    - Retries on transient errors.
    - Optional on-disk cache for deterministic re-runs and cost control.
    - Thread-safe for concurrent access.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        import threading
        self._lock = threading.Lock()
        self._local = threading.local()
        self.config = config or LLMConfig(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. pip install openai")
        kwargs: dict[str, Any] = {"timeout": self.config.timeout}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self.client = OpenAI(**kwargs)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Usage accounting. Cached responses contribute zero tokens and zero
        # latency, so a replayed run does not report phantom cost.
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_seconds = 0.0
        self.network_calls = 0
        # Per-thread usage, so a shared client can still attribute cost to one
        # sample when workers run concurrently. (``self._local`` is created in
        # __init__; it must not be re-created here, or earlier thread counters
        # would be discarded.)

    def _thread_usage(self) -> dict[str, float]:
        if not hasattr(self._local, "usage"):
            self._local.usage = {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "latency_seconds": 0.0,
                "network_calls": 0.0,
            }
        return self._local.usage

    def thread_usage(self) -> dict[str, float]:
        usage = dict(self._thread_usage())
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return usage

    def reset_thread_usage(self) -> None:
        self._local.usage = {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "latency_seconds": 0.0,
            "network_calls": 0.0,
        }

    def usage(self) -> dict[str, float]:
        return {
            "prompt_tokens": float(self.prompt_tokens),
            "completion_tokens": float(self.completion_tokens),
            "total_tokens": float(self.prompt_tokens + self.completion_tokens),
            "latency_seconds": float(self.latency_seconds),
            "network_calls": float(self.network_calls),
        }

    def reset_usage(self) -> None:
        self.reset_thread_usage()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_seconds = 0.0
        self.network_calls = 0

    # ------------------------------------------------------------------ cache

    def _cache_key(self, model: str, messages: list[dict], **kwargs: Any) -> str:
        import hashlib

        payload = json.dumps(
            {"model": model, "messages": messages, **kwargs},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def _cache_set(self, key: str, value: str) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / f"{key}.txt"
        with self._lock:
            p.write_text(value, encoding="utf-8")

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Single chat completion. Returns the response content string."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        cache_key = self._cache_key(model, messages, temperature=temperature)
        cached = self._cache_get(cache_key)
        if cached is not None:
            # Cached calls still cost tokens in an apples-to-apples budget
            # comparison, so estimate them rather than reporting zero.
            prompt_est = sum(_estimate_tokens(m.get("content", "")) for m in messages)
            completion_est = _estimate_tokens(cached)
            usage = self._thread_usage()
            usage["prompt_tokens"] += prompt_est
            usage["completion_tokens"] += completion_est
            usage["cache_hits"] = usage.get("cache_hits", 0.0) + 1.0
            with self._lock:
                self.prompt_tokens += prompt_est
                self.completion_tokens += completion_est
            return cached

        @retry(stop=stop_after_attempt(self.config.max_retries),
               wait=wait_exponential(min=1, max=8))
        def _call() -> str:
            started = time.monotonic()
            resp = self.client.chat.completions.create(**kwargs)
            elapsed = time.monotonic() - started
            usage = getattr(resp, "usage", None)
            with self._lock:
                self.network_calls += 1
                self.latency_seconds += elapsed
            thread_usage = self._thread_usage()
            thread_usage["network_calls"] += 1
            thread_usage["latency_seconds"] += elapsed
            if usage is not None:
                prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion = int(getattr(usage, "completion_tokens", 0) or 0)
                with self._lock:
                    self.prompt_tokens += prompt
                    self.completion_tokens += completion
                thread_usage["prompt_tokens"] += prompt
                thread_usage["completion_tokens"] += completion
            return resp.choices[0].message.content or ""

        content = _call()
        self._cache_set(cache_key, content)
        return content

    # ----------------------------------------------------------- JSON helpers

    @staticmethod
    def parse_json(content: str) -> Any:
        """Parse JSON from LLM output, handling fenced blocks and stray text."""
        content = content.strip()
        if not content:
            raise ValueError("Empty LLM output")
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        # Strip code fences
        m = JSON_FENCE_RE.search(content)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        # Find first { or [ and last } or ]
        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            start = content.find(open_ch)
            end = content.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"Could not parse JSON from LLM output: {content[:200]}")

    def chat_json(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Chat completion expecting JSON output; parses and returns the object."""
        content = self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=None,
            max_tokens=max_tokens,
        )
        return self.parse_json(content)


def make_client(
    cache_dir: Optional[str] = "data/cache/llm",
    config_path: Optional[str] = None,
) -> LLMClient:
    """Build a client without reading credentials from repository files.

    The API key is accepted only from the process environment.  A config may
    provide a non-secret base URL plus retry and timeout settings.
    """
    cfg_api: dict[str, Any] = {}
    if config_path:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_api = yaml.safe_load(f).get("api", {})

    llm_config = LLMConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or cfg_api.get("base_url"),
        max_retries=cfg_api.get("max_retries", 3),
        timeout=cfg_api.get("timeout", 60),
    )
    return LLMClient(config=llm_config, cache_dir=cache_dir)
