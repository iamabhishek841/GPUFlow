"""Ollama client used for real local LLM inference benchmarking.

The module calls a locally running Ollama server and extracts timing fields
returned by Ollama. Durations returned by Ollama are in nanoseconds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class OllamaResult:
    model: str
    prompt: str
    response: str
    wall_latency_ms: float
    total_duration_ms: Optional[float]
    load_duration_ms: Optional[float]
    prompt_eval_duration_ms: Optional[float]
    eval_duration_ms: Optional[float]
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    tokens_per_second: Optional[float]
    raw: Dict[str, Any]


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_sec: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None) -> OllamaResult:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_sec,
        )
        wall_latency_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        raw = response.json()

        def ns_to_ms(value: Optional[int]) -> Optional[float]:
            if value is None:
                return None
            return value / 1_000_000

        eval_count = raw.get("eval_count")
        eval_duration_ns = raw.get("eval_duration")
        tokens_per_second = None
        if eval_count and eval_duration_ns and eval_duration_ns > 0:
            tokens_per_second = eval_count / (eval_duration_ns / 1_000_000_000)

        return OllamaResult(
            model=model,
            prompt=prompt,
            response=raw.get("response", ""),
            wall_latency_ms=wall_latency_ms,
            total_duration_ms=ns_to_ms(raw.get("total_duration")),
            load_duration_ms=ns_to_ms(raw.get("load_duration")),
            prompt_eval_duration_ms=ns_to_ms(raw.get("prompt_eval_duration")),
            eval_duration_ms=ns_to_ms(eval_duration_ns),
            prompt_tokens=raw.get("prompt_eval_count"),
            output_tokens=eval_count,
            tokens_per_second=tokens_per_second,
            raw=raw,
        )
