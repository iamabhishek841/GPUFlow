"""Prompt workload generator for LLM inference benchmarking."""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class WorkloadPrompt:
    request_id: str
    prompt_type: str
    prompt: str
    expected_complexity: str


def _long_context_block(repeats: int = 16) -> str:
    base = (
        "Enterprise AI infrastructure teams operate model-serving systems where "
        "requests vary in prompt length, output length, cache locality, and latency "
        "SLO. Performance engineers monitor p95 latency, throughput, VRAM usage, "
        "cold starts, batching behavior, and queueing delay. "
    )
    return base * repeats


PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "short_qa": [
        "Explain LLM inference latency in five bullet points.",
        "What is GPU memory pressure in AI serving?",
        "Define tokens per second in model inference.",
    ],
    "medium_reasoning": [
        "Compare model cold start, queue delay, and decode latency in an LLM serving system. Give practical examples.",
        "Explain why p95 latency matters more than average latency for production inference workloads.",
        "Describe how batching improves throughput but can affect tail latency.",
    ],
    "long_context": [
        _long_context_block(18) + "\nSummarize the operational bottlenecks and suggest three optimizations.",
        _long_context_block(24) + "\nIdentify which metrics should be monitored in a GPU-backed inference service.",
    ],
    "code_generation": [
        "Write a Python function that records request latency, p95 latency, throughput, and error rate for an inference service.",
        "Create pseudocode for model-aware routing across a worker pool with cache-hit tracking.",
    ],
}


def build_workload(total_prompts: int = 24, seed: int = 42) -> List[WorkloadPrompt]:
    random.seed(seed)
    prompt_types = list(PROMPT_TEMPLATES.keys())
    weights = [30, 30, 25, 15]
    workload: List[WorkloadPrompt] = []

    for i in range(total_prompts):
        prompt_type = random.choices(prompt_types, weights=weights, k=1)[0]
        prompt = random.choice(PROMPT_TEMPLATES[prompt_type])
        complexity = {
            "short_qa": "low",
            "medium_reasoning": "medium",
            "long_context": "high",
            "code_generation": "medium-high",
        }[prompt_type]

        workload.append(
            WorkloadPrompt(
                request_id=f"gpu_req_{i}_{uuid.uuid4().hex[:8]}",
                prompt_type=prompt_type,
                prompt=prompt,
                expected_complexity=complexity,
            )
        )

    return workload
