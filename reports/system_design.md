# GPUFlow System Design

## 1. Overview

GPUFlow is a GPU-backed LLM inference performance lab. It uses a local Ollama server to execute real LLM inference and captures NVIDIA GPU telemetry during each request.

The system is designed to help answer performance-engineering questions such as:

- How does prompt length affect latency?
- How much VRAM is used during inference?
- How does cold start differ from warm execution?
- How does concurrent request pressure affect p95 latency?
- How does token generation speed vary by prompt type?

## 2. Components

### Prompt Workload Generator

Creates realistic prompt categories:

- short question answering
- medium reasoning
- long-context summarization
- code-generation style prompts

### Ollama Client

Calls the local Ollama `/api/generate` endpoint with `stream=false` and extracts timing data including:

- total duration
- model load duration
- prompt evaluation duration
- decode/eval duration
- prompt token count
- output token count
- tokens/sec

### GPU Telemetry Collector

Uses pynvml to sample:

- GPU utilization
- VRAM used
- VRAM total
- temperature
- power usage where available

### Benchmark Runner

Runs prompts, captures telemetry, and writes results to CSV.

### Dashboard

Streamlit dashboard visualizes:

- latency distribution
- prompt tokens vs latency
- tokens/sec by prompt type
- peak VRAM by prompt type
- cold vs warm runs
- concurrency behavior

## 3. Performance Engineering Concepts

### Latency Breakdown

Inference latency is not a single number. It can include:

- model loading
- prompt prefill
- token decoding
- client overhead
- queueing/concurrency delay

### Prompt Length Impact

Long prompts increase prefill work and can increase memory pressure. GPUFlow compares short, medium and long-context workloads to show this relationship.

### Cold vs Warm Execution

The first run may include model load and warmup costs. Later runs often reuse loaded model state and are faster.

### Concurrency

Concurrent client requests can increase throughput but may increase tail latency depending on backend scheduling and hardware limits.

## 4. Interview Positioning

GPUFlow is useful for AI infrastructure, ML platform, performance engineering and inference-serving roles because it focuses on measurement, bottleneck analysis, GPU telemetry and serving behavior rather than application UI.
