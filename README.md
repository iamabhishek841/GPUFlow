# GPUFlow — GPU-Backed LLM Inference Performance Lab

GPUFlow is a real local LLM inference benchmarking project focused on AI infrastructure and performance engineering. It is designed to profile how prompt length, cold starts, concurrency, token generation speed, GPU memory, and GPU utilization affect LLM serving behavior.

This is not a chatbot project. The goal is to measure and explain inference performance.

## Why This Project Exists

Modern AI infrastructure teams care about more than whether a model can generate an answer. They need to understand:

- end-to-end inference latency
- p50 / p95 / p99 latency
- tokens per second
- GPU utilization
- VRAM usage
- cold-start vs warm-run behavior
- prompt-length impact
- concurrency bottlenecks
- throughput vs latency trade-offs

GPUFlow builds a local benchmarking lab around these concerns.

## Architecture

```text
Prompt Workload Generator
        ↓
Ollama Local LLM Server
        ↓
Real LLM Inference on NVIDIA GPU
        ↓
GPU Telemetry Collector using pynvml
        ↓
Benchmark CSV Store
        ↓
Streamlit Performance Dashboard
```

## Features

- Real local LLM inference through Ollama
- NVIDIA GPU telemetry collection using pynvml
- Prompt workload generator: short, medium, long-context and code prompts
- Latency measurement
- Tokens/sec measurement
- Prompt token and output token analysis
- VRAM and GPU utilization tracking
- Cold vs warm inference benchmark
- Concurrent request benchmark
- Streamlit dashboard for performance analysis

## Recommended Hardware

This project was designed for local NVIDIA GPU systems such as RTX 4060 Laptop GPU with 8GB VRAM. Use small quantized models first.

Recommended models:

```powershell
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

## Setup

Create virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Check Ollama:

```powershell
ollama --version
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

## Run Benchmarks

Main inference benchmark:

```powershell
python scripts/run_benchmark.py --model qwen2.5:3b --prompts 24 --runs 2
```

Cold vs warm benchmark:

```powershell
python scripts/cold_warm_benchmark.py --model qwen2.5:3b --runs 8
```

Concurrent request benchmark:

```powershell
python scripts/concurrency_benchmark.py --model qwen2.5:3b --requests 24 --concurrency 4
```

Run dashboard:

```powershell
streamlit run dashboard/app.py
```

## What to Discuss in Interviews

A strong explanation:

> I built GPUFlow to understand the performance side of LLM serving. I ran real local LLM inference on an NVIDIA GPU and measured latency, tokens/sec, VRAM usage, GPU utilization, prompt-length impact, and cold/warm inference behavior. The goal was not to build another chatbot, but to profile inference bottlenecks and understand how prompt length, model size, GPU memory, and concurrency affect serving performance.

## Future Improvements

- Add vLLM or Triton backend
- Add real batching backend
- Add Prometheus/Grafana monitoring
- Add model comparison dashboard
- Add cost-per-1K-tokens estimation
- Add ONNX/PyTorch inference path
- Add Kubernetes deployment profile
