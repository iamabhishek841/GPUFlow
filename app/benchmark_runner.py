"""Real GPU-backed LLM inference benchmark runner.

Runs prompts through Ollama, samples NVIDIA GPU telemetry, and writes a CSV file
that can be visualized by the Streamlit dashboard.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from app.gpu_monitor import GPUMonitor
from app.ollama_client import OllamaClient
from app.workload_generator import build_workload


def benchmark_once(client: OllamaClient, model: str, request, sample_interval: float) -> Dict:
    monitor = GPUMonitor(sample_interval_sec=sample_interval)
    monitor.start()
    error = None
    result = None

    try:
        result = client.generate(model=model, prompt=request.prompt)
    except Exception as exc:  # capture but keep benchmark moving
        error = str(exc)
    finally:
        gpu_metrics = monitor.stop()

    row = {
        "request_id": request.request_id,
        "model": model,
        "prompt_type": request.prompt_type,
        "expected_complexity": request.expected_complexity,
        "prompt_chars": len(request.prompt),
        "status": "success" if error is None else "failed",
        "error": error,
    }

    if result is not None:
        row.update({
            "wall_latency_ms": result.wall_latency_ms,
            "total_duration_ms": result.total_duration_ms,
            "load_duration_ms": result.load_duration_ms,
            "prompt_eval_duration_ms": result.prompt_eval_duration_ms,
            "eval_duration_ms": result.eval_duration_ms,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
            "tokens_per_second": result.tokens_per_second,
            "response_chars": len(result.response),
        })
    else:
        row.update({
            "wall_latency_ms": None,
            "total_duration_ms": None,
            "load_duration_ms": None,
            "prompt_eval_duration_ms": None,
            "eval_duration_ms": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "tokens_per_second": None,
            "response_chars": None,
        })

    row.update(gpu_metrics)
    return row


def run_benchmark(model: str, total_prompts: int, runs_per_prompt: int, output_path: str, base_url: str, sample_interval: float):
    client = OllamaClient(base_url=base_url)
    if not client.health_check():
        raise RuntimeError(
            f"Ollama server not reachable at {base_url}. Start Ollama and pull the model first."
        )

    base_workload = build_workload(total_prompts=total_prompts)
    rows: List[Dict] = []

    expanded = []
    for run_id in range(runs_per_prompt):
        for item in base_workload:
            item.request_id = f"{item.request_id}_run{run_id+1}"
            expanded.append((run_id + 1, item))

    for run_id, request in tqdm(expanded, desc="Running GPUFlow benchmark"):
        row = benchmark_once(client, model, request, sample_interval)
        row["run_id"] = run_id
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run GPUFlow LLM inference benchmark")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--prompts", type=int, default=24, help="Number of base prompts")
    parser.add_argument("--runs", type=int, default=2, help="Runs per prompt")
    parser.add_argument("--output", default="data/llm_inference_benchmark.csv")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--sample-interval", type=float, default=0.2)
    args = parser.parse_args()

    df = run_benchmark(
        model=args.model,
        total_prompts=args.prompts,
        runs_per_prompt=args.runs,
        output_path=args.output,
        base_url=args.base_url,
        sample_interval=args.sample_interval,
    )

    successful = df[df["status"] == "success"]
    print("\n===== GPUFlow Benchmark Summary =====")
    print(f"Rows: {len(df)} | Successful: {len(successful)}")
    if not successful.empty:
        print(f"Avg latency ms: {successful['wall_latency_ms'].mean():.2f}")
        print(f"P95 latency ms: {successful['wall_latency_ms'].quantile(0.95):.2f}")
        print(f"Avg tokens/sec: {successful['tokens_per_second'].mean():.2f}")
        print(f"Max VRAM MB: {successful['gpu_max_memory_mb'].max():.2f}")
        print(f"Max GPU util %: {successful['gpu_max_util_pct'].max():.2f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
