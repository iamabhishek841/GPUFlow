"""Concurrency benchmark for local Ollama inference.

This is not true GPU batching; it measures end-to-end behavior when multiple
client requests are issued concurrently to the local LLM server.
"""
from __future__ import annotations

import argparse
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from app.gpu_monitor import GPUMonitor
from app.ollama_client import OllamaClient
from app.workload_generator import build_workload


def send_request(base_url: str, model: str, prompt: str, request_id: str):
    client = OllamaClient(base_url=base_url, timeout_sec=240)
    start = time.perf_counter()
    try:
        result = client.generate(model, prompt)
        return {
            "request_id": request_id,
            "status": "success",
            "wall_latency_ms": result.wall_latency_ms,
            "client_latency_ms": (time.perf_counter() - start) * 1000,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
            "tokens_per_second": result.tokens_per_second,
            "error": None,
        }
    except Exception as exc:
        return {
            "request_id": request_id,
            "status": "failed",
            "wall_latency_ms": None,
            "client_latency_ms": (time.perf_counter() - start) * 1000,
            "prompt_tokens": None,
            "output_tokens": None,
            "tokens_per_second": None,
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default="data/concurrency_benchmark.csv")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    workload = build_workload(total_prompts=args.requests)
    monitor = GPUMonitor()
    monitor.start()
    start = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(send_request, args.base_url, args.model, item.prompt, f"conc_{uuid.uuid4().hex[:8]}")
            for item in workload
        ]
        for future in as_completed(futures):
            results.append(future.result())

    duration_sec = time.perf_counter() - start
    gpu_summary = monitor.stop()
    df = pd.DataFrame(results)
    df["model"] = args.model
    df["concurrency"] = args.concurrency
    df["total_duration_sec"] = duration_sec
    df["throughput_rps"] = len(df) / duration_sec
    for key, value in gpu_summary.items():
        df[key] = value

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    successful = df[df["status"] == "success"]
    print("\n===== Concurrency Benchmark =====")
    print(f"Requests: {len(df)} | Success: {len(successful)} | Concurrency: {args.concurrency}")
    print(f"Throughput: {len(df)/duration_sec:.2f} req/sec")
    if not successful.empty:
        print(f"Avg client latency ms: {successful['client_latency_ms'].mean():.2f}")
        print(f"P95 client latency ms: {successful['client_latency_ms'].quantile(0.95):.2f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
