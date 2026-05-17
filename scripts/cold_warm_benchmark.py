"""Cold vs warm run benchmark.

The first call to a model often includes model load/warmup overhead. This script
runs the same prompt multiple times to compare the first run with later runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from app.gpu_monitor import GPUMonitor
from app.ollama_client import OllamaClient

PROMPT = "Explain how cold start, prompt prefill, decode latency, and GPU memory affect LLM inference performance."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--output", default="data/cold_warm_benchmark.csv")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    client = OllamaClient(base_url=args.base_url)
    rows = []

    for i in tqdm(range(args.runs), desc="Cold/warm benchmark"):
        monitor = GPUMonitor()
        monitor.start()
        result = client.generate(args.model, PROMPT)
        gpu = monitor.stop()
        rows.append({
            "run_number": i + 1,
            "phase": "cold_or_first" if i == 0 else "warm",
            "model": args.model,
            "wall_latency_ms": result.wall_latency_ms,
            "load_duration_ms": result.load_duration_ms,
            "prompt_eval_duration_ms": result.prompt_eval_duration_ms,
            "eval_duration_ms": result.eval_duration_ms,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
            "tokens_per_second": result.tokens_per_second,
            **gpu,
        })

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
