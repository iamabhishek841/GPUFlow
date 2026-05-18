import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from app.ollama_client import OllamaClient
from app.gpu_monitor import GPUMonitor
from app.workload_generator import build_workload

def get_gpu_metric(gpu_stats: dict, *keys, default=0):
    for key in keys:
        if key in gpu_stats and gpu_stats[key] is not None:
            return gpu_stats[key]
    return default

def estimate_cost_per_1k_tokens(tokens_per_second: float, gpu_hourly_cost_usd: float) -> float:
    """
    Estimate cost per 1K generated tokens.

    Formula:
        tokens_per_hour = tokens_per_second * 3600
        cost_per_1k_tokens = gpu_hourly_cost / (tokens_per_hour / 1000)
    """
    if tokens_per_second <= 0:
        return 0.0

    tokens_per_hour = tokens_per_second * 3600
    return gpu_hourly_cost_usd / (tokens_per_hour / 1000)


def run_model_benchmark(
    model_name: str,
    prompts: list,
    runs_per_prompt: int,
    gpu_hourly_cost_usd: float,
) -> list[dict]:
    client = OllamaClient()
    gpu_monitor = GPUMonitor()

    rows = []

    for prompt_item in tqdm(prompts, desc=f"Benchmarking {model_name}"):
        for run_id in range(1, runs_per_prompt + 1):
            prompt = prompt_item.prompt
            prompt_type = prompt_item.prompt_type

            gpu_monitor.start()
            start = time.perf_counter()

            try:
                response = client.generate(model=model_name, prompt=prompt)
                success = True
                error = None
            except Exception as exc:
                response = None
                success = False
                error = str(exc)

            wall_latency_ms = (time.perf_counter() - start) * 1000
            gpu_stats = gpu_monitor.stop()

            output_tokens = response.output_tokens if response else 0
            prompt_tokens = response.prompt_tokens if response else 0
            tokens_per_second = response.tokens_per_second if response and response.tokens_per_second else 0

            cost_per_1k_tokens = estimate_cost_per_1k_tokens(
                tokens_per_second=tokens_per_second,
                gpu_hourly_cost_usd=gpu_hourly_cost_usd,
            )

            rows.append({
                "model": model_name,
                "prompt_type": prompt_type,
                "run_id": run_id,
                "success": success,
                "error": error,
                "wall_latency_ms": round(wall_latency_ms, 2),
                "prompt_tokens": prompt_tokens or 0,
                "output_tokens": output_tokens or 0,
                "tokens_per_second": round(tokens_per_second, 2),
                "cost_per_1k_tokens_usd": round(cost_per_1k_tokens, 6),
                "gpu_available": gpu_stats.get("gpu_available", False),
                "gpu_avg_util_pct": get_gpu_metric(gpu_stats, "avg_util_pct", "gpu_avg_util_pct"),
                "gpu_max_util_pct": get_gpu_metric(gpu_stats, "max_util_pct", "gpu_max_util_pct"),
                "gpu_avg_memory_mb": get_gpu_metric(gpu_stats, "avg_memory_mb", "gpu_avg_memory_mb"),
                "gpu_max_memory_mb": get_gpu_metric(gpu_stats, "max_memory_mb", "gpu_max_memory_mb"),
                "gpu_total_memory_mb": get_gpu_metric(gpu_stats, "total_memory_mb", "gpu_total_memory_mb"),
                "gpu_max_temperature_c": get_gpu_metric(gpu_stats, "max_temperature_c", "gpu_max_temperature_c"),
                "gpu_avg_power_w": get_gpu_metric(gpu_stats, "avg_power_w", "gpu_avg_power_w"),
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5:3b", "llama3.2:3b", "phi3:mini"],
        help="List of Ollama models to benchmark"
    )
    parser.add_argument("--prompts", type=int, default=12)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--gpu-hourly-cost", type=float, default=0.50)
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    prompt_suite = build_workload(total_prompts=args.prompts)

    all_rows = []

    for model_name in args.models:
        rows = run_model_benchmark(
            model_name=model_name,
            prompts=prompt_suite,
            runs_per_prompt=args.runs,
            gpu_hourly_cost_usd=args.gpu_hourly_cost,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    output_path = "data/model_comparison_benchmark.csv"
    df.to_csv(output_path, index=False)

    successful = df[df["success"] == True].copy()

    if successful.empty:
        print("No successful benchmark runs. Check Ollama model names or server status.")
        return

    summary = (
        successful
        .groupby("model")
        .agg(
            total_runs=("model", "count"),
            avg_latency_ms=("wall_latency_ms", "mean"),
            p95_latency_ms=("wall_latency_ms", lambda x: x.quantile(0.95)),
            avg_tokens_per_second=("tokens_per_second", "mean"),
            avg_cost_per_1k_tokens_usd=("cost_per_1k_tokens_usd", "mean"),
            max_vram_mb=("gpu_max_memory_mb", "max"),
            max_gpu_util_pct=("gpu_max_util_pct", "max"),
            max_temperature_c=("gpu_max_temperature_c", "max"),
        )
        .reset_index()
    )

    summary = summary.round({
        "avg_latency_ms": 2,
        "p95_latency_ms": 2,
        "avg_tokens_per_second": 2,
        "avg_cost_per_1k_tokens_usd": 6,
        "max_vram_mb": 2,
        "max_gpu_util_pct": 2,
        "max_temperature_c": 2,
    })

    summary_path = "data/model_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n===== Model Comparison Summary =====")
    print(summary.to_string(index=False))
    print(f"\nSaved raw results: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()