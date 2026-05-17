from __future__ import annotations

import pandas as pd


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    successful = df[df["status"] == "success"].copy()
    if successful.empty:
        return pd.DataFrame()

    return (
        successful.groupby("prompt_type")
        .agg(
            requests=("request_id", "count"),
            avg_latency_ms=("wall_latency_ms", "mean"),
            p50_latency_ms=("wall_latency_ms", lambda s: s.quantile(0.50)),
            p95_latency_ms=("wall_latency_ms", lambda s: s.quantile(0.95)),
            avg_tokens_per_second=("tokens_per_second", "mean"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            max_vram_mb=("gpu_max_memory_mb", "max"),
            max_gpu_util_pct=("gpu_max_util_pct", "max"),
        )
        .reset_index()
    )
