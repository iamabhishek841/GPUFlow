from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BENCHMARK_PATH = Path("data/llm_inference_benchmark.csv")
COLD_WARM_PATH = Path("data/cold_warm_benchmark.csv")
CONCURRENCY_PATH = Path("data/concurrency_benchmark.csv")
MODEL_COMPARISON_PATH = Path("data/model_comparison_summary.csv")
MODEL_COMPARISON_RAW_PATH = Path("data/model_comparison_benchmark.csv")

st.set_page_config(
    page_title="GPUFlow Dashboard",
    page_icon="🚀",
    layout="wide"
)

# -----------------------------
# Dashboard styling
# -----------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            padding: 16px;
            border-radius: 14px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(15,23,42,.06);
        }

        .hero {
            padding: 24px 28px;
            border-radius: 18px;
            background: linear-gradient(90deg, #0f172a, #1e293b);
            color: white;
            margin-bottom: 18px;
        }

        .hero h1 {
            margin-bottom: 6px;
            font-size: 34px;
            font-weight: 800;
        }

        .hero p {
            color: #d1d5db;
            font-size: 15px;
            margin-bottom: 0;
        }

        .insight {
            padding: 15px 18px;
            border-left: 5px solid #f97316;
            background: #fff7ed;
            border-radius: 12px;
            margin: 12px 0 18px 0;
            color: #374151;
            font-size: 15px;
        }

        .note {
            padding: 14px 16px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            color: #475569;
            font-size: 14px;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def polish(fig, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=45, r=45, t=70, b=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=13, color="#374151"),
        title=dict(font=dict(size=18, color="#111827")),
        legend_title_text="",
        xaxis_title=None,
    )
    fig.update_xaxes(tickangle=-15, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig


def safe_pct(numerator, denominator):
    if denominator == 0:
        return 0
    return (numerator / denominator) * 100


# -----------------------------
# Load data
# -----------------------------
benchmark_df = load_csv(BENCHMARK_PATH)
cold_df = load_csv(COLD_WARM_PATH)
concurrency_df = load_csv(CONCURRENCY_PATH)
model_comparison_df = load_csv(MODEL_COMPARISON_PATH)
model_comparison_raw_df = load_csv(MODEL_COMPARISON_RAW_PATH)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🚀 GPUFlow: GPU-Backed LLM Inference Performance Lab</h1>
        <p>
            Local LLM inference benchmarking with NVIDIA GPU telemetry, latency profiling,
            prompt-length analysis, cold/warm comparison, concurrency testing, model comparison,
            and cost-per-token estimation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("GPUFlow")
    st.caption("LLM inference performance dashboard")

    st.markdown("### Data Status")
    st.write(f"Inference benchmark: **{'loaded' if not benchmark_df.empty else 'missing'}**")
    st.write(f"Cold/warm benchmark: **{'loaded' if not cold_df.empty else 'missing'}**")
    st.write(f"Concurrency benchmark: **{'loaded' if not concurrency_df.empty else 'missing'}**")
    st.write(f"Model comparison: **{'loaded' if not model_comparison_df.empty else 'missing'}**")

    st.markdown("---")
    st.markdown("### Useful Commands")
    st.code("ollama pull qwen2.5:3b", language="powershell")
    st.code("python scripts/run_benchmark.py --model qwen2.5:3b", language="powershell")
    st.code("python scripts/cold_warm_benchmark.py --model qwen2.5:3b", language="powershell")
    st.code("python scripts/concurrency_benchmark.py --model qwen2.5:3b", language="powershell")
    st.code(
        "python scripts/model_comparison_benchmark.py --models qwen2.5:3b llama3.2:3b phi3:mini",
        language="powershell"
    )

tabs = st.tabs([
    "📊 Inference Overview",
    "🧠 Prompt Analysis",
    "❄️ Cold vs Warm",
    "⚙️ Concurrency",
    "📈 Model Comparison",
    "💰 Cost Efficiency",
])


# -----------------------------
# Tab 1: Inference Overview
# -----------------------------
with tabs[0]:
    st.subheader("Inference Benchmark Overview")

    if benchmark_df.empty:
        st.warning("No benchmark data found. Run: python scripts/run_benchmark.py --model qwen2.5:3b")
    else:
        ok = benchmark_df[benchmark_df["status"] == "success"].copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Successful Requests", f"{len(ok):,}")
        c2.metric("Avg Latency", f"{ok['wall_latency_ms'].mean():.1f} ms")
        c3.metric("P95 Latency", f"{ok['wall_latency_ms'].quantile(.95):.1f} ms")
        c4.metric("Avg Tokens/sec", f"{ok['tokens_per_second'].mean():.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Max VRAM", f"{ok['gpu_max_memory_mb'].max():.0f} MB")
        c6.metric("Max GPU Util", f"{ok['gpu_max_util_pct'].max():.0f}%")
        c7.metric("Avg Prompt Tokens", f"{ok['prompt_tokens'].mean():.0f}")
        c8.metric("Avg Output Tokens", f"{ok['output_tokens'].mean():.0f}")

        st.markdown(
            """
            <div class="insight">
                <b>Performance lens:</b> This view separates end-to-end latency, token throughput,
                and GPU telemetry so inference bottlenecks can be analyzed beyond a simple chatbot response.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(
                ok,
                x="prompt_type",
                y="wall_latency_ms",
                color="prompt_type",
                title="Latency Distribution by Prompt Type",
                labels={"wall_latency_ms": "Latency (ms)", "prompt_type": "Prompt Type"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        with col2:
            fig = px.scatter(
                ok,
                x="prompt_tokens",
                y="wall_latency_ms",
                color="prompt_type",
                size="output_tokens",
                title="Prompt Tokens vs Latency",
                labels={
                    "prompt_tokens": "Prompt Tokens",
                    "wall_latency_ms": "Latency (ms)",
                    "prompt_type": "Prompt Type",
                },
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            prompt_tps = (
                ok.groupby("prompt_type", as_index=False)["tokens_per_second"]
                .mean()
                .sort_values("tokens_per_second", ascending=False)
            )
            fig = px.bar(
                prompt_tps,
                x="prompt_type",
                y="tokens_per_second",
                title="Average Decode Throughput by Prompt Type",
                text_auto=".2f",
                labels={"tokens_per_second": "Tokens/sec"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        with col4:
            prompt_vram = (
                ok.groupby("prompt_type", as_index=False)["gpu_max_memory_mb"]
                .max()
                .sort_values("gpu_max_memory_mb", ascending=False)
            )
            fig = px.bar(
                prompt_vram,
                x="prompt_type",
                y="gpu_max_memory_mb",
                title="Peak VRAM by Prompt Type",
                text_auto=".0f",
                labels={"gpu_max_memory_mb": "VRAM (MB)"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        st.subheader("Raw Benchmark Results")
        st.dataframe(ok, use_container_width=True, hide_index=True)


# -----------------------------
# Tab 2: Prompt Analysis
# -----------------------------
with tabs[1]:
    st.subheader("Prompt-Length and Token Behavior")

    if benchmark_df.empty:
        st.warning("Run the inference benchmark first.")
    else:
        ok = benchmark_df[benchmark_df["status"] == "success"].copy()

        grouped = ok.groupby("prompt_type", as_index=False).agg(
            requests=("request_id", "count"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            avg_latency_ms=("wall_latency_ms", "mean"),
            p95_latency_ms=("wall_latency_ms", lambda s: s.quantile(.95)),
            avg_tokens_per_second=("tokens_per_second", "mean"),
            max_vram_mb=("gpu_max_memory_mb", "max"),
        )

        grouped = grouped.round({
            "avg_prompt_tokens": 1,
            "avg_output_tokens": 1,
            "avg_latency_ms": 1,
            "p95_latency_ms": 1,
            "avg_tokens_per_second": 2,
            "max_vram_mb": 0,
        })

        st.dataframe(grouped, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="insight">
                <b>Prompt insight:</b> Longer prompts usually increase prefill work, while longer outputs
                increase decode time. This tab helps separate input-size impact from output-generation cost.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                grouped,
                x="prompt_type",
                y="avg_prompt_tokens",
                title="Average Prompt Tokens",
                text_auto=".0f",
                labels={"avg_prompt_tokens": "Prompt Tokens"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        with col2:
            fig = px.bar(
                grouped,
                x="prompt_type",
                y="p95_latency_ms",
                title="P95 Latency by Prompt Type",
                text_auto=".1f",
                labels={"p95_latency_ms": "P95 Latency (ms)"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)


# -----------------------------
# Tab 3: Cold vs Warm
# -----------------------------
with tabs[2]:
    st.subheader("Cold Start vs Warm Run Benchmark")

    if cold_df.empty:
        st.warning("No cold/warm data found. Run: python scripts/cold_warm_benchmark.py --model qwen2.5:3b")
    else:
        st.markdown(
            """
            <div class="insight">
                <b>Cold/warm insight:</b> The first run often includes model load overhead.
                Warm runs help evaluate steady-state serving behavior after the model is already loaded.
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Runs", f"{len(cold_df):,}")
        c2.metric("Avg Latency", f"{cold_df['wall_latency_ms'].mean():.1f} ms")
        c3.metric("Avg Tokens/sec", f"{cold_df['tokens_per_second'].mean():.2f}")
        c4.metric("Max Temp", f"{cold_df['gpu_max_temperature_c'].max():.0f} °C")

        st.dataframe(cold_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                cold_df,
                x="run_number",
                y="wall_latency_ms",
                color="phase",
                title="Cold/Warm Wall Latency",
                text_auto=".1f",
                labels={"wall_latency_ms": "Latency (ms)", "run_number": "Run"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        with col2:
            fig = px.line(
                cold_df,
                x="run_number",
                y="tokens_per_second",
                markers=True,
                title="Tokens/sec Across Repeated Runs",
                labels={"tokens_per_second": "Tokens/sec", "run_number": "Run"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)


# -----------------------------
# Tab 4: Concurrency
# -----------------------------
with tabs[3]:
    st.subheader("Concurrent Client Request Benchmark")

    if concurrency_df.empty:
        st.warning("No concurrency data found. Run: python scripts/concurrency_benchmark.py --model qwen2.5:3b")
    else:
        ok = concurrency_df[concurrency_df["status"] == "success"].copy()

        total = len(concurrency_df)
        success_rate = safe_pct(len(ok), total)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests", f"{total:,}")
        c2.metric("Success Rate", f"{success_rate:.1f}%")
        c3.metric("Throughput", f"{concurrency_df['throughput_rps'].iloc[0]:.2f} rps")
        c4.metric("P95 Client Latency", f"{ok['client_latency_ms'].quantile(.95):.1f} ms")

        st.markdown(
            """
            <div class="insight">
                <b>Concurrency insight:</b> Higher concurrency can increase latency if the GPU or serving backend
                becomes saturated. This benchmark helps identify the point where parallel traffic creates queueing delay.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                ok,
                x="client_latency_ms",
                nbins=30,
                title="Client Latency Distribution",
                labels={"client_latency_ms": "Client Latency (ms)"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        with col2:
            temp = ok.reset_index()
            fig = px.scatter(
                temp,
                x="index",
                y="client_latency_ms",
                title="Concurrent Request Latency",
                labels={"index": "Request Index", "client_latency_ms": "Client Latency (ms)"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        st.dataframe(concurrency_df, use_container_width=True, hide_index=True)


# -----------------------------
# Tab 5: Model Comparison
# -----------------------------
with tabs[4]:
    st.subheader("Model Comparison Dashboard")

    if model_comparison_df.empty:
        st.warning(
            "No model comparison data found. Run: "
            "python scripts/model_comparison_benchmark.py --models qwen2.5:3b llama3.2:3b phi3:mini"
        )
    else:
        st.markdown(
            """
            <div class="insight">
                <b>Model comparison insight:</b> This tab compares models using serving metrics instead of answer quality:
                p95 latency, tokens/sec, VRAM footprint, GPU utilization, and estimated cost per 1K tokens.
            </div>
            """,
            unsafe_allow_html=True,
        )

        best_latency = model_comparison_df.sort_values("p95_latency_ms").iloc[0]
        best_tps = model_comparison_df.sort_values("avg_tokens_per_second", ascending=False).iloc[0]
        best_cost = model_comparison_df.sort_values("avg_cost_per_1k_tokens_usd").iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Best P95 Latency", best_latency["model"])
        c2.metric("Best Tokens/sec", best_tps["model"])
        c3.metric("Best Cost Efficiency", best_cost["model"])

        st.dataframe(model_comparison_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                model_comparison_df,
                x="model",
                y="p95_latency_ms",
                title="P95 Latency by Model",
                text_auto=".1f",
                labels={"p95_latency_ms": "P95 Latency (ms)"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        with col2:
            fig = px.bar(
                model_comparison_df,
                x="model",
                y="avg_tokens_per_second",
                title="Average Tokens/sec by Model",
                text_auto=".2f",
                labels={"avg_tokens_per_second": "Tokens/sec"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig = px.bar(
                model_comparison_df,
                x="model",
                y="max_vram_mb",
                title="Max VRAM Usage by Model",
                text_auto=".0f",
                labels={"max_vram_mb": "VRAM (MB)"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        with col4:
            fig = px.bar(
                model_comparison_df,
                x="model",
                y="max_gpu_util_pct",
                title="Max GPU Utilization by Model",
                text_auto=".0f",
                labels={"max_gpu_util_pct": "GPU Utilization (%)"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        if not model_comparison_raw_df.empty:
            st.subheader("Raw Model Comparison Runs")
            st.dataframe(model_comparison_raw_df, use_container_width=True, hide_index=True)


# -----------------------------
# Tab 6: Cost Efficiency
# -----------------------------
with tabs[5]:
    st.subheader("Cost-per-1K-Tokens Estimation")

    if model_comparison_df.empty:
        st.warning("Run model comparison benchmark first to generate cost estimates.")
    else:
        st.markdown(
            """
            <div class="insight">
                <b>Cost insight:</b> Cost is estimated from measured tokens/sec and a configurable GPU-hour assumption.
                This is not a billing number; it is a performance-to-cost proxy for comparing serving efficiency.
            </div>
            """,
            unsafe_allow_html=True,
        )

        cost_df = model_comparison_df.sort_values("avg_cost_per_1k_tokens_usd").copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Lowest Cost Model", cost_df.iloc[0]["model"])
        c2.metric("Lowest Cost / 1K Tokens", f"${cost_df.iloc[0]['avg_cost_per_1k_tokens_usd']:.6f}")
        c3.metric("Models Compared", f"{len(cost_df)}")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                cost_df,
                x="model",
                y="avg_cost_per_1k_tokens_usd",
                title="Estimated Cost per 1K Tokens",
                text_auto=".6f",
                labels={"avg_cost_per_1k_tokens_usd": "USD / 1K tokens"},
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        with col2:
            fig = px.scatter(
                cost_df,
                x="avg_tokens_per_second",
                y="avg_cost_per_1k_tokens_usd",
                size="max_vram_mb",
                color="model",
                title="Throughput vs Cost Efficiency",
                labels={
                    "avg_tokens_per_second": "Tokens/sec",
                    "avg_cost_per_1k_tokens_usd": "USD / 1K tokens",
                    "max_vram_mb": "VRAM (MB)",
                },
            )
            st.plotly_chart(polish(fig), use_container_width=True)

        st.markdown(
            """
            <div class="note">
                <b>Formula:</b><br>
                tokens_per_hour = tokens_per_second × 3600<br>
                cost_per_1K_tokens = gpu_hourly_cost / (tokens_per_hour / 1000)
            </div>
            """,
            unsafe_allow_html=True,
        )