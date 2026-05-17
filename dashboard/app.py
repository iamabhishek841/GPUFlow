from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BENCHMARK_PATH = Path("data/llm_inference_benchmark.csv")
COLD_WARM_PATH = Path("data/cold_warm_benchmark.csv")
CONCURRENCY_PATH = Path("data/concurrency_benchmark.csv")

st.set_page_config(page_title="GPUFlow Dashboard", page_icon="🚀", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetric"] {
            background: white;
            padding: 16px;
            border-radius: 14px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(15,23,42,.06);
        }
        .hero {
            padding: 24px;
            border-radius: 20px;
            background: linear-gradient(90deg, #111827, #1f2937);
            color: white;
            margin-bottom: 18px;
        }
        .hero h1 { margin-bottom: 6px; }
        .hero p { color: #d1d5db; }
        .insight {
            padding: 15px 18px;
            border-left: 5px solid #f97316;
            background: #fff7ed;
            border-radius: 12px;
            margin: 12px 0 18px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def polish(fig, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=40, t=70, b=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=13),
        legend_title_text="",
        xaxis_title=None,
    )
    fig.update_xaxes(tickangle=-15, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig


benchmark_df = load_csv(BENCHMARK_PATH)
cold_df = load_csv(COLD_WARM_PATH)
concurrency_df = load_csv(CONCURRENCY_PATH)

st.markdown(
    """
    <div class="hero">
        <h1>🚀 GPUFlow: GPU-Backed LLM Inference Performance Lab</h1>
        <p>Real local LLM inference benchmarking with NVIDIA GPU telemetry, latency profiling, prompt-length analysis, cold/warm comparison, and concurrency testing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("GPUFlow")
    st.caption("LLM inference performance dashboard")
    st.markdown("### Data Files")
    st.write(f"Inference benchmark: **{'loaded' if not benchmark_df.empty else 'missing'}**")
    st.write(f"Cold/warm benchmark: **{'loaded' if not cold_df.empty else 'missing'}**")
    st.write(f"Concurrency benchmark: **{'loaded' if not concurrency_df.empty else 'missing'}**")
    st.markdown("### Commands")
    st.code("ollama pull qwen2.5:3b", language="powershell")
    st.code("python scripts/run_benchmark.py --model qwen2.5:3b", language="powershell")
    st.code("python scripts/cold_warm_benchmark.py --model qwen2.5:3b", language="powershell")
    st.code("python scripts/concurrency_benchmark.py --model qwen2.5:3b", language="powershell")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Inference Overview",
    "🧠 Prompt Analysis",
    "❄️ Cold vs Warm",
    "⚙️ Concurrency",
])

with tab1:
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
            <div class="insight"><b>Performance lens:</b> This view separates end-to-end latency, token throughput, and GPU telemetry so the bottleneck can be analyzed beyond a simple chatbot response.</div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(ok, x="prompt_type", y="wall_latency_ms", color="prompt_type", title="Latency Distribution by Prompt Type")
            st.plotly_chart(polish(fig), use_container_width=True)
        with col2:
            fig = px.scatter(ok, x="prompt_tokens", y="wall_latency_ms", color="prompt_type", size="output_tokens", title="Prompt Tokens vs Latency")
            st.plotly_chart(polish(fig), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.bar(ok.groupby("prompt_type", as_index=False)["tokens_per_second"].mean(), x="prompt_type", y="tokens_per_second", title="Average Decode Throughput by Prompt Type", text_auto=".2f")
            st.plotly_chart(polish(fig), use_container_width=True)
        with col4:
            fig = px.bar(ok.groupby("prompt_type", as_index=False)["gpu_max_memory_mb"].max(), x="prompt_type", y="gpu_max_memory_mb", title="Peak VRAM by Prompt Type", text_auto=".0f")
            st.plotly_chart(polish(fig), use_container_width=True)

        st.subheader("Raw Benchmark Results")
        st.dataframe(ok, use_container_width=True, hide_index=True)

with tab2:
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
        )
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(grouped, x="prompt_type", y="avg_prompt_tokens", title="Average Prompt Tokens", text_auto=".0f")
            st.plotly_chart(polish(fig), use_container_width=True)
        with col2:
            fig = px.bar(grouped, x="prompt_type", y="p95_latency_ms", title="P95 Latency by Prompt Type", text_auto=".1f")
            st.plotly_chart(polish(fig), use_container_width=True)

with tab3:
    st.subheader("Cold Start vs Warm Run Benchmark")
    if cold_df.empty:
        st.warning("No cold/warm data found. Run: python scripts/cold_warm_benchmark.py --model qwen2.5:3b")
    else:
        st.dataframe(cold_df, use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(cold_df, x="run_number", y="wall_latency_ms", color="phase", title="Cold/Warm Wall Latency", text_auto=".1f")
            st.plotly_chart(polish(fig), use_container_width=True)
        with col2:
            fig = px.line(cold_df, x="run_number", y="tokens_per_second", markers=True, title="Tokens/sec Across Repeated Runs")
            st.plotly_chart(polish(fig), use_container_width=True)

with tab4:
    st.subheader("Concurrent Client Request Benchmark")
    if concurrency_df.empty:
        st.warning("No concurrency data found. Run: python scripts/concurrency_benchmark.py --model qwen2.5:3b")
    else:
        ok = concurrency_df[concurrency_df["status"] == "success"].copy()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests", f"{len(concurrency_df):,}")
        c2.metric("Success Rate", f"{len(ok)/len(concurrency_df)*100:.1f}%")
        c3.metric("Throughput", f"{concurrency_df['throughput_rps'].iloc[0]:.2f} rps")
        c4.metric("P95 Client Latency", f"{ok['client_latency_ms'].quantile(.95):.1f} ms")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(ok, x="client_latency_ms", nbins=30, title="Client Latency Distribution")
            st.plotly_chart(polish(fig), use_container_width=True)
        with col2:
            fig = px.scatter(ok.reset_index(), x="index", y="client_latency_ms", title="Concurrent Request Latency")
            st.plotly_chart(polish(fig), use_container_width=True)
        st.dataframe(concurrency_df, use_container_width=True, hide_index=True)
