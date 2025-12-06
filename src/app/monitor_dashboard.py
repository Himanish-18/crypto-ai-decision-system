import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path

# Config
st.set_page_config(page_title="Crypto AI Bot Dashboard", layout="wide")
st.title("🤖 Crypto AI Trading Bot Dashboard")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOG_FILE = DATA_DIR / "execution" / "logs" / "trading_log.jsonl"
PAPER_LOG = DATA_DIR / "execution" / "paper_trades.jsonl"
GUARDIAN_STATE = DATA_DIR / "guardian" / "state.json"
ANALYTICS_DIR = DATA_DIR / "analytics"
RESEARCH_DIR = DATA_DIR / "research"

PERF_METRICS = ANALYTICS_DIR / "performance_metrics.csv"
PERF_CHARTS = ANALYTICS_DIR / "performance_charts.png"
DRIFT_REPORT = ANALYTICS_DIR / "drift_report.csv"
RL_RESULTS = RESEARCH_DIR / "rl_results.csv"

# Load Data
@st.cache_data(ttl=60)
def load_logs():
    # Fallback to paper logs if main log empty
    target_log = LOG_FILE if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0 else PAPER_LOG
    
    if not target_log.exists():
        return pd.DataFrame()
    
    data = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    if not data:
        return pd.DataFrame()
        
    # Flatten
    rows = []
    for entry in data:
        row = {}
        if "decision" in entry:
            row.update(entry["decision"])
        if "signal" in entry:
            row.update(entry["signal"])
            if "strategy_context" in entry["signal"]:
                row.update(entry["signal"]["strategy_context"])
        rows.append(row)
        
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    return df

def load_guardian_state():
    if not GUARDIAN_STATE.exists():
        return {}
    with open(GUARDIAN_STATE, "r") as f:
        return json.load(f)

# Sidebar
st.sidebar.header("Status")
guardian = load_guardian_state()

if guardian:
    st.sidebar.metric("Start Equity", f"${guardian.get('start_of_day_equity', 0):.2f}")
    
    if guardian.get("is_locked"):
        st.sidebar.error(f"🔒 LOCKED: {guardian.get('lock_reason')}")
    else:
        st.sidebar.success("✅ System Active")
        
    st.sidebar.write(f"Losing Streak: {guardian.get('losing_streak', 0)}")

# Main Dashboard
df = load_logs()

if not df.empty:
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    trades = df[df["action"].isin(["BUY", "SELL"])]
    total_trades = len(trades)
    
    # Calculate PnL if available (mock logic if not in logs yet)
    # Assuming 'net_pnl' column exists if we logged exits
    total_pnl = 0.0
    if "net_pnl" in df.columns:
        total_pnl = df["net_pnl"].sum()
        
    col1.metric("Total Trades", total_trades)
    col2.metric("Total PnL", f"${total_pnl:.2f}")
    
    last_price = df["btc_close"].iloc[-1] if "btc_close" in df.columns else 0
    col3.metric("BTC Price", f"${last_price:,.2f}")
    
    regime = df["regime"].iloc[-1] if "regime" in df.columns else "Unknown"
    col4.metric("Market Regime", regime)

    # Charts
    st.subheader("📈 Equity & Signals")
    
    # Signal Confidence
    fig_conf = px.line(df, x="timestamp", y="signal_confidence", title="Signal Confidence")
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Regime
    if "regime" in df.columns:
        fig_regime = px.scatter(df, x="timestamp", y="btc_close", color="regime", title="Regime Detection")
        st.plotly_chart(fig_regime, use_container_width=True)

    # Recent Logs
    st.subheader("📝 Recent Activity")
    st.dataframe(df.tail(10).sort_values("timestamp", ascending=False))

else:
    st.info("No trading logs found yet.")

# --- NEW PANELS ---

# 1. Performance Analytics
st.markdown("---")
st.header("📊 Performance Analytics")
col_p1, col_p2 = st.columns([1, 2])

with col_p1:
    if PERF_METRICS.exists():
        st.subheader("Key Metrics")
        metrics_df = pd.read_csv(PERF_METRICS)
        st.dataframe(metrics_df.T)
    else:
        st.info("No performance metrics yet.")

with col_p2:
    if PERF_CHARTS.exists():
        st.subheader("Equity & Drawdown")
        st.image(str(PERF_CHARTS))
    else:
        st.info("No performance charts yet.")

st.markdown("---")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.subheader("📊 Regime Distribution (Live)")
    if "regime" in df.columns:
        regime_counts = df["regime"].value_counts().reset_index()
        regime_counts.columns = ["Regime", "Count"]
        fig_reg = px.pie(regime_counts, values="Count", names="Regime", hole=0.4)
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("No regime data.")

with col_d2:
    st.subheader("🎯 Probability Distribution")
    if "prediction_prob" in df.columns:
        fig_hist = px.histogram(df, x="prediction_prob", nbins=20, title="Model Confidence")
        fig_hist.add_vline(x=0.55, line_dash="dash", line_color="green", annotation_text="Threshold")
        st.plotly_chart(fig_hist, use_container_width=True)
    elif "mf_score" in df.columns:
        fig_hist = px.histogram(df, x="mf_score", nbins=20, title="Model Confidence")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No probability data.")

st.markdown("---")
st.subheader("🛡️ Stress Test Benchmarks (Baseline)")
stress_data = {
    "Scenario": ["Base Case", "High Slippage", "High Fees", "Gap Risk"],
    "Profit Factor": [0.46, 0.23, 0.18, 0.48], # Hardcoded from V2 Audit
    "Max Drawdown": ["-26.0%", "-42.3%", "-51.3%", "-28.7%"],
    "Status": ["Baseline", "Critical", "Critical", "Robust"]
}
st.dataframe(pd.DataFrame(stress_data))
st.caption("Live metrics should ideally outperform 'Base Case' and avoid 'Critical' scenarios.")

# 2. Live Drift Detection
st.markdown("---")
st.header("⚠️ Live Drift Detection")
if DRIFT_REPORT.exists():
    drift_df = pd.read_csv(DRIFT_REPORT)
    critical_drift = drift_df[drift_df["status"] == "CRITICAL"]
    
    if not critical_drift.empty:
        st.error(f"🚨 CRITICAL DRIFT DETECTED in {len(critical_drift)} features!")
        st.dataframe(critical_drift)
    else:
        st.success("✅ No critical drift detected.")
        with st.expander("View Full Drift Report"):
            st.dataframe(drift_df)
else:
    st.info("No drift report available.")

# 3. RL Simulation Results
st.markdown("---")
st.header("🤖 RL Agent Simulation")
if RL_RESULTS.exists():
    rl_df = pd.read_csv(RL_RESULTS)
    st.line_chart(rl_df["equity"])
    st.caption("RL Agent Equity Curve (Simulation)")
else:
    st.info("No RL simulation results yet.")

if st.button("Refresh"):
    st.rerun()
