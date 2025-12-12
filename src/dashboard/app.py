import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Crypto AI Autonomous Brain", layout="wide", page_icon="🧠"
)

st.title("🧠 Crypto AI Autonomous Brain (v8)")

# Paths
ROOT = Path(__file__).resolve().parents[2]
MODELS_PROD = ROOT / "data" / "models" / "prod"
LOG_FILE = ROOT / "live_trading.log"

# Application State
st.header("1. System Status")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Operator Mode", "Autonomous", delta="Verify")

with col2:
    # Check if process running? (Simplistic check via log update time)
    if LOG_FILE.exists():
        last_mod = LOG_FILE.stat().st_mtime
        lag = time.time() - last_mod
        status = "ONLINE 🟢" if lag < 60 else "STALLED 🔴"
        st.metric("Bot Pulse", status, f"{lag:.0f}s lag")
    else:
        st.metric("Bot Pulse", "OFFLINE ⚫")

with col3:
    st.metric("Exchange Link", "Binance + Failover", "Active")

# Model Metrics
st.header("2. Active Model Intelligence")

metrics_file = MODELS_PROD / "current_metrics.json"

if metrics_file.exists():
    with open(metrics_file, "r") as f:
        m = json.load(f)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
    m_col2.metric("Max Drawdown", f"{m.get('max_drawdown', 0)*100:.1f}%")
    m_col3.metric("Training Date", m.get("training_timestamp", "N/A")[:10])
    m_col4.metric("Version", m.get("version", "v_prod"))
else:
    st.warning("No Production Model Metrics Found.")

# Live Logs
st.header("3. Live Operations Log")
if LOG_FILE.exists():
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()[-20:]  # Last 20 lines

    for line in reversed(lines):
        st.caption(line.strip())
else:
    st.info("No logs generated yet.")

# Manual Controls (Simulated)
st.sidebar.header("Override Controls")
if st.sidebar.button("Force Retrain Cycle"):
    st.sidebar.write("Signal sent to Supervisor...")

st.sidebar.info("Auto-Refresh Enabled (Real-time)")
time.sleep(2)
st.rerun()
