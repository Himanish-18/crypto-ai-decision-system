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
GUARDIAN_STATE = DATA_DIR / "guardian" / "state.json"

# Load Data
@st.cache_data(ttl=60)
def load_logs():
    if not LOG_FILE.exists():
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

if st.button("Refresh"):
    st.rerun()
