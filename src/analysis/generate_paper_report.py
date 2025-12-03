import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("report_generator")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOG_FILE = DATA_DIR / "execution" / "paper_trades.jsonl"
REPORT_FILE = PROJECT_ROOT / "paper_trading_report.md"
EQUITY_CURVE_FILE = PROJECT_ROOT / "equity_curve.png"

def load_logs():
    """Load logs from JSONL."""
    if not LOG_FILE.exists():
        logger.error(f"Log file not found: {LOG_FILE}")
        return pd.DataFrame()
        
    data = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
                
    df = pd.DataFrame(data)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    return df

def generate_report(df):
    """Generate markdown report and equity curve."""
    if df.empty:
        logger.warning("No data to generate report.")
        return

    # Metrics
    total_cycles = len(df)
    trades = df[df["action"] != "HOLD"]
    total_trades = len(trades)
    
    start_equity = df["equity"].iloc[0]
    end_equity = df["equity"].iloc[-1]
    total_return = (end_equity - start_equity) / start_equity * 100
    
    max_equity = df["equity"].cummax()
    drawdown = (df["equity"] - max_equity) / max_equity
    max_drawdown = drawdown.min() * 100
    
    regime_counts = df["regime"].value_counts().to_dict()
    
    # Guardian Stats
    locks = df["is_locked"].sum()
    max_losing_streak = df["losing_streak"].max()
    
    # Generate Markdown
    report = f"""# Paper Trading Execution Report

**Generated**: {pd.Timestamp.now()}
**Duration**: {df['timestamp'].min()} to {df['timestamp'].max()}

## 1. Performance Summary
| Metric | Value |
| :--- | :--- |
| **Total Cycles** | {total_cycles} |
| **Total Trades** | {total_trades} |
| **Start Equity** | ${start_equity:.2f} |
| **End Equity** | ${end_equity:.2f} |
| **Total Return** | **{total_return:.2f}%** |
| **Max Drawdown** | **{max_drawdown:.2f}%** |

## 2. Risk & Safety
| Metric | Value |
| :--- | :--- |
| **Guardian Locks** | {locks} |
| **Max Losing Streak** | {max_losing_streak} |
| **Max Exposure** | ${df['exposure'].max():.2f} |

## 3. Regime Distribution
{pd.DataFrame(list(regime_counts.items()), columns=['Regime', 'Count']).to_markdown(index=False)}

## 4. Trade Log (Last 10)
{trades.tail(10)[['timestamp', 'action', 'price', 'size', 'regime']].to_markdown(index=False)}
"""

    with open(REPORT_FILE, "w") as f:
        f.write(report)
    logger.info(f"✅ Report saved to {REPORT_FILE}")

    # Plot Equity Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp"], df["equity"], label="Equity")
    plt.title("Paper Trading Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity (USDT)")
    plt.grid(True)
    plt.legend()
    plt.savefig(EQUITY_CURVE_FILE)
    logger.info(f"✅ Equity curve saved to {EQUITY_CURVE_FILE}")

if __name__ == "__main__":
    df = load_logs()
    generate_report(df)
