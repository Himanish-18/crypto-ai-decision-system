import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("performance_analytics")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = DATA_DIR / "execution"
TRADING_LOG = LOG_DIR / "paper_trades.jsonl"  # Default to paper trades for now
OUTPUT_DIR = DATA_DIR / "analytics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trades(log_file: Path = TRADING_LOG) -> pd.DataFrame:
    """Load trading logs from JSONL."""
    if not log_file.exists():
        logger.warning(f"Log file not found: {log_file}")
        return pd.DataFrame()

    data = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Compute performance metrics."""
    if df.empty:
        return {}

    # Filter for closed trades or cycles where equity changed
    # For simplicity, we use the equity curve from the logs
    equity_curve = df["equity"]
    returns = equity_curve.pct_change().dropna()

    # 1. Total Return
    start_equity = equity_curve.iloc[0]
    end_equity = equity_curve.iloc[-1]
    total_return_pct = (end_equity - start_equity) / start_equity * 100

    # 2. Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown_pct = drawdown.min() * 100

    # 3. Sharpe Ratio (Annualized, assuming hourly data)
    # Risk-free rate = 0 for simplicity
    if returns.std() == 0:
        sharpe = 0
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365)

    # 4. Trade Statistics (Need actual trade execution logs, not just cycle logs)
    # We'll infer trades from "action" column if available
    trades = df[df["action"].isin(["BUY", "SELL"])]  # Assuming these are entries
    # Note: accurate trade stats need entry/exit pairing.
    # For now, we'll use cycle-based metrics where possible.

    # Win Rate & Profit Factor (Approximation from returns)
    winning_periods = returns[returns > 0]
    losing_periods = returns[returns < 0]

    win_rate = len(winning_periods) / len(returns) if len(returns) > 0 else 0

    gross_profit = winning_periods.sum()
    gross_loss = abs(losing_periods.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Kelly Criterion (Simple version: W - (1-W)/R)
    # R = Avg Win / Avg Loss
    avg_win = winning_periods.mean() if not winning_periods.empty else 0
    avg_loss = abs(losing_periods.mean()) if not losing_periods.empty else 0

    if avg_loss > 0:
        payoff_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / payoff_ratio
    else:
        kelly = 0

    metrics = {
        "Start Equity": start_equity,
        "End Equity": end_equity,
        "Total Return (%)": total_return_pct,
        "Max Drawdown (%)": max_drawdown_pct,
        "Sharpe Ratio": sharpe,
        "Win Rate (%)": win_rate * 100,
        "Profit Factor": profit_factor,
        "Kelly Criterion": kelly,
        "Total Cycles": len(df),
        "Total Trades (Approx)": len(trades),
    }

    return metrics


def generate_charts(df: pd.DataFrame):
    """Generate Equity Curve and Drawdown charts."""
    if df.empty:
        return

    plt.figure(figsize=(12, 8))

    # Equity Curve
    plt.subplot(2, 1, 1)
    plt.plot(df["timestamp"], df["equity"], label="Equity", color="blue")
    plt.title("Equity Curve")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.legend()

    # Drawdown
    rolling_max = df["equity"].cummax()
    drawdown = (df["equity"] - rolling_max) / rolling_max

    plt.subplot(2, 1, 2)
    plt.fill_between(
        df["timestamp"], drawdown, 0, color="red", alpha=0.3, label="Drawdown"
    )
    plt.title("Drawdown")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    plt.legend()

    output_path = OUTPUT_DIR / "performance_charts.png"
    plt.savefig(output_path)
    logger.info(f"Charts saved to {output_path}")
    plt.close()


def main():
    logger.info("📊 Starting Performance Analytics...")

    # Load Data
    df = load_trades()
    if df.empty:
        logger.warning("No data found. Exiting.")
        return

    # Compute Metrics
    metrics = calculate_metrics(df)

    # Display Metrics
    print("\n" + "=" * 30)
    print(" PERFORMANCE REPORT ")
    print("=" * 30)
    for k, v in metrics.items():
        print(f"{k:<25}: {v:.4f}")
    print("=" * 30 + "\n")

    # Save Metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = OUTPUT_DIR / "performance_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    # Generate Charts
    generate_charts(df)

    logger.info("✅ Analytics Complete.")


if __name__ == "__main__":
    main()
