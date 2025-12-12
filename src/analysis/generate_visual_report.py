import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Config
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (15, 8)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "execution"
ANALYTICS_DIR = DATA_DIR / "analytics"
RESEARCH_DIR = DATA_DIR / "research"
FEATURES_DIR = DATA_DIR / "features"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_trading_logs():
    # Try main log first, then paper log
    log_file = LOGS_DIR / "logs" / "trading_log.jsonl"
    if not log_file.exists() or log_file.stat().st_size == 0:
        log_file = LOGS_DIR / "paper_trades.jsonl"

    if not log_file.exists():
        return pd.DataFrame()

    data = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue

    # Flatten logs
    rows = []
    for entry in data:
        row = {}
        # Top level keys
        for k, v in entry.items():
            if k not in ["decision", "signal"]:
                row[k] = v
        # Nested keys
        if "decision" in entry:
            row.update(entry["decision"])
        if "signal" in entry:
            row.update(entry["signal"])
            if "strategy_context" in entry["signal"]:
                row.update(entry["signal"]["strategy_context"])
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    # Check for critical columns
    if "equity" not in df.columns:
        print("⚠️ 'equity' column missing in main log. Trying paper trades...")
        paper_log = LOGS_DIR / "paper_trades.jsonl"
        if paper_log.exists():
            # Load paper trades (simple structure)
            paper_data = []
            with open(paper_log, "r") as f:
                for line in f:
                    try:
                        paper_data.append(json.loads(line))
                    except:
                        continue
            df_paper = pd.DataFrame(paper_data)
            if "timestamp" in df_paper.columns:
                df_paper["timestamp"] = pd.to_datetime(df_paper["timestamp"])
                df_paper = df_paper.sort_values("timestamp")
            return df_paper

    return df


def plot_performance():
    print("📊 Plotting Performance...")
    df = load_trading_logs()
    if df.empty:
        print("No trading logs found.")
        return

    if "timestamp" not in df.columns or "equity" not in df.columns:
        print(f"Missing required columns. Available: {df.columns.tolist()}")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Equity
    ax1.plot(df["timestamp"], df["equity"], label="Equity", color="#00ff00")
    ax1.set_title("Equity Curve", fontsize=16)
    ax1.set_ylabel("Equity ($)")
    ax1.legend()

    # Drawdown
    rolling_max = df["equity"].cummax()
    drawdown = (df["equity"] - rolling_max) / rolling_max * 100

    ax2.fill_between(
        df["timestamp"], drawdown, 0, color="#ff0000", alpha=0.3, label="Drawdown %"
    )
    ax2.set_title("Drawdown", fontsize=14)
    ax2.set_ylabel("%")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "performance_analytics.png")
    plt.close()


def plot_drift():
    print("⚠️ Plotting Drift...")
    drift_file = ANALYTICS_DIR / "drift_report.csv"
    if not drift_file.exists():
        print("No drift report found.")
        return

    df = pd.read_csv(drift_file)
    top_drift = df.sort_values("psi", ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_drift, x="psi", y="feature", hue="status", dodge=False)
    plt.title("Top 10 Features by Drift (PSI)", fontsize=16)
    plt.xlabel("Population Stability Index (PSI)")
    plt.axvline(0.1, color="orange", linestyle="--", label="Warning (0.1)")
    plt.axvline(0.2, color="red", linestyle="--", label="Critical (0.2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "data_drift.png")
    plt.close()


def plot_rl():
    print("🤖 Plotting RL Results...")
    rl_file = RESEARCH_DIR / "rl_results.csv"
    if not rl_file.exists():
        print("No RL results found.")
        return

    df = pd.read_csv(rl_file)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["equity"], color="purple", label="RL Agent")
    plt.title("RL Agent Training Performance", fontsize=16)
    plt.ylabel("Equity ($)")
    plt.xlabel("Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rl_simulation.png")
    plt.close()


def plot_regimes():
    print("📈 Plotting Market Regimes...")
    features_file = FEATURES_DIR / "features_1H_advanced.parquet"
    if not features_file.exists():
        print("No features file found.")
        return

    df = pd.read_parquet(features_file)
    df = df.iloc[-500:].copy()  # Last 500 candles

    plt.figure(figsize=(15, 8))
    plt.plot(df["timestamp"], df["btc_close"], label="Price", color="black", alpha=0.5)

    # Mock regime if not present (for visualization demo)
    if "regime" not in df.columns:
        # Create mock regimes based on volatility for demo
        df["regime"] = pd.cut(
            df["btc_atr_14"], bins=3, labels=["Low Vol", "Med Vol", "High Vol"]
        )

    sns.scatterplot(
        data=df, x="timestamp", y="btc_close", hue="regime", palette="deep", s=30
    )

    plt.title("BTC Price Action & Market Regimes", fontsize=16)
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "market_regimes.png")
    plt.close()


def main():
    plot_performance()
    plot_drift()
    plot_rl()
    plot_regimes()
    print(f"✅ All plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
