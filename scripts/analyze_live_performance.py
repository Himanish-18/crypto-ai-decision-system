import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_FILE = "live_trading.log"


def parse_log_performance():
    data = []

    current_entry = {}

    # Regex patterns
    # 2025-12-10 20:10:28,979 - main - INFO - Processing Candle: 2025-12-10 14:39:00 | Price: 91863.55
    price_pattern = re.compile(r"Processing Candle: (.*?) \| Price: ([\d\.]+)")

    # 2025-12-10 20:10:29,083 - live_engine - INFO - 🧠 Hybrid v5 Stacker: 0.2546
    # Or "Score 0.2546"
    score_pattern = re.compile(r"(Stacker|Score) ([\d\.]+)")

    with open(LOG_FILE, "r") as f:
        for line in f:
            if "Processing Candle" in line:
                # If we have a pending entry with price but no score, skip it (or save as is)
                # But usually score comes right after.
                if current_entry.get("timestamp") and current_entry.get("score"):
                    data.append(current_entry)

                match = price_pattern.search(line)
                if match:
                    ts_str = match.group(1)  # This is the candle timestamp (UTC)
                    price = float(match.group(2))
                    current_entry = {
                        "timestamp": ts_str,
                        "price": price,
                        "score": None,  # Waiting for score
                        "log_line_ts": line.split(",")[0],  # Log timestamp
                    }

            if "Stacker:" in line or "Score" in line:
                # heuristic to catch score
                match = score_pattern.search(line)
                if match:
                    score = float(match.group(2))
                    # Only take the first score found for a candle
                    if current_entry.get("score") is None:
                        current_entry["score"] = score

    # Append last
    if current_entry.get("timestamp") and current_entry.get("score"):
        data.append(current_entry)

    df = pd.DataFrame(data)
    if df.empty:
        print("No valid data found in logs.")
        return

    df["price"] = pd.to_numeric(df["price"])
    df["score"] = pd.to_numeric(df["score"])

    # Calculate Future Returns (e.g. 5 candles later)
    # Assuming logs are sequential candles
    N_FORWARD = 5
    df["future_price"] = df["price"].shift(-N_FORWARD)
    df["future_ret"] = (df["future_price"] - df["price"]) / df["price"]

    # Calculate Accuracy
    # If Score > 0.5, we expect Ret > 0
    # If Score < 0.5, we expect Ret < 0 (or flat)

    # Define "Prediction"
    # Strong Buy: > 0.55
    # Weak/Neutral: 0.45 - 0.55
    # Bearish: < 0.45

    def get_pred_dir(s):
        if s > 0.52:
            return 1
        elif s < 0.48:
            return -1
        return 0

    df["pred_dir"] = df["score"].apply(get_pred_dir)
    df["actual_dir"] = np.sign(df["future_ret"])

    # Filter for rows where we have a future price and a non-neutral prediction
    valid = df.dropna(subset=["future_ret"])
    active_preds = valid[valid["pred_dir"] != 0]

    correct = active_preds[active_preds["pred_dir"] == active_preds["actual_dir"]]

    print(f"--- Analysis Report ---")
    print(f"Total Candles Processed: {len(df)}")
    print(f"Predictions with Outcome (Next {N_FORWARD} candles): {len(active_preds)}")
    print(f"Correct Predictions: {len(correct)}")

    if len(active_preds) > 0:
        acc = len(correct) / len(active_preds)
        print(f"Directional Accuracy: {acc:.2%}")
    else:
        print("Accuracy: N/A (No active trades/predictions)")

    print(f"\nAverage Score: {df['score'].mean():.4f}")
    print(f"Score Deviation: {df['score'].std():.4f}")

    # Correlation
    valid_corr = valid[["score", "future_ret"]].corr().iloc[0, 1]
    print(f"Score-Return Correlation: {valid_corr:.4f}")

    print("\n--- Recent Samples ---")
    print(valid[["log_line_ts", "price", "score", "future_ret"]].tail(5).to_string())


if __name__ == "__main__":
    parse_log_performance()
