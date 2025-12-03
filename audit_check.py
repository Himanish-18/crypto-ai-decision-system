import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
CLEAN_FILE = DATA_DIR / "clean" / "timeseries_clean_1H.csv"
FEATURES_FILE = DATA_DIR / "features" / "features_ml_1H.csv"

def audit():
    print(f"Loading {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE)
    
    # Check timestamp
    if "timestamp" not in df.columns:
        print("❌ Missing 'timestamp' column")
    else:
        # Check if it looks like UTC
        sample_ts = df["timestamp"].iloc[0]
        print(f"ℹ️ Sample timestamp: {sample_ts}")
        if "+00:00" in str(sample_ts) or "Z" in str(sample_ts):
             print("✅ Timestamp appears to be UTC")
        else:
             print("⚠️ Timestamp might not be UTC (check sample)")

    # Check missing values for OHLCV
    ohlcv_cols = [c for c in df.columns if any(x in c for x in ["open", "high", "low", "close", "volume"])]
    missing = df[ohlcv_cols].isnull().sum().sum()
    if missing == 0:
        print("✅ No missing values in OHLCV columns")
    else:
        print(f"❌ Found {missing} missing values in OHLCV columns")

    # Check alignment (simple check: do we have both btc and eth cols populated?)
    if "btc_close" in df.columns and "eth_close" in df.columns:
        print("✅ BTC and ETH columns present")
    
    # Check sentiment
    if "sentiment_mean" in df.columns and "sentiment_count" in df.columns:
        print("✅ Sentiment columns present")
        # Check if we have non-zero sentiment
        non_zero_sent = (df["sentiment_count"] > 0).sum()
        print(f"ℹ️ Rows with news: {non_zero_sent} / {len(df)}")
    else:
        print("❌ Missing sentiment columns")

    # Check returns
    if "btc_ret" in df.columns and "eth_ret" in df.columns:
        print("✅ Returns columns present")
    else:
        print("❌ Missing returns columns")

    # Check target
    if "btc_ret_fwd_1" in df.columns and "y_direction_up" in df.columns:
        print("✅ Target columns present")
    else:
        print("❌ Missing target columns")

    # Time index continuity
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    diffs = df["timestamp"].diff().dropna()
    # Assuming 1H freq
    expected_diff = pd.Timedelta(hours=1)
    gaps = (diffs != expected_diff).sum()
    if gaps == 0:
        print("✅ Time index is continuous (1H steps)")
    else:
        print(f"⚠️ Found {gaps} gaps in time index")
        print(diffs.value_counts().head())

if __name__ == "__main__":
    audit()
