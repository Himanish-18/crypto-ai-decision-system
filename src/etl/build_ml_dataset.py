import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_NEWS_DIR = RAW_DIR / "news"
CLEAN_DIR = DATA_DIR / "clean"
FEATURES_DIR = DATA_DIR / "features"


def ensure_dirs() -> None:
    """Ensure output directories exist."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_price_file(pattern: str) -> pd.DataFrame:
    """
    Load a single price file from data/raw that matches the given pattern.
    Supports .csv and .parquet.
    """
    candidates = list(RAW_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No price file found matching pattern: {pattern}")
    fp = candidates[0]
    if fp.suffix == ".csv":
        df = pd.read_csv(fp)
    elif fp.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(fp)
    else:
        raise ValueError(f"Unsupported price file format: {fp}")
    return df


def load_price_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load BTC and ETH OHLCV price data and standardize columns.

    Expected columns (will try to normalize):
    - timestamp / time / datetime
    - open, high, low, close, volume
    """
    btc = _load_price_file("*btc*.*")
    eth = _load_price_file("*eth*.*")

    def normalize_price_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # Try to infer timestamp column
        ts_col_candidates = ["timestamp", "time", "datetime", "date"]
        ts_col = None
        for c in ts_col_candidates:
            if c in df.columns:
                ts_col = c
                break
        if ts_col is None:
            raise KeyError(f"No timestamp-like column found in price data for {symbol}")

        df = df.copy()
        df.rename(
            columns={ts_col: "timestamp"},
            inplace=True,
        )

        # Standardize timestamp to UTC datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Rename price columns if needed
        col_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc.startswith("open"):
                col_map[c] = f"{symbol}_open"
            elif lc.startswith("high"):
                col_map[c] = f"{symbol}_high"
            elif lc.startswith("low"):
                col_map[c] = f"{symbol}_low"
            elif lc.startswith("close"):
                col_map[c] = f"{symbol}_close"
            elif "volume" in lc:
                col_map[c] = f"{symbol}_volume"
        df.rename(columns=col_map, inplace=True)

        # Keep only timestamp + relevant cols
        keep_cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
        df = (
            df[keep_cols].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        )

        return df

    btc = normalize_price_df(btc, "btc")
    eth = normalize_price_df(eth, "eth")

    return btc, eth


def load_news_sentiment() -> pd.DataFrame:
    """
    Load news+sentiment data from data/raw/news/.

    Expected columns (will try to normalize):
    - publishedAt / timestamp / datetime
    - sentiment (numeric, -1 to 1 ideally)
    - keyword / topic / asset (optional)
    """
    candidates = list(RAW_NEWS_DIR.glob("*.*"))
    if not candidates:
        raise FileNotFoundError(f"No news files found under {RAW_NEWS_DIR}")

    fp = candidates[0]
    if fp.suffix == ".csv":
        df = pd.read_csv(fp)
    elif fp.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(fp)
    else:
        raise ValueError(f"Unsupported news file format: {fp}")

    ts_col_candidates = [
        "published",
        "publishedAt",
        "timestamp",
        "time",
        "datetime",
        "date",
    ]
    ts_col = None
    for c in ts_col_candidates:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise KeyError("No timestamp-like column found in news data")

    df = df.copy()
    df.rename(columns={ts_col: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Try to detect sentiment column
    sentiment_col = None
    for c in df.columns:
        if "sentiment" in c.lower():
            sentiment_col = c
            break
    if sentiment_col is None:
        raise KeyError("No sentiment column found in news data (expected 'sentiment*')")

    df.rename(columns={sentiment_col: "sentiment"}, inplace=True)

    # Optional: keep minimal fields
    keep = ["timestamp", "sentiment"]
    if "keyword" in df.columns:
        keep.append("keyword")
    if "source" in df.columns:
        keep.append("source")
    df = df[keep]

    # Sort + drop dupes
    df = df.dropna(subset=["timestamp"]).drop_duplicates().sort_values("timestamp")
    return df


def aggregate_news_sentiment(
    news_df: pd.DataFrame,
    freq: str = "1H",
) -> pd.DataFrame:
    """
    Aggregate news sentiment into fixed time buckets.

    For each time window (e.g. 1H), compute:
    - mean sentiment
    - count of news items
    """
    df = news_df.copy()
    df.set_index("timestamp", inplace=True)

    agg = df.resample(freq).agg(
        sentiment_mean=("sentiment", "mean"),
        sentiment_count=("sentiment", "count"),
    )

    # Fill gaps with 0 sentiment and 0 count (meaning "no news")
    agg["sentiment_mean"] = agg["sentiment_mean"].fillna(0.0)
    agg["sentiment_count"] = agg["sentiment_count"].fillna(0)

    agg = agg.reset_index()
    return agg


def build_clean_timeseries(freq: str = "1H") -> pd.DataFrame:
    """
    Build a unified, cleaned time series of BTC + ETH OHLCV + aggregated sentiment.
    """
    btc, eth = load_price_data()
    news = load_news_sentiment()
    news_agg = aggregate_news_sentiment(news, freq=freq)

    # Resample BTC/ETH to same freq (if not already)
    def resample_price(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = df.copy().set_index("timestamp")
        price_cols = [c for c in df.columns if c.startswith(symbol)]
        # Use OHLC for close-like field if needed, but here we'll just take last
        resampled = df[price_cols].resample(freq).last()
        return resampled

    btc_r = resample_price(btc, "btc")
    eth_r = resample_price(eth, "eth")

    # Merge all on timestamp
    merged = (
        btc_r.join(eth_r, how="outer")
        .join(news_agg.set_index("timestamp"), how="left")
        .sort_index()
    )

    # Forward-fill prices where reasonable (for missing candles)
    price_cols = [
        c
        for c in merged.columns
        if c.endswith(("_open", "_high", "_low", "_close", "_volume"))
    ]
    merged[price_cols] = merged[price_cols].ffill()

    # Fill missing sentiment stats
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)
    merged["sentiment_count"] = merged["sentiment_count"].fillna(0)

    merged = merged.dropna(how="all")  # drop rows with all NaNs
    merged = merged.reset_index().rename(columns={"index": "timestamp"})
    return merged


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic ML features and target:
    - log returns for BTC/ETH close
    - future return (next-period) as target
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Identify close columns
    btc_close_col = None
    eth_close_col = None
    for c in df.columns:
        if c.lower().startswith("btc") and "close" in c.lower():
            btc_close_col = c
        if c.lower().startswith("eth") and "close" in c.lower():
            eth_close_col = c

    if btc_close_col is None or eth_close_col is None:
        raise KeyError("Could not find BTC/ETH close columns in dataframe")

    # Log returns
    df["btc_ret"] = np.log(df[btc_close_col] / df[btc_close_col].shift(1))
    df["eth_ret"] = np.log(df[eth_close_col] / df[eth_close_col].shift(1))

    # Future returns as target (e.g., next-period BTC return)
    df["btc_ret_fwd_1"] = df["btc_ret"].shift(-1)

    # Binary classification target: 1 if next return > 0, else 0
    df["y_direction_up"] = (df["btc_ret_fwd_1"] > 0).astype(int)

    # Drop initial rows with NaNs due to shifting
    df = df.dropna().reset_index(drop=True)

    return df


def run_etl(freq: str = "1H") -> None:
    """
    Orchestrator: builds clean time series + features and writes to disk.
    """
    ensure_dirs()
    print("📥 Building cleaned timeseries...")
    clean_ts = build_clean_timeseries(freq=freq)

    clean_path_csv = CLEAN_DIR / f"timeseries_clean_{freq}.csv"
    clean_path_parquet = CLEAN_DIR / f"timeseries_clean_{freq}.parquet"
    clean_ts.to_csv(clean_path_csv, index=False)
    clean_ts.to_parquet(clean_path_parquet, index=False)
    print(f"✅ Saved cleaned timeseries to: {clean_path_csv} and {clean_path_parquet}")

    print("🧠 Adding ML features...")
    features = add_basic_features(clean_ts)

    feat_path_csv = FEATURES_DIR / f"features_ml_{freq}.csv"
    feat_path_parquet = FEATURES_DIR / f"features_ml_{freq}.parquet"
    features.to_csv(feat_path_csv, index=False)
    features.to_parquet(feat_path_parquet, index=False)
    print(f"✅ Saved ML features to: {feat_path_csv} and {feat_path_parquet}")


if __name__ == "__main__":
    # Default to 1H candle frequency; can be changed to '15min', '30min', etc.
    run_etl(freq="1H")
