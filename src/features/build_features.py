import os
from pathlib import Path
import pandas as pd
import numpy as np
import ta

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_FILE = DATA_DIR / "features" / "features_ml_1H.csv"
FEATURES_DIR = DATA_DIR / "features"
OUTPUT_CSV = FEATURES_DIR / "features_1H_advanced.csv"
OUTPUT_PARQUET = FEATURES_DIR / "features_1H_advanced.parquet"

def load_data() -> pd.DataFrame:
    """Load cleaned time series data."""
    if not CLEAN_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {CLEAN_FILE}")
    
    print(f"📥 Loading data from {CLEAN_FILE}...")
    df = pd.read_csv(CLEAN_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add Technical Indicators (RSI, MACD, BB, ATR) for BTC and ETH."""
    print("📈 Adding Technical Indicators...")
    df = df.copy()
    
    for symbol in ["btc", "eth"]:
        close = df[f"{symbol}_close"]
        high = df[f"{symbol}_high"]
        low = df[f"{symbol}_low"]
        
        # RSI (14)
        df[f"{symbol}_rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        
        # MACD (12, 26, 9)
        macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df[f"{symbol}_macd"] = macd.macd()
        df[f"{symbol}_macd_signal"] = macd.macd_signal()
        df[f"{symbol}_macd_diff"] = macd.macd_diff()
        
        # Bollinger Bands (20, 2)
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df[f"{symbol}_bb_high"] = bb.bollinger_hband()
        df[f"{symbol}_bb_low"] = bb.bollinger_lband()
        df[f"{symbol}_bb_width"] = bb.bollinger_wband()
        
        # ATR (14)
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        df[f"{symbol}_atr_14"] = atr.average_true_range()
        
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Rolling Statistics (Mean, Std, Z-score, Momentum)."""
    print("🔄 Adding Rolling Statistics...")
    df = df.copy()
    
    windows = [5, 10, 20, 50]
    
    for symbol in ["btc", "eth"]:
        close = df[f"{symbol}_close"]
        
        for w in windows:
            # Rolling Mean & Std
            roll_mean = close.rolling(window=w).mean()
            roll_std = close.rolling(window=w).std()
            
            df[f"{symbol}_roll_mean_{w}"] = roll_mean
            df[f"{symbol}_roll_std_{w}"] = roll_std
            
            # Z-Score: (Price - Mean) / Std
            df[f"{symbol}_zscore_{w}"] = (close - roll_mean) / (roll_std + 1e-9)
            
            # Momentum: (Price / Mean) - 1
            df[f"{symbol}_momentum_{w}"] = (close / (roll_mean + 1e-9)) - 1
            
    return df

def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Lagged Features to prevent look-ahead bias."""
    print("⏳ Adding Lagged Features...")
    df = df.copy()
    
    lags = [1, 3, 6]
    
    # Features to lag
    cols_to_lag = []
    for symbol in ["btc", "eth"]:
        cols_to_lag.extend([
            f"{symbol}_close", 
            f"{symbol}_volume", 
            f"{symbol}_rsi_14", 
            f"{symbol}_macd",
            f"{symbol}_atr_14"
        ])
    
    # Also lag returns if they exist (calculated in previous step or here)
    # Let's calculate returns here first to be safe
    for symbol in ["btc", "eth"]:
        df[f"{symbol}_ret"] = np.log(df[f"{symbol}_close"] / df[f"{symbol}_close"].shift(1))
        cols_to_lag.append(f"{symbol}_ret")

    for col in cols_to_lag:
        if col not in df.columns:
            continue
            
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
    return df

def engineer_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer Sentiment Features."""
    print("🧠 Engineering Sentiment...")
    df = df.copy()
    
    # Fill missing sentiment with 0
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0)
    df["sentiment_count"] = df["sentiment_count"].fillna(0)
    
    # Rolling 24h features (assuming 1H data, window=24)
    df["sentiment_roll_mean_24h"] = df["sentiment_mean"].rolling(window=24).mean()
    
    # Weighted mean proxy: sum(mean * count) / sum(count) over window
    # We can approximate or just do rolling mean of (mean * count)
    # Let's do a simple rolling weighted average if possible, or just rolling sum of counts
    
    # Rolling sum of counts
    roll_count_sum = df["sentiment_count"].rolling(window=24).sum()
    
    # Rolling sum of (mean * count) -> total sentiment score mass
    total_sent_mass = (df["sentiment_mean"] * df["sentiment_count"]).rolling(window=24).sum()
    
    df["sentiment_roll_wmean_24h"] = total_sent_mass / (roll_count_sum + 1e-9)
    
    # Shock flag: if current sentiment mean abs > 0.5 (arbitrary threshold)
    df["sentiment_shock"] = (df["sentiment_mean"].abs() > 0.5).astype(int)
    
    return df

def export(df: pd.DataFrame) -> None:
    """Export final dataset."""
    # Drop rows with NaNs created by rolling/lagging
    # We want a clean ML dataset
    original_len = len(df)
    df = df.dropna()
    print(f"🧹 Dropped {original_len - len(df)} rows due to NaNs (rolling/lags).")
    
    print(f"💾 Saving to {OUTPUT_CSV} and {OUTPUT_PARQUET}...")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print("✅ Done.")

def main():
    try:
        df = load_data()
        
        # 1. Technical Indicators
        df = add_ta_indicators(df)
        
        # 2. Rolling Stats
        df = add_rolling_features(df)
        
        # 3. Lagged Features
        df = add_lagged_features(df)
        
        # 4. Sentiment
        df = engineer_sentiment(df)
        
        # 5. Export
        export(df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
