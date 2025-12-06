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
OUTPUT_CSV = FEATURES_DIR / "features_1H_mega_alpha.csv"
OUTPUT_PARQUET = FEATURES_DIR / "features_1H_mega_alpha.parquet"

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
        if f"{symbol}_close" not in df.columns:
            continue
            
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
        if f"{symbol}_close" not in df.columns:
            continue
            
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
        if f"{symbol}_close" not in df.columns:
            continue
            
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
        if f"{symbol}_close" not in df.columns:
            continue
            
        df[f"{symbol}_ret"] = np.log(df[f"{symbol}_close"] / df[f"{symbol}_close"].shift(1))
        cols_to_lag.append(f"{symbol}_ret")

    for col in cols_to_lag:
        if col not in df.columns:
            continue
            
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
    return df

def engineer_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer Sentiment Features (Upgraded)."""
    print("🧠 Engineering Sentiment...")
    df = df.copy()
    
    # Fill missing
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0)
    df["sentiment_count"] = df["sentiment_count"].fillna(0)
    
    # Base Rolling features
    windows = [24, 72]
    
    for w in windows:
        # Simple rolling mean
        df[f"sentiment_roll_mean_{w}h"] = df["sentiment_mean"].rolling(window=w).mean()
        
        # Weighted Mean proxy: Sum(mean*count) / Sum(count)
        # Total mass of sentiment "volume"
        sent_mass = (df["sentiment_mean"] * df["sentiment_count"]).rolling(window=w).sum()
        count_sum = df["sentiment_count"].rolling(window=w).sum()
        df[f"sentiment_roll_wmean_{w}h"] = sent_mass / (count_sum + 1e-9)
        
    # Shock Flags (Z-Score > 2)
    # Using 24h rolling stats for baseline
    roll_mean = df["sentiment_mean"].rolling(24).mean()
    roll_std = df["sentiment_mean"].rolling(24).std()
    z_score = (df["sentiment_mean"] - roll_mean) / (roll_std + 1e-9)
    df["sentiment_shock"] = (z_score.abs() > 2).astype(int)
    
    # Sentiment-Price Divergence
    # Price Trend (24h) vs Sentiment Trend (24h)
    # Divergence = Price Up & Sentiment Down (Bearish), or Price Down & Sentiment Up (Bullish)
    price_change = df["btc_close"].pct_change(24)
    sent_change = df["sentiment_roll_wmean_24h"].diff(24)
    
    # 1 if Bullish Div, -1 if Bearish Div, 0 otherwise
    df["sentiment_divergence"] = np.where(
        (price_change < 0) & (sent_change > 0), 1,
        np.where((price_change > 0) & (sent_change < 0), -1, 0)
    )
    
    return df

def main():
    try:
        df = load_data()
        
        # 1. Technical Indicators
        df = add_ta_indicators(df)
        
        # 2. Rolling Stats
        df = add_rolling_features(df)
        
        # 3. Lagged Features
        df = add_lagged_features(df)
        
        # 4. Sentiment (Upgraded)
        df = engineer_sentiment(df)

        # 5. Advanced Alpha Signals (New - Expanded)
        from src.features.alpha_signals import AlphaSignals
        alpha_signals = AlphaSignals()
        df = alpha_signals.compute_all(df, symbol="btc")
        if "eth_close" in df.columns:
             df = alpha_signals.compute_all(df, symbol="eth")

        # 6. Order Flow Features (New)
        print("🌊 Calculating Order Flow & Microstructure features...")
        from src.features.orderflow_features import OrderFlowFeatures
        of_feats = OrderFlowFeatures()
        df = of_feats.compute_all(df, symbol="btc")
        if "eth_close" in df.columns:
             df = of_feats.compute_all(df, symbol="eth")

        # 7. Order Book Features (From Parquet)
        ob_path = FEATURES_DIR / "orderbook_features.parquet"
        if ob_path.exists():
             print(f"📖 Merging OrderBook Features from {ob_path}...")
             df_ob = pd.read_parquet(ob_path)
             df_ob["timestamp"] = pd.to_datetime(df_ob["timestamp"], utc=True)
             df_ob = df_ob.sort_values("timestamp")
             
             df = pd.merge_asof(
                 df.sort_values("timestamp"),
                 df_ob.sort_values("timestamp"),
                 on="timestamp",
                 direction="backward",
                 tolerance=pd.Timedelta("3h") 
             )
             # Note: merge_asof output might have NaNs if out of tolerance.
             # OB data is likely sparse/new compared to historical candles.
             # FILL NaNs to preserve candle history!
             for col in df_ob.columns:
                 if col != "timestamp":
                     df[col] = df[col].fillna(0) # or ffill? 0 is safer for "No Info"
        else:
             print("⚠️ OrderBook Features not found. Creating placeholders.")
             for col in ["spread_pct", "obi", "impact_cost", "liquidity_ratio"]:
                 df[col] = 0.0 # Use 0.0 instead of NaN to avoid dropna later

        # We need to fill NaNs first before fitting regime model?
        # RegimeFilter handles internal fitting.
        # Check for NaNs/Infs
        df = df.replace([np.inf, -np.inf], np.nan)
        df_clean_for_regime = df.dropna()
        
        if not df_clean_for_regime.empty:
            from src.risk_engine.regime_filter import RegimeFilter
            regime_filter = RegimeFilter()
            # This saves 'regime_labels.parquet' internally
            regime_filter.fit_predict_and_save(df_clean_for_regime, symbol="btc")
        
        # 8. Export
        # Update output path for this specific request
        OUTPUT_ALPHA = FEATURES_DIR / "alpha_features.parquet"
        
        # Clean final DF
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"💾 Saving to {OUTPUT_ALPHA}...")
        df.to_parquet(OUTPUT_ALPHA, index=False)
        
        # Also save to legacy paths for compatibility
        df.to_csv(OUTPUT_CSV, index=False)
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print("✅ Done.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
