import pandas as pd
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"
MODELS_DIR = DATA_DIR / "models"
MODEL_PATH = MODELS_DIR / "model_xgb_v1.pkl"
SCALER_PATH = MODELS_DIR / "scaler_v1.pkl"

def debug_signals():
    print("Loading artifacts...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    df = pd.read_parquet(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Test set
    n = len(df)
    val_end = int(n * 0.85)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    print(f"Test set size: {len(test_df)}")
    
    # Prepare features
    exclude_cols = ["timestamp", "y_direction_up", "btc_ret_fwd_1"]
    feature_cols = [c for c in test_df.columns if c not in exclude_cols]
    
    X_test = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    probs = model.predict_proba(X_test_scaled)[:, 1]
    test_df["y_prob"] = probs
    
    print("\n--- Probability Distribution ---")
    print(test_df["y_prob"].describe())
    print(f"Count > 0.53: {(test_df['y_prob'] > 0.53).sum()}")
    print(f"Count > 0.50: {(test_df['y_prob'] > 0.50).sum()}")
    
    # Filters
    test_df["atr_pct"] = test_df["btc_atr_14"].expanding().rank(pct=True)
    test_df["is_shock"] = ((test_df["sentiment_shock"] == 1) | (test_df["atr_pct"] > 0.95)).astype(int)
    test_df["is_uptrend"] = ((test_df["btc_rsi_14"] > 50) | (test_df["btc_macd"] > 0)).astype(int)
    
    print("\n--- Filter Counts ---")
    print(f"Shock Events: {test_df['is_shock'].sum()}")
    print(f"Uptrend Events: {test_df['is_uptrend'].sum()}")
    
    # Combined
    test_df["entry_signal"] = (
        (test_df["y_prob"] > 0.53) &
        (test_df["is_shock"] == 0) &
        (test_df["is_uptrend"] == 1)
    ).astype(int)
    
    print(f"\nTotal Entry Signals: {test_df['entry_signal'].sum()}")
    
    if test_df['entry_signal'].sum() > 0:
        print("\nSample Signals:")
        print(test_df[test_df["entry_signal"] == 1][["timestamp", "y_prob", "btc_close"]].head())

if __name__ == "__main__":
    debug_signals()
