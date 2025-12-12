import pandas as pd
import numpy as np
import joblib
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")

DATA_PATH = "data/features/features_1H_mega_alpha.parquet"
MODEL_PATH = "data/models/multifactor_model_v3.pkl"
FEAT_PATH = "data/models/training_features.json"

# Core Features computable in Live Loop
# We map generic names (in main.py) to these specific names if needed, 
# or just train on generic names if we rename in script.
# The parquet has 'btc_' prefixes. We will train on these but expects inputs to match.
# In main.py, we will rename 'close' to 'btc_close' before inference.

FEATURES = [
    # Top Fundamental
    "btc_close", "btc_volume",
    
    # Returns
    "btc_ret", 
    
    # Technicals
    "btc_rsi_14", 
    "btc_macd", 
    "btc_macd_signal",
    "btc_macd_diff",
    "btc_bb_width", 
    "btc_atr_14",
    
    # Rolling Stats (Volatility & Mean Rev)
    "btc_roll_std_20", 
    "btc_zscore_20",
    "btc_momentum_20",
    
    # Lags (Short term context)
    "btc_ret_lag_1",
    "btc_close_lag_1",
    "btc_volume_lag_1",
]

def train():
    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Create Target: Next candle return > 0 ?
    # btc_ret_fwd_1 exists in csv header, let's check if in parquet.
    # Or just calc simple next return.
    if "btc_ret_fwd_1" in df.columns:
        df["target"] = (df["btc_ret_fwd_1"] > 0).astype(int)
    else:
        df["target"] = ((df["btc_close"].shift(-1) - df["btc_close"]) > 0).astype(int)
    
    # Filter
    df = df.dropna(subset=FEATURES + ["target"])
    
    X = df[FEATURES]
    y = df["target"]
    
    logger.info(f"Training on {len(X)} samples with {len(FEATURES)} features.")
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Base Model
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=7, 
        min_samples_leaf=20, 
        random_state=42, 
        n_jobs=-1
    )
    
    # Calibration
    logger.info("Calibrating model...")
    calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=3)
    calibrated.fit(X_train, y_train)
    
    # Validate
    probs = calibrated.predict_proba(X_test)
    preds = (probs[:, 1] > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, preds))
    
    # Save
    logger.info(f"Saving model to {MODEL_PATH}...")
    joblib.dump(calibrated, MODEL_PATH)
    
    # Save Feature List
    logger.info(f"Saving feature list to {FEAT_PATH}...")
    with open(FEAT_PATH, "w") as f:
        json.dump(FEATURES, f)
        
    logger.info("✅ Retraining Complete.")

if __name__ == "__main__":
    train()
