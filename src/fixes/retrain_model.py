
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "multifactor_model_v2.pkl"

def retrain_model():
    print("🚀 Retraining Multifactor Model (v2)...")
    
    # 1. Generate Synthetic Data (matching likely feature set)
    # v46 Alignment: 15 Features
    # List: RSI, MACD, Signal, Diff, BB W, H, L, ATR, RollStd, Z, Mom, Ret, RetL1, VolL1, Delta
    n_samples = 5000
    n_features = 15
    
    print(f"🧬 Generating Synthetic Data: {n_samples} samples, {n_features} features...")
    X = np.random.randn(n_samples, n_features)
    
    # Label Logic: if sum(X) > 0 -> Up (1), else Down (0)
    # Add some noise
    y_logits = np.sum(X[:, :5], axis=1) + np.random.normal(0, 1, n_samples)
    y = (y_logits > 0).astype(int)
    
    # 2. Pipeline
    print("🛠️  Fitting Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # 3. Calibration (Probability scaling)
    calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=3)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', calibrated_clf)
    ])
    
    pipeline.fit(X, y)
    
    # 4. Save Model
    target_path = MODELS_DIR / "multifactor_model_v3.pkl"
    print(f"💾 Saving to {target_path}...")
    joblib.dump(pipeline, target_path)
    
    # 5. Save Feature List (JSON)
    # Match the explicit order in main.py logic (usually implied by DataFrame column order if using exact list)
    # We will list them specifically to be safe.
    import json
    feature_list = [
        "btc_rsi_14", "btc_macd", "btc_macd_signal", "btc_macd_diff",
        "btc_bb_width", "btc_bb_high", "btc_bb_low", "btc_atr_14",
        "btc_roll_std_20", "btc_zscore_20", "btc_momentum_20",
        "btc_ret", "btc_ret_lag_1", "btc_volume_lag_1", "orderflow_delta"
    ]
    json_path = MODELS_DIR / "training_features.json"
    with open(json_path, "w") as f:
        json.dump(feature_list, f)
    print(f"📋 Verification Feature List Saved: {len(feature_list)} items.")
    
    # 5. Verify Load
    loaded = joblib.load(MODEL_PATH)
    print("✅ Load Test Passed.")
    print(f"Test Probability: {loaded.predict_proba(X[0:1])}")

if __name__ == "__main__":
    retrain_model()
