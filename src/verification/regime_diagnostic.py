import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import xgboost as xgb

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("regime_diagnostic")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "features" / "alpha_features.parquet"
LABELS_PATH = DATA_DIR / "features" / "regime_labels.parquet"

def run_diagnostic():
    logger.info("🔍 Starting Sideways & HighVol Diagnostic...")
    
    # 1. Load Data
    if not FEATURES_PATH.exists() or not LABELS_PATH.exists():
        logger.error("Data missing. Ensure features and labels exist.")
        return

    df_features = pd.read_parquet(FEATURES_PATH)
    df_labels = pd.read_parquet(LABELS_PATH)
    
    # Merge if needed (assuming timestamp alignment)
    # Check if 'regime' is already in features?
    if "regime" not in df_features.columns:
        # Merge on timestamp
        df = pd.merge(df_features, df_labels[['timestamp', 'regime']], on='timestamp', how='left')
    else:
        df = df_features

    # Filter targets
    df = df.dropna(subset=["y_direction_up"])
    
    # 2. Analyze Regimes
    target_regimes = ["Sideways", "High Volatility"]
    
    results = []
    
    for regime in target_regimes:
        logger.info(f"--- Analyzing Regime: {regime} ---")
        subset = df[df["regime"] == regime].copy()
        
        if subset.empty:
            logger.warning(f"No samples found for {regime}")
            continue
            
        logger.info(f"Samples: {len(subset)}")
        
        # Features (exclude non-numeric)
        exclude = ["timestamp", "y_direction_up", "btc_ret_fwd_1", "regime", "is_shock", "is_uptrend"]
        feature_cols = [c for c in subset.columns if c not in exclude and np.issubdtype(subset[c].dtype, np.number)]
        
        X = subset[feature_cols]
        y = subset["y_direction_up"]
        
        # 3. Train Quick Proxy Model for Feature Importance
        # We want to see what *could* work, so we train on this subset.
        # Use simple Train/Test split
        split = int(len(subset) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        if len(X_test) < 10:
            logger.warning("Not enough data for test.")
            continue
            
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)
        
        # Metrics
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        
        logger.info(f"Baseline AUC (Proxy Model): {auc:.4f}")
        
        # Feature Importance
        importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        top_10 = importance.head(10)
        
        logger.info(f"Top 5 Features:\n{top_10.head(5)}")
        
        results.append({
            "regime": regime,
            "samples": len(subset),
            "auc": auc,
            "top_features": top_10.index.tolist()
        })
        
    logger.info("✅ Diagnostic Complete.")
    return results

if __name__ == "__main__":
    run_diagnostic()
