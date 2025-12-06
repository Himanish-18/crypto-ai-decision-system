import pandas as pd
import numpy as np
import logging
import pickle
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import sys

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Imports
from src.features.alpha_signals import AlphaSignals
from src.risk_engine.regime_filter import RegimeFilter
from src.models.multifactor_model import MultiFactorModel

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_regime")

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_mega_alpha.parquet"
MODEL_OUT = DATA_DIR / "models" / "multifactor_model.pkl"

def main():
    logger.info("🚀 Starting Regime-Specific Training Pipeline...")
    
    # 1. Load Data
    if not FEATURES_FILE.exists():
        logger.error(f"Features file {FEATURES_FILE} not found!")
        return
        
    df = pd.read_parquet(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 2. Update Alphas (Compute New Requested Features)
    logger.info("🔧 Computing New Alpha Features...")
    alpha_eng = AlphaSignals()
    # We must ensure we compute for BTC (and ETH if present)
    df = alpha_eng.compute_all(df, "btc")
    if "eth_close" in df.columns:
        df = alpha_eng.compute_all(df, "eth")
        
    # 3. Label Regimes (Refined Logic)
    logger.info("🏷️ Refining Regime Labels...")
    rf = RegimeFilter()
    # Apply new heuristic labels
    labels_df = rf.fit_predict_and_save(df, "btc")
    df["regime"] = labels_df["regime"]
    
    logger.info("Regime Distribution:")
    print(df["regime"].value_counts())
    
    # 4. Train MultiFactor Model
    # Split Train/Test for Evaluation
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    logger.info(f"Training on {len(train_df)} rows. Testing on {len(test_df)} rows.")
    
    mf_model = MultiFactorModel()
    mf_model.train(train_df, "y_direction_up")
    
    # 5. Evaluate Per Regime
    logger.info("📊 Evaluating Performance per Regime...")
    test_df["score"] = mf_model.predict_composite_score(test_df)
    
    regimes = test_df["regime"].unique()
    metrics = {}
    
    print("\n--- PERFORMANCE SORECARD ---")
    for r in regimes:
        sub = test_df[test_df["regime"] == r]
        if len(sub) < 10: continue
        
        y_true = sub["y_direction_up"]
        y_score = sub["score"]
        y_pred = (y_score > 0.55).astype(int)
        
        auc = roc_auc_score(y_true, y_score) if len(y_true.unique()) > 1 else 0.5
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        metrics[r] = {"auc": auc, "precision": prec, "recall": rec, "count": len(sub)}
        print(f"Regime: {r:<15} | AUC: {auc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Count: {len(sub)}")
        
    # 6. Save
    mf_model.save(MODEL_OUT)
    logger.info(f"💾 Model saved to {MODEL_OUT}")
    
    # Save Feature Importance (High Vol)
    # Extract feature importance from the 'crisis' model's XGBoost/LGB if possible
    # This is tricky as AlphaEnsemble wraps them. 
    # Validating improvement:
    # We will generate the report in a separate step or print it here.
    
    # Save metrics to json for report generation
    import json
    with open(DATA_DIR / "models" / "regime_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
