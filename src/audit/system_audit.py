import pandas as pd
import pickle
import json
import logging
from pathlib import Path
import numpy as np

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("audit")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.csv"
MODEL_FILE = DATA_DIR / "models" / "best_model_xgb_opt.pkl"
SCALER_FILE = DATA_DIR / "models" / "scaler_opt.pkl"
REPORT_FILE = DATA_DIR / "execution" / "strategy_optimizer" / "performance_report.json"
TRADES_FILE = DATA_DIR / "execution" / "strategy_optimizer" / "trades.csv"

def audit_features():
    logger.info("🔍 Auditing Features...")
    if not FEATURES_FILE.exists():
        logger.error(f"❌ Feature file missing: {FEATURES_FILE}")
        return False
        
    df = pd.read_csv(FEATURES_FILE)
    
    # Check Alpha Columns
    alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
    logger.info(f"Found {len(alpha_cols)} alpha columns: {alpha_cols}")
    
    if len(alpha_cols) == 0:
        logger.error("❌ No alpha columns found!")
        return False
        
    # Check for NaNs in Alphas
    nan_counts = df[alpha_cols].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"⚠️ NaNs found in alpha columns:\n{nan_counts[nan_counts > 0]}")
    else:
        logger.info("✅ No NaNs in alpha columns.")
        
    # Check Value Ranges
    for col in alpha_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        logger.info(f"  - {col}: Range=[{min_val:.4f}, {max_val:.4f}], Mean={mean_val:.4f}")
        
        if min_val == 0 and max_val == 0:
            logger.warning(f"⚠️ Column {col} is all zeros!")

    return True

def audit_model():
    logger.info("🔍 Auditing Model...")
    if not MODEL_FILE.exists():
        logger.error(f"❌ Model file missing: {MODEL_FILE}")
        return False
        
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
        
    logger.info(f"✅ Model loaded: {type(model)}")
    
    if hasattr(model, "feature_names_in_"):
        feats = model.feature_names_in_
        alpha_feats = [f for f in feats if f.startswith("alpha_")]
        logger.info(f"Model expects {len(feats)} features.")
        logger.info(f"Model uses {len(alpha_feats)} alpha features.")
        
        if len(alpha_feats) == 0:
            logger.error("❌ Model does not use alpha features!")
            return False
    else:
        logger.warning("⚠️ Model does not store feature names.")
        
    return True

def audit_strategy():
    logger.info("🔍 Auditing Strategy Results...")
    if not REPORT_FILE.exists():
        logger.error(f"❌ Report file missing: {REPORT_FILE}")
        return False
        
    with open(REPORT_FILE, "r") as f:
        metrics = json.load(f)
        
    logger.info("Performance Metrics:")
    logger.info(json.dumps(metrics, indent=2))
    
    if metrics["total_trades"] == 0:
        logger.warning("⚠️ No trades executed!")
    else:
        logger.info(f"✅ Executed {metrics['total_trades']} trades.")
        
    if metrics["total_return_pct"] <= 0:
        logger.warning("⚠️ Strategy lost money.")
    else:
        logger.info("✅ Strategy is profitable.")
        
    return True

def main():
    logger.info("🚀 Starting System Audit...")
    
    f_ok = audit_features()
    m_ok = audit_model()
    s_ok = audit_strategy()
    
    if f_ok and m_ok and s_ok:
        logger.info("✅ SYSTEM AUDIT PASSED")
    else:
        logger.error("❌ SYSTEM AUDIT FAILED")

if __name__ == "__main__":
    main()
