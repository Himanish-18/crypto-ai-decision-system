import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, log_loss
import xgboost as xgb
import joblib

# Import Proxies
from src.models.hybrid.tiny_cnn import TinyCNNProxy
from src.models.hybrid.tcn_lite import TCNLiteProxy
from src.models.hybrid.dqn_mini import DQNMiniProxy

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainHybridV5")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models" / "hybrid"
FEATURES_FILE = DATA_DIR / "features" / "features_v5_expanded.parquet"

def train_v5_pipeline():
    logger.info("🚀 Starting Hybrid v5 Training (4-Layer Stack)...")
    
    # 1. Load Data
    df = pd.read_parquet(FEATURES_FILE).dropna()
    df = df.sort_values("timestamp")
    
    # Targets (Fee Aware)
    # Fees + Slippage approx 0.05% (0.0005). We target > 0.001 (0.1%) for safety.
    df["target_pnl"] = df["btc_close"].shift(-1) / df["btc_close"] - 1.0
    df["target"] = (df["target_pnl"] > 0.001).astype(int)
    df = df.dropna()
    
    # Check Class Imbalance
    pos_rate = df["target"].mean()
    logger.info(f"🎯 Target Rate (Ret > 0.1%): {pos_rate:.2%}")
    if pos_rate < 0.1:
        logger.warning("Target class rare! Reducing threshold to 0.0006.")
        df["target"] = (df["target_pnl"] > 0.0006).astype(int)
    
    # 2. Robust 5-Fold Stacking
    # Split: 80% Development, 20% Hold-out Test
    n = len(df)
    dev_idx = int(n * 0.80)
    dev_df = df.iloc[:dev_idx].reset_index(drop=True)
    test_df = df.iloc[dev_idx:].reset_index(drop=True)
    
    logger.info(f"📊 Split: Dev={len(dev_df)}, Test={len(test_df)}")
    
    # K-Fold for OOF Generation
    kf = TimeSeriesSplit(n_splits=5)
    
    # Store OOF predictions
    oof_cnn = np.zeros(len(dev_df))
    oof_tcn = np.zeros(len(dev_df))
    
    logger.info("🔄 Starting 5-Fold CV for Base Models...")
    
    fold = 0
    for train_ix, val_ix in kf.split(dev_df):
        fold += 1
        X_train, X_val = dev_df.iloc[train_ix], dev_df.iloc[val_ix]
        
        # --- CNN ---
        cnn_fold = TinyCNNProxy()
        cnn_fold.fit(X_train, target_col="target")
        
        # Predict Val
        X_val_cnn, _ = cnn_fold.create_dataset(X_val, target_col="target")
        # Align: CNN consumes window.
        # We need to fill OOF correctly. 
        # Window prediction returns len(X_val)-W. 
        # We fill the *end* of the validation indices.
        probs_cnn = cnn_fold.model.predict_proba(cnn_fold.scaler.transform(X_val_cnn))[:, 1]
        
        # Validation indices in OOF array
        # Create Dataset with window=10 means first 9 rows of val are skipped?
        # Typically TimeSeriesSplit val_idx are contiguous.
        # We assign to val_ix[window:]
        w = cnn_fold.window_size
        valid_indices = val_ix[w:]
        
        # Safety check lengths
        if len(probs_cnn) == len(valid_indices):
            oof_cnn[valid_indices] = probs_cnn
        else:
            logger.warning(f"Fold {fold}: CNN Shape Mismatch. {len(probs_cnn)} vs {len(valid_indices)}")
            
        # --- TCN ---
        tcn_fold = TCNLiteProxy()
        tcn_fold.fit(X_train, target_col="target")
        probs_tcn = tcn_fold.model.predict_proba(X_val[tcn_fold.feature_cols])[:, 1]
        
        oof_tcn[val_ix] = probs_tcn
        
        logger.info(f"   ✅ Fold {fold} Complete.")

    # 3. Train Stacker on OOF (Dev Set)
    # Filter out zeros (first fold/window gaps)
    mask = (oof_cnn > 0) & (oof_tcn > 0)
    X_stack_dev = dev_df.loc[mask, ["btc_alpha_of_imbalance", "btc_alpha_vwap_zscore", "btc_alpha_smart_money_delta"]].copy() # Reduced alpha set for simplicity
    if "btc_alpha_vol_flow_rsi" in dev_df.columns:
         X_stack_dev["btc_alpha_vol_flow_rsi"] = dev_df.loc[mask, "btc_alpha_vol_flow_rsi"]
    
    X_stack_dev["cnn_prob"] = oof_cnn[mask]
    X_stack_dev["tcn_prob"] = oof_tcn[mask]
    y_stack_dev = dev_df.loc[mask, "target"]
    
    logger.info("🚀 Layer 2: Training XGBoost Stacker on Full Dev Set OOF...")
    xgb_stacker = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1)
    xgb_stacker.fit(X_stack_dev, y_stack_dev)
    
    acc = xgb_stacker.score(X_stack_dev, y_stack_dev)
    logger.info(f"✅ Stacker Dev Accuracy: {acc:.4f}")
    
    # 4. Retrain Base Models on Full Dev Set (For Deployment)
    logger.info("🧠 Retraining Base Models on Full Dev Set...")
    final_cnn = TinyCNNProxy()
    final_cnn.fit(dev_df, target_col="target")
    
    final_tcn = TCNLiteProxy()
    final_tcn.fit(dev_df, target_col="target")
    
    # 5. Train DQN Policy
    # We use OOF predictions + targets
    logger.info("🤖 Layer 3: Training DQN-Mini Policy...")
    dqn = DQNMiniProxy()
    dqn.fit(
        dev_df.loc[mask],
        mf_scores=xgb_stacker.predict_proba(X_stack_dev)[:, 1],
        cnn_scores=oof_cnn[mask],
        tcn_scores=oof_tcn[mask],
        targets_pnl=dev_df.loc[mask, "target_pnl"].values
    )
    
    # --- FINAL: Save All Models ---
    logger.info("💾 Saving Hybrid v5 Models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    final_cnn.save(MODELS_DIR / "tiny_cnn_v2.h5") 
    final_tcn.save(MODELS_DIR / "tcn_lite_v2.h5") 
    dqn.save(MODELS_DIR / "dqn_mini_v2.pt")
    joblib.dump(xgb_stacker, MODELS_DIR / "hybrid_v5_xgb.bin")
    
    logger.info("🎉 DONE v5 Training (Combined).")

if __name__ == "__main__":
    train_v5_pipeline()
