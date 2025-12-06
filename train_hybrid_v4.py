import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.models.hybrid.tiny_cnn import TinyCNNProxy
from src.models.hybrid.tcn_lite import TCNLiteProxy
from src.models.hybrid.dqn_mini import DQNMiniProxy

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainHybridV4")

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models" / "hybrid"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"

def train_hybrid_system():
    logger.info("🚀 Starting Hybrid v4 Training Pipeline...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    if not FEATURES_FILE.exists():
        logger.error(f"❌ Features file not found: {FEATURES_FILE}")
        return
        
    df = pd.read_parquet(FEATURES_FILE)
    df = df.dropna()
    logger.info(f"📊 Loaded {len(df)} samples.")
    
    # Add Targets
    # Target 1: Simple Direction (Up/Down) for CNN/TCN
    df["target"] = (df["btc_close"].shift(-1) > df["btc_close"]).astype(int)
    # Target 2: PnL for DQN (Return - Fee)
    # Approx 6 basis points fee hurdle
    ret = df["btc_close"].shift(-1) / df["btc_close"] - 1
    df["target_pnl"] = ret # Raw return, let DQN learn the hurdle
    
    df = df.dropna()
    
    # ---------------------------
    # 2. Train Tiny-CNN
    # ---------------------------
    cnn = TinyCNNProxy()
    cnn.fit(df, target_col="target")
    cnn.save(MODELS_DIR / "tiny_cnn_weights.pth")
    
    # Generate CNN Scores for Meta-Training
    # Warning: Using training data for predictions introduces bias in stacking.
    # Ideally should use CV. For proxy speed, we use OOB or simple inference.
    # MLP doesn't have OOB. We'll reuse predict_proba on train set (risk of overfitting DQN).
    # Correct approach: Split Train/Meta-Train.
    # Let's split 70/30. 70 for Base Models, 30 for DQN.
    
    split = int(len(df) * 0.7)
    df_base = df.iloc[:split]
    df_meta = df.iloc[split:]
    
    logger.info("🔄 Retraining Base Models on Split (70%) for Meta-Stacking...")
    
    # Re-train CNN on Base
    cnn.fit(df_base, target_col="target")
    cnn_scores_meta = []
    
    # Predict on Meta (iterative to simulate window)
    # Slow loop? Vectorized prepare?
    # TinyCNN needs windows. We can optimize.
    # For now, quick loop on a subset if needed, or just batch.
    # Batch predict: create all windows first.
    X_meta, _ = cnn.create_dataset(df_meta, target_col="target")
    X_meta_scaled = cnn.scaler.transform(X_meta)
    cnn_scores_meta = cnn.model.predict_proba(X_meta_scaled)[:, 1]
    
    # Align df_meta to X_meta (TinyCNN loses first 20 rows)
    df_meta_aligned = df_meta.iloc[cnn.window_size:]
    
    # ---------------------------
    # 3. Train TCN-Lite
    # ---------------------------
    tcn = TCNLiteProxy()
    tcn.fit(df_base, target_col="target")
    tcn.save(MODELS_DIR / "tcn_lite_weights.pth")
    
    tcn_probs_meta = tcn.model.predict_proba(df_meta_aligned[tcn.feature_cols])[:, 1]
    
    # ---------------------------
    # 4. Train DQN-Mini
    # ---------------------------
    # We need MF Scores too. Assume we have a placeholder or use TCN as proxy for MF for this simulation.
    # In live system, MF is XGBoost. Here we can treat TCN as the Trend signal.
    # Let's use TCN score as MF score proxy for training DQN.
    
    dqn = DQNMiniProxy()
    dqn.fit(
        df_meta_aligned, 
        mf_scores=tcn_probs_meta, # Proxy
        cnn_scores=cnn_scores_meta, 
        tcn_scores=tcn_probs_meta, 
        targets_pnl=df_meta_aligned["target_pnl"].values
    )
    dqn.save(MODELS_DIR / "dqn_mini_rl.pth")
    
    logger.info("✅ Hybrid System Training Complete.")

if __name__ == "__main__":
    train_hybrid_system()
