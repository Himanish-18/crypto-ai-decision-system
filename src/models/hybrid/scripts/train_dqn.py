import logging
from pathlib import Path

import pandas as pd

from src.models.hybrid.dqn_mini import DQNMiniProxy
from src.models.hybrid.tcn_lite import TCNLiteProxy
from src.models.hybrid.tiny_cnn import TinyCNNProxy

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainDQN")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models" / "hybrid"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"


def train():
    logger.info("🤖 Starting DQN-Mini Training...")
    df = pd.read_parquet(FEATURES_FILE).dropna()
    ret = df["btc_close"].shift(-1) / df["btc_close"] - 1
    df["target_pnl"] = ret
    df = df.dropna()

    # Load Base Models
    try:
        cnn = TinyCNNProxy.load(MODELS_DIR / "tiny_cnn_weights.pth")
        tcn = TCNLiteProxy.load(MODELS_DIR / "tcn_lite_weights.pth")
    except Exception as e:
        logger.error(f"Base models missing. Train CNN/TCN first. Error: {e}")
        return

    # Generate Meta Features
    # Split for Meta-Training (Last 30%)
    split = int(len(df) * 0.7)
    df_meta = df.iloc[split:].copy()

    # Generate Scores
    # TCN (Batch)
    tcn_probs = tcn.model.predict_proba(df_meta[tcn.feature_cols])[:, 1]

    # CNN (Batch - Approximation)
    # MLP expects scaled window.
    # For speed in this script, we might re-use simple features or do proper windowing.
    # Proper windowing is slow in Python loop.
    # We will use TCN score as CNN proxy for now or skip CNN input to DQN?
    # Or just use TCN score twice?
    # Let's generate properly if possible.
    # Actually, TinyCNN has `create_dataset`.
    X_meta, _ = cnn.create_dataset(df_meta, target_col="target_pnl")
    # Align df_meta
    df_meta_aligned = df_meta.iloc[cnn.window_size :]
    tcn_probs_aligned = tcn_probs[cnn.window_size :]

    X_meta_scaled = cnn.scaler.transform(X_meta)
    cnn_probs = cnn.model.predict_proba(X_meta_scaled)[:, 1]

    dqn = DQNMiniProxy()
    dqn.fit(
        df_meta_aligned,
        mf_scores=tcn_probs_aligned,  # Proxy for MF
        cnn_scores=cnn_probs,
        tcn_scores=tcn_probs_aligned,
        targets_pnl=df_meta_aligned["target_pnl"].values,
    )
    dqn.save(MODELS_DIR / "dqn_mini_rl.pth")


if __name__ == "__main__":
    train()
