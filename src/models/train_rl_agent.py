import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_rl_agent_sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_rl_proxy():
    """
    Train a 'Policy Network' using classification.
    Target: 1 if (Close[t+1] - Close[t])/Close[t] > Taker_Fee * 2 (Profit Target)
    This approximates an RL agent optimizing for 1-step immediate reward > cost.
    """
    logger.info("🚀 Starting RL Policy Training (RF Proxy)...")

    if not FEATURES_FILE.exists():
        logger.error("Features file missing.")
        return

    df = pd.read_parquet(FEATURES_FILE)
    df = df.sort_values("timestamp")

    # 1. Define Reward / Target
    # Reward = (Next_Close - Current_Close) / Current_Close
    df["fwd_ret"] = df["btc_close"].shift(-1) / df["btc_close"] - 1

    # Action Target: Buy (1) if Return > Transaction Cost Barrier (0.1%), else Hold (0)
    # We treat Short as separate or simplify to Long/Neutral for now.
    # Let's do Long-Only RL Agent since MF is often Long-biased.
    FEE_BARRIER = 0.001
    df["target_action"] = (df["fwd_ret"] > FEE_BARRIER).astype(int)

    # Drop NaNs (last row)
    df = df.dropna()

    # 2. Features
    exclude = [
        "timestamp",
        "y_direction_up",
        "btc_ret_fwd_1",
        "fwd_ret",
        "target_action",
        "regime",
    ]
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
    ]

    # Split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train = train_df[feature_cols]
    y_train = train_df["target_action"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_action"]

    # 3. Train Policy
    # Using Random Forest as Policy Network (Non-linear, robust)
    policy_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Shallow trees to prevent overfitting, akin to simple policy
        min_samples_leaf=50,
        random_state=42,
        class_weight="balanced",  # Handle imbalances
        n_jobs=-1,
    )

    logger.info("🧠 Training Policy Model...")
    policy_model.fit(X_train, y_train)

    # 4. Evaluate
    logger.info("🧪 Evaluating Policy...")
    y_pred = policy_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"\n{report}")

    # 5. Save
    save_path = MODELS_DIR / "rl_agent_rf.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(policy_model, f)
    logger.info(f"💾 Policy Model saved to {save_path}")

    # Save Feature List
    with open(MODELS_DIR / "rl_features.json", "w") as f:
        json.dump(feature_cols, f)


if __name__ == "__main__":
    train_rl_proxy()
