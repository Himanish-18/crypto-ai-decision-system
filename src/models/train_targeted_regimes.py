import logging
from pathlib import Path

import joblib
import pandas as pd

from src.models.multifactor_model import MultiFactorModel

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_sideways_highvol")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "features" / "alpha_features.parquet"
LABELS_PATH = DATA_DIR / "features" / "regime_labels.parquet"
MODEL_PATH = DATA_DIR / "models" / "multifactor_model.pkl"


def train_targeted_regimes():
    logger.info("🚀 Starting Targeted Regime Training (Sideways & HighVol)...")

    # 1. Load Data
    df_features = pd.read_parquet(FEATURES_PATH)
    df_labels = pd.read_parquet(LABELS_PATH)

    # Merge
    if "regime" not in df_features.columns:
        df = pd.merge(
            df_features, df_labels[["timestamp", "regime"]], on="timestamp", how="left"
        )
        logger.info(
            f"Merged features {df_features.shape} with labels {df_labels.shape}. Result: {df.shape}"
        )
    else:
        df = df_features
        logger.info(f"Features already contain regime: {df.shape}")

    logger.info(
        f"Regime counts before dropna:\n{df['regime'].value_counts(dropna=False)}"
    )

    df = df.dropna(subset=["y_direction_up"])
    logger.info(f"Shape after dropna target: {df.shape}")
    logger.info(
        f"Regime counts after dropna:\n{df['regime'].value_counts(dropna=False)}"
    )

    # 2. Add Alpha Features Mask (Important for cleaning)
    # The MF Model might want specific cols, but AlphaEnsemble usually takes all numeric.
    # New features are already in DF.

    # 3. Load Existing Model (to preserve other regimes if any) or Train New?
    # User said "Retrain regime-specific models... and update MultiFactorModel".
    # MultiFactorModel wrapper usually trains everything.
    # Let's instantiate a new one or load existing.

    if MODEL_PATH.exists():
        import pickle

        with open(MODEL_PATH, "rb") as f:
            mf_model = pickle.load(f)
        logger.info("Loaded existing MultiFactorModel.")
    else:
        mf_model = MultiFactorModel()
        logger.info("Created new MultiFactorModel.")

    # 4. targeted Training
    # The current `train` method in MF Model trains "Normal" (includes Sideways) and "Crisis" (HighVol).
    # We need to ensure "Sideways" gets special treatment or is part of a newly defined cluster?
    # User asked for "Sideways" and "High Volatility" specifically.
    # MF Model currently groups:
    #   Normal Cluster: Bull, Bear, Sideways
    #   Crisis Cluster: HighVol, LowLiq, Macro
    # This grouping dilutes Sideways performance.

    # UPGRADE: Refactor MF Model training to separate Sideways?
    # Or just Retrain the 'Normal' cluster (which has Sideways) and 'Crisis' cluster (HighVol) with NEW features.
    # Since new features are specific to these regimes, retraining the CLUSTERS should capture the value.
    # Let's run the standard `train` method, which will use the new features.

    mf_model.train(df, target_col="y_direction_up")

    # 5. Save
    with open(MODEL_PATH, "wb") as f:
        import pickle

        pickle.dump(mf_model, f)

    logger.info(f"✅ Updated MultiFactorModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_targeted_regimes()
