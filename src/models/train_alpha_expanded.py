import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.multifactor_model import MultiFactorModel

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_alpha_expanded")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "features" / "alpha_features.parquet"
LABELS_PATH = DATA_DIR / "features" / "regime_labels.parquet"
MODEL_PATH = DATA_DIR / "models" / "multifactor_model.pkl"
FEATURE_MASK_PATH = DATA_DIR / "models" / "selected_alpha_features.json"


def train_expanded_model():
    logger.info("🚀 Starting Multi-Factor Model Training (Expanded Alphas)...")

    # 1. Load Data
    if not FEATURES_PATH.exists():
        logger.error(f"Features file not found: {FEATURES_PATH}")
        return

    df_features = pd.read_parquet(FEATURES_PATH)
    logger.info(f"Loaded Features: {df_features.shape}")

    # Merge Regime Labels
    if "regime" not in df_features.columns:
        if LABELS_PATH.exists():
            df_labels = pd.read_parquet(LABELS_PATH)
            # Use merge logic from build_features if available or simple merge
            # Assuming timestamps match or we merge.
            # Ideally Labels should be re-generated if data changes, but let's assume valid.
            df = pd.merge(
                df_features,
                df_labels[["timestamp", "regime"]],
                on="timestamp",
                how="left",
            )
        else:
            logger.warning("Regime labels missing. Running RegimeFilter...")
            from src.risk_engine.regime_filter import RegimeFilter

            rf = RegimeFilter()
            df_labels = rf.fit_predict_and_save(df_features, symbol="btc")
            df = pd.merge(
                df_features,
                df_labels[["timestamp", "regime"]],
                on="timestamp",
                how="left",
            )
    else:
        df = df_features

    # Clean
    df = df.dropna(subset=["y_direction_up"])

    # 2. Train MultiFactor Model
    # This trains Normal and Crisis models
    mf_model = MultiFactorModel()
    mf_model.train(df, target_col="y_direction_up")

    # 3. Feature Selection (Top 60)
    # Extract importance from 'Normal' regime XGBoost model
    # Path: mf_model.models['normal'].models['xgb']

    selected_features = []

    try:
        if "normal" in mf_model.models:
            ensemble = mf_model.models["normal"]
            if "xgb" in ensemble.models:
                xgb_model = ensemble.models["xgb"]

                # Get Feature Names from Scaler
                if hasattr(ensemble.scaler, "feature_names_in_"):
                    feature_names = ensemble.scaler.feature_names_in_
                else:
                    # Fallback
                    exclude = [
                        "timestamp",
                        "y_direction_up",
                        "regime",
                        "symbol",
                        "btc_ret_fwd_1",
                    ]
                    feature_names = [
                        c
                        for c in df.columns
                        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
                    ]

                importances = xgb_model.feature_importances_

                # Zip and Sort
                feat_imp = pd.Series(importances, index=feature_names).sort_values(
                    ascending=False
                )

                # Select Top 60
                top_features = feat_imp.head(60).index.tolist()

                # Ensure critical signals are kept if missed (e.g. regime specific)
                # But importance should reflect that.

                selected_features = top_features

                logger.info(f"Top 5 Features:\n{feat_imp.head(5)}")
                logger.info(
                    f"Selected {len(selected_features)} features for live inference."
                )

            else:
                logger.warning("XGBoost model not found in Normal ensemble.")
        else:
            logger.warning("Normal model not found in MultiFactorModel.")

    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        # Fallback to all numeric cols
        exclude = ["timestamp", "y_direction_up", "regime", "symbol", "btc_ret_fwd_1"]
        selected_features = [
            c
            for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    # 4. Save Artifacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(mf_model, f)
    logger.info(f"✅ Model saved to {MODEL_PATH}")

    # Save Feature Mask
    with open(FEATURE_MASK_PATH, "w") as f:
        json.dump(selected_features, f, indent=2)
    logger.info(f"✅ Feature Selection Mask saved to {FEATURE_MASK_PATH}")


if __name__ == "__main__":
    train_expanded_model()
