import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("alpha_diag")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "features" / "alpha_features.parquet"
LABELS_PATH = DATA_DIR / "features" / "regime_labels.parquet"


def run_diagnostic():
    logger.info("🔍 Starting Alpha Sources Diagnostic (OrderFlow & Sentiment)...")

    # 1. Load Data
    df_features = pd.read_parquet(FEATURES_PATH)
    if "regime" not in df_features.columns:
        df_labels = pd.read_parquet(LABELS_PATH)
        df = pd.merge(
            df_features, df_labels[["timestamp", "regime"]], on="timestamp", how="left"
        )
    else:
        df = df_features

    df = df.dropna(subset=["y_direction_up"])

    logger.info(f"Loaded {len(df)} samples.")

    # 2. Define New Feature Groups
    of_features = [c for c in df.columns if "_of_" in c]
    sent_features = [c for c in df.columns if "sentiment_" in c]
    new_alphas = of_features + sent_features

    logger.info(
        f"Targeting {len(of_features)} Order Flow and {len(sent_features)} Sentiment features."
    )

    # 3. Analyze per Regime
    regimes = ["Normal", "High Volatility", "Macro Event"]

    for regime in regimes:
        logger.info(f"\n--- Regime: {regime} ---")

        # Filter (Macro/HighVol are explicit. Normal is implicit or explicitly valid)
        if regime == "Normal":
            # Exclude crisis
            subset = df[
                ~df["regime"].isin(["High Volatility", "Low Liquidity", "Macro Event"])
            ].copy()
        else:
            subset = df[df["regime"] == regime].copy()

        if subset.empty:
            logger.warning("Empty subset.")
            continue

        logger.info(f"Samples: {len(subset)}")

        # Train Proxy
        exclude = [
            "timestamp",
            "y_direction_up",
            "btc_ret_fwd_1",
            "regime",
            "is_shock",
            "is_uptrend",
            "symbol",
        ]
        features = [
            c
            for c in subset.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(subset[c])
        ]

        X = subset[features]
        y = subset["y_direction_up"]

        split = int(len(subset) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        logger.info(f"AUC: {auc:.4f}")

        # Importance
        imp = pd.Series(model.feature_importances_, index=features).sort_values(
            ascending=False
        )

        # Check specific new features rank
        logger.info("Top 10 Features:")
        logger.info(imp.head(10))

        logger.info("\nNew Alpha Rankings:")
        for f in new_alphas:
            if f in imp:
                rank = imp.index.get_loc(f) + 1
                score = imp[f]
                if rank <= 50:  # Only show if relevant
                    logger.info(f"  {f}: Rank {rank} ({score:.4f})")


if __name__ == "__main__":
    run_diagnostic()
