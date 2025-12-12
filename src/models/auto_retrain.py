import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import (add_lagged_features,
                                         add_rolling_features,
                                         add_ta_indicators, engineer_sentiment)
from src.ingest.live_market_data import LiveMarketData
from src.models.train_model import (evaluate_model, prepare_features,
                                    save_artifacts, split_data, train_xgboost)

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("auto_retrain")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
CURRENT_MODEL = MODELS_DIR / "model_xgb_v1.pkl"
BACKUP_DIR = MODELS_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


class RetrainingPipeline:
    def __init__(self):
        self.market_data_btc = LiveMarketData(symbol="BTC/USDT", timeframe="1h")
        self.market_data_eth = LiveMarketData(symbol="ETH/USDT", timeframe="1h")

    def fetch_recent_data(self, days=30) -> pd.DataFrame:
        """Fetch recent market data for retraining."""
        logger.info(f"📥 Fetching last {days} days of data...")
        limit = days * 24

        df_btc = self.market_data_btc.fetch_candles(limit=limit)
        df_eth = self.market_data_eth.fetch_candles(limit=limit)

        if df_btc is None or df_eth is None:
            raise ValueError("Failed to fetch market data")

        df = pd.merge(df_btc, df_eth, on="timestamp", how="inner")
        return df

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for training."""
        logger.info("🧠 Generating features...")
        if "sentiment_mean" not in df.columns:
            df["sentiment_mean"] = 0.0
            df["sentiment_count"] = 0.0

        df = add_ta_indicators(df)
        df = add_rolling_features(df)
        df = add_lagged_features(df)
        df = engineer_sentiment(df)

        # Add Target
        df["btc_ret_fwd_1"] = np.log(df["btc_close"].shift(-1) / df["btc_close"])
        df["y_direction_up"] = (df["btc_ret_fwd_1"] > 0).astype(int)

        df = df.dropna().reset_index(drop=True)
        return df

    def train_and_evaluate(self, df: pd.DataFrame):
        """Train new model and compare with current."""
        logger.info("🚀 Training Candidate Model...")

        train_df, val_df, test_df = split_data(df)
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols = (
            prepare_features(train_df, val_df, test_df)
        )

        model = train_xgboost(X_train, y_train, X_val, y_val)

        # Evaluate Candidate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_prob, "candidate_xgb")

        logger.info(
            f"Candidate Metrics: F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}"
        )

        return model, scaler, metrics

    def promote_model(self, model, scaler, metrics):
        """Promote candidate model to production."""
        logger.info("🏆 Promoting Candidate Model to Production...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Backup Current
        if CURRENT_MODEL.exists():
            backup_path = BACKUP_DIR / f"model_xgb_v1_{timestamp}.pkl"
            shutil.copy(CURRENT_MODEL, backup_path)
            logger.info(f"Backed up current model to {backup_path}")

        # Save New Model (Overwrite v1 for simplicity, or increment version)
        # Using save_artifacts from train_model.py which saves as v1
        # We might want to modify save_artifacts to accept version or path
        # For now, let's manually save to overwrite

        import pickle

        with open(CURRENT_MODEL, "wb") as f:
            pickle.dump(model, f)

        scaler_path = MODELS_DIR / "scaler_v1.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        logger.info("✅ Model Promoted Successfully.")

    def run(self):
        try:
            # 1. Fetch Data
            df = self.fetch_recent_data(days=30)

            # 2. Features
            df = self.generate_features(df)

            # 3. Train
            model, scaler, metrics = self.train_and_evaluate(df)

            # 4. Compare (Simplified: If F1 > 0.55, promote)
            # In real scenario, load current model and evaluate on same test set
            if metrics["f1_score"] > 0.55:
                self.promote_model(model, scaler, metrics)
            else:
                logger.info(
                    f"Candidate F1 ({metrics['f1_score']:.4f}) below threshold. No promotion."
                )

        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)


if __name__ == "__main__":
    pipeline = RetrainingPipeline()
    pipeline.run()
