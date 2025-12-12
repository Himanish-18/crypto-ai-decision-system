import logging
from pathlib import Path

import pandas as pd

from src.features.alpha_signals import AlphaSignals
from src.features.build_features import (add_lagged_features,
                                         add_rolling_features,
                                         add_ta_indicators, engineer_sentiment)
from src.ingest.live_market_data import LiveMarketData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BuildFeaturesV5")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "features" / "features_v5_expanded.parquet"


def build_v5_dataset():
    logger.info("🏭 Starting Feature Generation Pipeline (v5)...")

    # 1. Load Raw History (Use existing parquet dump or fetch?)
    # Prefer loading from existing raw dump if available, else fetch.
    raw_btc_path = DATA_DIR / "historical_btc_1h.parquet"  # Approx path
    if not raw_btc_path.exists():
        # Fallback: Load current features to get raw OHLCV?
        # Or re-fetch.
        logger.warning("Raw data not found. Loading from features_1H_advanced as base.")
        base_df = pd.read_parquet(
            DATA_DIR / "features" / "features_1H_advanced.parquet"
        )
        # Ensure we have essential cols
        req = [
            "btc_open",
            "btc_high",
            "btc_low",
            "btc_close",
            "btc_volume",
            "timestamp",
        ]
        if not all(c in base_df.columns for c in req):
            logger.error("Base DF missing OHLCV")
            return
        df = base_df
    else:
        df = pd.read_parquet(raw_btc_path)

    # 2. Compute Alphas (New v5 Logic)
    logger.info("🧠 Computing v5 Alphas...")
    alpha_eng = AlphaSignals()

    # Ensure sorted
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute
    df = alpha_eng.compute_all(df, "btc")
    if "eth_close" in df.columns:
        df = alpha_eng.compute_all(df, "eth")

    # 3. Post-Processing
    # Drop NaNs created by new windows (max window ~24-50)
    df = df.dropna()

    logger.info(f"✅ Generated {len(df)} rows with {len(df.columns)} features.")

    # Save
    df.to_parquet(OUTPUT_FILE)
    logger.info(f"💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_v5_dataset()
