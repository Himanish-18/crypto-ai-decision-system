import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - LABELER - %(message)s")
logger = logging.getLogger("feature_labeler")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUFFER_DIR = PROJECT_ROOT / "data" / "live_buffer"
TRAINING_DIR = PROJECT_ROOT / "data" / "training_ready"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

class FeatureLabeler:
    """
    Consumes Raw Stream Buffers -> Produces Labeled Training Data.
    """
    def __init__(self):
        pass
        
    def process_daily_buffer(self):
        logger.info("🏷️ Starting Labeling Job...")
        
        # 1. Load all buffer files
        files = sorted(list(BUFFER_DIR.glob("stream_*.parquet")))
        if not files:
            logger.info("No buffer files to process.")
            return
            
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception:
                continue
                
        if not dfs: return

        raw_df = pd.concat(dfs).sort_values("timestamp")
        
        # 2. Resample to 1m Candles
        raw_df = raw_df.set_index("timestamp")
        ohlc = raw_df["global_price"].resample("1min").ohlc()
        ohlc.columns = ["btc_open", "btc_high", "btc_low", "btc_close"] # Assuming BTC global
        
        # 3. Labeling (Forward Returns)
        # Target: Return 5m into future
        ohlc["target_ret_5m"] = ohlc["btc_close"].shift(-5) / ohlc["btc_close"] - 1
        ohlc["target_ret_1m"] = ohlc["btc_close"].shift(-1) / ohlc["btc_close"] - 1
        
        # Volatility Class
        # ATR-like (High-Low) / Close
        ohlc["vol_1m"] = (ohlc["btc_high"] - ohlc["btc_low"]) / ohlc["btc_close"]
        ohlc["vol_state"] = pd.qcut(ohlc["vol_1m"], 3, labels=["LOW", "MED", "HIGH"])
        
        # Drop NaNs (last 5 mins)
        labeled_df = ohlc.dropna()
        
        # Save
        timestamp = pd.Timestamp.now().strftime("%Y%m%d")
        out_path = TRAINING_DIR / f"labeled_{timestamp}.parquet"
        labeled_df.to_parquet(out_path)
        logger.info(f"✅ Saved {len(labeled_df)} labeled samples to {out_path}")
        
        # Cleanup processed buffers (in real system, maybe archive instead of delete)
        # for f in files:
        #     os.remove(f)

if __name__ == "__main__":
    labeler = FeatureLabeler()
    labeler.process_daily_buffer()
